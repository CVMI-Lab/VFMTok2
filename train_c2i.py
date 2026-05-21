# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import random, math
import numpy as np
import pickle as pkl
import os, torch, pdb
from PIL import Image
from glob import glob
from time import time
import os.path as osp
from copy import deepcopy
from einops import rearrange
import argparse, logging, math
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from collections import OrderedDict

from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from tokenizer.vfmae.vfmae import AE_models
from commons.engine.util import create_logger
from denoise.models import Stage2ModelProtocol
from commons.engine.util import parse_configs
from denoise.transport import create_transport, Sampler
from denoise.utils.model_utils import instantiate_from_config
from commons.engine.distributed import init_distributed_mode
from denoise.utils.optim_utils import build_optimizer, build_scheduler
from commons.data.augmentation import random_crop_arr, center_crop_arr
from commons.engine.ema import update_ema, requires_grad
from commons.data.imagenet_lmdb import ImageNetLmdbDataset as ImageNetDataset
from commons.engine.misc import (is_main_process, get_rank, get_world_size, concat_all_gather, all_reduce_mean, cleanup)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    (   model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    if model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))

    ckpt_every = args.ckpt_every
    sample_every = int(training_cfg.get("sample_every", 10_000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the RAE encoder.")

    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()

    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")

    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if is_main_process():
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32
    
    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    if is_main_process():
        os.makedirs(args.results_dir, exist_ok=True)
        log_dir = osp.join(args.results_dir, '../logs')
        snapshot_dir = args.results_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        logger = create_logger(log_dir)
        
        logger.info(f"Experiment directory created at {args.results_dir}")
  
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)

    transformer_config = OmegaConf.load(args.transformer_config_file)
    vfmae = AE_models[args.ae_model](
        embed_dim=args.embed_dim,
        transformer_config=transformer_config,
        z_channels=args.z_channels,)

    vfmae.eval()
    checkpoint = torch.load(args.ae_ckpt, map_location="cpu")

    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missings, unexpected = vfmae.load_state_dict(model_weight, strict=False)
    assert sum(['backbone' in p for p in missings]) == len(missings), 'Please check the state_dict since necessary parameters are missed.'
    
    del checkpoint
    #* Set arguments generation params
    vfmae.to(device)
    vfmae.eval()

    assert osp.exists(args.stats_file)
    stats = torch.load(args.stats_file, map_location='cpu')
    mean, var = torch.tensor(stats['mean'], device=device), torch.tensor(stats['var'], device=device)

    H = W = int(math.sqrt(mean.size(1)))
    mean = rearrange(mean, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)
    var = rearrange(var, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    ema.eval()
    requires_grad(ema, False)

    opt_state = None
    sched_state = None
    train_steps = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    if args.compile:
        try:
            vfmae.encode_transformer.backbone = torch.compile(vfmae.encode_transformer.backbone,)
        except:
            print('VFMAE ENCODE compile meets error, falling back to no compile')
        try:
            model.forward = torch.compile(model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')
    else:
        raise NotImplementedError('ARGS>COMPILE')
    
    ddp_model = DDP(model, device_ids=[device_idx], broadcast_buffers=False, gradient_as_bucket_view=False)
    model = ddp_model.module

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    dataset = ImageNetDataset(args.data_path, image_size=args.image_size, is_train=True)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,)

    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,)

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}")

    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)

    if is_main_process():
        logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model, decay=0)
    ddp_model.train()

    running_loss = 0.0
    start_time = time()

    ys = torch.randint(num_classes, size=(4,), device=device)
    using_cfg = guidance_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

    if using_cfg:
        zs = torch.cat([zs, zs], dim=0)
        y_null = torch.full((n,), null_label, device=device)
        ys = torch.cat([ys, y_null], dim=0)
        sample_model_kwargs = dict(
            y=ys,
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs["additional_model_forward"] = guid_model_forward
            model_fn = ema.forward_with_autoguidance
        else:
            model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {epochs} epochs...")
    model_dirs = osp.splitext(osp.realpath(__file__))[0].split('/')
    this_model_dir = '/'.join(model_dirs[-3:-1])

    assert grad_accum_steps == 1

    start_epoch = max(0, train_steps // steps_per_epoch)
    if is_main_process():
        logger.info(f'Start epoch is {start_epoch}, steps_per_epoch is {steps_per_epoch}.')

    for epoch in range(start_epoch + 1, epochs):

        logger.info(f"Beginning epoch {epoch}...")

        for indice, (x, y) in enumerate(loader):

            x = x.to(device)
            y = y.to(device)

            imgs = concat_all_gather(x)
            labels = concat_all_gather(y)
            
            imgs = imgs.to('cpu',dtype=torch.float32)
            labels = labels.to('cpu', dtype=torch.int)
        
            with torch.no_grad():
                x = vfmae.encode_imgs(x)

            x = (x - mean.to(x)) / (var.to(x) + 1e-5)

            # latents = x * (var + 1e-5) + mean
            # recons = vfmae.decode_to_imgs(latents.float(), 256)
            # recon_np = recons.to('cpu', dtype=torch.uint8).numpy()
            # saveDir = 'recons'
            # os.makedirs(saveDir,exist_ok=True)
            # for i, recon in enumerate(recon_np):
            #     recon = Image.fromarray(recon)
            #     filename = osp.join(saveDir, f'{i+1:06d}.png')
            #     recon.save(filename)
            # pdb.set_trace()

            opt.zero_grad()
            model_kwargs = dict(y=y)
            with autocast(**autocast_kwargs):
                loss_tensor = transport.training_losses(ddp_model, x, model_kwargs)["loss"].mean()
            loss_tensor = loss_tensor.float()

            flag = False
            if torch.any(torch.isnan(loss_tensor)) and is_main_process():
                checkpoint = {
                    "model": ddp_model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    "cli_overrides": {
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
                checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                saveDir = 'data'
                os.makedirs(saveDir, exist_ok=True)
                fpath = osp.join(saveDir,f'{indice:06d}.pt')
                torch.save({'images':imgs, 'labels': labels}, fpath)
                pdb.set_trace()

            (loss_tensor / grad_accum_steps).backward()

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            opt.step()
            schedl.step(train_steps + 1)
            
            cur_lr = opt.param_groups[0]['lr']
            train_lr = schedl.get_lr()[0]
            update_ema(ema, ddp_model.module, decay=ema_decay)
            torch.cuda.synchronize()

            avg_loss = all_reduce_mean(loss_tensor / grad_accum_steps)
            train_steps += 1

            if is_main_process() & (train_steps % log_every == 0):

                end_time = time()
                steps_per_sec = log_every / (end_time - start_time)
                avg_loss = avg_loss.item()
                logger.info(f"[Epoch {epoch} | Step {train_steps % steps_per_epoch} / {steps_per_epoch} | Iter {train_steps}], " + \
                            f"Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Train/Schedl-LR:{train_lr:.8f}, this_model:{this_model_dir}.")

                start_time = time()

            if (train_steps % ckpt_every == 0) and (train_steps > 0) and is_main_process():

                checkpoint = {
                    "model": ddp_model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": schedl.state_dict(),
                    "train_steps": train_steps,
                    "config_path": args.config,
                    "training_cfg": training_cfg,
                    "cli_overrides": {
                        "image_size": args.image_size,
                        "precision": args.precision,
                        "global_seed": global_seed,
                    },
                }
                checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            dist.barrier()


    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="output", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 336, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vfmae/vfmae_config.yaml',)
    parser.add_argument("--ae-model", type=str, choices=list(AE_models.keys()), default="AE-16")
    parser.add_argument("--ae-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument('--stats-file', type=str, default='stats/stats-250.pt', required=False)
    parser.add_argument("--z-channels", type=int, default=512, help="z-channels")
    
    parser.add_argument("--embed-dim", type=int, default=32)
    args = parser.parse_args()
    main(args)
