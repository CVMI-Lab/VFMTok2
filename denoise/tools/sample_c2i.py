# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained stage-2 model using DDP and
stores results for downstream metrics. For single-device sampling, use sample.py.
"""
import os, sys, math
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import argparse, json, math
from typing import Callable, Optional
import numpy as np
import torch, pdb
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.cuda.amp import autocast

import tensorflow.compat.v1 as tf
from evaluations.c2i.evaluator import Evaluator

from tokenizer.vfmae.vfmae import AE_models
from commons.engine.ema import requires_grad
from commons.engine.util import parse_configs
from denoise.models import Stage2ModelProtocol
from commons.engine.util import instantiate_from_config
from denoise.transport import create_transport, Sampler
from commons.engine.distributed import init_distributed_mode
from commons.engine.misc import (is_main_process, get_rank, get_world_size, cleanup, concat_all_gather)


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
    # label_counts_path:  Optional[str]=None,
) -> Callable[[int], torch.Tensor]:
    """Create a callable that returns a batch of labels for the given step index."""

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)

        return random_sampler

    if sampling_mode == "equal":
        if num_fid_samples % num_classes != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) to be divisible by num_classes ({num_classes})."
            )

        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[permutation]

        if total_samples > num_fid_samples:
            tail = torch.randint(0, num_classes, (total_samples - num_fid_samples,), generator=generator)
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        start = rank * samples_needed_this_device
        end = start + samples_needed_this_device
        device_pool = global_pool[start:end]
        device_pool = device_pool.view(iterations, batch_size)

        def equal_sampler(step_idx: int) -> torch.Tensor:
            labels = device_pool[step_idx]
            return labels.to(device)

        return equal_sampler
    raise ValueError(f"Unknown label sampling mode: {sampling_mode}")

def main(args):
    """Run sampling with distributed execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU. Use sample.py for single-device usage.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if is_main_process():
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)

    if model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")
    misc = {} if misc is None else dict(misc)

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if is_main_process():
        print(f"Using time_dist_shift={time_dist_shift:.4f}.")

    #* Create AE
    transformer_config = OmegaConf.load(args.transformer_config_file)
    ae_model = AE_models[args.ae_model](
        embed_dim=args.embed_dim,
        transformer_config=transformer_config,
        z_channels=args.z_channels,).to(device)

    ae_model.eval()
    ae_model.freeze()
    assert osp.exists(args.ae_ckpt)
    checkpoint = torch.load(args.ae_ckpt, map_location="cpu")


    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missings, unexpected = ae_model.load_state_dict(model_weight, strict=False)
    assert sum(['backbone' in p for p in missings]) == len(missings), 'Please check the state_dict since necessary parameters are missed.'
    del checkpoint

    #* Create generative model
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)

    assert osp.exists(args.ckpt)
    ckpt = torch.load(args.ckpt, 'cpu')
    model.load_state_dict(ckpt['ema'], strict=True)
    model.eval()
    
    assert osp.exists(args.stats_file)
    stats = torch.load(args.stats_file, map_location='cpu')
    mean, var = torch.tensor(stats['mean'], device=device), torch.tensor(stats['var'], device=device)

    H = W = int(math.sqrt(mean.size(1)))
    mean = rearrange(mean, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)
    var = rearrange(var, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)

    if args.compile:
        print(f"compiling the model...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True,) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    sampler = Sampler(transport)

    sampler_config = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config.get("mode", "ODE")
    sampler_params = dict(sampler_config.get("params", {}))
    mode = sampler_mode.upper()
    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guidance_config = {} if guidance_config is None else dict(guidance_config)

    def guidance_value(key: str, default: float):
        if key in guidance_config:
            return guidance_config[key]
        dashed_key = key.replace("_", "-")
        return guidance_config.get(dashed_key, default)

    guidance_scale = args.cfg_scale
    guidance_method = guidance_config.get("method", "cfg")
    t_min = guidance_value("t_min", 0.0)
    t_max = guidance_value("t_max", 1.0)

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guid_model_config = guidance_config.get("guidance_model")
        if guid_model_config is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)

        guided_ckpt = guid_model_config.get('ckpt')
        assert osp.exists(guided_ckpt)
        guided_stats = torch.load(guided_ckpt, 'cpu')
        guid_model.load_state_dict(guided_stats['ema'], strict=True)

        guid_model.eval()
        guid_model_forward = guid_model.forward

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))

    model_target = model_config.get("target", "stage2")
    model_string_name = str(model_target).split(".")[-1]
    ckpt_path = model_config.get("ckpt")
    ckpt_string_name = "pretrained" if not ckpt_path else osp.splitext(osp.basename(str(args.ckpt)))[0]
    sampling_method = sampler_params.get("sampling_method", "na")
    num_steps = sampler_params.get("num_steps", sampler_params.get("steps", "na"))
    guidance_tag = f"cfg-{guidance_scale:.2f}"
    base_components = [model_string_name, ckpt_string_name, guidance_tag, f"bs{args.per_proc_batch_size}"]
    if mode == "ODE":
        detail_components = [mode, str(num_steps), str(sampling_method), args.precision]
    else:
        diffusion_form = sampler_params.get("diffusion_form", "na")
        last_step = sampler_params.get("last_step", "na")
        last_step_size = sampler_params.get("last_step_size", "na")
        detail_components = [mode, str(num_steps), str(sampling_method), str(diffusion_form), str(last_step), str(last_step_size), args.precision]

    folder_name = "-".join(component.replace(os.sep, "-") for component in base_components + detail_components)
    sample_folder_dir = osp.join(args.sample_dir, folder_name)

    if is_main_process():
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    
    num_samples = 0
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if is_main_process():
        print(f"Total number of images that will be sampled: {total_samples}")
    if total_samples % world_size != 0:
        raise ValueError("Total samples must be divisible by world size.")
    samples_needed_this_gpu = total_samples // world_size
    if samples_needed_this_gpu % n != 0:
        raise ValueError("Per-rank sample count must be divisible by the per-GPU batch size.")
    iterations = samples_needed_this_gpu // n
    pbar = tqdm(range(iterations)) if is_main_process() else range(iterations)
    total = (num_samples // world_size) * world_size

    label_sampler = build_label_sampler(
        args.label_sampling,
        num_classes,
        args.num_fid_samples,
        total_samples,
        samples_needed_this_gpu,
        n,
        device,
        rank,
        iterations,
        args.global_seed,)

    gen_samples = []
    using_cfg = guidance_scale > 1.0
    for step_idx in pbar:
        
        if is_main_process():
            print('')

        with autocast(**autocast_kwargs):

            z = torch.randn(n, *latent_size, device=device)
            y = label_sampler(step_idx)
            model_kwargs = dict(y=y)
            model_fn = model.forward

            if using_cfg:
                z = torch.cat([z, z], dim=0)
                y_null = torch.full((n,), null_label, device=device)
                y = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(
                    y=y,
                    cfg_scale=guidance_scale,
                    cfg_interval=(t_min, t_max),
                )
                if guidance_method == "autoguidance":
                    if guid_model_forward is None:
                        raise RuntimeError("Guidance model forward is not initialized.")
                    model_kwargs["additional_model_forward"] = guid_model_forward
                    model_fn = model.forward_with_autoguidance
                else:
                    model_fn = model.forward_with_cfg

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)
            
            x = samples * (var.to(samples) + 1e-5) + mean.to(samples)

            gen_images = samples = ae_model.decode_to_imgs(x)
            samples = concat_all_gather(samples)
            samples = samples.to('cpu', dtype=torch.uint8).numpy()
            gen_images = gen_images.to('cpu', dtype=torch.uint8).numpy()
            gen_samples.append(samples)
            bs = samples.shape[0]
            
            # Save samples to disk as individual .png files
            # saveDir = 'samples/gen'
            # os.makedirs(saveDir, exist_ok=True)
            # for local_idx, sample in enumerate(samples):
            #     index = local_idx * world_size + rank + step_idx * bs
            #     Image.fromarray(sample).save(f"{saveDir}/{index:06d}.png")
            # pdb.set_trace()

        dist.barrier()

    dist.barrier()
    if is_main_process():

        gen_samples = np.concatenate(gen_samples, axis=0)[:50_000]

        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()

        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print("computing/reading reference batch statistics...")
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(gen_samples)
        print("computing/reading sample batch statistics...")
        sample_stats, sample_stats_spatial = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)
        sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)

        IS = evaluator.compute_inception_score(sample_acts[0])
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

        print("Computing evaluations...")
        print("Inception Score:", IS)
        print("FID:", FID)
        print("sFID:", sFID)
        print("Precision:", prec)
        print("Recall:", recall)

        txt_path = sample_folder_dir + '.txt'
        print("writing to {}".format(txt_path))
        with open(txt_path, 'w') as f:
            print("Inception Score:", IS, file=f)
            print("FID:", FID, file=f)
            print("sFID:", sFID, file=f)
            print("Precision:", prec, file=f)
            print("Recall:", recall, file=f)

        print("Done.")
    

    cleanup()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument(
        "--label-sampling",
        type=str,
        choices=["random", "equal"],
        default="equal",
        help="Choose how to sample class labels when generating images.",
    )
    parser.add_argument('--cfg-scale', type=float, default=1.45)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vfmae/vfmae_config.yaml',)
    parser.add_argument("--ae-model", type=str, choices=list(AE_models.keys()), default="AE-16")
    parser.add_argument("--ae-ckpt", type=str, help="ckpt path for resume training", default='pretrained_models/vit_vqgan_step_125000.pt')
    parser.add_argument("--z-channels", type=int, default=512, help="z-channels")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument('--stats-file', type=str, default='stats/stats-500.pt')
    parser.add_argument("--ref-batch", type=str, default='imagenet/VIRTUAL_imagenet256_labeled.npz', help="path to reference batch npz file")

    args = parser.parse_args()
    main(args)
