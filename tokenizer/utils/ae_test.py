import argparse
import json, time
import torch, timm
import numpy as np
import os.path as osp
from PIL import Image
from omegaconf import OmegaConf
import tensorflow.compat.v1 as tf
from torchvision import transforms
import os, sys, warnings, pdb, time
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from tokenizer.vfmae.vfmae import AE_models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from commons.evaluations.evaluator import Evaluator
from commons.data.augmentation import center_crop_arr
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from commons.engine.distributed import init_distributed_mode
from commons.data.imagenet_lmdb import LMDBImageNet as ImageNet
from commons.engine.misc import (is_main_process, get_rank, get_world_size, concat_all_gather)

warnings.filterwarnings('ignore')

def get_args_parser():

    parser = argparse.ArgumentParser('VFMTok testing', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    #* Dataset parameters
    parser.add_argument('--output_dir', default='./recons',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/logs/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    #* Feature genration
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")
    parser.add_argument('--eval-coco', action='store_true', default=False,)

    #* distributed training parameters
    parser.add_argument("--ae-model", type=str, choices=list(AE_models.keys()), default="AE-16")
    parser.add_argument("--ae-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--image-size", type=int, choices=[256, 336, 384, 448, 512, 1024], default=512)
    parser.add_argument("--transformer-config-file", type=str, default='configs/vfmae/vfmae_config.yaml')
    parser.add_argument("--z-channels", type=int, default=512,)
    parser.add_argument('--embed-dim', type=int, default=32,)
    parser.add_argument("--anno-file", type=str, default='imagenet/lmdb/val_lmdb')
    return parser

def main(args):

    init_distributed_mode(args)
    print('job dir: {}'.format(osp.dirname(osp.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False

    num_tasks = get_world_size()
    global_rank = get_rank()

    assert osp.exists(args.anno_file)
    transform = transforms.Compose([
                 transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])

    dataset = ImageNet(args.anno_file, transform=transform,)

    sampler = DistributedSampler(dataset, rank=global_rank, shuffle=False)
    data_loader = DataLoader(
        dataset,  sampler=sampler,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False, drop_last=False,)

    transformer_config = OmegaConf.load(args.transformer_config_file)
    model = AE_models[args.ae_model](
        embed_dim = args.embed_dim,
        transformer_config=transformer_config,
        z_channels=args.z_channels,)

    model.eval()
    checkpoint = torch.load(args.ae_ckpt, map_location="cpu")

    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missings, unexpected = model.load_state_dict(model_weight, strict=False)
    assert sum(['backbone' in p for p in missings]) == len(missings), 'Please check the state_dict since necessary parameters are missed.'

    del checkpoint
    
    #* Set arguments generation params
    model.to(device)

    #* Log parameters
    if is_main_process():
        print('#DataLoader: {}, num_tasks: {}'.format(len(data_loader), num_tasks))

    (samples, gt, psnr_val_rgb, ssim_val_rgb) = gen_images(model, data_loader, device, args)
    if is_main_process():
        eps = 1e-6
        samples = np.concatenate(samples, axis=0)
        gt = np.concatenate(gt, axis=0)

        print(f'len(samples):{samples.shape[0]}, len(gt): {len(gt)}')
        config = tf.ConfigProto(
                allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        config.gpu_options.allow_growth = True

        evaluator = Evaluator(tf.Session(config=config),batch_size=64)
        evaluator.warmup()
        print("computing reference batch activations...")
        ref_acts = evaluator.read_activations(gt)
        print("computing/reading reference batch statistics...")
        ref_stats, _ = evaluator.read_statistics(gt, ref_acts)
        print("computing sample batch activations...")
        sample_acts = evaluator.read_activations(samples)
        print("computing/reading sample batch statistics...")
        sample_stats, _ = evaluator.read_statistics(samples, sample_acts)
        FID = sample_stats.frechet_distance(ref_stats)

        IS = evaluator.compute_inception_score(sample_acts[0])

        print(f"rFID: {FID:04f}, rIS: {IS:04f}.")

        psnr_val_rgb = np.concatenate(psnr_val_rgb, axis=0).mean()
        ssim_val_rgb = np.concatenate(ssim_val_rgb, axis=0).mean()

        print('PSNR: {:.4f}, SSIM: {:.4f}'.format(psnr_val_rgb, ssim_val_rgb))
        filename = osp.basename(args.ae_ckpt).split('.')[0]
        with open('results.md', 'a') as fid:
            fid.write(f'\n{filename}:\n')
            fid.write(f'rFID: {FID:04f}, rIS: {IS:04f}.\n')
            fid.write('PSNR: {:.4f}, SSIM: {:.4f}.\n'.format(psnr_val_rgb, ssim_val_rgb))

@torch.no_grad()
def gen_images(model, dataloader, device, args):

    model.eval()
    saveDir = args.output_dir
    prev, total = 0, len(dataloader)
    
    model_dirs = osp.realpath(__file__).split('/')
    idx = np.argmax([len(p) for p in model_dirs])
    this_model_dir = model_dirs[idx]

    samples, gt = [], []
    psnr_val_rgb, ssim_val_rgb = [], []
    rank, world_size = get_rank(), get_world_size()
    for i, (images, labels,) in enumerate(dataloader):

        
        images = images.to(device)
        (gen_imgs, _, _), posterior = model(images)

        gen_images = concat_all_gather(gen_imgs)
        gt_imgs = concat_all_gather(images)

        bs = gen_images.size(0)
        np_gens = torch.clamp(127.5 * gen_imgs.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu', dtype=torch.uint8).numpy()
        np_images = torch.clamp(127.5 * images.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu', dtype=torch.uint8).numpy()
        
        if is_main_process():
            print('{}, iter-{}/{}, gen_imgs.shape:{}'.format(this_model_dir, i, total, gen_images.shape))

        psnr_val_rgb_gpu, ssim_val_rgb_gpu = [], []
        for k, re in enumerate(np_gens):

            rec = Image.fromarray(re)
            img = Image.fromarray(np_images[k])

            rec = rec.resize((256, 256))
            img = img.resize((256, 256))

            # recon = np.concatenate((np.array(rec), np.array(img)), axis=1)
            # index = k * world_size + rank + i * bs
            # filename = osp.join(saveDir, f'{index:06d}.png')
            # Image.fromarray(recon).save(filename)

            rgb_restored = np.array(rec).astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            rgb_gt = np.array(img).astype(np.float32) / 255.
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb_gpu.append(psnr)
            ssim_val_rgb_gpu.append(ssim)
        
        psnr_val_rgb_gpu = torch.tensor(psnr_val_rgb_gpu, device=device)
        ssim_val_rgb_gpu = torch.tensor(ssim_val_rgb_gpu, device = device)

        psnr_val = concat_all_gather(psnr_val_rgb_gpu)
        ssim_val = concat_all_gather(ssim_val_rgb_gpu)
        
        psnr_val = psnr_val.to('cpu', dtype=torch.float32).numpy()
        ssim_val = ssim_val.to('cpu', dtype=torch.float32).numpy()
        
        gen_images = torch.clamp(127.5 * gen_images.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu', dtype=torch.uint8).numpy()
        np_images = torch.clamp(127.5 * gt_imgs.permute(0, 2, 3, 1) + 128.0, 0, 255).to('cpu', dtype=torch.uint8).numpy()
        
        
        psnr_val_rgb.append(psnr_val)
        ssim_val_rgb.append(ssim_val)

        samples.append(gen_images)
        gt.append(np_images)
        
    return samples, gt, psnr_val_rgb, ssim_val_rgb

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
 
    main(args)
