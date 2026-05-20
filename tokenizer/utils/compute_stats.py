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

        posteriors = torch.cat(posteriors, dim=0)
        posteriors = posteriors.to(dtype=torch.float32)
        mean, var = torch.mean(posteriors, dim=0, keepdim=False), torch.var(posteriors, dim=0, keepdim=False)

        mean = mean.flatten(1)
        var = torch.sqrt(var.flatten(1))
        mean, var = mean.to('cpu', dtype=torch.float32).numpy(), var.to('cpu', dtype=torch.float32).numpy()
        stats = {'mean': mean, 'var': var}
        saveDir = 'stats'
        os.makedirs(saveDir, exist_ok=True)
        torch.save(stats, 'stats/stats-500.pt')

@torch.no_grad()
def gen_images(model, dataloader, device, args=None):

    model.eval()
    prev, total = 0, len(dataloader)
    
    model_dirs = osp.realpath(__file__).split('/')
    idx = np.argmax([len(p) for p in model_dirs])
    this_model_dir = model_dirs[idx]

    posteriors = []
    for i, (images, labels,) in enumerate(dataloader):

        images = images.to(device, non_blocking=True)

        posterior = model.encode_imgs(images)

        posterior = concat_all_gather(posterior)

        posterior = posterior.to('cpu', dtype=torch.float32)
    
        if is_main_process():
            print('{}, iter-{}/{}, gen_imgs.shape:{}'.format(this_model_dir, i, total, posterior.shape))

        posteriors.append(posterior)

        if i > 999:
            break

    return posteriors

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
 
    main(args)
