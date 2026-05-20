# !/bin/bash
export NODE_COUNT=$1
export NODE_RANK=$2
export PROC_PER_NODE=8
export MASTER_PORT=22331
export MASTER_ADDR=localhost
config_file='configs/denoise/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml'
scripts/autoregressive/torchrun.sh train_c2i.py --config ${config_file} --data-path imagenet/lmdb/train_lmdb       \
    --image-size 256 --results-dir output/snapshot --precision fp32 --embed-dim 32  --z-channels 512      \
    --ae-ckpt DINOv2/tokenizer/vfmae-tokenizer.pt --stats-file stats/stats-500.pt --compile       \
    2>&1 | tee 'train2.log'