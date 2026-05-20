# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
export MASTER_PORT=65331
scripts/autoregressive/torchrun.sh ae_test.py --transformer-config-file configs/vfmae/vfmae_config.yaml --image-size 256 \
        --batch-size $1 --anno-file imagenet/lmdb/val_lmdb --ae-ckpt DINOv2/tokenizer/vfmae-tokenizer.pt \
        --embed-dim 32 2>&1 | tee 'test2.log'
