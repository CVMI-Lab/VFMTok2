# !/bin/bash
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=1
export MASTER_PORT=65331
/jobutils/scripts/torchrun.sh  \
        compute_stats.py --ae-model AE-16 --image-size 256 --batch-size $2 --embed-dim 32 --z-channels 512     \
        --anno-file imagenet/lmdb/train_lmdb --ae-ckpt DINOv2/tokenizer/vfmae-tokenizer.pt  \
        2>&1 | tee 'test.log'
