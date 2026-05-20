# mkdir -p output/logs
export NODE_COUNT=$1
export NODE_RANK=$2
export PROC_PER_NODE=8
export MASTER_PORT=12333
export MASTER_ADDR=localhost
/jobutils/scripts/torchrun.sh  \
    ae_train.py  --image-size 256 --results-dir output --mixed-precision none --global-batch-size 256 --num-workers 4  \
    --data-path imagenet/lmdb/train_lmdb --ckpt-every 5000 --transformer-config configs/vfmae/vfmae_config.yaml  \
    --epochs 50 --log-every 1 --lr 1e-4 --ema --z-channels 512 --embed-dim 32 --disc-start 20000 \
    2>&1 | tee 'train.log'
