# !/bin/bash
eval $(curl -s http://deploy.i.shaipower.com/httpproxy)
export NODE_COUNT=1
export NODE_RANK=0
export PROC_PER_NODE=8
iters="$(printf "%07d" "$1")"
saveDir='samples'
saveDir='AutoGuidance-50k'
mkdir -p ${saveDir}
# cfg_file='configs/denoise/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml'
cfg_file='configs/denoise/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml'
scripts/autoregressive/torchrun.sh test_c2i.py --config ${cfg_file} --compile --sample-dir ${saveDir} --precision bf16  \
    --label-sampling equal --ckpt output/denoise/${iters}.pt --per-proc-batch-size $2 --embed-dim 32      \
    --stats-file stats/stats-500.pt --ae-ckpt DINOv2/tokenizer/vfmae-tokenizer.pt --cfg-scale $3       \
    2>&1 | tee 'hello.log'