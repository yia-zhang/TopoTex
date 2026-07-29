#!/usr/bin/env bash
# Official multi-GPU training entry: torchrun DDP over the packed-group
# trainer (group K, bf16, face-count buckets — unchanged recipe; groups are
# sharded across ranks, effective batch = K * world_size meshes/step).
#   GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_8gpu.sh \
#     --samples 2000 --run-name fm_2k [--resume] [--steps N]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/root/miniconda3/envs/geomae/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
N=$(awk -F, '{print NF}' <<< "$GPU_IDS")
echo "train_8gpu: GPUs=$GPU_IDS (world_size=$N) | args: $*"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node="$N" train.py "$@"
