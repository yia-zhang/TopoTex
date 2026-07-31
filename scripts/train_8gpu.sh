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
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  "$PY" scripts/preflight_training.py --gpus "$GPU_IDS" --skip-tests \
    || { echo "preflight FAILED — refusing to launch (SKIP_PREFLIGHT=1 to override)"; exit 1; }
fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node="$N" train.py "$@"
