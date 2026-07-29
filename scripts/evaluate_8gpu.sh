#!/usr/bin/env bash
# Sharded evaluation: rank i evaluates samples[i::world] and writes
# eval_rank_i.json; a final merge produces the unified eval.json
# (UV PSNR / render PSNR / heldout / consistency / seam).
#   GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/evaluate_8gpu.sh <run_dir> <n_total> [offset]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/root/miniconda3/envs/geomae/bin/python}"
RUN="${1:?run dir}"
NTOT="${2:?total samples}"
OFFSET="${3:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
W=${#GPUS[@]}
echo "evaluate_8gpu: GPUs=$GPU_IDS (world=$W) | run=$RUN n=$NTOT offset=$OFFSET"
PIDS=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" evaluate.py --run "$RUN" \
    --n "$NTOT" --offset "$OFFSET" --world_size "$W" --rank "$i" \
    --out "$RUN/eval_rank_$i.json" > "/tmp/eval8_rank_$i.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "eval rank $i FAILED (/tmp/eval8_rank_$i.log)"; FAIL=1; }
done
"$PY" evaluate.py --run "$RUN" --merge
exit $FAIL
