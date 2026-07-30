#!/usr/bin/env bash
# 8-GPU source dataset build: one process per GPU, ids[rank::8], atomic publish,
# resume-skip, per-sample error logs; merges per-rank manifests at the end
# (datasets/merge_manifest.py: no duplicate / no missing / schema valid).
#   GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/build_dataset_8gpu.sh \
#     output/source_manifests/glbs.jsonl output/topotex_source [LIMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/root/miniconda3/envs/geomae/bin/python}"
MANIFEST="${1:?input manifest jsonl}"
OUTPUT="${2:?output dataset dir}"
LIMIT="${3:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
N=${#GPUS[@]}
LOG_DIR="${LOG_DIR:-/tmp/topotex_build_logs}"; mkdir -p "$LOG_DIR"
echo "build: $MANIFEST -> $OUTPUT on GPUs ${GPUS[*]} ($N ranks)"
PIDS=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m topotex.data.builder source \
    --input-manifest "$MANIFEST" --output "$OUTPUT" \
    --world_size "$N" --rank "$i" \
    --scratch-root "/tmp/topotex_source_rank_$i" \
    ${LIMIT:+--limit "$LIMIT"} \
    > "$LOG_DIR/rank_$i.log" 2>&1 &
  PIDS+=($!)
  sleep 20   # stagger checkpoint loads
done
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "rank $i FAILED ($LOG_DIR/rank_$i.log)"; FAIL=1; }
done
MERGE_MANIFEST="$MANIFEST"
if [ -n "$LIMIT" ]; then
  MERGE_MANIFEST="/tmp/topotex_build_input_head.jsonl"
  head -n "$LIMIT" "$MANIFEST" > "$MERGE_MANIFEST"
fi
"$PY" -m topotex.data.builder merge --output "$OUTPUT" --input-manifest "$MERGE_MANIFEST"

# stage 2: UV query set (canonical/alternative/partial/held-out per mesh),
# sharded across the same workers (bpy/xatlas-bound; rasterizer uses the GPU)
UVQ_OUTPUT="${UVQ_OUTPUT:-output/topotex_dataset}"
PIDS2=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m topotex.data.builder queries \
    --output "$UVQ_OUTPUT" --world_size "$N" --rank "$i" \
    ${LIMIT:+--limit "$LIMIT"} \
    > "$LOG_DIR/uvq_rank_$i.log" 2>&1 &
  PIDS2+=($!)
done
for i in "${!PIDS2[@]}"; do
  wait "${PIDS2[$i]}" || { echo "uvq rank $i FAILED ($LOG_DIR/uvq_rank_$i.log)"; FAIL=1; }
done
"$PY" -m topotex.data.builder queries --output "$UVQ_OUTPUT" --finalize ${LIMIT:+--limit "$LIMIT"}
exit $FAIL
