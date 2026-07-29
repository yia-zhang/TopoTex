#!/usr/bin/env bash
# Experiment launcher: every official job declares its GPU plan up front.
#   GPU_IDS=0,1,2,3 bash scripts/run_experiment.sh <task-name> <command...>
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?task name}"; shift
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
echo "=============================================="
echo " task               : $TASK"
echo " selected GPUs      : $GPU_IDS"
echo " CUDA_VISIBLE_DEVICES: $GPU_IDS"
echo " command            : $*"
echo " busy GPUs right now:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | awk -F', ' '$2+0>5 || $3+0>1000 {print "   GPU"$0}'
echo "=============================================="
export GPU_IDS
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$@"
