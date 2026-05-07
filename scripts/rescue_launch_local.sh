#!/usr/bin/env bash
# Launch a single rescue-campaign variant locally on a pinned GPU.
#
# Usage:
#   scripts/rescue_launch_local.sh <algo> <gpu> [t_max] [batch_size_run]
# Example:
#   scripts/rescue_launch_local.sh maddpg_M3_lowcriticlr 0 500000 4
set -euo pipefail
cd /home/rahul/qaduub-mappo

ALGO=${1:?"need algo name (e.g. maddpg_M3_lowcriticlr)"}
GPU=${2:?"need GPU index"}
TMAX=${3:-500000}
BSR=${4:-}

LOG_DIR=/tmp/claude_runs/rescue
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${ALGO}.log"

WITH_ARGS=("t_max=$TMAX")
if [ -n "$BSR" ]; then
  WITH_ARGS+=("batch_size_run=$BSR")
fi
# Note: use_cuda defaults to True in vendor/config/default.yaml so torch runs on
# GPU. We deliberately do NOT pass use_cuda=True on the CLI because train.py
# would then force JAX_PLATFORMS=cuda, which breaks under ParallelRunner forks.
# JAX falls back to CPU; the env-side qarray sim runs on CPU, but the policy
# training (the bottleneck) runs on GPU regardless.

echo "[rescue_launch] $(date '+%H:%M:%S') algo=$ALGO gpu=$GPU t_max=$TMAX bsr=${BSR:-default} → $LOG" | tee "$LOG"

CUDA_VISIBLE_DEVICES=$GPU nohup uv run --extra facmac python benchmarks/MARL/facmac/train.py \
  --config=$ALGO \
  --env-config=env_quantum \
  with "${WITH_ARGS[@]}" \
  >> "$LOG" 2>&1 &
PID=$!
echo "[rescue_launch] pid=$PID" | tee -a "$LOG"
disown
