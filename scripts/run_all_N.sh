#!/usr/bin/env bash
# Orchestrate 8-GPU sharded rollouts at N ∈ {2, 4, 6, 8}.
#
# Each N is run sequentially. Within each N, the requested seed budget is split
# across 8 GPUs and run in parallel. After all GPUs finish, shards are merged
# into plots_supersims_diagnostic/staircase_scan_N{N}.npz.
#
# Auto-mode invocation:
#   nohup bash scripts/run_all_N.sh > /tmp/claude_runs/run_all_N.log 2>&1 &
set -euo pipefail
cd <repo>
source .venv/bin/activate

CKPT=checkpoints_supersims_grouped/iteration_28
OUT_DIR=plots_supersims_diagnostic
SHARD_DIR=/tmp/claude_runs/shards
mkdir -p "$SHARD_DIR" /tmp/claude_runs

# (N, total_seeds, param_cfg)
N_VALUES=(2 4 6 8)
SEEDS_BY_N=(100 100 100 50)
CFGS=(parameter_config_N2.json parameter_config.json parameter_config_N6.json parameter_config_N8.json)

NUM_GPUS=8

for idx in 0 1 2 3; do
  N=${N_VALUES[$idx]}
  TOTAL_SEEDS=${SEEDS_BY_N[$idx]}
  CFG=${CFGS[$idx]}
  echo "=========================================="
  echo "N=$N  seeds=$TOTAL_SEEDS  cfg=$CFG"
  echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="

  # Compute per-GPU seed counts (distribute remainder).
  base=$((TOTAL_SEEDS / NUM_GPUS))
  rem=$((TOTAL_SEEDS - base * NUM_GPUS))

  PIDS=()
  offset=0
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    if [ "$gpu" -lt "$rem" ]; then
      n_this=$((base + 1))
    else
      n_this=$base
    fi
    if [ "$n_this" -le 0 ]; then
      continue
    fi
    shard_out="$SHARD_DIR/staircase_scan_N${N}_gpu${gpu}.npz"
    log="$SHARD_DIR/N${N}_gpu${gpu}.log"
    echo "  GPU$gpu  seeds=$n_this  offset=$offset  → $shard_out"
    CUDA_VISIBLE_DEVICES=$gpu SUPERSIMS_PARAM_CFG=$CFG \
      python scripts/eval_multi_N.py \
        --checkpoint "$CKPT" \
        --n-seeds $n_this \
        --seed-offset $offset \
        --out "$shard_out" \
        > "$log" 2>&1 &
    PIDS+=($!)
    offset=$((offset + n_this))
  done

  echo "  Spawned ${#PIDS[@]} GPU shards. Waiting..."
  fail=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "  PID $pid FAILED for N=$N"
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "ERROR: at least one shard failed at N=$N. Bailing."
    exit 1
  fi

  echo "  All shards finished at $(date '+%H:%M:%S'). Merging..."
  shards_glob=("$SHARD_DIR"/staircase_scan_N${N}_gpu*.npz)
  python scripts/merge_eval_shards.py \
    --shards "${shards_glob[@]}" \
    --out "$OUT_DIR/staircase_scan_N${N}.npz"
  echo "N=$N complete."
done

echo "=========================================="
echo "ALL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
ls -la "$OUT_DIR"/staircase_scan_N*.npz
