#!/usr/bin/env bash
# Eval orchestration for the rescue campaign.
# After Modal runs commit, this:
#   1. Pulls each algo's checkpoints from the facmac-results volume
#   2. Runs run_eval_trials.py against the latest checkpoint
#   3. Also runs the random baseline once for comparison
#
# Output: /tmp/eval_results/{algo}.json + random_baseline.json
#
# Edit ALGOS to match the variants you want to evaluate.

set -euo pipefail

ALGOS=(
  "maddpg_M1_td3"
  "maddpg_M3_lowcriticlr"
  "maddpg_M6_antizero"
  "maddpg_M6b_strongantizero"
  "facmac_F1_lowcriticlr"
  "facmac_F2_vdn"
  "facmac_F2b_nomixer"
  "facmac_F3_slowtau"
  "facmac_F4_rewardnorm"
)

REPO=/home/rahul/qaduub-mappo
CKPT_ROOT=/tmp/eval_ckpts
RESULT_DIR=/tmp/eval_results
# Per-trial .npy distance trajectories live here in the layout
# ablation_metrics.py expects: <NPY_ROOT>/run_<algo>/plunger_<i>/<NNNN>_*.npy
NPY_ROOT=/tmp/eval_npy
# IMPORTANT: env-config must be absolute — env.py resolves it relative to its own
# package dir, not the cwd, so a relative path silently breaks.
ENV_CFG="$REPO/benchmarks/MARL/facmac/configs/env_config_smoke.yaml"
NUM_TRIALS=${NUM_TRIALS:-100}

mkdir -p "$CKPT_ROOT" "$RESULT_DIR" "$NPY_ROOT"

cd "$REPO"

# --- 1. Pull checkpoints from Modal volume -----------------------------------
for algo in "${ALGOS[@]}"; do
  dst="$CKPT_ROOT/$algo"
  if [ -d "$dst" ] && [ -n "$(find "$dst" -name 'agent_plunger.th' -print -quit)" ]; then
    echo "[skip pull] $algo already has checkpoints in $dst"
    continue
  fi
  rm -rf "$dst"
  echo "[pull] $algo from facmac-results:/$algo -> $dst"
  uv run --extra facmac modal volume get facmac-results "/$algo" "$dst" \
    || { echo "[warn] no commit yet for $algo, skipping"; continue; }
done

# --- 2. Random baseline (run once at num_dots=4) ------------------------------
echo "[eval] random baseline"
uv run --extra facmac python benchmarks/MARL/facmac/run_eval_trials.py \
  --checkpoint-dir /dev/null \
  --env-config "$ENV_CFG" \
  --num-trials "$NUM_TRIALS" \
  --output "$RESULT_DIR/random_baseline.json" \
  --npy-output-dir "$NPY_ROOT/run_random_baseline" \
  --random-baseline

# --- 3. Per-algo eval at latest step -----------------------------------------
for algo in "${ALGOS[@]}"; do
  dst="$CKPT_ROOT/$algo"
  # Find latest step dir: results/models/<run>/<step>/agent_plunger.th
  latest=$(find "$dst" -name 'agent_plunger.th' 2>/dev/null \
           | xargs -I{} dirname {} \
           | awk -F/ '{print $NF, $0}' \
           | sort -n -k1 \
           | tail -1 \
           | cut -d' ' -f2-)
  if [ -z "$latest" ]; then
    echo "[skip eval] $algo: no checkpoint found in $dst"
    continue
  fi
  step=$(basename "$latest")
  echo "[eval] $algo @ step=$step"
  uv run --extra facmac python benchmarks/MARL/facmac/run_eval_trials.py \
    --checkpoint-dir "$latest" \
    --env-config "$ENV_CFG" \
    --num-trials "$NUM_TRIALS" \
    --output "$RESULT_DIR/${algo}.json" \
    --npy-output-dir "$NPY_ROOT/run_${algo}" \
    || echo "[warn] eval failed for $algo (continuing)"
done

echo "[done] JSON results in $RESULT_DIR/"
ls -l "$RESULT_DIR/"

# --- 4. Convergence-percent table at radii 2, 5, 10V (paper table format) -----
echo "[summary] computing convergence percentages"
uv run python "$REPO/benchmarks/Ablations/ablation_metrics.py" \
  --data-dir "$NPY_ROOT" \
  --radius 2,5,10 \
  --length 30 \
  --out "$RESULT_DIR/convergence_table.json" \
  || echo "[warn] convergence table failed (rerun manually if needed)"

echo "[done] convergence table in $RESULT_DIR/convergence_table.json"
