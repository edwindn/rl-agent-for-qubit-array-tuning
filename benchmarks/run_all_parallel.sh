#!/bin/bash
# Run benchmarks in parallel, 8 at a time (one per GPU)
# Methods: nelder_mead, lbfgs, bayesian, random
# Dots: 2, 4, 6, 8
# Trials: 100, Steps: 2000

TRIALS=100
MAX_ITER=2000
MAX_SAMPLES=2000
SEED=42

cd /home/rahul/rl-agent-for-qubit-array-tuning/benchmarks

echo "========================================"
echo "Starting benchmark suite at $(date)"
echo "Trials: $TRIALS, Max Iter/Samples: $MAX_ITER"
echo "Dots: 2, 4, 6, 8"
echo "Using 8 GPUs in parallel"
echo "========================================"
echo ""

# Function to run a benchmark on specific GPU
run_on_gpu() {
    local gpu=$1
    local name=$2
    shift 2
    echo "[GPU $gpu] Starting: $name at $(date)"
    CUDA_VISIBLE_DEVICES=$gpu "$@"
    echo "[GPU $gpu] Completed: $name at $(date)"
}

# Batch 1: Nelder-Mead (all 4) + L-BFGS (all 4) = 8 jobs
echo "=== BATCH 1: Nelder-Mead + L-BFGS ==="
run_on_gpu 0 "NM-2dots"    uv run python nelder_mead/run.py --mode joint --num_dots 2 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 1 "NM-4dots"    uv run python nelder_mead/run.py --mode joint --num_dots 4 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 2 "NM-6dots"    uv run python nelder_mead/run.py --mode joint --num_dots 6 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 3 "NM-8dots"    uv run python nelder_mead/run.py --mode joint --num_dots 8 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 4 "LBFGS-2dots" uv run python lbfgs/run.py --mode joint --num_dots 2 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 5 "LBFGS-4dots" uv run python lbfgs/run.py --mode joint --num_dots 4 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 6 "LBFGS-6dots" uv run python lbfgs/run.py --mode joint --num_dots 6 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
run_on_gpu 7 "LBFGS-8dots" uv run python lbfgs/run.py --mode joint --num_dots 8 --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED &
wait
echo "=== BATCH 1 COMPLETE ==="
echo ""

# Batch 2: Random (all 4) + Bayesian (all 4) = 8 jobs
echo "=== BATCH 2: Random + Bayesian ==="
run_on_gpu 0 "Random-2dots" uv run python random/run.py --num_dots 2 --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED &
run_on_gpu 1 "Random-4dots" uv run python random/run.py --num_dots 4 --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED &
run_on_gpu 2 "Random-6dots" uv run python random/run.py --num_dots 6 --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED &
run_on_gpu 3 "Random-8dots" uv run python random/run.py --num_dots 8 --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED &
run_on_gpu 4 "Bayes-2dots"  uv run python bayesian/run.py --mode joint --num_dots 2 --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED &
run_on_gpu 5 "Bayes-4dots"  uv run python bayesian/run.py --mode joint --num_dots 4 --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED &
run_on_gpu 6 "Bayes-6dots"  uv run python bayesian/run.py --mode joint --num_dots 6 --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED &
run_on_gpu 7 "Bayes-8dots"  uv run python bayesian/run.py --mode joint --num_dots 8 --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED &
wait
echo "=== BATCH 2 COMPLETE ==="
echo ""

echo "========================================"
echo "All benchmarks completed at $(date)"
echo "Results saved to: benchmarks/results/"
echo "========================================"
