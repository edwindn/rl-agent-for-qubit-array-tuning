#!/bin/bash
# Run all benchmark combinations SEQUENTIALLY to avoid GPU OOM
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
echo "========================================"
echo ""

# Run each benchmark sequentially
run_benchmark() {
    local name=$1
    shift
    echo "----------------------------------------"
    echo "[$name] Starting at $(date)"
    echo "----------------------------------------"
    "$@"
    echo "[$name] Completed at $(date)"
    echo ""
}

# Faster methods first (Nelder-Mead, L-BFGS, Random), then Bayesian

# Nelder-Mead
for dots in 2 4 6 8; do
    run_benchmark "Nelder-Mead ${dots} dots" uv run python nelder_mead/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED
done

# L-BFGS
for dots in 2 4 6 8; do
    run_benchmark "L-BFGS ${dots} dots" uv run python lbfgs/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED
done

# Random
for dots in 2 4 6 8; do
    run_benchmark "Random ${dots} dots" uv run python random/run.py --num_dots $dots --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED
done

# Bayesian (slowest - runs last)
for dots in 2 4 6 8; do
    run_benchmark "Bayesian ${dots} dots" uv run python bayesian/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED
done

echo "========================================"
echo "All benchmarks completed at $(date)"
echo "Results saved to: benchmarks/results/"
echo "========================================"
