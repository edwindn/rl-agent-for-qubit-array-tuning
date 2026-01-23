#!/bin/bash
# Run all benchmark combinations in parallel tmux sessions
# Methods: nelder_mead, lbfgs, bayesian, random
# Dots: 2, 4, 6, 8
# Trials: 100, Steps: 2000

SESSION_NAME="benchmarks"
TRIALS=100
MAX_ITER=2000
MAX_SCANS=2000
MAX_SAMPLES=2000
SEED=42

cd /home/rahul/rl-agent-for-qubit-array-tuning/benchmarks

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -n "main"

# Function to create a new tmux window and run command
run_benchmark() {
    local method=$1
    local dots=$2
    local cmd=$3
    local window_name="${method}_${dots}dots"

    tmux new-window -t $SESSION_NAME -n "$window_name"
    tmux send-keys -t $SESSION_NAME:"$window_name" "cd /home/rahul/rl-agent-for-qubit-array-tuning/benchmarks && $cmd" C-m
}

# Nelder-Mead runs (joint mode - faster)
for dots in 2 4 6 8; do
    run_benchmark "nm" $dots "uv run python nelder_mead/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED"
done

# L-BFGS runs (joint mode - faster)
for dots in 2 4 6 8; do
    run_benchmark "lbfgs" $dots "uv run python lbfgs/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --seed $SEED"
done

# Random sampling runs
for dots in 2 4 6 8; do
    run_benchmark "random" $dots "uv run python random/run.py --num_dots $dots --num_trials $TRIALS --max_samples $MAX_SAMPLES --seed $SEED"
done

# Bayesian optimization runs (slowest - will take longer)
for dots in 2 4 6 8; do
    run_benchmark "bayes" $dots "uv run python bayesian/run.py --mode joint --num_dots $dots --num_trials $TRIALS --max_iter $MAX_ITER --n_initial 20 --seed $SEED"
done

# Close the initial empty "main" window
tmux kill-window -t $SESSION_NAME:main

echo "All benchmarks launched in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME     # Attach to session"
echo "  tmux ls                          # List sessions"
echo "  Ctrl+b n / Ctrl+b p              # Next/prev window"
echo "  Ctrl+b d                         # Detach"
echo ""
echo "Running 16 benchmarks total:"
echo "  - Nelder-Mead: 2, 4, 6, 8 dots"
echo "  - L-BFGS: 2, 4, 6, 8 dots"
echo "  - Random: 2, 4, 6, 8 dots"
echo "  - Bayesian: 2, 4, 6, 8 dots (slowest)"
