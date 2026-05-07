#!/bin/bash
# Hyperparameter sweep for L-BFGS-B on 2 dots (pairwise mode)
# Results saved to results/lbfgs_ablation/
#
# Parameters swept:
#   - ftol: Function value convergence tolerance [1e-7, 1e-5, 1e-3]
#   - gtol: Gradient convergence tolerance [1e-6, 1e-4, 1e-2]
#   - maxcor: Memory size / Hessian approximation quality [5, 10, 20]
#
# Total combinations: 3 x 3 x 3 = 27

# Force CPU-only (no GPU usage)
export CUDA_VISIBLE_DEVICES=""

cd <repo>/benchmarks/lbfgs

OUTPUT_DIR="../results/lbfgs_ablation"
NUM_TRIALS=100
NUM_DOTS=2
MAX_SCANS=500
SEED=42

# Hyperparameter grid
FTOL=(1e-7 1e-5 1e-3)
GTOL=(1e-6 1e-4 1e-2)
MAXCOR=(5 10 20)

# Fixed parameters
CAP_PLUNGER=20.0
CAP_BARRIER=10.0
THRESH_PLUNGER=0.5
THRESH_BARRIER=1.0

# Run sweep in parallel (limit concurrent jobs)
MAX_JOBS=8

mkdir -p "$OUTPUT_DIR"

for ftol in "${FTOL[@]}"; do
    for gtol in "${GTOL[@]}"; do
        for maxcor in "${MAXCOR[@]}"; do
            # Wait if we have too many background jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done

            OUTPUT_FILE="${OUTPUT_DIR}/lbfgs_2dots_ftol${ftol}_gtol${gtol}_maxcor${maxcor}.json"

            echo "Running: ftol=${ftol}, gtol=${gtol}, maxcor=${maxcor}"

            python run.py \
                --mode pairwise \
                --num_dots $NUM_DOTS \
                --num_trials $NUM_TRIALS \
                --max_scans $MAX_SCANS \
                --seed $SEED \
                --ftol $ftol \
                --gtol $gtol \
                --maxcor $maxcor \
                --threshold_per_plunger $THRESH_PLUNGER \
                --threshold_per_barrier $THRESH_BARRIER \
                --cap_per_plunger $CAP_PLUNGER \
                --cap_per_barrier $CAP_BARRIER \
                --output "$OUTPUT_FILE" \
                > /dev/null 2>&1 &
        done
    done
done

# Wait for all jobs to complete
wait

echo "Sweep complete. Results in ${OUTPUT_DIR}/"
echo "Run 'python ../make_table.py ${OUTPUT_DIR}' to view results"
