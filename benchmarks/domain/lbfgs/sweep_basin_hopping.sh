#!/bin/bash
# Hyperparameter sweep for Basin Hopping with L-BFGS-B on 2 dots
# Results saved to results/basin_hopping_ablation/
#
# Parameters swept:
#   - bh_T: Temperature for Metropolis acceptance [0.5, 1.0, 2.0]
#   - bh_stepsize: Perturbation step size in volts [2.0, 5.0, 10.0]
#   - bh_niter: Number of basin hopping iterations [50, 100, 200]
#
# L-BFGS params fixed to best values from pairwise sweep (UPDATE THESE):
#   - ftol: 1e-5 (default, update after pairwise sweep)
#   - gtol: 1e-4 (default, update after pairwise sweep)
#   - maxcor: 10 (default, update after pairwise sweep)
#
# Total combinations: 3 x 3 x 3 = 27

# Force CPU-only (no GPU usage)
export CUDA_VISIBLE_DEVICES=""

cd <repo>/benchmarks/lbfgs

OUTPUT_DIR="../results/basin_hopping_ablation"
NUM_TRIALS=100
NUM_DOTS=2
MAX_SCANS=500
SEED=42

# Basin hopping parameters to sweep
BH_T=(0.5 1.0 2.0)
BH_STEPSIZE=(2.0 5.0 10.0)
BH_NITER=(50 100 200)

# L-BFGS params - from pairwise sweep (best was 1% with these)
FTOL=1e-5
GTOL=1e-2
MAXCOR=10

# Run sweep in parallel (limit concurrent jobs)
MAX_JOBS=8

mkdir -p "$OUTPUT_DIR"

for T in "${BH_T[@]}"; do
    for stepsize in "${BH_STEPSIZE[@]}"; do
        for niter in "${BH_NITER[@]}"; do
            # Wait if we have too many background jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done

            OUTPUT_FILE="${OUTPUT_DIR}/bh_2dots_T${T}_step${stepsize}_niter${niter}.json"

            echo "Running: T=${T}, stepsize=${stepsize}, niter=${niter}"

            python run.py \
                --mode basin_hopping \
                --num_dots $NUM_DOTS \
                --num_trials $NUM_TRIALS \
                --max_scans $MAX_SCANS \
                --seed $SEED \
                --bh_T $T \
                --bh_stepsize $stepsize \
                --bh_niter $niter \
                --ftol $FTOL \
                --gtol $GTOL \
                --maxcor $MAXCOR \
                --output "$OUTPUT_FILE" \
                > /dev/null 2>&1 &
        done
    done
done

# Wait for all jobs to complete
wait

echo "Sweep complete. Results in ${OUTPUT_DIR}/"
echo "Run 'python ../make_table.py ${OUTPUT_DIR}' to view results"
