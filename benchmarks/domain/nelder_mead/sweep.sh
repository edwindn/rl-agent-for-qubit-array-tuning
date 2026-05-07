#!/bin/bash
# Hyperparameter sweep for Nelder-Mead on 2 dots
# Results saved to results/nelder_mead_ablation/

# Force CPU-only (no GPU usage)
export CUDA_VISIBLE_DEVICES=""

cd <repo>/benchmarks/nelder_mead

OUTPUT_DIR="../results/nelder_mead_ablation"
NUM_TRIALS=100
NUM_DOTS=2
MAX_SCANS=500
SEED=42

# Hyperparameter grid
SIMPLEX_PLUNGER=(20 35 50 70)
SIMPLEX_BARRIER=(2 4 8)
TOL=(0.05 0.1 0.5)

# Run sweep in parallel (limit concurrent jobs)
MAX_JOBS=8

for sp in "${SIMPLEX_PLUNGER[@]}"; do
    for sb in "${SIMPLEX_BARRIER[@]}"; do
        for tol in "${TOL[@]}"; do
            # Wait if we have too many background jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1
            done

            OUTPUT_FILE="${OUTPUT_DIR}/nm_2dots_sp${sp}_sb${sb}_tol${tol}.json"

            echo "Running: simplex_plunger=${sp}, simplex_barrier=${sb}, tol=${tol}"

            python run.py \
                --mode pairwise \
                --num_dots $NUM_DOTS \
                --num_trials $NUM_TRIALS \
                --max_scans $MAX_SCANS \
                --seed $SEED \
                --simplex_step_plunger $sp \
                --simplex_step_barrier $sb \
                --xatol $tol \
                --fatol $tol \
                --output "$OUTPUT_FILE" \
                > /dev/null 2>&1 &
        done
    done
done

# Wait for all jobs to complete
wait

echo "Sweep complete. Results in ${OUTPUT_DIR}/"
