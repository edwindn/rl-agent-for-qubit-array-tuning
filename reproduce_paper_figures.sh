#!/usr/bin/env bash
#
# Regenerate the 4 paper figures from existing data.
#
# Assumes:
#   - data/episode_data_1000ep.npy exists (panels a, b, c)
#   - data/scaling_10dot/, data/scaling_12dot/ exist (panel d)
#   - benchmarks/results/final_{2,4,6,8}dot/ppo_*.json exist (panel d)
#
# See src/swarm/capacitance_model/README.md for how to regenerate the data.

set -euo pipefail

cd "$(dirname "$0")"

echo "=== Panels a, b, c: Kalman calibration + CNN convergence ==="
uv run python src/swarm/capacitance_model/plot_kalman_calibration.py \
    --data data/episode_data_1000ep.npy \
    --output-dir .

echo ""
echo "=== Panel d: Scaling ==="
uv run python src/swarm/capacitance_model/plot_scaling.py \
    --output scaling_paper.svg

echo ""
echo "Outputs:"
ls -la kalman_calibration_nn.{svg,png} \
       kalman_calibration_nnn.{svg,png} \
       model_convergence_paper.{svg,png} \
       scaling_paper.{svg,png}
