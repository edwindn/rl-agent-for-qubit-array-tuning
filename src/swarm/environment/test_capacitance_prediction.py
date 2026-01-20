"""
Test capacitance prediction by sampling quantum device environment in the middle quarter.

Creates a grid of charge stability diagram scans, limited to the middle half of each
voltage range (middle quarter of the total space), to test capacitance predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent
src_dir = swarm_dir.parent
sys.path.insert(0, str(src_dir))

from env import QuantumDeviceEnv
from swarm.capacitance_model.CapacitancePrediction import CapacitancePredictionModel
from swarm.capacitance_model.dataloader import PercentileNormalize


def test_capacitance_prediction(step_size=10, output_path="capacitance_prediction_test.png",
                                  weights_path=None, v0_min=-60, v0_max=60, v1_min=-60, v1_max=60):
    """
    Sample the quantum device environment at intervals across the voltage range.

    Sweeps the full space symmetrically about zero, just like map_device_range.

    Args:
        step_size: Voltage step size for sampling (default: 10V)
        output_path: Path to save the output image
        weights_path: Path to capacitance model weights (default: best_model_barriers.pth)
        v0_min, v0_max: Voltage range for gate 0 (default: -60 to 60V)
        v1_min, v1_max: Voltage range for gate 1 (default: -60 to 60V)
    """
    # Load capacitance model
    if weights_path is None:
        weights_path = os.path.join(current_dir.parent, 'capacitance_model', 'weights', 'best_model_barriers.pth')

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model = CapacitancePredictionModel(output_size=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded capacitance model from: {weights_path}")

    normalizer = PercentileNormalize()

    # Initialize environment
    env = QuantumDeviceEnv(training=True)
    obs, info = env.reset()

    print(f"Using voltage ranges:")
    print(f"  Gate 0: [{v0_min:.2f}, {v0_max:.2f}] V")
    print(f"  Gate 1: [{v1_min:.2f}, {v1_max:.2f}] V")
    print(f"  Ground truth: {info['current_device_state']['gate_ground_truth'][:2]}")

    # Create voltage grids
    v0_points = np.arange(v0_min, v0_max + step_size, step_size)
    v1_points = np.arange(v1_min, v1_max + step_size, step_size)

    n_cols = len(v0_points)  # Gate 0 increases as we move right
    n_rows = len(v1_points)  # Gate 1 increases as we move up

    print(f"\nCreating {n_rows} x {n_cols} grid ({n_rows * n_cols} scans)")
    print(f"Step size: {step_size} V")

    # Get other gate voltages (keep them at initial values)
    other_gates = env.device_state["current_gate_voltages"][2:]
    barriers = env.device_state["current_barrier_voltages"]

    # Collect all scans and predictions
    scans = []
    predictions = []
    uncertainties = []

    for v1 in v1_points:
        row_scans = []
        row_preds = []
        row_uncerts = []
        for v0 in v0_points:
            # Set voltages
            gate_voltages = np.array([v0, v1] + list(other_gates))

            # Get observation (first channel is scan between gates 0 and 1)
            raw_obs = env.array._get_obs(gate_voltages, barriers)
            scan = raw_obs["image"][:, :, 0]  # First channel

            # Normalize for visualization
            p_low = np.percentile(scan, 0.5)
            p_high = np.percentile(scan, 99.5)
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scan_norm = np.clip(scan_norm, 0, 1)

            # Run through capacitance model
            with torch.no_grad():
                # Prepare image for model (apply same normalization as training)
                model_input = normalizer(torch.from_numpy(scan).float())
                model_input = model_input.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

                # Get predictions
                values, log_vars = model(model_input)
                values = values.cpu().numpy().flatten()
                log_vars = log_vars.cpu().numpy().flatten()

                # Convert log variance to standard deviation: std = exp(log_var / 2) = sqrt(exp(log_var))
                stds = np.exp(log_vars / 2)

            row_scans.append(scan_norm)
            row_preds.append(values)
            row_uncerts.append(stds)
        scans.append(row_scans)
        predictions.append(row_preds)
        uncertainties.append(row_uncerts)

    # Create figure with grid of scans
    # Bottom row (index 0) should be minimum v1, top row should be maximum v1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    # Ensure axes is 2D array even for single row/column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot scans (flip vertically so bottom row has min v1)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[n_rows - 1 - row_idx, col_idx]  # Flip vertically
            scan = scans[row_idx][col_idx]
            preds = predictions[row_idx][col_idx]
            uncerts = uncertainties[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add voltage labels and predictions
            v0 = v0_points[col_idx]
            v1 = v1_points[row_idx]
            ax.set_title(f"({v0:.0f}, {v1:.0f})", fontsize=8, pad=2)

            # Add predictions below image
            pred_text = f"C: [{preds[0]:.3f}, {preds[1]:.3f}]\nstd: [{uncerts[0]:.3f}, {uncerts[1]:.3f}]"
            ax.text(0.5, -0.05, pred_text, transform=ax.transAxes,
                   ha='center', va='top', fontsize=6, family='monospace')

    # Add overall labels
    fig.text(0.5, 0.01, f'Gate 0 Voltage (V) →', ha='center', fontsize=12)
    fig.text(0.01, 0.5, f'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.98])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add space for text below images
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved capacitance prediction test to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test capacitance prediction across voltage range")
    parser.add_argument("--step-size", type=float, default=10.0,
                        help="Voltage step size in volts (default: 10)")
    parser.add_argument("--output", type=str, default="capacitance_prediction_test.png",
                        help="Output image path (default: capacitance_prediction_test.png)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (default: best_model_barriers.pth)")
    parser.add_argument("--v0-min", type=float, default=-60,
                        help="Minimum voltage for gate 0 (default: -60)")
    parser.add_argument("--v0-max", type=float, default=60,
                        help="Maximum voltage for gate 0 (default: 60)")
    parser.add_argument("--v1-min", type=float, default=-60,
                        help="Minimum voltage for gate 1 (default: -60)")
    parser.add_argument("--v1-max", type=float, default=60,
                        help="Maximum voltage for gate 1 (default: 60)")

    args = parser.parse_args()

    test_capacitance_prediction(
        step_size=args.step_size,
        output_path=args.output,
        weights_path=args.weights,
        v0_min=args.v0_min,
        v0_max=args.v0_max,
        v1_min=args.v1_min,
        v1_max=args.v1_max
    )
