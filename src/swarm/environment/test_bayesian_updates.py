"""
Test script for Bayesian capacitance model updates.

Performs 5 update steps and saves a 3x3 grid of scans centered around zero volts
for the left dot pair (gates 0 and 1) at each step. Passes all 9 scans through the
capacitance model and updates the virtual gate matrix with the weighted mean prediction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def main():
    import torch

    # Initialize environment with Bayesian capacitance model
    env = QuantumDeviceEnv(training=False)

    # Reset to get initial state
    obs, info = env.reset(seed=42)

    # Reset VGM to identity AFTER the automatic update in reset()
    # This ensures the first set of scans uses identity VGM (no virtualization)
    env.array._reset_virtual_gate_matrix_to_identity()

    # Get other gate and barrier voltages
    other_gates = env.device_state["current_gate_voltages"][2:]
    barriers = env.device_state["current_barrier_voltages"]

    print(f"Number of gates: {env.num_dots}")
    print(f"Number of barriers: {len(barriers)}")
    print(f"\nVirtual gate matrix after reset to identity:")
    print(env.array.model.gate_voltage_composer.virtual_gate_matrix)

    # Check if capacitance model is available
    use_ml_model = env.capacitance_model is not None and isinstance(env.capacitance_model, dict)

    if not use_ml_model:
        print("WARNING: Capacitance model not available or not an ML model")
        print("Will plot first set of scans only without predictions")
        num_update_steps = 1
    else:
        num_update_steps = 5

    # Define the 3x3 grid centered at zero volts
    # Using 10V steps as in map_device_range
    step_size = 10.0
    v0_points = np.array([-step_size, 0.0, step_size])
    v1_points = np.array([-step_size, 0.0, step_size])

    print(f"\nVoltage grid for gates 0 and 1:")
    print(f"  Gate 0 values: {v0_points}")
    print(f"  Gate 1 values: {v1_points}")

    # Perform update steps (5 if ML model available, 1 otherwise)
    for update_step in range(num_update_steps):
        print(f"\n=== Update Step {update_step + 1} ===")

        # Collect 9 scans for the 3x3 grid
        scans_raw = []
        scans_norm = []
        predictions = []
        uncertainties = []

        for v1 in v1_points:
            row_scans_raw = []
            row_scans_norm = []
            row_predictions = []
            row_uncertainties = []

            for v0 in v0_points:
                # Set voltages for left dot pair
                gate_voltages = np.array([v0, v1] + list(other_gates))

                # Get observation directly from array
                raw_obs = env.array._get_obs(gate_voltages, barriers)

                # Store raw observation for ML model
                row_scans_raw.append(raw_obs)

                # Extract first channel (left dot pair: gates 0 and 1)
                scan = raw_obs["image"][:, :, 0].copy()

                # Normalize for visualization (same as map_device_range)
                p_low = np.percentile(scan, 0.5)
                p_high = np.percentile(scan, 99.5)
                if p_high > p_low:
                    scan_norm = (scan - p_low) / (p_high - p_low)
                else:
                    scan_norm = np.zeros_like(scan)
                scan_norm = np.clip(scan_norm, 0, 1)

                row_scans_norm.append(scan_norm)

                if use_ml_model:
                    # Normalize for ML model
                    obs_normalized = env._normalise_obs(raw_obs)

                    # Get capacitance prediction from the model
                    image = obs_normalized["image"]
                    batch_tensor = (
                        torch.from_numpy(image)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(1)
                        .to(env.capacitance_model["device"])
                    )

                    with torch.no_grad():
                        values, log_vars = env.capacitance_model["ml_model"](batch_tensor)

                    values_np = values.cpu().numpy()  # Shape: (num_dots-1, num_outputs)
                    log_vars_np = log_vars.cpu().numpy()

                    # Extract predictions for the first channel (left dot pair)
                    pred = values_np[0]  # First channel's predictions
                    log_var = log_vars_np[0]
                    uncertainty = np.exp(0.5 * log_var)  # Convert log variance to std

                    row_predictions.append(pred)
                    row_uncertainties.append(uncertainty)

                print(f"  Collected scan at ({v0:.0f}, {v1:.0f})")

            scans_raw.append(row_scans_raw)
            scans_norm.append(row_scans_norm)
            predictions.append(row_predictions)
            uncertainties.append(row_uncertainties)

        if use_ml_model:
            # Flatten lists for weighted mean calculation
            predictions_flat = [pred for row in predictions for pred in row]
            uncertainties_flat = [unc for row in uncertainties for unc in row]

            # Calculate weighted mean using a = 1 / (1 + uncertainty)
            # predictions_flat[i] has shape (num_outputs,)
            # uncertainties_flat[i] has shape (num_outputs,)

            num_outputs = predictions_flat[0].shape[0]
            weighted_predictions = np.zeros(num_outputs)

            for output_idx in range(num_outputs):
                weights = []
                values = []

                for pred, unc in zip(predictions_flat, uncertainties_flat):
                    # Weight for this prediction
                    a = 1.0 / (1.0 + unc[output_idx])
                    weights.append(a)
                    values.append(pred[output_idx])

                # Weighted mean: sum(pred * weight) / sum(weights)
                weighted_predictions[output_idx] = np.sum(np.array(values) * np.array(weights)) / np.sum(weights)

            print(f"\n  Weighted mean prediction: {weighted_predictions}")
        else:
            weighted_predictions = None
            num_outputs = None

        # Create 3x3 grid figure with predictions
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        fig.suptitle(f'Bayesian Update Step {update_step + 1} - Left Dot Pair', fontsize=14)

        # Plot scans (flip vertically so bottom row has min v1)
        for row_idx in range(3):
            for col_idx in range(3):
                ax = axes[2 - row_idx, col_idx]  # Flip vertically
                scan = scans_norm[row_idx][col_idx]

                ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
                ax.set_xticks([])
                ax.set_yticks([])

                # Add voltage labels and predictions
                v0 = v0_points[col_idx]
                v1 = v1_points[row_idx]

                if use_ml_model:
                    pred = predictions[row_idx][col_idx]
                    unc = uncertainties[row_idx][col_idx]

                    # Format prediction and uncertainty strings
                    pred_str = ', '.join([f'{p:.3f}' for p in pred])
                    unc_str = ', '.join([f'{u:.3f}' for u in unc])

                    title = f"({v0:.0f}, {v1:.0f})\nP: [{pred_str}]\nU: [{unc_str}]"
                else:
                    title = f"({v0:.0f}, {v1:.0f})"

                ax.set_title(title, fontsize=7)

        # Add axis labels
        fig.text(0.5, 0.04, 'Gate 0 Voltage (V) →', ha='center', fontsize=11)
        fig.text(0.04, 0.5, 'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=11)

        if use_ml_model:
            # Add weighted mean to the figure
            weighted_str = ', '.join([f'{w:.3f}' for w in weighted_predictions])
            fig.text(0.5, 0.97, f'Weighted Mean: [{weighted_str}]', ha='center', fontsize=10, weight='bold')
            plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        else:
            plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

        # Save figure
        output_dir = Path(__file__).parent
        output_path = output_dir / f'bayesian_update_step_{update_step + 1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved grid to {output_path}")
        plt.close()

        # Update VGM for all steps EXCEPT the first one
        # The first step shows scans with identity VGM (no virtualization)
        # Subsequent steps show progressive virtualization as the model learns
        if use_ml_model and update_step < num_update_steps - 1:
            # Update using the bottom left scan (-step_size, -step_size)
            # This will process the scan through the ML model and update the Bayesian predictor
            bottom_left_gate_voltages = np.array([-step_size, -step_size] + list(other_gates))
            bottom_left_raw_obs = env.array._get_obs(bottom_left_gate_voltages, barriers)
            bottom_left_obs_normalized = env._normalise_obs(bottom_left_raw_obs)

            # Use the environment's built-in update method
            env._update_virtual_gate_matrix(bottom_left_obs_normalized)

            print(f"  Updated virtual gate matrix using bottom left scan at ({-step_size:.0f}, {-step_size:.0f})")
            print(f"  VGM after update:")
            print(env.array.model.gate_voltage_composer.virtual_gate_matrix)

    if use_ml_model:
        print(f"\n=== Completed {num_update_steps} update steps ===")
    else:
        print(f"\n=== Plotted first set of scans (no ML model available) ===")


if __name__ == "__main__":
    main()
