import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_random_origin():
    """
    Test script that creates an environment instance, sets all plunger and gate
    values to zero, generates a 3x3 grid of scans for the first dot pair, and
    prints out the random virtual gate origin that was set. Also generates a
    second set of scans centered at the ground truth.
    """

    # Create environment instance
    env = QuantumDeviceEnv()
    obs, info = env.reset()

    # Set all gate voltages to zero
    env.device_state["current_gate_voltages"] = np.zeros(env.num_dots)

    # Set all barrier voltages to zero
    if env.use_barriers:
        env.device_state["current_barrier_voltages"] = np.zeros(env.num_dots - 1)

    # Print the random virtual gate origin
    virtual_gate_origin = env.array.model.gate_voltage_composer.virtual_gate_origin
    print(f"Random Virtual Gate Origin: {virtual_gate_origin}")

    # Get gate ground truth
    gate_ground_truth = env.device_state["gate_ground_truth"]
    print(f"Gate Ground Truth: {gate_ground_truth}")

    # Get gate and barrier voltages (all zeros)
    gate_voltages = env.device_state["current_gate_voltages"]
    barriers = env.device_state["current_barrier_voltages"] if env.use_barriers else None

    # Define the 5x5 grid centered at zero
    step_size = 10.0
    v0_points_zero = np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])
    v1_points_zero = np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])

    print(f"\n=== First set: centered at zero ===")
    print(f"Voltage grid for gates 0 and 1:")
    print(f"  Gate 0 values: {v0_points_zero}")
    print(f"  Gate 1 values: {v1_points_zero}")

    # Get other gates (gates 2 onward)
    other_gates = gate_voltages[2:]

    # Collect 25 scans for the 5x5 grid centered at zero (only first channel - left dot pair)
    scans_norm_zero = []

    for v1 in v1_points_zero:
        row_scans_norm = []

        for v0 in v0_points_zero:
            # Set voltages for left dot pair
            gate_voltages_scan = np.array([v0, v1] + list(other_gates))

            # Get observation directly from array
            raw_obs = env.array._get_obs(gate_voltages_scan, barriers)

            # Extract first channel (left dot pair: gates 0 and 1)
            scan = raw_obs["image"][:, :, 0].copy()

            # Normalize for visualization
            p_low = np.percentile(scan, 0.5)
            p_high = np.percentile(scan, 99.5)
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scan_norm = np.clip(scan_norm, 0, 1)

            row_scans_norm.append(scan_norm)
            print(f"  Collected scan at ({v0:.0f}, {v1:.0f})")

        scans_norm_zero.append(row_scans_norm)

    # Create 5x5 grid figure for zero-centered scans
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Random Origin Test - Centered at Zero', fontsize=16)

    # Plot scans (flip vertically so bottom row has min v1)
    for row_idx in range(5):
        for col_idx in range(5):
            ax = axes[4 - row_idx, col_idx]  # Flip vertically
            scan = scans_norm_zero[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add voltage labels
            v0 = v0_points_zero[col_idx]
            v1 = v1_points_zero[row_idx]
            title = f"({v0:.0f}, {v1:.0f})"
            ax.set_title(title, fontsize=8)

    # Add axis labels
    fig.text(0.5, 0.04, 'Gate 0 Voltage (V) →', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

    # Save to random_origin_zero.png in the environment directory
    output_path = Path(__file__).parent / 'random_origin_zero.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved zero-centered scans to {output_path}")
    plt.close()

    # Now generate second set of scans centered at ground truth
    print(f"\n=== Second set: centered at ground truth ===")
    v0_points_gt = gate_ground_truth[0] + np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])
    v1_points_gt = gate_ground_truth[1] + np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])

    print(f"Voltage grid for gates 0 and 1:")
    print(f"  Gate 0 values: {v0_points_gt}")
    print(f"  Gate 1 values: {v1_points_gt}")

    # Collect 25 scans for the 5x5 grid centered at ground truth
    scans_norm_gt = []

    for v1 in v1_points_gt:
        row_scans_norm = []

        for v0 in v0_points_gt:
            # Set voltages for left dot pair
            gate_voltages_scan = np.array([v0, v1] + list(other_gates))

            # Get observation directly from array
            raw_obs = env.array._get_obs(gate_voltages_scan, barriers)

            # Extract first channel (left dot pair: gates 0 and 1)
            scan = raw_obs["image"][:, :, 0].copy()

            # Normalize for visualization
            p_low = np.percentile(scan, 0.5)
            p_high = np.percentile(scan, 99.5)
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scan_norm = np.clip(scan_norm, 0, 1)

            row_scans_norm.append(scan_norm)
            print(f"  Collected scan at ({v0:.0f}, {v1:.0f})")

        scans_norm_gt.append(row_scans_norm)

    # Create 5x5 grid figure for ground truth-centered scans
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Random Origin Test - Centered at Ground Truth', fontsize=16)

    # Plot scans (flip vertically so bottom row has min v1)
    for row_idx in range(5):
        for col_idx in range(5):
            ax = axes[4 - row_idx, col_idx]  # Flip vertically
            scan = scans_norm_gt[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add voltage labels
            v0 = v0_points_gt[col_idx]
            v1 = v1_points_gt[row_idx]
            title = f"({v0:.0f}, {v1:.0f})"
            ax.set_title(title, fontsize=8)

    # Add axis labels
    fig.text(0.5, 0.04, 'Gate 0 Voltage (V) →', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

    # Save to random_origin_gt.png in the environment directory
    output_path_gt = Path(__file__).parent / 'random_origin_gt.png'
    plt.savefig(output_path_gt, dpi=150, bbox_inches='tight')
    print(f"\nSaved ground truth-centered scans to {output_path_gt}")


if __name__ == "__main__":
    test_random_origin()
