import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_coupling():
    """
    Test script that creates an environment instance, initializes near ground truth,
    updates the virtual gate matrix with a custom hardcoded matrix, and saves
    a 3x3 grid of scans for the first dot pair to coupling.png
    """

    # Create environment instance
    env = QuantumDeviceEnv()
    obs, info = env.reset()

    # Initialize near ground truth
    gate_ground_truth = env.device_state["gate_ground_truth"]
    barrier_ground_truth = env.device_state["barrier_ground_truth"]

    env.device_state["current_gate_voltages"] = gate_ground_truth.copy()
    env.device_state["current_barrier_voltages"] = np.zeros_like(barrier_ground_truth)

    # Define custom virtual gate matrix (hardcoded)
    # This is a 4x4 matrix for a 4-dot system with strong coupling between dots
    custom_vgm = np.array([
        [1.0,  -0.2,  0.2,  0.1],
        [-0.2,  1.0,  0.4,  0.2],
        [0.2,  0.4,  1.0,  0.4],
        [0.1,  0.2,  0.4,  1.0]
    ])

    custom_vgm = np.array([
        [1.0,  0.,  0.1,  0],
        [0.,  1.0,  0.4,  0.2],
        [0.1,  0.4,  1.0,  0.4],
        [0,  0.2,  0.4,  1.0]
    ])

    custom_vgm = np.array([
        [1.0,  0.,  0,  0],
        [0.,  1.0,  0.,  0.],
        [0.,  0.,  1.0,  0.],
        [0,  0.,  0.,  1.0]
    ])

    # Update the virtual gate matrix in the environment
    # env.array._update_virtual_gate_matrix(custom_vgm)

    print(f"Number of gates: {env.num_dots}")
    print(f"Gate ground truth: {gate_ground_truth}")
    print(f"Barrier ground truth: {barrier_ground_truth}")
    print(f"\nCustom Virtual Gate Matrix:")
    print(custom_vgm)

    # Get other gate and barrier voltages
    other_gates = env.device_state["current_gate_voltages"][2:]
    barriers = np.zeros_like(env.device_state["current_barrier_voltages"])

    # Define the 5x5 grid centered at ground truth
    step_size = 10.0
    v0_points = gate_ground_truth[0] + np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])
    v1_points = gate_ground_truth[1] + np.array([-2*step_size, -step_size, 0.0, step_size, 2*step_size])

    print(f"\nVoltage grid for gates 0 and 1 (centered at ground truth):")
    print(f"  Gate 0 values: {v0_points}")
    print(f"  Gate 1 values: {v1_points}")

    # Collect 25 scans for the 5x5 grid (only first channel - left dot pair)
    scans_norm = []

    for v1 in v1_points:
        row_scans_norm = []

        for v0 in v0_points:
            # Set voltages for left dot pair
            gate_voltages = np.array([v0, v1] + list(other_gates))

            # Get observation directly from array
            raw_obs = env.array._get_obs(gate_voltages, barriers)

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

        scans_norm.append(row_scans_norm)

    # Create 5x5 grid figure
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle('Coupling Test - Left Dot Pair (Gates 0-1)', fontsize=16)

    # Plot scans (flip vertically so bottom row has min v1)
    for row_idx in range(5):
        for col_idx in range(5):
            ax = axes[4 - row_idx, col_idx]  # Flip vertically
            scan = scans_norm[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add voltage labels
            v0 = v0_points[col_idx]
            v1 = v1_points[row_idx]
            title = f"({v0:.0f}, {v1:.0f})"
            ax.set_title(title, fontsize=8)

    # Add axis labels
    fig.text(0.5, 0.04, 'Gate 0 Voltage (V) →', ha='center', fontsize=11)
    fig.text(0.04, 0.5, 'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=11)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])

    # Save to coupling.png in the environment directory
    output_path = Path(__file__).parent / 'coupling.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved coupling scans to {output_path}")


if __name__ == "__main__":
    test_coupling()
