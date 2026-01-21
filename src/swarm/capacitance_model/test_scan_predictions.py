"""
Test script to compare real vs virtualized scans in a grid layout.

Generates 20 pairs of scans:
- First scan: obtained directly from reset (real scan)
- Second scan: obtained with same input voltages but without reset (virtualized scan)

The grid layout is 4x10 (4 rows, 10 columns) where each pair is stacked vertically.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def generate_scan_grid(num_scans=20, num_dots=6, seed=42):
    """
    Generate a grid of real vs virtualized scan comparisons.

    Args:
        num_scans (int): Number of scan pairs to generate (default 20)
        num_dots (int): Number of dots in the quantum device (default 6)
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)

    # Initialize environment
    env = QuantumDeviceEnv(num_dots=num_dots, training=False)

    # Storage for scan pairs
    real_scans = []
    virtual_scans = []

    for i in range(num_scans):
        print(f"Generating scan pair {i+1}/{num_scans}")

        # Reset environment to randomize parameters
        obs, info = env.reset()

        # Get plunger ground truth with small random negative offset
        plunger_gt = info['current_device_state']['gate_ground_truth']
        offset = np.random.uniform(-0.1, 0.0, size=plunger_gt.shape)
        plunger_voltages = plunger_gt + offset

        # Set barrier voltages to zero
        barrier_voltages = np.zeros(env.num_barrier_voltages)

        # Get the real scan (from reset)
        real_scan = obs['image']
        real_scans.append(real_scan)

        # Now get virtualized scan with same voltages (without reset)
        # Use env.array._get_obs directly to bypass reset
        raw_virtual_obs = env.array._get_obs(plunger_voltages, barrier_voltages)
        virtual_obs = env._normalise_obs(raw_virtual_obs)
        virtual_scan = virtual_obs['image']
        virtual_scans.append(virtual_scan)

    # Plot the grid: 4 rows x 10 columns (each pair is 2 rows)
    # So we have 2 rows per pair, 10 pairs per row-set, 2 row-sets total
    num_cols = 10
    num_pairs_per_row = num_cols
    num_row_sets = (num_scans + num_pairs_per_row - 1) // num_pairs_per_row  # Ceiling division

    # Each pair takes 2 rows (real + virtual)
    num_rows = num_row_sets * 2

    # Assuming scans have shape (resolution, resolution, num_channels)
    # We'll show the first channel only for simplicity
    channel_idx = 0

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))

    # Ensure axes is 2D even if only one row/column
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    for pair_idx in range(num_scans):
        # Calculate position in grid
        row_set = pair_idx // num_pairs_per_row
        col = pair_idx % num_pairs_per_row

        # Real scan goes in first row of the row_set
        real_row = row_set * 2
        # Virtual scan goes in second row of the row_set
        virtual_row = row_set * 2 + 1

        # Plot real scan
        axes[real_row, col].imshow(
            real_scans[pair_idx][:, :, channel_idx],
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=1
        )
        axes[real_row, col].axis('off')
        if col == 0:
            axes[real_row, col].set_ylabel('Real', fontsize=8, rotation=0, ha='right', va='center')

        # Plot virtual scan
        axes[virtual_row, col].imshow(
            virtual_scans[pair_idx][:, :, channel_idx],
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=1
        )
        axes[virtual_row, col].axis('off')
        if col == 0:
            axes[virtual_row, col].set_ylabel('Virtual', fontsize=8, rotation=0, ha='right', va='center')

    # Hide unused subplots if num_scans is not a multiple of num_pairs_per_row
    for pair_idx in range(num_scans, num_row_sets * num_pairs_per_row):
        col = pair_idx % num_pairs_per_row
        row_set = pair_idx // num_pairs_per_row
        real_row = row_set * 2
        virtual_row = row_set * 2 + 1
        axes[real_row, col].axis('off')
        axes[virtual_row, col].axis('off')

    plt.suptitle('Real vs Virtualized Scans (Channel 0)', fontsize=14, y=0.995)
    plt.tight_layout()

    # Save to capacitance_model directory
    output_dir = Path(__file__).parent
    output_path = output_dir / 'scan_predictions_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scan grid to {output_path}")
    plt.close()


if __name__ == "__main__":
    generate_scan_grid(num_scans=20, num_dots=6, seed=42)
