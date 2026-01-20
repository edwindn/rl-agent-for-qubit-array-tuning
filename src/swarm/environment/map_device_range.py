"""
Map device range by sampling quantum device environment at different voltage positions.

Creates a grid of charge stability diagram scans across the full voltage range,
showing how the CSD changes as we move through the search space.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import QuantumDeviceEnv


def map_device_range(step_size=10, output_path="device_range_map.png",
                     v0_min=None, v0_max=None, v1_min=None, v1_max=None):
    """
    Sample the quantum device environment at intervals across the voltage range.

    Args:
        step_size: Voltage step size for sampling (default: 10V)
        output_path: Path to save the output image
        v0_min, v0_max: Override min/max voltage for gate 0 (default: use env range)
        v1_min, v1_max: Override min/max voltage for gate 1 (default: use env range)
    """
    # Initialize environment
    env = QuantumDeviceEnv(training=True)
    obs, info = env.reset()

    # Get voltage ranges for first two plunger gates
    plunger_min = env.plunger_min[:2].copy()  # First two gates
    plunger_max = env.plunger_max[:2].copy()  # First two gates

    print(f"Environment voltage ranges:")
    print(f"  Gate 0: [{plunger_min[0]:.2f}, {plunger_max[0]:.2f}] V")
    print(f"  Gate 1: [{plunger_min[1]:.2f}, {plunger_max[1]:.2f}] V")

    # Override with user-specified ranges if provided
    if v0_min is not None:
        plunger_min[0] = v0_min
        print(f"  Overriding gate 0 min: {v0_min} V")
    if v0_max is not None:
        plunger_max[0] = v0_max
        print(f"  Overriding gate 0 max: {v0_max} V")
    if v1_min is not None:
        plunger_min[1] = v1_min
        print(f"  Overriding gate 1 min: {v1_min} V")
    if v1_max is not None:
        plunger_max[1] = v1_max
        print(f"  Overriding gate 1 max: {v1_max} V")

    print(f"\nUsing voltage ranges:")
    print(f"  Gate 0: [{plunger_min[0]:.2f}, {plunger_max[0]:.2f}] V")
    print(f"  Gate 1: [{plunger_min[1]:.2f}, {plunger_max[1]:.2f}] V")

    # Get ground truth
    gt_v0 = info['current_device_state']['gate_ground_truth'][0]
    gt_v1 = info['current_device_state']['gate_ground_truth'][1]
    print(f"  Ground truth: [{gt_v0:.2f}, {gt_v1:.2f}]")

    # Calculate voltage grids centered on ground truth
    # For 13 scans, we want 6 steps on each side of center: [-6, -5, ..., 0, ..., 5, 6]
    n_scans_per_side = 6  # Number of scans on each side of center (13 total = 6 + 1 + 6)

    # Create voltage points centered exactly on ground truth
    v0_offsets = np.arange(-n_scans_per_side, n_scans_per_side + 1) * step_size
    v1_offsets = np.arange(-n_scans_per_side, n_scans_per_side + 1) * step_size

    v0_points = gt_v0 + v0_offsets
    v1_points = gt_v1 + v1_offsets

    print(f"\nVoltage grid centered on ground truth:")
    print(f"  Gate 0: [{v0_points[0]:.2f}, {v0_points[-1]:.2f}] V (center: {gt_v0:.2f})")
    print(f"  Gate 1: [{v1_points[0]:.2f}, {v1_points[-1]:.2f}] V (center: {gt_v1:.2f})")

    n_cols = len(v0_points)  # Gate 0 increases as we move right
    n_rows = len(v1_points)  # Gate 1 increases as we move up

    print(f"\nCreating {n_rows} x {n_cols} grid ({n_rows * n_cols} scans)")
    print(f"Step size: {step_size} V")

    # Get other gate voltages (keep them at initial values)
    other_gates = env.device_state["current_gate_voltages"][2:]
    barriers = env.device_state["current_barrier_voltages"]

    # Collect all scans
    scans = []
    for v1 in v1_points:
        row_scans = []
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

            row_scans.append(scan_norm)
        scans.append(row_scans)

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

    # Get observation voltage range for each scan window
    obs_v_min = env.array.obs_voltage_min
    obs_v_max = env.array.obs_voltage_max

    # Plot scans (flip vertically so bottom row has min v1)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[n_rows - 1 - row_idx, col_idx]  # Flip vertically
            scan = scans[row_idx][col_idx]

            # Center voltages of this scan
            v0_center = v0_points[col_idx]
            v1_center = v1_points[row_idx]

            # Calculate scan window bounds
            v0_scan_min = v0_center + obs_v_min
            v0_scan_max = v0_center + obs_v_max
            v1_scan_min = v1_center + obs_v_min
            v1_scan_max = v1_center + obs_v_max

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto',
                     extent=[v0_scan_min, v0_scan_max, v1_scan_min, v1_scan_max])

            # Check if ground truth is within this scan window
            if (v0_scan_min <= gt_v0 <= v0_scan_max and
                v1_scan_min <= gt_v1 <= v1_scan_max):
                # Plot ground truth as red dot
                ax.plot(gt_v0, gt_v1, 'ro', markersize=4, markeredgewidth=0.5,
                       markeredgecolor='white', zorder=10)

            ax.set_xticks([])
            ax.set_yticks([])

            # Add voltage labels (center voltages)
            ax.set_title(f"({v0_center:.0f}, {v1_center:.0f})", fontsize=8)

    # Add overall labels
    fig.text(0.5, 0.02, f'Gate 0 Voltage (V) →', ha='center', fontsize=12)
    fig.text(0.02, 0.5, f'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved device range map to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map quantum device voltage range")
    parser.add_argument("--step-size", type=float, default=10.0,
                        help="Voltage step size in volts (default: 10)")
    parser.add_argument("--output", type=str, default="device_range_map.png",
                        help="Output image path (default: device_range_map.png)")
    parser.add_argument("--v0-min", type=float, default=-60,
                        help="Minimum voltage for gate 0")
    parser.add_argument("--v0-max", type=float, default=60,
                        help="Maximum voltage for gate 0")
    parser.add_argument("--v1-min", type=float, default=-60,
                        help="Minimum voltage for gate 1")
    parser.add_argument("--v1-max", type=float, default=60,
                        help="Maximum voltage for gate 1")

    args = parser.parse_args()

    map_device_range(
        step_size=args.step_size,
        output_path=args.output,
        v0_min=args.v0_min,
        v0_max=args.v0_max,
        v1_min=args.v1_min,
        v1_max=args.v1_max
    )
