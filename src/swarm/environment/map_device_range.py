"""
Map device range by sampling quantum device environment at different voltage positions.

Creates a grid of charge stability diagram scans across the full voltage range,
showing how the CSD changes as we move through the search space.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import QuantumDeviceEnv


def map_device_range(output_path="device_range_map.png", n_grid=11, percentile=None):
    """
    Generate a grid of scans sampling the device voltage range.

    Creates an n_grid × n_grid grid of scans centered at ground truth,
    with symmetric voltage range of ±60V on both axes.

    Args:
        output_path: Path to save the output image
        n_grid: Number of scans per axis (default: 11)
        percentile: Optional percentile (0-100) to cap values after normalization
    """
    # Initialize environment
    env = QuantumDeviceEnv(training=True)
    obs, info = env.reset()

    # Get ground truth for all gates and barriers
    gt_gates = info['current_device_state']['gate_ground_truth']
    gt_barriers = info['current_device_state']['barrier_ground_truth']

    print(f"Ground truth gates: {gt_gates}")
    print(f"Ground truth barriers: {gt_barriers}")

    # Set symmetric voltage range centered at ground truth
    v0_min = gt_gates[0] - 20
    v0_max = gt_gates[0] + 20
    v1_min = gt_gates[1] - 20
    v1_max = gt_gates[1] + 20

    print(f"\nVoltage ranges (centered at ground truth):")
    print(f"  Gate 0: [{v0_min:.2f}, {v0_max:.2f}] V")
    print(f"  Gate 1: [{v1_min:.2f}, {v1_max:.2f}] V")

    # Set scan window size to 3V
    scan_window_size = 3.0
    print(f"\nScan window size: {scan_window_size:.2f} V")

    # Calculate how many scans needed to tile the space without gaps
    v0_range = v0_max - v0_min
    v1_range = v1_max - v1_min

    n_scans_x = int(np.ceil(v0_range / scan_window_size))
    n_scans_y = int(np.ceil(v1_range / scan_window_size))

    print(f"Number of scans to tile the range: {n_scans_x} × {n_scans_y} = {n_scans_x * n_scans_y} total scans")

    # Calculate step size to evenly tile the space
    # Scans are positioned edge-to-edge with no gaps
    step_x = scan_window_size
    step_y = scan_window_size

    print(f"Step size between scans: ({step_x:.2f}, {step_y:.2f}) V")

    # Store all scans and their positions
    all_scans = []
    scan_positions = []

    # Use standard resolution
    resolution = env.array.obs_image_size
    print(f"Scan resolution: {resolution}x{resolution}")

    # Generate scans at grid positions that tile the space
    scan_count = 0
    for i in range(n_scans_x):
        for j in range(n_scans_y):
            scan_count += 1

            # Calculate scan boundaries - scans tile edge-to-edge
            scan_min_v0 = v0_min + i * step_x
            scan_max_v0 = scan_min_v0 + scan_window_size
            scan_min_v1 = v1_min + j * step_y
            scan_max_v1 = scan_min_v1 + scan_window_size

            # Center of this scan
            center_v0 = (scan_min_v0 + scan_max_v0) / 2
            center_v1 = (scan_min_v1 + scan_max_v1) / 2

            # Temporarily set the environment's observation range for this scan
            env.array.obs_voltage_min = scan_min_v0
            env.array.obs_voltage_max = scan_max_v0

            # Get the scan with plunger gates at (center_v0, center_v1)
            temp_gates = gt_gates.copy()
            temp_gates[0] = center_v0
            temp_gates[1] = center_v1

            raw_obs = env.array._get_obs(temp_gates, gt_barriers)
            scan = raw_obs["image"][:, :, 0]  # First channel

            all_scans.append(scan)
            scan_positions.append((center_v0, center_v1, scan_min_v0, scan_max_v0, scan_min_v1, scan_max_v1))

            print(f"  Scan {scan_count}/{n_scans_x * n_scans_y}: "
                  f"v0=[{scan_min_v0:.1f}, {scan_max_v0:.1f}], "
                  f"v1=[{scan_min_v1:.1f}, {scan_max_v1:.1f}]")

    # Save scan data
    import pickle

    data_path = output_path.replace('.png', '_data.pkl')
    scan_data = {
        'scans': all_scans,
        'positions': scan_positions,
        'n_scans_x': n_scans_x,
        'n_scans_y': n_scans_y,
        'resolution': resolution,
        'v0_min': v0_min,
        'v0_max': v0_max,
        'v1_min': v1_min,
        'v1_max': v1_max,
        'scan_window_size': scan_window_size,
        'step_x': step_x,
        'step_y': step_y,
        'gt_gates': gt_gates,
        'gt_barriers': gt_barriers
    }

    with open(data_path, 'wb') as f:
        pickle.dump(scan_data, f)
    print(f"\nSaved scan data to: {data_path}")

    # Normalize each scan individually
    normalized_scans = []
    for scan in all_scans:
        # Calculate percentiles for this scan
        p_low = np.percentile(scan, 0.5)
        p_high = np.percentile(scan, 99.5)

        if p_high > p_low:
            scan_norm = (scan - p_low) / (p_high - p_low)
        else:
            scan_norm = np.zeros_like(scan)
        scan_norm = np.clip(scan_norm, 0, 1)
        normalized_scans.append(scan_norm)

    # Optionally cap values at specified percentile (computed globally across all scans)
    if percentile is not None:
        all_normalized = np.array(normalized_scans)
        p_cap = np.percentile(all_normalized, percentile)
        normalized_scans = [np.clip(scan, 0, p_cap) / p_cap for scan in normalized_scans]
        print(f"Capping values at {percentile}th percentile: {p_cap:.4f}")

    # Stitch scans together into a single continuous image
    # Organize scans into a 2D grid structure
    scan_grid = [[None for _ in range(n_scans_x)] for _ in range(n_scans_y)]

    for idx, scan_norm in enumerate(normalized_scans):
        i = idx // n_scans_y  # x index
        j = idx % n_scans_y   # y index
        scan_grid[n_scans_y - 1 - j][i] = scan_norm  # Flip j for correct orientation

    # Concatenate scans: first horizontally within each row, then vertically
    stitched_rows = []
    for row in scan_grid:
        stitched_row = np.hstack(row)
        stitched_rows.append(stitched_row)

    stitched_image = np.vstack(stitched_rows)

    # Create figure with stitched image
    fig, ax = plt.subplots(figsize=(12, 12 * n_scans_y / n_scans_x))

    # Plot the stitched image with correct extent
    im = ax.imshow(stitched_image, cmap='viridis', origin='lower', aspect='auto',
                   extent=[v0_min, v0_max, v1_min, v1_max])

    # Mark ground truth
    ax.plot(gt_gates[0], gt_gates[1], 'r*', markersize=15, markeredgewidth=2,
            markeredgecolor='white', zorder=10, label='Ground Truth')

    # Add labels
    ax.set_xlabel('Gate 0 Voltage (V)', fontsize=14)
    ax.set_ylabel('Gate 1 Voltage (V)', fontsize=14)

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved stitched device range map to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map quantum device voltage range with grid of scans")
    parser.add_argument("--output", type=str, default="device_range_map.png",
                        help="Output image path (default: device_range_map.png)")
    parser.add_argument("--n-grid", type=int, default=11,
                        help="Number of scans per axis (default: 11)")
    parser.add_argument("--percentile", type=int, default=None,
                        help="Cap values at this percentile after normalization (0-100)")

    args = parser.parse_args()

    map_device_range(
        output_path=args.output,
        n_grid=args.n_grid,
        percentile=args.percentile
    )
