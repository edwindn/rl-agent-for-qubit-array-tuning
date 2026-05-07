"""
Map device range by sampling quantum device environment at different voltage positions.

Creates a grid of charge stability diagram scans across the full voltage range,
showing how the CSD changes as we move through the search space.
Scans are tiled to fully cover the space without gaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import QuantumDeviceEnv


def map_device_range(output_path="device_range_map.png",
                     v0_min=None, v0_max=None, v1_min=None, v1_max=None):
    """
    Generate tiled scans covering the entire device voltage range without gaps.

    Creates a grid of overlapping scans that completely tile the voltage space.
    Each scan has the standard observation window size, and scans are positioned
    to ensure no gaps exist between them.

    Args:
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

    # Get ground truth for all gates and barriers
    gt_gates = info['current_device_state']['gate_ground_truth']
    gt_barriers = info['current_device_state']['barrier_ground_truth']

    print(f"  Ground truth gates: {gt_gates}")
    print(f"  Ground truth barriers: {gt_barriers}")

    # Calculate voltage range
    v0_range = plunger_max[0] - plunger_min[0]
    v1_range = plunger_max[1] - plunger_min[1]

    # Get the observation window size (window_delta_range from config)
    # The window goes from center - delta to center + delta
    # So total window size is 2 * delta
    obs_v_min = env.array.obs_voltage_min
    obs_v_max = env.array.obs_voltage_max
    obs_window_size = obs_v_max - obs_v_min  # Should be ~3-4V (2 * window_delta_range)

    print(f"\nObservation window size: {obs_window_size:.2f} V")

    # Calculate how many scans we need to tile the space without gaps
    # We need ceiling division to ensure complete coverage
    n_scans_x = int(np.ceil(v0_range / obs_window_size))
    n_scans_y = int(np.ceil(v1_range / obs_window_size))

    print(f"Number of scans to tile the range: {n_scans_x} x {n_scans_y} = {n_scans_x * n_scans_y} total scans")

    # Calculate step size to evenly distribute scans
    # If we have n scans to cover a range, we space them such that:
    # - First scan starts at the minimum
    # - Last scan ends at the maximum
    # - No gaps exist between scan coverage
    if n_scans_x > 1:
        step_x = v0_range / n_scans_x
    else:
        step_x = 0

    if n_scans_y > 1:
        step_y = v1_range / n_scans_y
    else:
        step_y = 0

    print(f"Step size between scan centers: ({step_x:.2f}, {step_y:.2f}) V")

    # Use standard resolution
    resolution = env.array.obs_image_size
    print(f"Scan resolution: {resolution}x{resolution}")

    # Store all scans and their positions
    all_scans = []
    scan_positions = []

    # Generate scans at a grid of positions that tile the space
    for i in range(n_scans_x):
        for j in range(n_scans_y):
            # Calculate center position for this scan
            # Position scans so they tile without gaps
            center_v0 = plunger_min[0] + (i + 0.5) * step_x
            center_v1 = plunger_min[1] + (j + 0.5) * step_y

            # For edge scans, adjust to not exceed the boundary
            scan_min_v0 = center_v0 - obs_window_size / 2
            scan_max_v0 = center_v0 + obs_window_size / 2
            scan_min_v1 = center_v1 - obs_window_size / 2
            scan_max_v1 = center_v1 + obs_window_size / 2

            # Temporarily set the environment's observation range for this scan
            env.array.obs_voltage_min = scan_min_v0
            env.array.obs_voltage_max = scan_max_v0

            # Get the scan with plunger gates at (center_v0, center_v1)
            # We need to temporarily set the plunger voltages for the scan
            temp_gates = gt_gates.copy()
            temp_gates[0] = center_v0
            temp_gates[1] = center_v1

            raw_obs = env.array._get_obs(temp_gates, gt_barriers)
            scan = raw_obs["image"][:, :, 0]  # First channel

            all_scans.append(scan)
            scan_positions.append((scan_min_v0, scan_max_v0, scan_min_v1, scan_max_v1))

            print(f"  Scan {i*n_scans_y + j + 1}/{n_scans_x*n_scans_y}: "
                  f"v0=[{scan_min_v0:.1f}, {scan_max_v0:.1f}], "
                  f"v1=[{scan_min_v1:.1f}, {scan_max_v1:.1f}]")

    # Save scan data
    import pickle
    import os

    data_path = output_path.replace('.png', '_data.pkl')
    scan_data = {
        'scans': all_scans,
        'positions': scan_positions,
        'n_scans_x': n_scans_x,
        'n_scans_y': n_scans_y,
        'resolution': resolution,
        'plunger_min': plunger_min,
        'plunger_max': plunger_max,
        'obs_window_size': obs_window_size,
        'step_x': step_x,
        'step_y': step_y,
        'gt_gates': gt_gates,
        'gt_barriers': gt_barriers
    }

    with open(data_path, 'wb') as f:
        pickle.dump(scan_data, f)
    print(f"\nSaved scan data to: {data_path}")

    # Create composite image by stitching scans together
    # Calculate total dimensions needed
    total_width = resolution * n_scans_x
    total_height = resolution * n_scans_y
    composite = np.zeros((total_height, total_width))

    # Place each scan in the composite
    for idx, (scan, pos) in enumerate(zip(all_scans, scan_positions)):
        i = idx // n_scans_y
        j = idx % n_scans_y

        # Calculate pixel position in composite
        x_start = i * resolution
        x_end = (i + 1) * resolution
        y_start = j * resolution
        y_end = (j + 1) * resolution

        composite[y_start:y_end, x_start:x_end] = scan

    # Normalize the composite for visualization
    p_low = np.percentile(composite, 0.5)
    p_high = np.percentile(composite, 99.5)
    if p_high > p_low:
        composite_norm = (composite - p_low) / (p_high - p_low)
    else:
        composite_norm = np.zeros_like(composite)
    composite_norm = np.clip(composite_norm, 0, 1)

    # Create figure - single continuous stitched image
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))

    # Plot the stitched composite as a single continuous image
    ax.imshow(composite_norm, cmap='viridis', origin='lower', aspect='auto',
             extent=[plunger_min[0], plunger_max[0], plunger_min[1], plunger_max[1]])

    # Plot ground truth as red dot
    ax.plot(gt_gates[0], gt_gates[1], 'ro', markersize=10, markeredgewidth=2,
           markeredgecolor='white', zorder=10, label='Ground Truth')

    ax.set_xlabel('Gate 0 Voltage (V)', fontsize=14)
    ax.set_ylabel('Gate 1 Voltage (V)', fontsize=14)
    ax.set_title(f'Full Device Range Map - Stitched ({n_scans_x}×{n_scans_y} scans, {total_width}×{total_height} pixels)',
                 fontsize=16, pad=20)
    ax.legend(fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle=':', color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved stitched device range map to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map quantum device voltage range")
    parser.add_argument("--output", type=str, default="device_range_map.png",
                        help="Output image path (default: device_range_map.png)")
    parser.add_argument("--v0-min", type=float, default=None,
                        help="Minimum voltage for gate 0")
    parser.add_argument("--v0-max", type=float, default=None,
                        help="Maximum voltage for gate 0")
    parser.add_argument("--v1-min", type=float, default=None,
                        help="Minimum voltage for gate 1")
    parser.add_argument("--v1-max", type=float, default=None,
                        help="Maximum voltage for gate 1")

    args = parser.parse_args()

    map_device_range(
        output_path=args.output,
        v0_min=args.v0_min,
        v0_max=args.v0_max,
        v1_min=args.v1_min,
        v1_max=args.v1_max
    )
