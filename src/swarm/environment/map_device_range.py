"""
Map device range by sampling quantum device environment at different voltage positions.

Creates a grid of charge stability diagram scans across the full voltage range,
showing how the CSD changes as we move through the search space.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import QuantumDeviceEnv


def map_device_range(output_path="device_range_map.png",
                     v0_min=None, v0_max=None, v1_min=None, v1_max=None):
    """
    Generate one large scan covering the entire device voltage range.

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

    # Get the observation window size
    obs_v_min = env.array.obs_voltage_min
    obs_v_max = env.array.obs_voltage_max
    obs_window_size = obs_v_max - obs_v_min  # Should be ~3V

    print(f"\nObservation window size: {obs_window_size:.2f} V")

    # Calculate how many scans we'd need to cross the voltage space
    n_scans_x = int(np.ceil(v0_range / obs_window_size))
    n_scans_y = int(np.ceil(v1_range / obs_window_size))

    print(f"Number of scans needed to cover range: {n_scans_x} x {n_scans_y}")

    # Use base resolution of 100
    base_resolution = 100
    print(f"Base resolution: {base_resolution}")

    # Multiply resolution to cover the entire range in one scan
    new_resolution_x = base_resolution * n_scans_x
    new_resolution_y = base_resolution * n_scans_y

    print(f"New resolution for full range scan: {new_resolution_x} x {new_resolution_y}")

    # Temporarily modify the environment's resolution and voltage range
    original_image_size = env.array.obs_image_size
    original_v_min = env.array.obs_voltage_min
    original_v_max = env.array.obs_voltage_max

    env.array.obs_image_size = new_resolution_x  # Assuming square images
    env.array.obs_voltage_min = plunger_min[0]
    env.array.obs_voltage_max = plunger_max[0]

    # Get the large scan with all voltages set to ground truth
    raw_obs = env.array._get_obs(gt_gates, gt_barriers)
    scan = raw_obs["image"][:, :, 0]  # First channel

    # Restore original settings
    env.array.obs_image_size = original_image_size
    env.array.obs_voltage_min = original_v_min
    env.array.obs_voltage_max = original_v_max

    # Normalize for visualization
    p_low = np.percentile(scan, 0.5)
    p_high = np.percentile(scan, 99.5)
    if p_high > p_low:
        scan_norm = (scan - p_low) / (p_high - p_low)
    else:
        scan_norm = np.zeros_like(scan)
    scan_norm = np.clip(scan_norm, 0, 1)

    # Create figure with single large scan
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.imshow(scan_norm, cmap='viridis', origin='lower', aspect='auto',
             extent=[plunger_min[0], plunger_max[0], plunger_min[1], plunger_max[1]])

    # Plot ground truth as red dot
    ax.plot(gt_gates[0], gt_gates[1], 'ro', markersize=8, markeredgewidth=1,
           markeredgecolor='white', zorder=10, label='Ground Truth')

    ax.set_xlabel('Gate 0 Voltage (V)', fontsize=12)
    ax.set_ylabel('Gate 1 Voltage (V)', fontsize=12)
    ax.set_title(f'Full Device Range Scan ({new_resolution_x}x{new_resolution_y} pixels)', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved device range map to: {output_path}")
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
