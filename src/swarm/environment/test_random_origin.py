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
    Test script that creates 10 environment instances, each time generating
    three scans centered at ground truth: one with barriers at zero, one
    with barriers at ground truth, and one with random barriers. All scans
    are displayed in a 10x3 grid.
    """

    num_initializations = 10
    all_scans = []
    all_extents = []
    all_ground_truths = []
    all_virtual_gate_origins = []

    # Collect scans from 10 different initializations
    for i in range(num_initializations):
        print(f"\n{'='*60}")
        print(f"Initialization {i+1}/{num_initializations}")
        print(f"{'='*60}")

        # Create new environment instance
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

        # Get barrier ground truth if using barriers
        barrier_ground_truth = env.device_state["barrier_ground_truth"] if env.use_barriers else None
        if barrier_ground_truth is not None:
            print(f"Barrier Ground Truth: {barrier_ground_truth}")

        # Get sensor ground truth
        sensor_ground_truth = env.device_state["sensor_ground_truth"]

        # Override observation voltage range to -10 to +10
        env.array.obs_voltage_min = -10.0
        env.array.obs_voltage_max = 10.0

        # Get ground truth center
        v0_center_gt = gate_ground_truth[0]
        v1_center_gt = gate_ground_truth[1]

        # Use ALL gate ground truths (not just gates 0-1)
        gate_voltages_scan = gate_ground_truth.copy()

        # Normalize gate voltages to [-1, 1] for step action
        # The step method expects normalized actions in [-1, 1]
        normalized_gates = (gate_voltages_scan - env.plunger_min) / (env.plunger_max - env.plunger_min)
        normalized_gates = normalized_gates * 2 - 1

        # First scan: centered at ground truth with zero barriers
        print(f"  Scan 1: GT center with barriers at zero")
        barriers_zero = np.zeros(env.num_dots - 1) if env.use_barriers else None
        normalized_barriers_zero = (barriers_zero - env.barrier_min) / (env.barrier_max - env.barrier_min)
        normalized_barriers_zero = normalized_barriers_zero * 2 - 1

        action = {
            "action_gate_voltages": normalized_gates,
            "action_barrier_voltages": normalized_barriers_zero
        }
        raw_obs, _, _, _, _ = env.step(action)
        scan_zero_barriers = raw_obs["image"][:, :, 0].copy()

        # Cap values to 75th percentile
        p_high = np.percentile(scan_zero_barriers, 75)
        scan_zero_barriers = np.clip(scan_zero_barriers, None, p_high)

        # Second scan: centered at ground truth with barriers at ground truth
        if barrier_ground_truth is not None:
            print(f"  Scan 2: GT center with barriers at GT")
            normalized_barriers_gt = (barrier_ground_truth - env.barrier_min) / (env.barrier_max - env.barrier_min)
            normalized_barriers_gt = normalized_barriers_gt * 2 - 1

            action = {
                "action_gate_voltages": normalized_gates,
                "action_barrier_voltages": normalized_barriers_gt
            }
            raw_obs, _, _, _, _ = env.step(action)
            scan_gt_barriers = raw_obs["image"][:, :, 0].copy()

            # Cap values to 75th percentile
            p_high = np.percentile(scan_gt_barriers, 75)
            scan_gt_barriers = np.clip(scan_gt_barriers, None, p_high)
        else:
            scan_gt_barriers = scan_zero_barriers

        # Third scan: centered at ground truth with random barriers
        if barrier_ground_truth is not None:
            print(f"  Scan 3: GT center with random barriers")
            # Sample random barrier voltages from the barrier voltage range
            # barrier_min and barrier_max are arrays, sample element-wise
            random_barriers = np.random.uniform(
                low=env.barrier_min,
                high=env.barrier_max
            )
            print(f"    Barrier min: {env.barrier_min}")
            print(f"    Barrier max: {env.barrier_max}")
            print(f"    Barrier GT: {barrier_ground_truth}")
            print(f"    Random barriers: {random_barriers}")

            normalized_barriers_random = (random_barriers - env.barrier_min) / (env.barrier_max - env.barrier_min)
            normalized_barriers_random = normalized_barriers_random * 2 - 1

            action = {
                "action_gate_voltages": normalized_gates,
                "action_barrier_voltages": normalized_barriers_random
            }
            raw_obs, _, _, _, _ = env.step(action)
            scan_random_barriers = raw_obs["image"][:, :, 0].copy()

            # Cap values to 75th percentile
            p_high = np.percentile(scan_random_barriers, 75)
            scan_random_barriers = np.clip(scan_random_barriers, None, p_high)
        else:
            scan_random_barriers = scan_zero_barriers

        # Store scans and metadata
        all_scans.append([scan_zero_barriers, scan_gt_barriers, scan_random_barriers])
        extent = [v0_center_gt - 10, v0_center_gt + 10, v1_center_gt - 10, v1_center_gt + 10]
        all_extents.append(extent)
        all_ground_truths.append(gate_ground_truth[:2])
        all_virtual_gate_origins.append(virtual_gate_origin)

    # Create 10x3 grid figure
    print(f"\n{'='*60}")
    print(f"Creating 10x3 grid of all scans")
    print(f"{'='*60}")

    fig, axes = plt.subplots(num_initializations, 3, figsize=(18, 5 * num_initializations))
    fig.suptitle('Random Origin Test - 10 Initializations: Barriers at Zero (left) vs GT Barriers (middle) vs Random Barriers (right)',
                 fontsize=16, y=0.995)

    for i in range(num_initializations):
        for j in range(3):
            ax = axes[i, j]
            scan = all_scans[i][j]
            extent = all_extents[i]
            gt = all_ground_truths[i]
            vg_origin = all_virtual_gate_origins[i]

            im = ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto', extent=extent)

            # Plot ground truth position as red dot
            ax.plot(gt[0], gt[1], 'ro', markersize=8, markeredgewidth=2, markerfacecolor='none')

            # Plot red square centered at ground truth with width/height = 3V
            square_size = 3.0
            from matplotlib.patches import Rectangle
            rect = Rectangle((gt[0] - square_size/2, gt[1] - square_size/2),
                           square_size, square_size,
                           linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Add virtual gate origin offset text under left scans
            if j == 0:
                vg_text = f'VG Origin: [{vg_origin[0]:.3f}, {vg_origin[1]:.3f}]'
                ax.text(0.5, -0.15, vg_text, transform=ax.transAxes,
                       fontsize=8, ha='center', va='top')

            # Labels
            if i == num_initializations - 1:
                ax.set_xlabel('Gate 0 Voltage (V)', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'Init {i+1}\nGate 1 Voltage (V)', fontsize=10)

            # Title for each column
            if i == 0:
                if j == 0:
                    ax.set_title('Barriers at Zero', fontsize=12)
                elif j == 1:
                    ax.set_title('Barriers at GT', fontsize=12)
                else:
                    ax.set_title('Random Barriers', fontsize=12)

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Signal' if j == 2 else '')

    plt.tight_layout()

    # Save to file
    output_path = Path(__file__).parent / 'random_origin_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved 10x3 grid to {output_path}")
    plt.close()


if __name__ == "__main__":
    test_random_origin()
