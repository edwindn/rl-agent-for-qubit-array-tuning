import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for clean imports
from pathlib import Path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_barrier_range(num_steps=10, seed=42):
    """
    Test script that steps through env 10 times, varying barrier voltages.
    Left scan: barriers at min, Right scan: barriers at max, Middle: linspace between.
    Plots all scans in a 3x10 grid (3 pairs, 10 steps).
    """
    print("Initializing environment...")
    np.random.seed(seed)
    env = QuantumDeviceEnv(training=False, num_dots=4, use_barriers=True)
    obs, info = env.reset(seed=seed)

    num_channels = obs["image"].shape[2]  # Should be 3 for 4 dots
    num_barriers = env.num_barrier_voltages  # Should be 3 for 4 dots

    # Storage for scans
    all_scans = {i: [] for i in range(num_channels)}

    # Create linspace for barrier voltages from min to max
    barrier_voltages_linspace = np.linspace(env.barrier_min, env.barrier_max, num_steps)

    print(f"Barrier range: min={env.barrier_min}, max={env.barrier_max}")
    print(f"Running {num_steps} steps with varying barrier voltages...")

    for step in range(num_steps):
        # Get ground truth plungers
        gt_plungers = env.device_state["gate_ground_truth"]

        # Set barriers to linspace values for this step
        barrier_voltages = barrier_voltages_linspace[step]

        # Normalize plungers to action space [-1, 1]
        normalized_gates = (gt_plungers - env.plunger_min) / (env.plunger_max - env.plunger_min)
        normalized_gates = normalized_gates * 2 - 1
        normalized_gates = np.clip(normalized_gates, -1, 1)

        # Normalize barriers to action space [-1, 1]
        normalized_barriers = (barrier_voltages - env.barrier_min) / (env.barrier_max - env.barrier_min)
        normalized_barriers = normalized_barriers * 2 - 1
        normalized_barriers = np.clip(normalized_barriers, -1, 1)

        # Create action dictionary
        action = {
            "action_gate_voltages": normalized_gates,
            "action_barrier_voltages": normalized_barriers,
        }

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Store each channel's scan
        for channel_idx in range(num_channels):
            all_scans[channel_idx].append(obs["image"][:, :, channel_idx])

        print(f"  Step {step + 1}/{num_steps}: barriers={barrier_voltages}")

    # Add 1 more scan with ground truth barrier voltages (for all 3 channels)
    print(f"\nAdding 1 scan with ground truth barrier voltages...")

    # Get ground truth values
    gt_plungers = env.device_state["gate_ground_truth"]
    gt_barriers = env.device_state["barrier_ground_truth"]

    # Normalize to action space [-1, 1]
    normalized_gates = (gt_plungers - env.plunger_min) / (env.plunger_max - env.plunger_min)
    normalized_gates = normalized_gates * 2 - 1
    normalized_gates = np.clip(normalized_gates, -1, 1)

    normalized_barriers = (gt_barriers - env.barrier_min) / (env.barrier_max - env.barrier_min)
    normalized_barriers = normalized_barriers * 2 - 1
    normalized_barriers = np.clip(normalized_barriers, -1, 1)

    # Create action dictionary
    action = {
        "action_gate_voltages": normalized_gates,
        "action_barrier_voltages": normalized_barriers,
    }

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Store ground truth scans for all channels
    gt_barrier_scans = []
    for channel_idx in range(num_channels):
        gt_barrier_scans.append(obs["image"][:, :, channel_idx])

    print(f"  GT scan: barriers={gt_barriers}")

    # Create 3x11 grid plot (10 varying + 1 ground truth)
    total_cols = num_steps + 1
    fig, axes = plt.subplots(num_channels, total_cols, figsize=(2.5 * total_cols, 2.5 * num_channels))

    # Plot varying barrier scans (first 10 columns)
    for row in range(num_channels):
        for col in range(num_steps):
            ax = axes[row, col]
            scan = all_scans[row][col]
            ax.imshow(scan, cmap='viridis', origin='lower', aspect='equal', vmin=0, vmax=1)
            ax.axis('off')

            # Add labels
            if col == 0:
                pair_label = f'Dots {row}-{row+1}'
                ax.set_ylabel(pair_label, fontsize=10, rotation=90, labelpad=10)
            if row == 0:
                ax.set_title(f'Step {col+1}', fontsize=8)

        # Add ground truth scan in last column
        ax = axes[row, num_steps]
        scan = gt_barrier_scans[row]
        ax.imshow(scan, cmap='viridis', origin='lower', aspect='equal', vmin=0, vmax=1)
        ax.axis('off')

        # Add red border to ground truth scan
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
            spine.set_visible(True)

        if row == 0:
            ax.set_title('GT', fontsize=8, color='red')

    # Add a vertical red line between varying and ground truth scans
    fig.add_artist(plt.Line2D([num_steps / total_cols, num_steps / total_cols], [0, 1],
                               transform=fig.transFigure, color='red', linewidth=3))

    plt.suptitle('Barrier Range Test: Min → Max (left) | Ground Truth (right, red border)', fontsize=12)
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'test_barrier_range.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved test scans to {output_path}")
    plt.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_barrier_range(num_steps=10, seed=42)
