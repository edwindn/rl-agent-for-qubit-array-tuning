"""
Test script that loads the environment, calls step with ground truth voltages,
and saves the resulting scan.
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_env():
    """
    Create environment, step with ground truth voltages, and save scan.
    """
    print("=" * 80)
    print("Creating environment")
    print("=" * 80)

    # Create environment
    env = QuantumDeviceEnv(training=True)
    obs, info = env.reset(seed=42)

    # Get ground truth voltages from initial state
    gate_gt = env.device_state["gate_ground_truth"].copy()
    barrier_gt = env.device_state["barrier_ground_truth"].copy()

    print(f"Gate ground truth: {gate_gt}")
    print(f"Barrier ground truth: {barrier_gt}")

    # Normalize ground truth voltages to [-1, 1] for action space
    gate_normalized = (gate_gt - env.plunger_min) / (env.plunger_max - env.plunger_min)
    gate_normalized = gate_normalized * 2 - 1

    barrier_normalized = (barrier_gt - env.barrier_min) / (env.barrier_max - env.barrier_min)
    barrier_normalized = barrier_normalized * 2 - 1

    # Create action with ground truth voltages
    action = {
        "action_gate_voltages": gate_normalized,
        "action_barrier_voltages": barrier_normalized
    }

    # Step with ground truth voltages
    print("\n" + "=" * 80)
    print("Stepping with ground truth voltages")
    print("=" * 80)

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Reward (gates): {reward['gates']}")
    print(f"Reward (barriers): {reward['barriers']}")

    # Get the scans
    scans = observation["image"]
    num_channels = scans.shape[2]

    print(f"\nScan shape: {scans.shape}")

    # Save scan
    print("\n" + "=" * 80)
    print("Saving scan")
    print("=" * 80)

    fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        im = axes[i].imshow(scans[:, :, i], cmap='viridis', origin='lower')
        axes[i].set_title(f'Scan {i+1} (Dots {i}-{i+1})')
        axes[i].set_xlabel('Gate Voltage')
        axes[i].set_ylabel('Gate Voltage')

        center_idx = env.resolution // 2
        axes[i].plot(center_idx, center_idx, 'r+', markersize=20, markeredgewidth=3)

        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig('test_env_scan.png', dpi=150, bbox_inches='tight')
    print(f"Saved scan to: test_env_scan.png")
    plt.close()


if __name__ == "__main__":
    test_env()
