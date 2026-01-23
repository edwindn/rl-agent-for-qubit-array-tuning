"""
Test script that loads the environment, passes ground truth voltages to step(),
and plots the resulting scan.
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_transformation():
    """
    Create environment, pass ground truth voltages to step(), and plot the scan.
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
    print(f"Gate voltage range: [{env.plunger_min}, {env.plunger_max}]")
    print(f"Barrier voltage range: [{env.barrier_min}, {env.barrier_max}]")

    # Normalize ground truth voltages to [-1, 1] for action space
    # Gates
    gate_normalized = (gate_gt - env.plunger_min) / (env.plunger_max - env.plunger_min)
    gate_normalized = gate_normalized * 2 - 1  # Scale to [-1, 1]

    # Barriers
    barrier_normalized = (barrier_gt - env.barrier_min) / (env.barrier_max - env.barrier_min)
    barrier_normalized = barrier_normalized * 2 - 1  # Scale to [-1, 1]

    print(f"\nNormalized gate action: {gate_normalized}")
    print(f"Normalized barrier action: {barrier_normalized}")

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

    # Print reward
    print(f"\nReward (gates): {reward['gates']}")
    print(f"Reward (barriers): {reward['barriers']}")
    print(f"Mean gate reward: {np.mean(reward['gates']):.4f}")
    print(f"Mean barrier reward: {np.mean(reward['barriers']):.4f}")

    # Get the scans with offset
    scans_with_offset = observation["image"].copy()  # Shape: (resolution, resolution, num_channels)
    num_channels = scans_with_offset.shape[2]

    print(f"\nScan shape: {scans_with_offset.shape}")
    print(f"Number of channels: {num_channels}")

    # Set offset to zero and get scans again
    print("\n" + "=" * 80)
    print("Setting offset to zero")
    print("=" * 80)

    original_offset = env.array.model.gate_voltage_composer.virtual_gate_origin.copy()
    print(f"Original virtual_gate_origin: {original_offset}")

    env.array._update_virtual_gate_origin(np.zeros(env.num_dots + 1))
    print(f"New virtual_gate_origin: {env.array.model.gate_voltage_composer.virtual_gate_origin}")

    # Get the same voltages from device state
    current_gate = env.device_state["current_gate_voltages"]
    current_barrier = env.device_state["current_barrier_voltages"]
    sensor_gt = env.device_state["sensor_ground_truth"]

    print(f"\nCurrent gate voltages: {current_gate}")
    print(f"Current barrier voltages: {current_barrier}")

    # Get observation with zero offset
    raw_obs_zero = env.array._get_obs(current_gate, current_barrier, sensor_gt)
    scans_zero_offset = raw_obs_zero["image"].copy()

    # Plot the scans
    print("\n" + "=" * 80)
    print("Plotting scans")
    print("=" * 80)

    fig, axes = plt.subplots(2, num_channels, figsize=(5 * num_channels, 10))

    # Handle single channel case
    if num_channels == 1:
        axes = axes.reshape(2, 1)

    # Plot scans with offset (row 0) and zero offset (row 1)
    for i in range(num_channels):
        # Row 0: With offset
        im0 = axes[0, i].imshow(scans_with_offset[:, :, i], cmap='viridis', origin='lower')
        axes[0, i].set_title(f'With Offset - Scan {i+1} (Dots {i}-{i+1})\nGate GT: [{gate_gt[i]:.2f}, {gate_gt[i+1]:.2f}]')
        axes[0, i].set_xlabel('Gate Voltage')
        axes[0, i].set_ylabel('Gate Voltage')
        center_idx = env.resolution // 2
        axes[0, i].plot(center_idx, center_idx, 'r+', markersize=20, markeredgewidth=3, label='Ground Truth')
        axes[0, i].legend()
        plt.colorbar(im0, ax=axes[0, i])

        # Row 1: Zero offset
        im1 = axes[1, i].imshow(scans_zero_offset[:, :, i], cmap='viridis', origin='lower')
        axes[1, i].set_title(f'Zero Offset - Scan {i+1} (Dots {i}-{i+1})\nGate: [{current_gate[i]:.2f}, {current_gate[i+1]:.2f}]')
        axes[1, i].set_xlabel('Gate Voltage')
        axes[1, i].set_ylabel('Gate Voltage')
        axes[1, i].plot(center_idx, center_idx, 'r+', markersize=20, markeredgewidth=3, label='Ground Truth')
        axes[1, i].legend()
        plt.colorbar(im1, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('test_transformation_scans.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved scans to: test_transformation_scans.png")
    plt.close()

    # Print final device state
    print("\n" + "=" * 80)
    print("Final device state")
    print("=" * 80)
    print(f"Current gate voltages: {env.device_state['current_gate_voltages']}")
    print(f"Current barrier voltages: {env.device_state['current_barrier_voltages']}")
    print(f"Gate ground truth: {env.device_state['gate_ground_truth']}")
    print(f"Barrier ground truth: {env.device_state['barrier_ground_truth']}")


if __name__ == "__main__":
    test_transformation()
