#!/usr/bin/env python3
"""
Test script to compare deterministic simple env vs standard env.
Creates two environments, steps them with identical actions, and compares the scans.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv as StandardEnv
from swarm.algo_ablations.simple_env import QuantumDeviceEnv as DeterministicEnv


def normalize_action(value, min_val, max_val):
    """Convert a value from [min, max] range to [-1, 1] normalized range."""
    normalized = (value - min_val) / (max_val - min_val)  # [0, 1]
    normalized = normalized * 2 - 1  # [-1, 1]
    return normalized


def test_deterministic_environments():
    """Test and compare deterministic vs standard environments."""

    # Configuration - use absolute path so both envs can find it
    config_path = str(Path(__file__).parent / "configs" / "sac_env_config.yaml")
    num_steps = 9  # 9 steps + 1 initial = 10 scans total

    # Initialize both environments
    print("Initializing environments...")
    det_env = DeterministicEnv(config_path=config_path)
    std_env = StandardEnv(config_path=config_path)

    # Reset both environments (NO SEED - testing determinism without explicit seeding)
    print("Resetting environments...")
    det_obs, det_info = det_env.reset()
    std_obs, std_info = std_env.reset()

    # Get ground truths from device state
    det_gt_plungers = det_info["current_device_state"]["gate_ground_truth"]
    det_gt_barriers = det_info["current_device_state"]["barrier_ground_truth"]

    std_gt_plungers = std_info["current_device_state"]["gate_ground_truth"]
    std_gt_barriers = std_info["current_device_state"]["barrier_ground_truth"]

    print(f"\nDeterministic env ground truth plungers: {det_gt_plungers}")
    print(f"Standard env ground truth plungers: {std_gt_plungers}")
    print(f"\nDeterministic env ground truth barriers: {det_gt_barriers}")
    print(f"Standard env ground truth barriers: {std_gt_barriers}")

    # Get voltage ranges for normalization
    det_plunger_min = det_env.plunger_min
    det_plunger_max = det_env.plunger_max
    det_barrier_min = det_env.barrier_min
    det_barrier_max = det_env.barrier_max

    std_plunger_min = std_env.plunger_min
    std_plunger_max = std_env.plunger_max
    std_barrier_min = std_env.barrier_min
    std_barrier_max = std_env.barrier_max

    # Set both environments to ground truth
    print("\nSetting environments to ground truth...")

    # Normalize ground truth values to [-1, 1] for actions
    det_action_plungers = normalize_action(det_gt_plungers, det_plunger_min, det_plunger_max)
    det_action_barriers = normalize_action(det_gt_barriers, det_barrier_min, det_barrier_max)

    std_action_plungers = normalize_action(std_gt_plungers, std_plunger_min, std_plunger_max)
    std_action_barriers = normalize_action(std_gt_barriers, std_barrier_min, std_barrier_max)

    # Step to ground truth
    det_action = {
        "action_gate_voltages": det_action_plungers,
        "action_barrier_voltages": det_action_barriers
    }

    std_action = {
        "action_gate_voltages": std_action_plungers,
        "action_barrier_voltages": std_action_barriers
    }

    det_obs, _, _, _, det_info = det_env.step(det_action)
    std_obs, _, _, _, std_info = std_env.step(std_action)

    # Store initial scans (at ground truth)
    det_scans = [det_obs["image"][:, :, 0].copy()]  # First channel only
    std_scans = [std_obs["image"][:, :, 0].copy()]

    print(f"Initial scan shape: {det_scans[0].shape}")

    # Define 9 hardcoded action values (normalized to [-1, 1])
    # These are perturbations from ground truth
    hardcoded_actions = [
        np.array([-0.8, -0.6, -0.4, -0.2]),
        np.array([-0.6, -0.4, -0.2, 0.0]),
        np.array([-0.4, -0.2, 0.0, 0.2]),
        np.array([-0.2, 0.0, 0.2, 0.4]),
        np.array([0.0, 0.2, 0.4, 0.6]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.array([0.4, 0.6, 0.8, 0.9]),
        np.array([0.6, 0.7, 0.8, 0.9]),
        np.array([0.8, 0.85, 0.9, 0.95]),
    ]

    # Keep barriers at ground truth for all steps
    det_barrier_action = det_action_barriers
    std_barrier_action = std_action_barriers

    # Step through hardcoded actions
    print(f"\nStepping through {num_steps} hardcoded actions...")
    for i, plunger_action in enumerate(hardcoded_actions):
        det_action = {
            "action_gate_voltages": plunger_action,
            "action_barrier_voltages": det_barrier_action
        }

        std_action = {
            "action_gate_voltages": plunger_action,
            "action_barrier_voltages": std_barrier_action
        }

        det_obs, det_reward, det_term, det_trunc, det_info = det_env.step(det_action)
        std_obs, std_reward, std_term, std_trunc, std_info = std_env.step(std_action)

        # Store scans (first channel only)
        det_scans.append(det_obs["image"][:, :, 0].copy())
        std_scans.append(std_obs["image"][:, :, 0].copy())

        print(f"  Step {i+1}/{num_steps} - Plunger action: {plunger_action}")

    print(f"\nCollected {len(det_scans)} scans from each environment")

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 10, figsize=(25, 5))

    # Plot deterministic env scans (top row)
    for i, scan in enumerate(det_scans):
        axes[0, i].imshow(scan, cmap='viridis', origin='lower', aspect='auto')
        axes[0, i].set_title(f'Det-{i}', fontsize=8)
        axes[0, i].axis('off')

    # Plot standard env scans (bottom row)
    for i, scan in enumerate(std_scans):
        axes[1, i].imshow(scan, cmap='viridis', origin='lower', aspect='auto')
        axes[1, i].set_title(f'Std-{i}', fontsize=8)
        axes[1, i].axis('off')

    # Add row labels
    axes[0, 0].set_ylabel('Deterministic\nEnv', fontsize=10, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Standard\nEnv', fontsize=10, rotation=0, ha='right', va='center')

    plt.suptitle('Environment Comparison: Deterministic vs Standard (10 scans each)', fontsize=14, y=0.98)
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "deterministic_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")

    # Calculate and print statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)

    for i in range(len(det_scans)):
        det_scan = det_scans[i]
        std_scan = std_scans[i]

        # Calculate statistics
        det_mean = np.mean(det_scan)
        std_mean = np.mean(std_scan)
        det_std = np.std(det_scan)
        std_std = np.std(std_scan)

        mse = np.mean((det_scan - std_scan) ** 2)
        mae = np.mean(np.abs(det_scan - std_scan))

        print(f"\nScan {i}:")
        print(f"  Deterministic - Mean: {det_mean:.4f}, Std: {det_std:.4f}")
        print(f"  Standard      - Mean: {std_mean:.4f}, Std: {std_std:.4f}")
        print(f"  Difference    - MSE: {mse:.6f}, MAE: {mae:.6f}")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_deterministic_environments()
