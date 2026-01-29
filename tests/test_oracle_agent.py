"""
Oracle Agent Test - Verifies environment correctness by submitting ground truth actions.

This test creates a "fake" agent that reads ground truth from the environment
and submits it as its action. This verifies:
1. Ground truth is correctly accessible
2. Reward is maximal when at ground truth
3. Scans look correct when at ground truth
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper


def normalize_voltage(voltage, v_min, v_max):
    """Convert voltage from physical range to [-1, 1] action space."""
    return 2.0 * (voltage - v_min) / (v_max - v_min) - 1.0


def run_oracle_episode(env, verbose=True, save_scans=False, output_dir=None):
    """
    Run a single episode where the agent always outputs ground truth.

    Returns:
        dict with episode statistics
    """
    obs, info = env.reset()

    episode_rewards = []
    episode_gate_distances = []
    episode_barrier_distances = []
    scans = []

    done = False
    step = 0

    while not done:
        # Get ground truth from base environment
        base_env = env.base_env
        gate_gt = base_env.device_state["gate_ground_truth"]
        barrier_gt = base_env.device_state["barrier_ground_truth"]

        # Normalize to [-1, 1] action space
        gate_action_normalized = normalize_voltage(
            gate_gt, base_env.plunger_min, base_env.plunger_max
        )
        barrier_action_normalized = normalize_voltage(
            barrier_gt, base_env.barrier_min, base_env.barrier_max
        )

        # Build action based on env mode
        if env.single_gate_mode:
            # Only output gate 0
            action = np.array([gate_action_normalized[0]], dtype=np.float32)
        elif env.bypass_barriers:
            # Only output gates
            action = gate_action_normalized.astype(np.float32)
        elif env.use_barriers:
            # Output gates + barriers
            action = np.concatenate([
                gate_action_normalized,
                barrier_action_normalized
            ]).astype(np.float32)
        else:
            action = gate_action_normalized.astype(np.float32)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record metrics
        episode_rewards.append(reward)

        # Get distance info from device state
        device_state = info.get("current_device_state", {})
        if device_state:
            gate_dist = np.abs(
                device_state["gate_ground_truth"] - device_state["current_gate_voltages"]
            )
            barrier_dist = np.abs(
                device_state["barrier_ground_truth"] - device_state["current_barrier_voltages"]
            )
            episode_gate_distances.append(gate_dist)
            episode_barrier_distances.append(barrier_dist)

        # Save scan if requested
        if save_scans and output_dir:
            scans.append(obs['image'].copy())

        step += 1

        if verbose and step <= 5:  # Print first 5 steps
            print(f"\nStep {step}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.4f}")
            if device_state:
                print(f"  Gate distances: {gate_dist}")
                print(f"  Barrier distances: {barrier_dist}")

    # Calculate statistics
    stats = {
        "total_reward": sum(episode_rewards),
        "mean_reward": np.mean(episode_rewards),
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "num_steps": step,
        "final_gate_distance": episode_gate_distances[-1] if episode_gate_distances else None,
        "final_barrier_distance": episode_barrier_distances[-1] if episode_barrier_distances else None,
    }

    # Save scans as a grid
    if save_scans and output_dir and scans:
        save_scan_grid(scans, output_dir, "oracle_scans.png")

    return stats


def save_scan_grid(scans, output_dir, filename):
    """Save scans as a grid image."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_scans = min(len(scans), 10)  # Show at most 10 scans
    n_channels = scans[0].shape[-1]

    fig, axes = plt.subplots(n_channels, n_scans, figsize=(2*n_scans, 2*n_channels))
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_scans):
        for c in range(n_channels):
            axes[c, i].imshow(scans[i][:, :, c], cmap='viridis', vmin=0, vmax=1)
            axes[c, i].axis('off')
            if i == 0:
                axes[c, i].set_ylabel(f'Ch {c}')
            if c == 0:
                axes[c, i].set_title(f'Step {i+1}')

    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150)
    plt.close()
    print(f"Saved scan grid to {output_path / filename}")


def test_oracle_agent():
    """Main test function."""
    print("=" * 60)
    print("ORACLE AGENT TEST")
    print("=" * 60)
    print("\nThis test verifies the environment by having an 'oracle' agent")
    print("that always outputs the ground truth action.")
    print("Expected: Reward should be maximal, distances should be ~0")
    print("=" * 60)

    # Get config path
    config_path = Path(__file__).parent.parent / "src" / "swarm" / "single_agent_ablations" / "single_agent_env_config.yaml"

    # Test with single_gate_mode (simplest case)
    print("\n\n>>> Testing with single_gate_mode=True <<<")
    print("-" * 40)

    env = SingleAgentEnvWrapper(
        training=True,
        config_path=str(config_path),
    )

    print(f"Environment config:")
    print(f"  num_gates: {env.num_gates}")
    print(f"  num_barriers: {env.num_barriers}")
    print(f"  num_actions: {env.num_actions}")
    print(f"  single_gate_mode: {env.single_gate_mode}")
    print(f"  bypass_barriers: {env.bypass_barriers}")
    print(f"  use_barriers: {env.use_barriers}")

    # Run multiple episodes
    all_stats = []
    for ep in range(3):
        print(f"\n--- Episode {ep+1} ---")
        stats = run_oracle_episode(
            env,
            verbose=(ep == 0),  # Verbose for first episode only
            save_scans=(ep == 0),
            output_dir="/tmp/oracle_test"
        )
        all_stats.append(stats)
        print(f"Episode {ep+1} summary:")
        print(f"  Total reward: {stats['total_reward']:.4f}")
        print(f"  Mean reward per step: {stats['mean_reward']:.4f}")
        print(f"  Min/Max reward: {stats['min_reward']:.4f} / {stats['max_reward']:.4f}")
        print(f"  Final gate distance: {stats['final_gate_distance']}")
        print(f"  Final barrier distance: {stats['final_barrier_distance']}")

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    mean_total_reward = np.mean([s['total_reward'] for s in all_stats])
    mean_step_reward = np.mean([s['mean_reward'] for s in all_stats])
    print(f"Mean total reward across episodes: {mean_total_reward:.4f}")
    print(f"Mean reward per step: {mean_step_reward:.4f}")

    # Check if rewards are as expected
    # With single_gate_mode and bypass_barriers, at ground truth we should get max reward
    # Max reward per gate = 1.0, so with 2 gates + 1 barrier = 3.0 per step
    # But in single_gate_mode with bypass_barriers, all are at ground truth
    expected_max_per_step = env.num_gates + (env.num_barriers if env.use_barriers else 0)
    print(f"\nExpected max reward per step (all at GT): {expected_max_per_step}")

    if mean_step_reward < expected_max_per_step * 0.9:  # Within 90%
        print(f"\n⚠️  WARNING: Mean reward ({mean_step_reward:.4f}) is less than 90% of expected max ({expected_max_per_step})")
        print("This suggests the oracle is NOT achieving ground truth, or reward function has issues.")
        return False
    else:
        print(f"\n✓ Rewards look correct! Oracle achieves {mean_step_reward/expected_max_per_step*100:.1f}% of max.")
        return True


def test_random_vs_oracle():
    """Compare random agent vs oracle to show the difference."""
    print("\n" + "=" * 60)
    print("RANDOM vs ORACLE COMPARISON")
    print("=" * 60)

    config_path = Path(__file__).parent.parent / "src" / "swarm" / "single_agent_ablations" / "single_agent_env_config.yaml"

    env = SingleAgentEnvWrapper(
        training=True,
        config_path=str(config_path),
    )

    # Run random agent
    print("\n>>> Random Agent <<<")
    obs, _ = env.reset()
    random_rewards = []
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        random_rewards.append(reward)
        if terminated or truncated:
            break
    print(f"Random agent mean reward: {np.mean(random_rewards):.4f}")

    # Run oracle agent
    print("\n>>> Oracle Agent <<<")
    obs, _ = env.reset()
    oracle_rewards = []
    for _ in range(50):
        # Get ground truth
        gate_gt = env.base_env.device_state["gate_ground_truth"]
        barrier_gt = env.base_env.device_state["barrier_ground_truth"]
        gate_action = normalize_voltage(gate_gt, env.base_env.plunger_min, env.base_env.plunger_max)

        if env.single_gate_mode:
            action = np.array([gate_action[0]], dtype=np.float32)
        else:
            action = gate_action.astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        oracle_rewards.append(reward)
        if terminated or truncated:
            break
    print(f"Oracle agent mean reward: {np.mean(oracle_rewards):.4f}")

    env.close()

    print(f"\nDifference: {np.mean(oracle_rewards) - np.mean(random_rewards):.4f}")
    print("(Oracle should be significantly higher)")


if __name__ == "__main__":
    success = test_oracle_agent()
    test_random_vs_oracle()

    if not success:
        print("\n❌ TEST FAILED - Environment may have issues")
        sys.exit(1)
    else:
        print("\n✓ TEST PASSED - Environment appears correct")
        sys.exit(0)
