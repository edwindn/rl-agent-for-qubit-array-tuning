"""
Test inference script - plots distance from ground truth for each plunger over an episode.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add paths
benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(benchmarks_dir))
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
from policy_loader import load_policies, get_deterministic_action


def run_episode_from_obs(env, policies, device, obs_dict, max_steps=50):
    """Run episode from given observation state and collect distance data."""
    num_plungers = env.num_gates
    num_barriers = env.num_barriers

    # Storage for distances
    plunger_distances = {f"plunger_{i}": [] for i in range(num_plungers)}
    barrier_distances = {f"barrier_{i}": [] for i in range(num_barriers)}

    # Record initial distances
    plunger_v = env.base_env.device_state["current_gate_voltages"]
    barrier_v = env.base_env.device_state["current_barrier_voltages"]
    plunger_gt = env.base_env.device_state["gate_ground_truth"]
    barrier_gt = env.base_env.device_state["barrier_ground_truth"]

    for i in range(num_plungers):
        plunger_distances[f"plunger_{i}"].append(plunger_v[i] - plunger_gt[i])
    for i in range(num_barriers):
        barrier_distances[f"barrier_{i}"].append(barrier_v[i] - barrier_gt[i])

    print(f"Initial state:")
    print(f"  Plunger voltages: {plunger_v}")
    print(f"  Plunger GT: {plunger_gt}")
    print(f"  Initial plunger distances: {plunger_v - plunger_gt}")

    for step in range(max_steps):
        # Get actions
        actions = {}
        for agent_id, agent_obs in obs_dict.items():
            policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
            torch_obs = {
                "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
                "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
            }
            action = get_deterministic_action(policies[policy_name], torch_obs)
            actions[agent_id] = action.cpu().numpy().flatten()

        # Step
        obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

        # Record distances
        plunger_v = env.base_env.device_state["current_gate_voltages"]
        barrier_v = env.base_env.device_state["current_barrier_voltages"]
        plunger_gt = env.base_env.device_state["gate_ground_truth"]
        barrier_gt = env.base_env.device_state["barrier_ground_truth"]

        for i in range(num_plungers):
            plunger_distances[f"plunger_{i}"].append(plunger_v[i] - plunger_gt[i])
        for i in range(num_barriers):
            barrier_distances[f"barrier_{i}"].append(barrier_v[i] - barrier_gt[i])

        # Check termination
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
        if done:
            break

    print(f"\nFinal state after {step+1} steps:")
    print(f"  Plunger voltages: {plunger_v}")
    print(f"  Final plunger distances: {plunger_v - plunger_gt}")

    return plunger_distances, barrier_distances


def plot_distances(plunger_distances, barrier_distances, save_path):
    """Plot distance from ground truth over episode."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plunger plot
    ax1 = axes[0]
    for agent_id, distances in plunger_distances.items():
        ax1.plot(distances, label=agent_id, linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.5, color='g', linestyle=':', linewidth=1, alpha=0.5, label='Success threshold')
    ax1.axhline(y=-0.5, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Distance from Ground Truth (V)')
    ax1.set_title('Plunger Distances Over Episode')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Barrier plot
    ax2 = axes[1]
    for agent_id, distances in barrier_distances.items():
        ax2.plot(distances, label=agent_id, linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.5, color='g', linestyle=':', linewidth=1, alpha=0.5, label='Success threshold')
    ax2.axhline(y=-0.5, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance from Ground Truth (V)')
    ax2.set_title('Barrier Distances Over Episode')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()

    print("Creating environment...")
    env = MultiAgentEnvWrapper(training=True, return_voltage=True)

    print(f"  num_dots: {env.num_gates}")
    print(f"  num_barriers: {env.num_barriers}")

    print("\nLoading policies...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "../../artifacts/rl_checkpoint_best:v3482"
    policies = load_policies(checkpoint_path, device=device)
    print(f"  Loaded: {list(policies.keys())}")

    print(f"\nRunning episode with seed={args.seed}...")

    # Modify run_episode to accept seed
    obs_dict, info = env.reset(seed=args.seed)
    plunger_distances, barrier_distances = run_episode_from_obs(env, policies, device, obs_dict, max_steps=args.max_steps)

    # Plot and save
    save_path = Path(__file__).parent / f"inference_distances_seed{args.seed}.png"
    plot_distances(plunger_distances, barrier_distances, save_path)


if __name__ == "__main__":
    main()
