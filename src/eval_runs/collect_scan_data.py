#!/usr/bin/env python3
"""
Simple data collection script for capacitance model validation.
Collects: scans + step position + CGD matrices without Ray overhead.
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup paths
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


def create_env(env_config_path: str = None):
    """Create the multi-agent environment."""
    config_path = env_config_path or str(current_dir / "env_config.yaml")

    env = MultiAgentEnvWrapper(
        training=False,
        return_voltage=False,
        gif_config=None,
        distance_data_dir=None,
        env_config_path=config_path,
    )
    return env


def collect_data(
    num_episodes: int = 10,
    max_steps: int = 100,
    output_dir: str = None,
    env_config_path: str = None,
    random_actions: bool = True,
):
    """
    Collect scan data from environment rollouts.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        output_dir: Where to save data
        env_config_path: Path to environment config
        random_actions: If True, use random actions; if False, need policy

    Saves:
        - scans/episode_XXXX/step_YYYYYY.npy: Raw scans (num_channels, H, W)
        - capacitance/cgd_true_XXXX.npy: True CGD matrix for each episode
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = current_dir / f"scan_data_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    scans_dir = output_dir / "scans"
    scans_dir.mkdir(exist_ok=True)
    capacitance_dir = output_dir / "capacitance"
    capacitance_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Collecting {num_episodes} episodes with max {max_steps} steps each")

    # Create environment
    print("Creating environment...")
    env = create_env(env_config_path)

    # Get action space info
    action_spaces = env.action_spaces
    print(f"Agent IDs: {list(action_spaces.keys())}")

    total_steps = 0

    for episode in range(num_episodes):
        episode_dir = scans_dir / f"episode_{episode:04d}"
        episode_dir.mkdir(exist_ok=True)

        # Reset environment
        obs, info = env.reset()

        # Get CGD matrix at start of episode (before any actions)
        cgd_matrix = env.base_env.array.model.cgd_full.copy()

        for step in range(max_steps):
            # Extract and save scans from observations
            # Barrier agents have the scans we want
            num_scans = env.num_gates - 1
            scans = []

            for i in range(num_scans):
                barrier_id = f"barrier_{i}"
                if barrier_id in obs:
                    agent_obs = obs[barrier_id]
                    if isinstance(agent_obs, dict):
                        scan = agent_obs['image'][:, :, 0]
                    else:
                        scan = agent_obs[:, :, 0]
                    scans.append(scan)

            if scans:
                # Save as (num_scans, H, W) array
                scan_array = np.stack(scans, axis=0)
                scan_path = episode_dir / f"step_{step:06d}.npy"
                np.save(scan_path, scan_array)

            # Generate random actions
            actions = {}
            for agent_id, space in action_spaces.items():
                actions[agent_id] = space.sample()

            # Step environment
            obs, rewards, terminated, truncated, infos = env.step(actions)
            total_steps += 1

            # Check if episode is done
            done = any(terminated.values()) or any(truncated.values())
            if done:
                break

        # Save CGD matrix for this episode
        cgd_path = capacitance_dir / f"cgd_true_{episode:04d}.npy"
        np.save(cgd_path, cgd_matrix)

        print(f"Episode {episode}: {step+1} steps, CGD shape: {cgd_matrix.shape}")

    print(f"\nCollection complete!")
    print(f"Total steps: {total_steps}")
    print(f"Scans saved to: {scans_dir}")
    print(f"CGD matrices saved to: {capacitance_dir}")

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Collect scan data for capacitance model validation')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--env-config', type=str, default=None, help='Path to env config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    collect_data(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        env_config_path=args.env_config,
        random_actions=True,
    )


if __name__ == "__main__":
    main()
