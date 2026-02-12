#!/usr/bin/env python3
"""
Simple data collection: runs episodes with trained policy, saves scans + CGD + VGM.
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

# Setup paths
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
sys.path.insert(0, str(src_dir))


def collect_episode_data(
    checkpoint_path: str,
    num_episodes: int = 5,
    output_dir: str = None,
    gpu: int = 7,
):
    """
    Collect scans + CGD + VGM from episodes using trained policy.

    Saves:
        output_dir/
            episode_XXXX/
                step_YYYYYY.npy   # Raw scans (num_channels, H, W)
            cgd_true_XXXX.npy     # True CGD matrix
            vgm_XXXX.npy          # Virtual gate matrix estimate
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / f"simple_data_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print()

    # Initialize Ray in local mode (no workers, no overhead)
    import ray
    ray.init(local_mode=True, ignore_reinit_error=True, num_gpus=1)

    # Load algorithm from checkpoint
    from ray.rllib.algorithms.algorithm import Algorithm
    print("Loading checkpoint...")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded.")

    # Get the environment from the algorithm's local worker
    env = algo.env_creator(algo.config.env_config)

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep} ===")

        # Create episode directory
        ep_dir = output_dir / f"episode_{ep:04d}"
        ep_dir.mkdir(exist_ok=True)

        # Reset
        obs, info = env.reset()

        # Get CGD at start (true capacitance matrix)
        cgd = env.base_env.array.model.cgd_full.copy()

        step = 0
        done = False

        while not done:
            # Extract and save scans from observations
            scans = []
            num_channels = env.num_gates - 1
            for i in range(num_channels):
                agent_id = f"barrier_{i}"
                if agent_id in obs:
                    agent_obs = obs[agent_id]
                    if isinstance(agent_obs, dict):
                        scan = agent_obs['image'][:, :, 0]
                    else:
                        scan = agent_obs[:, :, 0]
                    scans.append(scan)

            if scans:
                scan_array = np.stack(scans, axis=0)
                np.save(ep_dir / f"step_{step:06d}.npy", scan_array)

            # Get actions from trained policy
            actions = {}
            for agent_id in obs.keys():
                policy_id = "plunger_policy" if "plunger" in agent_id else "barrier_policy"
                action = algo.compute_single_action(obs[agent_id], policy_id=policy_id)
                actions[agent_id] = action

            # Step
            obs, rewards, terminated, truncated, infos = env.step(actions)
            step += 1

            done = any(terminated.values()) or any(truncated.values())

            if step % 20 == 0:
                print(f"  Step {step}")

        # Save CGD and VGM at end of episode
        np.save(output_dir / f"cgd_true_{ep:04d}.npy", cgd)

        vgm = env.base_env.device_state.get("virtual_gate_matrix", None)
        if vgm is not None:
            np.save(output_dir / f"vgm_{ep:04d}.npy", vgm)

        print(f"  Done: {step} steps, CGD shape: {cgd.shape}")
        if vgm is not None:
            print(f"  VGM shape: {vgm.shape}")

    ray.shutdown()
    print(f"\nData saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/rl_checkpoint_best:v3482")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()

    collect_episode_data(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        output_dir=args.output,
        gpu=args.gpu,
    )
