#!/usr/bin/env python3
"""
Minimal inference script for running RL agent rollouts.

Runs in a SINGLE PROCESS (num_env_runners=0, num_learners=0) to avoid
hanging issues from Ray worker coordination.

Usage:
    uv run python src/eval_runs/inference.py --checkpoint /path/to/checkpoint --num-dots 8 --num-rollouts 10
"""
import os
import sys
import warnings
import argparse
import time
from datetime import datetime
from functools import partial
from pathlib import Path

# Suppress warnings before any other imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set JAX memory settings BEFORE any imports that might initialize JAX
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
os.environ.setdefault("JAX_ENABLE_X64", "true")

# Add src to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import yaml
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from swarm.voltage_model import create_rl_module_spec
from swarm.training.train_utils import fix_optimizer_betas_after_checkpoint_load
from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs."""
    if agent_id.startswith("plunger"):
        return "plunger_policy"
    elif agent_id.startswith("barrier"):
        return "barrier_policy"
    raise ValueError(f"Unknown agent type: {agent_id}")


def create_env(config=None, env_config_path=None, distance_data_dir=None):
    """Create multi-agent environment."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    return MultiAgentEnvWrapper(
        training=False,
        return_voltage=True,
        gif_config=None,
        distance_data_dir=distance_data_dir,
        env_config_path=env_config_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--num-dots", type=int, default=8, help="Number of quantum dots")
    parser.add_argument("--num-rollouts", type=int, default=10, help="Number of rollouts")
    parser.add_argument("--env-config", type=str, default=None, help="Path to env config (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for distance data")
    parser.add_argument("--upload-to-wandb", action="store_true", help="Upload results to wandb")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Num dots: {args.num_dots}")
    print(f"Num rollouts: {args.num_rollouts}")

    # Load training config from checkpoint
    training_config_path = checkpoint_dir / "training_config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(f"training_config.yaml not found in: {checkpoint_dir}")

    with open(training_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded training config from checkpoint")

    # Load env config
    if args.env_config:
        env_config_path = Path(args.env_config)
    else:
        env_config_path = Path(__file__).parent / "env_config.yaml"

    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    # Override num_dots
    original_num_dots = env_config["simulator"]["num_dots"]
    env_config["simulator"]["num_dots"] = args.num_dots
    print(f"Overriding num_dots: {original_num_dots} -> {args.num_dots}")

    # Write modified env config to temp file
    import tempfile
    temp_env_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir='/tmp')
    yaml.dump(env_config, temp_env_config, default_flow_style=False)
    temp_env_config_path = temp_env_config.name
    temp_env_config.close()

    # Create distance data directory
    if args.output_dir:
        distance_data_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = checkpoint_dir.name.replace(":", "_")
        distance_data_dir = Path(__file__).parent / "collected_data" / f"{timestamp}_{checkpoint_name}_{args.num_dots}dots"

    distance_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Distance data directory: {distance_data_dir}")

    # Initialize wandb if requested
    wandb_run = None
    if args.upload_to_wandb:
        import wandb
        wandb_run = wandb.init(
            project="RLModel",
            entity="rl_agents_for_tuning",
            job_type="inference",
            config={
                "checkpoint": str(checkpoint_dir),
                "num_dots": args.num_dots,
                "num_rollouts": args.num_rollouts,
            }
        )

    # Initialize Ray (minimal config for single-process execution)
    print("Initializing Ray...")
    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        logging_level=40,  # ERROR only
    )

    try:
        # Register environment
        create_env_fn = partial(
            create_env,
            env_config_path=temp_env_config_path,
            distance_data_dir=str(distance_data_dir),
        )
        register_env("qarray_multiagent_env", create_env_fn)

        # Create RL module spec
        rl_module_config = {
            "plunger_policy": {
                **config['neural_networks']['plunger_policy'],
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            },
            "barrier_policy": {
                **config['neural_networks']['barrier_policy'],
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            }
        }
        rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

        # Build algorithm with LOCAL execution (no distributed workers)
        print("Building PPO algorithm (local execution mode)...")
        algo_config = (
            PPOConfig()
            .environment(env="qarray_multiagent_env")
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=["plunger_policy", "barrier_policy"],
                policies_to_train=[],  # No training during inference
                count_steps_by="agent_steps",
            )
            .rl_module(rl_module_spec=rl_module_spec)
            .env_runners(
                num_env_runners=0,  # LOCAL execution - single process!
                rollout_fragment_length=100,
                sample_timeout_s=600,
            )
            .learners(
                num_learners=0,  # No remote learners
                num_gpus_per_learner=1,  # Use GPU for model inference
            )
            .evaluation(
                evaluation_num_env_runners=0,  # Local evaluation
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_sample_timeout_s=600,
            )
        )

        algo = algo_config.build()
        print("Algorithm built successfully")

        # Load checkpoint
        print(f"Loading checkpoint...")
        algo.restore_from_path(str(checkpoint_dir))
        fix_optimizer_betas_after_checkpoint_load(algo)
        print("Checkpoint loaded successfully")

        # Run rollouts
        print(f"\nStarting {args.num_rollouts} rollouts with {args.num_dots} dots...")
        print("-" * 60)

        start_time = time.time()
        all_rewards = []

        for i in range(args.num_rollouts):
            rollout_start = time.time()
            print(f"Rollout {i+1}/{args.num_rollouts}...", end=" ", flush=True)

            result = algo.evaluate()

            # Extract metrics
            env_runners = result.get("env_runners", {})
            episode_reward = env_runners.get("episode_return_mean", 0)
            episode_len = env_runners.get("episode_len_mean", 0)
            all_rewards.append(episode_reward)

            rollout_time = time.time() - rollout_start
            print(f"reward={episode_reward:.2f}, len={episode_len:.0f}, time={rollout_time:.1f}s")

        total_time = time.time() - start_time
        avg_reward = np.mean(all_rewards) if all_rewards else 0

        print("-" * 60)
        print(f"Completed {args.num_rollouts} rollouts in {total_time:.1f}s")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Distance data saved to: {distance_data_dir}")

        # Upload to wandb if requested
        if wandb_run:
            import wandb
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_name = f"inference_distances_{args.num_dots}dots_{timestamp}"
            distance_artifact = wandb.Artifact(
                artifact_name,
                type="distance_data",
                metadata={
                    "checkpoint": str(checkpoint_dir),
                    "num_dots": args.num_dots,
                    "num_rollouts": args.num_rollouts,
                    "avg_reward": avg_reward,
                }
            )

            # Add all distance files
            for agent_dir in distance_data_dir.iterdir():
                if agent_dir.is_dir():
                    for npy_file in agent_dir.glob("*.npy"):
                        artifact_path = f"{agent_dir.name}/{npy_file.name}"
                        distance_artifact.add_file(str(npy_file), name=artifact_path)

            wandb.log_artifact(distance_artifact)
            print(f"Uploaded artifact: {artifact_name}")

            wandb.log({
                "inference/num_rollouts": args.num_rollouts,
                "inference/num_dots": args.num_dots,
                "inference/avg_reward": avg_reward,
                "inference/total_time_s": total_time,
            })

    finally:
        if ray.is_initialized():
            ray.shutdown()
        if wandb_run:
            import wandb
            wandb.finish()

    print("\nInference completed!")


if __name__ == "__main__":
    main()
