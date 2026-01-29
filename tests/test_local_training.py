#!/usr/bin/env python3
"""
Local training test with detailed logging.

Runs a few iterations of single-agent PPO locally to debug why training doesn't converge.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import yaml

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from swarm.single_agent_ablations.utils.factory import create_rl_module_spec
from swarm.single_agent_ablations.utils.custom_ppo_learner import PPOLearnerWithValueStats


def create_env(config=None):
    """Create single-agent environment."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
    config_path = str(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml")
    return SingleAgentEnvWrapper(training=True, config_path=config_path)


def main():
    print("=" * 70)
    print("LOCAL TRAINING DEBUG TEST")
    print("=" * 70)

    # Load configs
    config_path = src_dir / "swarm/single_agent_ablations/training_config.yaml"
    env_config_path = src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    print(f"\nEnvironment config:")
    print(f"  num_dots: {env_config['simulator']['num_dots']}")
    print(f"  single_gate_mode: {env_config['simulator'].get('single_gate_mode', False)}")
    print(f"  bypass_barriers: {env_config['simulator'].get('bypass_barriers', False)}")

    # Initialize Ray (minimal config for local testing)
    ray.init(
        include_dashboard=False,
        log_to_driver=False,
        logging_level=40,  # ERROR only
    )

    try:
        # Register environment
        register_env("test_env", create_env)

        # Create module spec
        rl_module_config = {
            **config['neural_networks']['single_agent_policy'],
            "free_log_std": config['rl_config']['single_agent']['free_log_std'],
            "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
        }
        rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

        # Build algorithm with LOCAL execution (no remote workers) for fast debugging
        algo_config = (
            PPOConfig()
            .environment(env="test_env")
            .rl_module(rl_module_spec=rl_module_spec)
            .env_runners(
                num_env_runners=0,  # LOCAL execution - no remote workers
                rollout_fragment_length=50,
                sample_timeout_s=300,
            )
            .learners(
                num_learners=0,  # Local learner
                num_gpus_per_learner=0,
            )
            .training(
                train_batch_size_per_learner=200,  # Small for local
                minibatch_size=50,
                num_epochs=4,
                lr=config['rl_config']['training']['lr'],
                gamma=config['rl_config']['training']['gamma'],
                lambda_=config['rl_config']['training']['lambda_'],
                clip_param=config['rl_config']['training']['clip_param'],
                entropy_coeff=config['rl_config']['training']['entropy_coeff'],
                vf_loss_coeff=config['rl_config']['training']['vf_loss_coeff'],
                learner_class=PPOLearnerWithValueStats,
            )
        )

        print("\nBuilding algorithm...")
        algo = algo_config.build()

        print("\nStarting training iterations...")
        print("-" * 70)

        for i in range(5):  # Just 5 iterations for debugging
            print(f"\n>>> Iteration {i+1} <<<")
            result = algo.train()

            # Extract key metrics
            env_runners = result.get("env_runners", {})
            learners = result.get("learners", {})

            print(f"\n  Episode metrics:")
            print(f"    reward_mean: {env_runners.get('episode_return_mean', 'N/A'):.4f}" if env_runners.get('episode_return_mean') else "    reward_mean: N/A")
            print(f"    episode_len_mean: {env_runners.get('episode_len_mean', 'N/A'):.1f}" if env_runners.get('episode_len_mean') else "    episode_len_mean: N/A")

            # Get learner metrics (contains our debug logs)
            if learners:
                # Find the policy key
                for key in learners.keys():
                    if key.startswith("module_"):
                        continue
                    learner_info = learners.get(key, {})
                    if isinstance(learner_info, dict):
                        # Print policy behavior
                        print(f"\n  Policy behavior:")
                        for metric in ["policy/action_mean_mean", "policy/action_mean_std",
                                      "policy/action_std_mean", "policy/log_std_mean"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.4f}")

                        # Print loss info
                        print(f"\n  Losses:")
                        for metric in ["total_loss", "policy_loss", "vf_loss", "entropy"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.4f}")

                        # Print value function info
                        print(f"\n  Value function:")
                        for metric in ["vf_predictions_mean", "vf_predictions_std"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.4f}")

                        # Print gradient norms
                        print(f"\n  Gradient norms:")
                        for metric in ["grad_norm/encoder_mean", "grad_norm/pi_mean", "grad_norm/vf_mean"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.6f}")

                        # Print rewards from batch
                        print(f"\n  Batch rewards:")
                        for metric in ["rewards/mean", "rewards/min", "rewards/max"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.4f}")

                        # Print actions
                        print(f"\n  Actions taken:")
                        for metric in ["actions/mean", "actions/std", "actions/min", "actions/max"]:
                            if metric in learner_info:
                                print(f"    {metric}: {learner_info[metric]:.4f}")

            print("-" * 70)

        algo.stop()

    finally:
        ray.shutdown()

    print("\n✓ Local training test completed")


if __name__ == "__main__":
    main()
