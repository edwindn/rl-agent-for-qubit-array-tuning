#!/usr/bin/env python3
"""
Single-agent PPO training with Ray RLlib on SingleAgentWrapper.

Uses RLlib's PPO implementation (no custom training loop beyond calling algo.train()).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.algo_ablations.ppo_single_ray_agent import (
    build_single_agent_ppo_module_spec,
    DEFAULT_PPO_SINGLE_AGENT_MODEL_CONFIG,
)
from swarm.training.utils.metrics_logger import print_training_progress

ENV_ID = "SingleAgentRayPPO-v0"


def make_env(env_ctx):
    """RLlib environment creator."""
    # Keep JAX settings consistent with other training scripts
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    from swarm.algo_ablations.single_agent_wrapper import SingleAgentWrapper

    env_config_path = env_ctx["env_config_path"]
    capacitance_model_checkpoint = env_ctx["capacitance_model_checkpoint"]
    deterministic = env_ctx["deterministic"]

    env = SingleAgentWrapper(
        training=True,
        return_voltage=True,
        env_config_path=env_config_path,
        capacitance_model_checkpoint=capacitance_model_checkpoint,
        deterministic=deterministic,
    )

    return env


def parse_args():
    default_env_config = str(current_dir / "configs" / "env_config.yaml")
    parser = argparse.ArgumentParser(description="RLlib PPO on SingleAgentWrapper")
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--env-config", type=str, default=default_env_config)
    parser.add_argument("--capacitance-model-checkpoint", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--stop-timesteps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()

def load_training_config():
    config_path = current_dir / "training_config_single.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    training_config = load_training_config()
    ray_config = training_config["ray"]
    rl_config = training_config["rl_config"]
    ppo_config = training_config["ppo"]
    defaults = training_config["defaults"]
    if args.num_iterations is None:
        args.num_iterations = defaults["num_iterations"]
    init_kwargs = {
        "ignore_reinit_error": True,
        "include_dashboard": ray_config["include_dashboard"],
        "log_to_driver": ray_config["log_to_driver"],
        "logging_level": ray_config["logging_level"],
        "runtime_env": ray_config["runtime_env"],
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    ray.init(**init_kwargs)

    register_env(args.env_id, make_env)

    env_config = {
        "env_config_path": args.env_config,
        "capacitance_model_checkpoint": args.capacitance_model_checkpoint,
        "deterministic": args.deterministic,
    }

    temp_env = make_env(env_config)
    try:
        rl_module_spec = build_single_agent_ppo_module_spec(
            observation_space=temp_env.observation_space,
            action_space=temp_env.action_space,
            model_config=DEFAULT_PPO_SINGLE_AGENT_MODEL_CONFIG,
        )
    finally:
        if hasattr(temp_env, "close"):
            temp_env.close()

    env_runner_cfg = dict(rl_config["env_runners"])
    learner_cfg = dict(rl_config["learners"])
    if args.num_gpus is not None:
        learner_cfg["num_gpus_per_learner"] = args.num_gpus

    config = (
        PPOConfig()
        .environment(env=args.env_id, env_config=env_config)
        .rl_module(rl_module_spec=rl_module_spec)
        .framework("torch")
        .env_runners(**env_runner_cfg)
        .learners(**learner_cfg)
        .training(**ppo_config["training"])
        .debugging(seed=args.seed)
    )

    algo = config.build()

    start_time = time.time()
    try:
        for _ in range(args.num_iterations):
            result = algo.train()
            env_steps = result["env_runners"]["num_env_steps_sampled_lifetime"]
            print_training_progress(result, result["training_iteration"] - 1, start_time)
            if args.stop_timesteps and env_steps >= args.stop_timesteps:
                break
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
