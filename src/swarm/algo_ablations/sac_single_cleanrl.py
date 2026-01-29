#!/usr/bin/env python3
"""
Thin wrapper that registers SingleAgentWrapper as a Gymnasium env and then
executes CleanRL's SAC training module with that env-id.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

ENV_ID = "SwarmSingleAgent-v0"


def make_env():
    """Gymnasium entry-point for CleanRL."""
    from swarm.algo_ablations.single_agent_wrapper import SingleAgentWrapper

    env_config_path = os.environ.get("SWARM_ENV_CONFIG_PATH") or None
    capacitance_model_checkpoint = os.environ.get("SWARM_CAP_MODEL_CHECKPOINT") or None
    deterministic = os.environ.get("SWARM_DETERMINISTIC", "0") == "1"

    env = SingleAgentWrapper(
        training=True,
        return_voltage=False,
        env_config_path=env_config_path,
        capacitance_model_checkpoint=capacitance_model_checkpoint,
        deterministic=deterministic,
    )

    # CleanRL SAC MLP expects flat observations.
    env = gym.wrappers.FlattenObservation(env)
    return env


def _register_env(env_id: str):
    try:
        gym.spec(env_id)
        return
    except Exception:
        pass
    register(id=env_id, entry_point="swarm.algo_ablations.sac_single:make_env")


def _has_env_id(args):
    for idx, arg in enumerate(args):
        if arg == "--env-id":
            return idx + 1 < len(args)
        if arg.startswith("--env-id="):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--capacitance-model-checkpoint", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args, passthrough = parser.parse_known_args()

    os.environ["SWARM_ENV_CONFIG_PATH"] = args.env_config or ""
    os.environ["SWARM_CAP_MODEL_CHECKPOINT"] = args.capacitance_model_checkpoint or ""
    if args.deterministic:
        os.environ["SWARM_DETERMINISTIC"] = "1"

    _register_env(args.env_id)

    cleanrl_args = list(passthrough)
    if not _has_env_id(cleanrl_args):
        cleanrl_args = ["--env-id", args.env_id] + cleanrl_args

    sys.argv = ["cleanrl.sac_continuous_action"] + cleanrl_args
    runpy.run_module("cleanrl.sac_continuous_action", run_name="__main__")


if __name__ == "__main__":
    main()
