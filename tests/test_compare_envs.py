#!/usr/bin/env python3
"""
Compare multi-agent vs single-agent environment and module setups.
"""
import sys
from pathlib import Path
import numpy as np
import yaml

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def test_env_comparison():
    """Compare observation formats between multi-agent and single-agent envs."""
    print("=" * 70)
    print("ENVIRONMENT COMPARISON")
    print("=" * 70)

    # Multi-agent env
    print("\n>>> MULTI-AGENT ENV <<<")
    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper

    multi_env = MultiAgentEnvWrapper(return_voltage=True)
    multi_obs, _ = multi_env.reset()

    print(f"Agent IDs: {multi_env.get_agent_ids()}")

    # Pick first plunger agent
    agent_id = "plunger_0"
    agent_obs = multi_obs[agent_id]
    print(f"\nObservation for {agent_id}:")
    print(f"  Type: {type(agent_obs)}")
    if isinstance(agent_obs, dict):
        for k, v in agent_obs.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")
            else:
                print(f"  {k}: {v}")

    print(f"\nAction space for {agent_id}: {multi_env.action_space[agent_id]}")
    print(f"Observation space for {agent_id}: {multi_env.observation_space[agent_id]}")

    multi_env.close()

    # Single-agent env
    print("\n>>> SINGLE-AGENT ENV <<<")
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml")
    single_env = SingleAgentEnvWrapper(training=True, config_path=config_path)
    single_obs, _ = single_env.reset()

    print(f"\nObservation:")
    print(f"  Type: {type(single_obs)}")
    if isinstance(single_obs, dict):
        for k, v in single_obs.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.3f}, {v.max():.3f}]")
            else:
                print(f"  {k}: {v}")

    print(f"\nAction space: {single_env.action_space}")
    print(f"Observation space: {single_env.observation_space}")

    single_env.close()


def test_module_comparison():
    """Compare module forward passes."""
    print("\n" + "=" * 70)
    print("MODULE COMPARISON")
    print("=" * 70)

    import torch

    # Load configs
    with open(src_dir / "swarm/training/configs/ppo_impala.yaml") as f:
        multi_config = yaml.safe_load(f)
    with open(src_dir / "swarm/single_agent_ablations/training_config.yaml") as f:
        single_config = yaml.safe_load(f)
    with open(src_dir / "swarm/environment/env_config.yaml") as f:
        multi_env_config = yaml.safe_load(f)
    with open(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml") as f:
        single_env_config = yaml.safe_load(f)

    # Multi-agent module
    print("\n>>> MULTI-AGENT MODULE <<<")
    from swarm.voltage_model import create_rl_module_spec as multi_factory

    multi_rl_config = {
        "plunger_policy": {
            **multi_config['neural_networks']['plunger_policy'],
            "free_log_std": multi_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": multi_config['rl_config']['multi_agent']['log_std_bounds'],
        },
        "barrier_policy": {
            **multi_config['neural_networks']['barrier_policy'],
            "free_log_std": multi_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": multi_config['rl_config']['multi_agent']['log_std_bounds'],
        }
    }

    multi_spec = multi_factory(multi_env_config, algo="ppo", config=multi_rl_config)
    multi_module = multi_spec.build()

    print(f"Module type: {type(multi_module)}")
    print(f"Module IDs: {list(multi_module.keys())}")

    # Get plunger policy
    plunger_module = multi_module["plunger_policy"]
    print(f"\nPlunger policy type: {type(plunger_module)}")

    # Test forward pass
    resolution = multi_env_config['simulator']['resolution']
    dummy_obs = {
        "image": torch.rand(4, resolution, resolution, 2),  # 2 channels for plunger
        "voltage": torch.rand(4, 1) * 2 - 1,
    }

    with torch.no_grad():
        out = plunger_module.forward_train({"obs": dummy_obs})

    print(f"Forward output keys: {list(out.keys())}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")

    # Single-agent module
    print("\n>>> SINGLE-AGENT MODULE <<<")
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec as single_factory

    single_rl_config = {
        **single_config['neural_networks']['single_agent_policy'],
        "free_log_std": single_config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": single_config['rl_config']['single_agent']['log_std_bounds'],
    }

    single_spec = single_factory(single_env_config, algo="ppo", config=single_rl_config)
    single_module = single_spec.build()

    print(f"Module type: {type(single_module)}")

    # Test forward pass
    resolution = single_env_config['simulator']['resolution']
    num_channels = single_env_config['simulator']['num_dots'] - 1

    # Single gate mode = 1 action
    dummy_obs = {
        "image": torch.rand(4, resolution, resolution, num_channels),
        "voltage": torch.rand(4, 1) * 2 - 1,  # single_gate_mode = 1 voltage
    }

    with torch.no_grad():
        out = single_module.forward_train({"obs": dummy_obs})

    print(f"Forward output keys: {list(out.keys())}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")

    # Compare action dist inputs
    print("\n>>> COMPARING ACTION DISTRIBUTIONS <<<")

    # Multi-agent
    with torch.no_grad():
        multi_out = plunger_module.forward_train({"obs": {
            "image": torch.rand(4, resolution, resolution, 2),
            "voltage": torch.rand(4, 1) * 2 - 1,
        }})

    multi_dist = multi_out["action_dist_inputs"]
    multi_mean = multi_dist[:, 0]
    multi_log_std = multi_dist[:, 1]

    print(f"Multi-agent (plunger_0):")
    print(f"  action_dist_inputs shape: {multi_dist.shape}")
    print(f"  mean: {multi_mean.mean():.4f} ± {multi_mean.std():.4f}")
    print(f"  log_std: {multi_log_std.mean():.4f} ± {multi_log_std.std():.4f}")
    print(f"  std: {multi_log_std.exp().mean():.4f}")

    # Single-agent
    with torch.no_grad():
        single_out = single_module.forward_train({"obs": dummy_obs})

    single_dist = single_out["action_dist_inputs"]
    single_mean = single_dist[:, 0]
    single_log_std = single_dist[:, 1]

    print(f"\nSingle-agent:")
    print(f"  action_dist_inputs shape: {single_dist.shape}")
    print(f"  mean: {single_mean.mean():.4f} ± {single_mean.std():.4f}")
    print(f"  log_std: {single_log_std.mean():.4f} ± {single_log_std.std():.4f}")
    print(f"  std: {single_log_std.exp().mean():.4f}")


if __name__ == "__main__":
    test_env_comparison()
    test_module_comparison()
    print("\n✓ Comparison complete")
