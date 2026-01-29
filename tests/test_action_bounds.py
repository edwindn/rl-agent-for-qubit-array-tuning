"""
Test to verify policy action outputs are correctly bounded to [-1, 1].

This checks if the model outputs are properly scaled, which could explain
why training doesn't work.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import yaml


def test_action_bounds():
    """Test that policy outputs are bounded correctly."""
    print("=" * 60)
    print("ACTION BOUNDS TEST")
    print("=" * 60)

    # Load configs
    config_path = src_dir / "swarm" / "single_agent_ablations" / "training_config.yaml"
    env_config_path = src_dir / "swarm" / "single_agent_ablations" / "single_agent_env_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(env_config_path) as f:
        env_config = yaml.safe_load(f)

    # Create module spec
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec

    rl_module_config = {
        **config['neural_networks']['single_agent_policy'],
        "free_log_std": config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

    # Build module
    module = rl_module_spec.build()
    module.eval()

    print(f"\nModule type: {type(module)}")
    print(f"Action space: {rl_module_spec.action_space}")
    print(f"Observation space keys: {list(rl_module_spec.observation_space.spaces.keys())}")

    # Create dummy observation
    batch_size = 32
    resolution = env_config['simulator']['resolution']
    num_channels = env_config['simulator']['num_dots'] - 1

    # Determine action dim based on single_gate_mode
    if env_config['simulator'].get('single_gate_mode', False):
        action_dim = 1
    elif env_config['simulator'].get('bypass_barriers', False):
        action_dim = env_config['simulator']['num_dots']
    else:
        action_dim = env_config['simulator']['num_dots'] + (env_config['simulator']['num_dots'] - 1)

    dummy_obs = {
        "image": torch.rand(batch_size, resolution, resolution, num_channels),
        "voltage": torch.rand(batch_size, action_dim) * 2 - 1,  # [-1, 1]
    }

    print(f"\nDummy obs shapes:")
    print(f"  image: {dummy_obs['image'].shape}")
    print(f"  voltage: {dummy_obs['voltage'].shape}")

    # Forward pass through module
    with torch.no_grad():
        output = module.forward_inference({"obs": dummy_obs})

    print(f"\nModule output keys: {list(output.keys())}")

    # Get actions
    actions = output.get("actions", output.get("action_dist_inputs"))
    print(f"Actions shape: {actions.shape if actions is not None else 'None'}")

    if actions is not None:
        print(f"\nAction statistics:")
        print(f"  Min: {actions.min().item():.4f}")
        print(f"  Max: {actions.max().item():.4f}")
        print(f"  Mean: {actions.mean().item():.4f}")
        print(f"  Std: {actions.std().item():.4f}")

        # Check if bounded
        in_bounds = (actions >= -1.0).all() and (actions <= 1.0).all()
        print(f"\n  All actions in [-1, 1]: {in_bounds}")

        if not in_bounds:
            out_of_bounds = ((actions < -1.0) | (actions > 1.0)).sum().item()
            print(f"  ⚠️  {out_of_bounds} out of {actions.numel()} actions are out of bounds!")
            print(f"  This could explain why training doesn't work - actions are not bounded!")

    # Also check action_dist_inputs if different from actions
    if "action_dist_inputs" in output and output["action_dist_inputs"] is not actions:
        dist_inputs = output["action_dist_inputs"]
        print(f"\nAction dist inputs shape: {dist_inputs.shape}")
        print(f"Action dist inputs stats:")
        print(f"  Min: {dist_inputs.min().item():.4f}")
        print(f"  Max: {dist_inputs.max().item():.4f}")

        # For Gaussian, first half is mean, second half is log_std
        if dist_inputs.shape[-1] == action_dim * 2:
            means = dist_inputs[..., :action_dim]
            log_stds = dist_inputs[..., action_dim:]
            print(f"\n  Means - min: {means.min().item():.4f}, max: {means.max().item():.4f}")
            print(f"  Log_stds - min: {log_stds.min().item():.4f}, max: {log_stds.max().item():.4f}")
            print(f"  Stds - min: {log_stds.exp().min().item():.4f}, max: {log_stds.exp().max().item():.4f}")

            means_bounded = (means >= -1.0).all() and (means <= 1.0).all()
            print(f"\n  Means in [-1, 1]: {means_bounded}")
            if not means_bounded:
                print("  ⚠️  Means are not bounded! The policy network outputs unbounded means.")

    # Test exploration forward as well
    print("\n" + "-" * 40)
    print("Testing forward_exploration:")
    with torch.no_grad():
        explore_output = module.forward_exploration({"obs": dummy_obs})

    explore_actions = explore_output.get("actions")
    if explore_actions is not None:
        print(f"  Exploration actions shape: {explore_actions.shape}")
        print(f"  Min: {explore_actions.min().item():.4f}")
        print(f"  Max: {explore_actions.max().item():.4f}")

        in_bounds = (explore_actions >= -1.0).all() and (explore_actions <= 1.0).all()
        print(f"  All in [-1, 1]: {in_bounds}")

        if not in_bounds:
            out_low = (explore_actions < -1.0).sum().item()
            out_high = (explore_actions > 1.0).sum().item()
            print(f"  ⚠️  {out_low} below -1, {out_high} above 1")


def test_with_real_env():
    """Test actions in context of real environment."""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL ENVIRONMENT")
    print("=" * 60)

    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = src_dir / "swarm" / "single_agent_ablations" / "single_agent_env_config.yaml"
    env = SingleAgentEnvWrapper(training=True, config_path=str(config_path))

    print(f"\nAction space: {env.action_space}")
    print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")

    # Sample random actions and see if env clips them
    obs, _ = env.reset()

    # Test with out-of-bounds action
    print("\n--- Testing out-of-bounds action ---")
    oob_action = np.array([5.0])  # Way outside [-1, 1]
    print(f"Submitting action: {oob_action}")

    try:
        obs, reward, term, trunc, info = env.step(oob_action)
        print(f"Env accepted action (may have clipped internally)")
        print(f"Reward: {reward}")

        # Check what voltage was actually applied
        device_state = info.get("current_device_state", {})
        if device_state:
            print(f"Applied gate voltage: {device_state['current_gate_voltages']}")
            print(f"Gate ground truth: {device_state['gate_ground_truth']}")
    except Exception as e:
        print(f"Error: {e}")

    env.close()


if __name__ == "__main__":
    test_action_bounds()
    test_with_real_env()
