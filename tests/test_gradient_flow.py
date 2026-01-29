"""
Test gradient flow through single-agent PPO module.

This verifies that:
1. Forward pass works with real observations
2. Loss can be computed
3. Gradients flow back through all parameters
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def test_gradient_flow():
    """Test that gradients flow correctly through the module."""
    print("=" * 60)
    print("GRADIENT FLOW TEST")
    print("=" * 60)

    # Load configs
    with open(src_dir / "swarm/single_agent_ablations/training_config.yaml") as f:
        config = yaml.safe_load(f)
    with open(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml") as f:
        env_config = yaml.safe_load(f)

    # Create environment and get real observation
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
    env = SingleAgentEnvWrapper(
        training=True,
        config_path=str(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml")
    )

    # Collect a few transitions
    obs_list = []
    action_list = []
    reward_list = []

    obs, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    env.close()

    # Stack into batch
    batch_obs = {
        "image": torch.tensor(np.stack([o["image"] for o in obs_list]), dtype=torch.float32),
        "voltage": torch.tensor(np.stack([o["voltage"] for o in obs_list]), dtype=torch.float32),
    }
    batch_actions = torch.tensor(np.stack(action_list), dtype=torch.float32)
    batch_rewards = torch.tensor(reward_list, dtype=torch.float32)

    print(f"\nBatch shapes:")
    print(f"  image: {batch_obs['image'].shape}")
    print(f"  voltage: {batch_obs['voltage'].shape}")
    print(f"  actions: {batch_actions.shape}")
    print(f"  rewards: {batch_rewards.shape}")

    # Create module
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec

    rl_module_config = {
        **config['neural_networks']['single_agent_policy'],
        "free_log_std": config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
    }

    spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)
    module = spec.build()
    module.train()

    print(f"\nModule parameters: {sum(p.numel() for p in module.parameters())}")

    # Forward pass
    print("\n--- Forward Pass ---")
    output = module.forward_train({"obs": batch_obs})

    print(f"Output keys: {list(output.keys())}")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, requires_grad={v.requires_grad}")

    # Get action distribution parameters
    action_dist_inputs = output["action_dist_inputs"]
    action_dim = batch_actions.shape[-1]
    means = action_dist_inputs[:, :action_dim]
    log_stds = action_dist_inputs[:, action_dim:]

    print(f"\nAction distribution:")
    print(f"  means: {means.mean().item():.4f} ± {means.std().item():.4f}")
    print(f"  log_stds: {log_stds.mean().item():.4f} ± {log_stds.std().item():.4f}")
    print(f"  stds: {log_stds.exp().mean().item():.4f} ± {log_stds.exp().std().item():.4f}")

    # Value estimates
    vf_preds = output.get("vf_preds")
    if vf_preds is not None:
        print(f"  value_preds: {vf_preds.mean().item():.4f} ± {vf_preds.std().item():.4f}")

    # Compute simple loss (negative log likelihood)
    print("\n--- Loss Computation ---")
    stds = log_stds.exp()
    dist = torch.distributions.Normal(means, stds)
    log_prob = dist.log_prob(batch_actions).sum(dim=-1)

    # Simple policy gradient loss
    advantages = batch_rewards - batch_rewards.mean()  # Simple baseline
    policy_loss = -(log_prob * advantages).mean()

    # Value loss (if we have value predictions)
    value_loss = torch.tensor(0.0)
    if vf_preds is not None:
        value_loss = ((vf_preds.squeeze() - batch_rewards) ** 2).mean()

    total_loss = policy_loss + 0.5 * value_loss

    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")

    # Backward pass
    print("\n--- Backward Pass ---")
    total_loss.backward()

    # Check gradients
    grad_stats = {"has_grad": 0, "no_grad": 0, "zero_grad": 0}
    grad_norms = {}

    for name, param in module.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                grad_stats["has_grad"] += 1
                grad_norms[name] = grad_norm
            else:
                grad_stats["zero_grad"] += 1
        else:
            grad_stats["no_grad"] += 1

    print(f"Gradient stats:")
    print(f"  Parameters with gradients: {grad_stats['has_grad']}")
    print(f"  Parameters with zero gradients: {grad_stats['zero_grad']}")
    print(f"  Parameters without gradients: {grad_stats['no_grad']}")

    if grad_norms:
        print(f"\nTop 5 gradient norms:")
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, norm in sorted_grads:
            print(f"  {name}: {norm:.6f}")

        print(f"\nBottom 5 gradient norms (non-zero):")
        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1])[:5]
        for name, norm in sorted_grads:
            print(f"  {name}: {norm:.6f}")

    # Check for any issues
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []
    if grad_stats["no_grad"] > 0:
        issues.append(f"⚠️  {grad_stats['no_grad']} parameters have no gradients!")
    if grad_stats["zero_grad"] > 0:
        issues.append(f"⚠️  {grad_stats['zero_grad']} parameters have zero gradients!")
    if not any(v.requires_grad for v in output.values() if isinstance(v, torch.Tensor)):
        issues.append("⚠️  No output tensors require gradients!")

    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("✓ All parameters have non-zero gradients")
        print("✓ Gradient flow looks correct")
        return True


def test_optimization_step():
    """Test that an optimization step actually changes parameters."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION STEP TEST")
    print("=" * 60)

    # Quick setup
    with open(src_dir / "swarm/single_agent_ablations/training_config.yaml") as f:
        config = yaml.safe_load(f)
    with open(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml") as f:
        env_config = yaml.safe_load(f)

    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    # Create module
    rl_module_config = {
        **config['neural_networks']['single_agent_policy'],
        "free_log_std": config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
    }
    spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)
    module = spec.build()
    module.train()

    # Create optimizer
    optimizer = torch.optim.Adam(module.parameters(), lr=0.001)

    # Get dummy data
    batch_obs = {
        "image": torch.rand(8, 100, 100, 1),
        "voltage": torch.rand(8, 1) * 2 - 1,
    }
    batch_actions = torch.rand(8, 1) * 2 - 1

    # Store initial parameter values
    initial_params = {name: param.clone() for name, param in module.named_parameters()}

    # Forward pass
    output = module.forward_train({"obs": batch_obs})
    action_dist_inputs = output["action_dist_inputs"]
    means = action_dist_inputs[:, :1]
    log_stds = action_dist_inputs[:, 1:]

    # Compute loss
    stds = log_stds.exp()
    dist = torch.distributions.Normal(means, stds)
    log_prob = dist.log_prob(batch_actions).sum(dim=-1)
    loss = -log_prob.mean()

    # Backward and step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check parameters changed
    changed = 0
    unchanged = 0
    for name, param in module.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            changed += 1
        else:
            unchanged += 1

    print(f"Parameters changed: {changed}")
    print(f"Parameters unchanged: {unchanged}")

    if unchanged > 0:
        print(f"⚠️  {unchanged} parameters didn't change after optimization!")
        return False
    else:
        print("✓ All parameters updated correctly")
        return True


if __name__ == "__main__":
    success1 = test_gradient_flow()
    success2 = test_optimization_step()

    if success1 and success2:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
