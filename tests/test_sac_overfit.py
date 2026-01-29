"""
Unit test: Can the SAC policy network overfit on a single episode?

This tests whether the SAC network architecture is capable of learning the task,
independent of training hyperparameters and exploration.

If this test fails: The network architecture can't learn the mapping
If this test passes but training doesn't converge: Issue is with training setup
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def collect_expert_episode_multiagent(env, num_steps=50):
    """Collect an episode using optimal actions for multi-agent env."""
    observations = []
    actions = []
    rewards = []

    obs_dict, info = env.reset()

    for step in range(num_steps):
        # Get ground truth from base env
        device_state = env.base_env.device_state
        gate_gt = device_state["gate_ground_truth"]
        barrier_gt = device_state["barrier_ground_truth"]
        plunger_min = env.base_env.plunger_min
        plunger_max = env.base_env.plunger_max
        barrier_min = env.base_env.barrier_min
        barrier_max = env.base_env.barrier_max

        # Compute optimal actions for each agent
        optimal_gate_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0
        optimal_barrier_action = 2.0 * (barrier_gt - barrier_min) / (barrier_max - barrier_min) - 1.0

        # Build action dict
        action_dict = {}
        for i in range(len(gate_gt)):
            action_dict[f"plunger_{i}"] = np.array([optimal_gate_action[i]], dtype=np.float32)
        for i in range(len(barrier_gt)):
            action_dict[f"barrier_{i}"] = np.array([optimal_barrier_action[i]], dtype=np.float32)

        # Store observations and actions for plunger agents
        step_obs = {}
        step_actions = {}
        for agent_id in obs_dict:
            if agent_id.startswith("plunger_"):
                step_obs[agent_id] = {
                    'image': obs_dict[agent_id]['image'].copy(),
                    'voltage': obs_dict[agent_id]['voltage'].copy(),
                }
                step_actions[agent_id] = action_dict[agent_id].copy()

        observations.append(step_obs)
        actions.append(step_actions)

        obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)
        rewards.append(sum(reward_dict.values()))

        if terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False):
            break

    return observations, actions, rewards


def test_sac_policy_can_overfit():
    """Test that the SAC policy network can overfit on a single expert episode."""
    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
    from swarm.voltage_model import create_rl_module_spec
    import yaml

    print("=" * 70)
    print("SAC POLICY OVERFIT TEST")
    print("=" * 70)

    # Load configs
    config_path = Path(__file__).parent.parent / "src/swarm/algo_ablations/configs/sac_env_config.yaml"
    training_config_path = Path(__file__).parent.parent / "src/swarm/algo_ablations/configs/sac_training_config.yaml"

    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    # Create environment
    gif_config = {"enabled": False, "save_dir": "/tmp", "target_agent_type": "plunger", "target_agent_indices": [0]}
    env = MultiAgentEnvWrapper(return_voltage=True, gif_config=gif_config)

    print(f"\nEnvironment setup:")
    print(f"  Num dots: {env.base_env.num_dots}")

    # Collect expert episode
    print("\nCollecting expert episode...")
    observations, actions, rewards = collect_expert_episode_multiagent(env, num_steps=20)

    print(f"  Collected {len(observations)} steps")
    print(f"  Total reward: {sum(rewards):.2f}")
    print(f"  Mean reward per step: {np.mean(rewards):.4f}")

    # Create RLModule for SAC
    print("\nCreating SAC policy network...")
    # SAC requires additional config params
    sac_training = training_config['rl_config']['training']
    rl_module_config = {
        "plunger_policy": {
            **training_config['neural_networks']['plunger_policy'],
            "free_log_std": training_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": training_config['rl_config']['multi_agent']['log_std_bounds'],
            "twin_q": sac_training.get("twin_q", True),
            "initial_alpha": sac_training.get("initial_alpha", 1.0),
        },
        "barrier_policy": {
            **training_config['neural_networks']['barrier_policy'],
            "free_log_std": training_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": training_config['rl_config']['multi_agent']['log_std_bounds'],
            "twin_q": sac_training.get("twin_q", True),
            "initial_alpha": sac_training.get("initial_alpha", 1.0),
        }
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="sac", config=rl_module_config)

    # Build the multi-agent module and extract plunger policy
    multi_module = rl_module_spec.build()
    plunger_module = multi_module["plunger_policy"]
    print(f"  Module type: {type(plunger_module)}")

    # Get encoder and policy head from SAC module
    # SAC module has: pi (policy), qf (Q-function), pi_encoder, qf_encoder
    encoder = plunger_module.pi_encoder
    pi_head = plunger_module.pi

    # Prepare training data - use first plunger agent's observations
    agent_id = "plunger_0"
    images = torch.tensor(np.array([o[agent_id]['image'] for o in observations]), dtype=torch.float32)
    voltages = torch.tensor(np.array([o[agent_id]['voltage'] for o in observations]), dtype=torch.float32)
    target_actions = torch.tensor(np.array([a[agent_id] for a in actions]), dtype=torch.float32)

    print(f"\nTraining data shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Voltages: {voltages.shape}")
    print(f"  Target actions: {target_actions.shape}")

    # Create a simple supervised learning setup
    print("\n" + "=" * 70)
    print("SUPERVISED OVERFIT TEST (using SAC RLModule)")
    print("=" * 70)

    class SupervisedSACPolicy(nn.Module):
        def __init__(self, encoder, pi_head, action_dim):
            super().__init__()
            self.encoder = encoder
            self.pi_head = pi_head
            self.action_dim = action_dim

        def forward(self, images, voltages):
            batch_size = images.shape[0]

            # Permute images from (B, H, W, C) to (B, C, H, W) for CNN
            images_permuted = images.permute(0, 3, 1, 2)

            # Forward through encoder
            encoder_out = self.encoder({"image": images_permuted, "voltage": voltages})

            # Handle nested dict output from encoder
            if isinstance(encoder_out, dict):
                enc_inner = encoder_out.get("encoder_out", encoder_out)
                if isinstance(enc_inner, dict) and "image_features" in enc_inner:
                    image_features = enc_inner["image_features"]
                elif isinstance(enc_inner, dict):
                    image_features = list(enc_inner.values())[0]
                    if isinstance(image_features, dict):
                        image_features = image_features.get("image_features", list(image_features.values())[0])
                else:
                    image_features = enc_inner
            else:
                image_features = encoder_out

            # Forward through policy head
            pi_out = self.pi_head({"image_features": image_features, "voltage": voltages})

            # pi_out contains mean and log_std concatenated
            action_mean = pi_out[:, :self.action_dim]

            return action_mean

    action_dim = 1  # Single gate action
    model = SupervisedSACPolicy(encoder, pi_head, action_dim)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print(f"\nTraining to overfit on {len(observations)} samples...")
    print("-" * 50)

    # Train for many epochs to see if it can overfit
    num_epochs = 500
    log_interval = 50

    initial_loss = None
    final_loss = None

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        predicted_actions = model(images, voltages)

        # Compute loss
        loss = loss_fn(predicted_actions, target_actions)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            # Compute action prediction error
            with torch.no_grad():
                pred = model(images, voltages)
                mae = torch.mean(torch.abs(pred - target_actions)).item()
                max_err = torch.max(torch.abs(pred - target_actions)).item()

            print(f"  Epoch {epoch+1:4d}: loss={loss.item():.6f}, MAE={mae:.4f}, max_err={max_err:.4f}")

    final_loss = loss.item()

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    with torch.no_grad():
        final_predictions = model(images, voltages)
        final_mae = torch.mean(torch.abs(final_predictions - target_actions)).item()
        final_max_err = torch.max(torch.abs(final_predictions - target_actions)).item()

    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    print(f"\nFinal MAE: {final_mae:.4f}")
    print(f"Final max error: {final_max_err:.4f}")

    # Show some predictions vs targets
    print("\nSample predictions vs targets:")
    print("-" * 50)
    for i in range(min(5, len(observations))):
        pred = final_predictions[i].numpy()
        target = target_actions[i].numpy()
        print(f"  Step {i}: pred={pred}, target={target}, err={np.abs(pred-target)}")

    # Determine if test passed
    print("\n" + "=" * 70)
    print("TEST RESULT")
    print("=" * 70)

    # Criteria: loss should decrease significantly and MAE should be small
    loss_decreased = final_loss < initial_loss * 0.1  # 90% reduction
    mae_small = final_mae < 0.1  # Less than 0.1 action units

    if loss_decreased and mae_small:
        print("✓ PASSED: SAC policy network CAN overfit on expert data")
        print("  → Issue is likely in training setup (hyperparameters, exploration)")
    elif loss_decreased:
        print("⚠ PARTIAL: Loss decreased but MAE still high")
        print(f"  → Network is learning but not accurately (MAE={final_mae:.4f})")
    else:
        print("✗ FAILED: SAC policy network CANNOT overfit on expert data")
        print("  → Issue is likely in network architecture or observation processing")

    env.close()
    return loss_decreased and mae_small


def test_sac_encoder_output():
    """Test that the SAC encoder produces meaningful features."""
    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
    from swarm.voltage_model import create_rl_module_spec
    import yaml

    print("\n" + "=" * 70)
    print("SAC ENCODER OUTPUT TEST")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "src/swarm/algo_ablations/configs/sac_env_config.yaml"
    training_config_path = Path(__file__).parent.parent / "src/swarm/algo_ablations/configs/sac_training_config.yaml"

    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    gif_config = {"enabled": False, "save_dir": "/tmp", "target_agent_type": "plunger", "target_agent_indices": [0]}
    env = MultiAgentEnvWrapper(return_voltage=True, gif_config=gif_config)

    sac_training = training_config['rl_config']['training']
    rl_module_config = {
        "plunger_policy": {
            **training_config['neural_networks']['plunger_policy'],
            "free_log_std": training_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": training_config['rl_config']['multi_agent']['log_std_bounds'],
            "twin_q": sac_training.get("twin_q", True),
            "initial_alpha": sac_training.get("initial_alpha", 1.0),
        },
        "barrier_policy": {
            **training_config['neural_networks']['barrier_policy'],
            "free_log_std": training_config['rl_config']['multi_agent']['free_log_std'],
            "log_std_bounds": training_config['rl_config']['multi_agent']['log_std_bounds'],
            "twin_q": sac_training.get("twin_q", True),
            "initial_alpha": sac_training.get("initial_alpha", 1.0),
        }
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="sac", config=rl_module_config)
    multi_module = rl_module_spec.build()
    encoder = multi_module["plunger_policy"].pi_encoder

    # Get observations at different positions
    positions = []
    encoder_outputs = []

    # Use plunger_0's observations
    agent_id = "plunger_0"

    # Take different actions and collect encoder outputs
    for _ in range(4):
        obs_dict, _ = env.reset()

        # Random action to move to different positions
        action_dict = {}
        for key in obs_dict:
            if key.startswith("plunger_"):
                action_dict[key] = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
            else:
                action_dict[key] = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)

        obs_dict, _, _, _, _ = env.step(action_dict)

        # Get plunger_0 observation
        obs = obs_dict[agent_id]
        img = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
        volt = torch.tensor(obs['voltage'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            enc_out = encoder({"image": img, "voltage": volt})

        # Handle nested dict output
        if isinstance(enc_out, dict):
            enc_inner = enc_out.get("encoder_out", enc_out)
            if isinstance(enc_inner, dict) and "image_features" in enc_inner:
                enc_out = enc_inner["image_features"]
            elif isinstance(enc_inner, dict):
                enc_out = list(enc_inner.values())[0]
                if isinstance(enc_out, dict):
                    enc_out = enc_out.get("image_features", list(enc_out.values())[0])

        positions.append(action_dict[agent_id])
        encoder_outputs.append(enc_out.numpy().flatten())

    # Check if encoder outputs vary with position
    encoder_outputs = np.array(encoder_outputs)

    print(f"\nEncoder output shape: {encoder_outputs.shape}")
    print(f"Encoder output stats: mean={encoder_outputs.mean():.4f}, std={encoder_outputs.std():.4f}")

    # Compute pairwise distances between encoder outputs
    print("\nPairwise distances between encoder outputs at different positions:")
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(encoder_outputs[i] - encoder_outputs[j])
            print(f"  pos {i} vs pos {j}: dist={dist:.4f}")

    # Check if outputs are diverse (not collapsed)
    output_variance = np.var(encoder_outputs, axis=0).mean()
    print(f"\nMean variance across encoder dimensions: {output_variance:.6f}")

    if output_variance < 1e-6:
        print("⚠ WARNING: Encoder outputs have very low variance - might be collapsed")
    else:
        print("✓ Encoder outputs show variation across positions")

    env.close()


if __name__ == "__main__":
    test_sac_encoder_output()
    print("\n")
    test_sac_policy_can_overfit()
