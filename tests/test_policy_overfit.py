"""
Unit test: Can the policy network overfit on a single episode?

This tests whether the network architecture is capable of learning the task,
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


def collect_expert_episode(env, num_steps=50):
    """Collect an episode using optimal actions."""
    observations = []
    actions = []
    rewards = []

    obs, info = env.reset()

    for step in range(num_steps):
        # Compute optimal action (move to ground truth)
        device_state = env.base_env.device_state
        gate_gt = device_state["gate_ground_truth"]
        plunger_min = env.base_env.plunger_min
        plunger_max = env.base_env.plunger_max

        optimal_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0
        optimal_action = optimal_action.astype(np.float32)

        observations.append({
            'image': obs['image'].copy(),
            'voltage': obs['voltage'].copy(),
        })
        actions.append(optimal_action.copy())

        obs, reward, terminated, truncated, info = env.step(optimal_action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return observations, actions, rewards


def test_policy_can_overfit():
    """Test that the policy network can overfit on a single expert episode."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec
    import yaml

    print("=" * 70)
    print("POLICY OVERFIT TEST")
    print("=" * 70)

    # Load configs
    config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml"
    training_config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/training_config.yaml"

    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    # Create environment
    env = SingleAgentEnvWrapper(training=True, config_path=str(config_path))

    print(f"\nEnvironment setup:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Collect expert episode
    print("\nCollecting expert episode...")
    observations, actions, rewards = collect_expert_episode(env, num_steps=20)

    print(f"  Collected {len(observations)} steps")
    print(f"  Total reward: {sum(rewards):.2f}")
    print(f"  Mean reward per step: {np.mean(rewards):.4f}")

    # Create RLModule
    print("\nCreating policy network...")
    rl_module_config = {
        **training_config['neural_networks']['single_agent_policy'],
        "free_log_std": training_config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": training_config['rl_config']['single_agent']['log_std_bounds'],
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

    # Build the module
    rl_module = rl_module_spec.build()
    print(f"  Module type: {type(rl_module)}")

    # Extract the encoder and policy head for direct training
    # We'll do supervised learning: given observation, predict optimal action

    # Convert observations to tensors
    images = torch.tensor(np.array([o['image'] for o in observations]), dtype=torch.float32)
    voltages = torch.tensor(np.array([o['voltage'] for o in observations]), dtype=torch.float32)
    target_actions = torch.tensor(np.array(actions), dtype=torch.float32)

    print(f"\nTraining data shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Voltages: {voltages.shape}")
    print(f"  Target actions: {target_actions.shape}")

    # Create a simple supervised learning setup using the RLModule's forward pass
    # We'll use MSE loss between predicted action mean and target action

    print("\n" + "=" * 70)
    print("SUPERVISED OVERFIT TEST (using RLModule)")
    print("=" * 70)

    # Get the encoder and policy head
    encoder = rl_module.encoder
    pi_head = rl_module.pi

    # Combine into a simple model for supervised learning
    class SupervisedPolicy(nn.Module):
        def __init__(self, encoder, pi_head, action_dim):
            super().__init__()
            self.encoder = encoder
            self.pi_head = pi_head
            self.action_dim = action_dim

        def forward(self, images, voltages):
            # Encoder expects dict observation
            # Need to handle the observation format correctly
            batch_size = images.shape[0]

            # Permute images from (B, H, W, C) to (B, C, H, W) for CNN
            images_permuted = images.permute(0, 3, 1, 2)

            # Forward through encoder
            encoder_out = self.encoder({"image": images_permuted, "voltage": voltages})

            # Handle nested dict output from encoder:
            # {ENCODER_OUT: {"actor": {"image_features": ..., "voltage": ...}, "critic": {...}}}
            if isinstance(encoder_out, dict):
                enc_inner = encoder_out.get("encoder_out", encoder_out)
                if isinstance(enc_inner, dict) and "actor" in enc_inner:
                    encoder_out = enc_inner["actor"]["image_features"]
                elif isinstance(enc_inner, dict) and "image_features" in enc_inner:
                    encoder_out = enc_inner["image_features"]
                else:
                    encoder_out = list(enc_inner.values())[0]
                    if isinstance(encoder_out, dict):
                        encoder_out = encoder_out.get("image_features", list(encoder_out.values())[0])

            # Forward through policy head - needs both image features and voltage
            pi_out = self.pi_head({"image_features": encoder_out, "voltage": voltages})

            # pi_out contains mean and log_std concatenated
            # For Gaussian policy: first half is mean, second half is log_std
            action_mean = pi_out[:, :self.action_dim]

            return action_mean

    action_dim = env.action_space.shape[0]
    model = SupervisedPolicy(encoder, pi_head, action_dim)

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
        print("✓ PASSED: Policy network CAN overfit on expert data")
        print("  → Issue is likely in training setup (hyperparameters, exploration)")
    elif loss_decreased:
        print("⚠ PARTIAL: Loss decreased but MAE still high")
        print(f"  → Network is learning but not accurately (MAE={final_mae:.4f})")
    else:
        print("✗ FAILED: Policy network CANNOT overfit on expert data")
        print("  → Issue is likely in network architecture or observation processing")

    env.close()
    return loss_decreased and mae_small


def test_encoder_output():
    """Test that the encoder produces meaningful features."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec
    import yaml

    print("\n" + "=" * 70)
    print("ENCODER OUTPUT TEST")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml"
    training_config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/training_config.yaml"

    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    env = SingleAgentEnvWrapper(training=True, config_path=str(config_path))

    rl_module_config = {
        **training_config['neural_networks']['single_agent_policy'],
        "free_log_std": training_config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": training_config['rl_config']['single_agent']['log_std_bounds'],
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)
    rl_module = rl_module_spec.build()
    encoder = rl_module.encoder

    # Get observations at different positions
    positions = []
    encoder_outputs = []

    test_actions = [
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([-0.5, 0.5], dtype=np.float32),
    ]

    for action in test_actions:
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(action)

        # Convert to tensor
        img = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
        volt = torch.tensor(obs['voltage'], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            enc_out = encoder({"image": img, "voltage": volt})

        # Handle nested dict output from encoder:
        # {ENCODER_OUT: {"actor": {"image_features": ..., "voltage": ...}, "critic": {...}}}
        if isinstance(enc_out, dict):
            enc_inner = enc_out.get("encoder_out", enc_out)
            if isinstance(enc_inner, dict) and "actor" in enc_inner:
                enc_out = enc_inner["actor"]["image_features"]
            elif isinstance(enc_inner, dict) and "image_features" in enc_inner:
                enc_out = enc_inner["image_features"]
            else:
                enc_out = list(enc_inner.values())[0]
                if isinstance(enc_out, dict):
                    enc_out = enc_out.get("image_features", list(enc_out.values())[0])

        positions.append(action)
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
            print(f"  {positions[i]} vs {positions[j]}: dist={dist:.4f}")

    # Check if outputs are diverse (not collapsed)
    output_variance = np.var(encoder_outputs, axis=0).mean()
    print(f"\nMean variance across encoder dimensions: {output_variance:.6f}")

    if output_variance < 1e-6:
        print("⚠ WARNING: Encoder outputs have very low variance - might be collapsed")
    else:
        print("✓ Encoder outputs show variation across positions")

    env.close()


if __name__ == "__main__":
    test_encoder_output()
    print("\n")
    test_policy_can_overfit()
