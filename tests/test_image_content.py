"""
Check what the CSD image actually shows - does it contain ground truth information?
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def visualize_image_vs_position():
    """Visualize how the image changes with position."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    print("=" * 60)
    print("IMAGE CONTENT ANALYSIS")
    print("=" * 60)

    obs, info = env.reset()
    img = obs['image']

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min: {img.min():.4f}, max: {img.max():.4f}")
    print(f"Image mean: {img.mean():.4f}, std: {img.std():.4f}")

    # Get ground truth info
    device_state = env.base_env.device_state
    gate_gt = device_state["gate_ground_truth"]
    current_gate = device_state["current_gate_voltages"]

    print(f"\nGround truth: {gate_gt}")
    print(f"Current position: {current_gate}")
    print(f"Distance from GT: {np.abs(current_gate - gate_gt)}")

    # Take optimal action
    plunger_min = env.base_env.plunger_min
    plunger_max = env.base_env.plunger_max
    optimal_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0

    obs_after, reward, _, _, _ = env.step(optimal_action.astype(np.float32))
    img_after = obs_after['image']

    print(f"\nAfter optimal action:")
    print(f"Reward: {reward:.4f}")
    print(f"Image min: {img_after.min():.4f}, max: {img_after.max():.4f}")
    print(f"Image mean: {img_after.mean():.4f}, std: {img_after.std():.4f}")

    # Compare images
    img_diff = np.abs(img - img_after)
    print(f"\nImage difference (before vs after optimal):")
    print(f"Max diff: {img_diff.max():.4f}")
    print(f"Mean diff: {img_diff.mean():.4f}")

    # Check if image is informative
    print("\n" + "=" * 60)
    print("CHECKING IF IMAGE VARIES WITH POSITION")
    print("=" * 60)

    # Take several different actions and see how image varies
    obs, info = env.reset()
    base_img = obs['image'].copy()

    actions_to_test = [
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float32),
    ]

    print("\nImage statistics at different positions:")
    for action in actions_to_test:
        obs, info = env.reset()
        obs_step, reward, _, _, _ = env.step(action)
        img = obs_step['image']
        print(f"  Action {action}: mean={img.mean():.4f}, std={img.std():.4f}, reward={reward:.4f}")

    # Save sample images
    print("\n" + "=" * 60)
    print("SAVING SAMPLE IMAGES")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Show images at different positions
    positions = [
        ("Random init", None),
        ("Optimal", "optimal"),
        ("Corner [-1,-1]", np.array([-1.0, -1.0], dtype=np.float32)),
        ("Center [0,0]", np.array([0.0, 0.0], dtype=np.float32)),
        ("Corner [1,1]", np.array([1.0, 1.0], dtype=np.float32)),
        ("Corner [-1,1]", np.array([-1.0, 1.0], dtype=np.float32)),
    ]

    for idx, (title, action) in enumerate(positions):
        ax = axes[idx // 3, idx % 3]
        obs, info = env.reset()

        if action is None:
            img = obs['image'][:, :, 0]
        elif isinstance(action, str) and action == "optimal":
            device_state = env.base_env.device_state
            gate_gt = device_state["gate_ground_truth"]
            plunger_min = env.base_env.plunger_min
            plunger_max = env.base_env.plunger_max
            optimal = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0
            obs, _, _, _, _ = env.step(optimal.astype(np.float32))
            img = obs['image'][:, :, 0]
        else:
            obs, _, _, _, _ = env.step(action)
            img = obs['image'][:, :, 0]

        ax.imshow(img, cmap='plasma')
        ax.set_title(f"{title}\nmean={img.mean():.3f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/single_agent_image_analysis.png', dpi=150)
    print(f"Saved image analysis to: /tmp/single_agent_image_analysis.png")

    env.close()


def check_image_gradient_signal():
    """Check if the image provides a gradient signal towards optimal."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    print("\n" + "=" * 60)
    print("IMAGE GRADIENT SIGNAL CHECK")
    print("=" * 60)

    # Do multiple episodes with same starting position
    # and see if image correlates with distance to optimal

    correlations = []

    for episode in range(5):
        obs, info = env.reset()

        device_state = env.base_env.device_state
        gate_gt = device_state["gate_ground_truth"]
        plunger_min = env.base_env.plunger_min
        plunger_max = env.base_env.plunger_max

        img_means = []
        distances = []

        # Take 10 random steps
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            img = obs['image']
            device_state = env.base_env.device_state
            dist = np.mean(np.abs(device_state["current_gate_voltages"] - device_state["gate_ground_truth"]))

            img_means.append(img.mean())
            distances.append(dist)

            if terminated or truncated:
                break

        # Compute correlation between image mean and distance
        if len(img_means) > 2:
            corr = np.corrcoef(img_means, distances)[0, 1]
            correlations.append(corr)
            print(f"  Episode {episode+1}: correlation(img_mean, distance) = {corr:.4f}")

    if correlations:
        print(f"\nMean correlation: {np.mean(correlations):.4f}")
        print("(Strong negative correlation = image gets brighter near optimal)")
        print("(Near-zero correlation = image doesn't encode position information)")

    env.close()


if __name__ == "__main__":
    visualize_image_vs_position()
    check_image_gradient_signal()
