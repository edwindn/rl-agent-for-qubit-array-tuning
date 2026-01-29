"""
Quick diagnostic tests for single-agent environment wrapper.
Run with: python tests/test_single_agent_env.py
"""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import numpy as np


def test_env_shapes_and_rewards():
    """Test that observation/action shapes match and rewards are sensible."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    print("=" * 60)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 60)
    print(f"num_dots: {env.num_gates}")
    print(f"num_barriers: {env.num_barriers}")
    print(f"num_actions: {env.num_actions}")
    print(f"use_barriers: {env.use_barriers}")
    print(f"bypass_barriers: {env.bypass_barriers}")
    print()

    print("=" * 60)
    print("SPACE DEFINITIONS")
    print("=" * 60)
    print(f"Action space: {env.action_space}")
    print(f"Observation space['image']: {env.observation_space['image']}")
    print(f"Observation space['voltage']: {env.observation_space['voltage']}")
    print()

    # Reset and check initial observation
    obs, info = env.reset()
    print("=" * 60)
    print("AFTER RESET")
    print("=" * 60)
    print(f"Observation['image'] shape: {obs['image'].shape}")
    print(f"Observation['voltage'] shape: {obs['voltage'].shape}")
    print(f"Observation['voltage'] values: {obs['voltage']}")

    # Check if observation matches space
    assert obs['image'].shape == env.observation_space['image'].shape, \
        f"Image shape mismatch: {obs['image'].shape} vs {env.observation_space['image'].shape}"
    assert obs['voltage'].shape == env.observation_space['voltage'].shape, \
        f"Voltage shape mismatch: {obs['voltage'].shape} vs {env.observation_space['voltage'].shape}"
    print("✓ Observation shapes match space definitions")
    print()

    # Get ground truth info
    device_state = env.base_env.device_state
    gate_gt = device_state["gate_ground_truth"]
    barrier_gt = device_state["barrier_ground_truth"]
    print(f"Gate ground truth: {gate_gt}")
    print(f"Barrier ground truth: {barrier_gt}")
    print(f"Current gate voltages: {device_state['current_gate_voltages']}")
    print(f"Current barrier voltages: {device_state['current_barrier_voltages']}")
    print()

    # Test stepping with random action
    print("=" * 60)
    print("STEPPING WITH RANDOM ACTION")
    print("=" * 60)
    action = env.action_space.sample()
    print(f"Random action: {action}")
    print(f"Action shape: {action.shape}")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Observation['voltage'] after step: {obs['voltage']}")
    print()

    # Test stepping with "perfect" action (try to hit ground truth)
    print("=" * 60)
    print("STEPPING WITH 'OPTIMAL' ACTION")
    print("=" * 60)

    # Reset to get fresh state
    obs, info = env.reset()
    device_state = env.base_env.device_state
    gate_gt = device_state["gate_ground_truth"]

    # Compute what action would move gates to ground truth
    # Action is in [-1, 1], maps to [plunger_min, plunger_max]
    plunger_min = env.base_env.plunger_min
    plunger_max = env.base_env.plunger_max

    # Normalize ground truth to [-1, 1]
    optimal_gate_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0
    print(f"Gate ground truth: {gate_gt}")
    print(f"Plunger range: [{plunger_min}, {plunger_max}]")
    print(f"Optimal gate action (normalized): {optimal_gate_action}")

    # If bypass_barriers, action is just gates
    if env.bypass_barriers:
        optimal_action = optimal_gate_action.astype(np.float32)
    else:
        barrier_gt = device_state["barrier_ground_truth"]
        barrier_min = env.base_env.barrier_min
        barrier_max = env.base_env.barrier_max
        optimal_barrier_action = 2.0 * (barrier_gt - barrier_min) / (barrier_max - barrier_min) - 1.0
        optimal_action = np.concatenate([optimal_gate_action, optimal_barrier_action]).astype(np.float32)

    print(f"Optimal action: {optimal_action}")

    obs, reward, terminated, truncated, info = env.step(optimal_action)
    print(f"Reward after optimal action: {reward}")
    print(f"Expected max reward (approx): {env.num_gates + (env.num_barriers if env.use_barriers else 0)}")

    # Check new positions
    device_state = env.base_env.device_state
    print(f"New gate voltages: {device_state['current_gate_voltages']}")
    print(f"Gate ground truth: {device_state['gate_ground_truth']}")
    print(f"Gate distances: {np.abs(device_state['current_gate_voltages'] - device_state['gate_ground_truth'])}")

    if env.use_barriers:
        print(f"New barrier voltages: {device_state['current_barrier_voltages']}")
        print(f"Barrier ground truth: {device_state['barrier_ground_truth']}")
        print(f"Barrier distances: {np.abs(device_state['current_barrier_voltages'] - device_state['barrier_ground_truth'])}")

    print()

    # Run multiple optimal steps
    print("=" * 60)
    print("RUNNING 10 OPTIMAL STEPS")
    print("=" * 60)
    obs, info = env.reset()
    total_reward = 0
    for i in range(10):
        device_state = env.base_env.device_state
        gate_gt = device_state["gate_ground_truth"]
        plunger_min = env.base_env.plunger_min
        plunger_max = env.base_env.plunger_max
        optimal_gate_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0

        if env.bypass_barriers:
            optimal_action = optimal_gate_action.astype(np.float32)
        else:
            barrier_gt = device_state["barrier_ground_truth"]
            barrier_min = env.base_env.barrier_min
            barrier_max = env.base_env.barrier_max
            optimal_barrier_action = 2.0 * (barrier_gt - barrier_min) / (barrier_max - barrier_min) - 1.0
            optimal_action = np.concatenate([optimal_gate_action, optimal_barrier_action]).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(optimal_action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.4f}, gate_dist={np.abs(device_state['current_gate_voltages'] - gate_gt)}")

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward/10:.4f}")

    env.close()
    print("\n✓ All tests passed!")


def test_bypass_barriers_action_mapping():
    """Test that bypass_barriers correctly maps barrier actions to ground truth."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    if not env.bypass_barriers:
        print("bypass_barriers is False, skipping this test")
        return

    print("=" * 60)
    print("TESTING BYPASS_BARRIERS ACTION MAPPING")
    print("=" * 60)

    obs, info = env.reset()

    # Get ground truth
    device_state = env.base_env.device_state
    barrier_gt = device_state["barrier_ground_truth"]

    # Take a step with arbitrary gate action
    gate_action = np.array([0.5] * env.num_gates, dtype=np.float32)
    print(f"Gate action: {gate_action}")

    obs, reward, terminated, truncated, info = env.step(gate_action)

    # Check that barriers are at ground truth
    device_state = env.base_env.device_state
    current_barriers = device_state["current_barrier_voltages"]
    barrier_gt = device_state["barrier_ground_truth"]

    print(f"Barrier ground truth: {barrier_gt}")
    print(f"Current barrier voltages: {current_barriers}")
    print(f"Barrier distance from GT: {np.abs(current_barriers - barrier_gt)}")

    # They should be exactly equal (or very close due to floating point)
    if np.allclose(current_barriers, barrier_gt, atol=1e-3):
        print("✓ Barriers are at ground truth as expected!")
    else:
        print("✗ ERROR: Barriers are NOT at ground truth!")
        print(f"  Difference: {current_barriers - barrier_gt}")

    env.close()


def test_factory_space_consistency():
    """Test that factory creates spaces consistent with env wrapper."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
    from swarm.single_agent_ablations.utils.factory import create_rl_module_spec
    import yaml

    config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml"
    training_config_path = Path(__file__).parent.parent / "src/swarm/single_agent_ablations/training_config.yaml"

    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    with open(training_config_path) as f:
        training_config = yaml.safe_load(f)

    # Create env
    env = SingleAgentEnvWrapper(training=True, config_path=str(config_path))

    # Create RL module spec
    rl_module_config = {
        **training_config['neural_networks']['single_agent_policy'],
        "free_log_std": training_config['rl_config']['single_agent']['free_log_std'],
        "log_std_bounds": training_config['rl_config']['single_agent']['log_std_bounds'],
    }

    rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

    print("=" * 60)
    print("FACTORY VS ENV WRAPPER SPACE COMPARISON")
    print("=" * 60)

    # Get spaces from factory spec
    factory_obs_space = rl_module_spec.observation_space
    factory_action_space = rl_module_spec.action_space

    print(f"Factory observation space: {factory_obs_space}")
    print(f"Env wrapper observation space: {env.observation_space}")
    print()
    print(f"Factory action space: {factory_action_space}")
    print(f"Env wrapper action space: {env.action_space}")
    print()

    # Compare
    factory_voltage_shape = factory_obs_space["voltage"].shape
    env_voltage_shape = env.observation_space["voltage"].shape

    if factory_voltage_shape == env_voltage_shape:
        print(f"✓ Voltage observation shapes match: {factory_voltage_shape}")
    else:
        print(f"✗ MISMATCH! Factory: {factory_voltage_shape}, Env: {env_voltage_shape}")

    factory_action_shape = factory_action_space.shape
    env_action_shape = env.action_space.shape

    if factory_action_shape == env_action_shape:
        print(f"✓ Action shapes match: {factory_action_shape}")
    else:
        print(f"✗ MISMATCH! Factory: {factory_action_shape}, Env: {env_action_shape}")

    env.close()


def test_reward_landscape():
    """Test reward at various action values to understand the reward landscape."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    print("=" * 60)
    print("REWARD LANDSCAPE ANALYSIS")
    print("=" * 60)

    obs, info = env.reset()

    # Get ground truth
    device_state = env.base_env.device_state
    gate_gt = device_state["gate_ground_truth"]
    plunger_min = env.base_env.plunger_min
    plunger_max = env.base_env.plunger_max
    optimal_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0

    print(f"Ground truth: {gate_gt}")
    print(f"Optimal action (normalized): {optimal_action}")
    print()

    # Test reward at various distances from optimal
    print("Reward vs distance from optimal:")
    print("-" * 40)

    for offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        # Reset to get consistent ground truth
        obs, info = env.reset()
        device_state = env.base_env.device_state
        gate_gt = device_state["gate_ground_truth"]
        optimal_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0

        # Add offset to action
        test_action = np.clip(optimal_action + offset, -1.0, 1.0).astype(np.float32)
        obs, reward, _, _, info = env.step(test_action)

        # Get actual distance
        device_state = env.base_env.device_state
        gate_dist = np.abs(device_state["current_gate_voltages"] - device_state["gate_ground_truth"])

        print(f"  Offset: {offset:.2f} | Reward: {reward:.4f} | Gate dist: {gate_dist}")

    print()

    # Test random policy episode
    print("=" * 60)
    print("RANDOM POLICY EPISODE")
    print("=" * 60)

    obs, info = env.reset()
    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Total reward with random policy: {total_reward:.2f}")
    print(f"Expected for uniform random in [-1,1]: ~depends on reward shape")

    env.close()


def test_policy_gradient_signal():
    """Test that there's a clear gradient signal for learning."""
    from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper

    config_path = str(Path(__file__).parent.parent / "src/swarm/single_agent_ablations/single_agent_env_config.yaml")

    env = SingleAgentEnvWrapper(training=True, config_path=config_path)

    print("=" * 60)
    print("POLICY GRADIENT SIGNAL TEST")
    print("=" * 60)

    # Test: does reward increase as we get closer to optimal?
    obs, info = env.reset()

    device_state = env.base_env.device_state
    gate_gt = device_state["gate_ground_truth"]
    plunger_min = env.base_env.plunger_min
    plunger_max = env.base_env.plunger_max
    optimal_action = 2.0 * (gate_gt - plunger_min) / (plunger_max - plunger_min) - 1.0

    print(f"Optimal action: {optimal_action}")
    print()

    # Starting from random point, take steps towards optimal
    current_action = np.array([0.0, 0.0], dtype=np.float32)

    print("Gradient descent towards optimal:")
    print("-" * 50)

    for step in range(10):
        obs, info = env.reset()  # Reset each time to get same ground truth behavior

        # Interpolate towards optimal
        alpha = step / 9.0  # 0.0 to 1.0
        test_action = ((1 - alpha) * current_action + alpha * optimal_action).astype(np.float32)

        obs, reward, _, _, info = env.step(test_action)

        device_state = env.base_env.device_state
        gate_dist = np.mean(np.abs(device_state["current_gate_voltages"] - device_state["gate_ground_truth"]))

        print(f"  alpha={alpha:.2f} | action={test_action} | reward={reward:.4f} | dist={gate_dist:.4f}")

    print()
    print("If reward increases monotonically with alpha → good gradient signal")

    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SINGLE-AGENT ENVIRONMENT DIAGNOSTIC TESTS")
    print("=" * 60 + "\n")

    test_env_shapes_and_rewards()
    print("\n")

    test_bypass_barriers_action_mapping()
    print("\n")

    test_factory_space_consistency()
    print("\n")

    test_reward_landscape()
    print("\n")

    test_policy_gradient_signal()
