"""
Single-agent environment wrapper for quantum device tuning.

This wrapper modifies the action space to only accept gate voltages,
automatically setting barrier voltages to zero for benchmarking purposes.
"""

import sys
import numpy as np
from gymnasium import spaces
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
benchmarks_dir = current_dir.parent
src_dir = benchmarks_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


class SingleAgentEnvWrapper:
    """
    Wrapper for QuantumDeviceEnv that simplifies to single-agent control.

    - Action space: Only gate voltages (barriers are always set to zero)
    - Observation space: Unchanged from base env
    - Rewards: Combined gate rewards (sum or mean)
    """

    def __init__(self, training=True, config_path="env_config.yaml"):
        """
        Initialize single-agent wrapper.

        Args:
            training: Whether in training mode
            config_path: Path to environment config yaml file
        """
        self.base_env = QuantumDeviceEnv(training=training, config_path=config_path)

        self.num_gates = self.base_env.num_dots
        self.num_barriers = self.base_env.num_dots - 1

        # Simplified action space: only gate voltages
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_gates,),
            dtype=np.float32,
        )

        # Observation space unchanged from base env
        self.observation_space = self.base_env.observation_space

    def reset(self, *, seed=None, options=None):
        """
        Reset environment and return initial observation.

        Returns:
            Tuple of (observation, info)
        """
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action):
        """
        Step environment with gate-only actions.

        Args:
            action: Array of gate voltages (num_gates,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to expected format
        action = np.array(action).flatten().astype(np.float32)

        if len(action) != self.num_gates:
            raise ValueError(
                f"Expected {self.num_gates} gate voltages, got {len(action)}"
            )

        # Create full action dict with barriers set to zero
        full_action = {
            "action_gate_voltages": action,
            "action_barrier_voltages": np.zeros(self.num_barriers, dtype=np.float32),
        }

        # Step base environment
        obs, rewards, terminated, truncated, info = self.base_env.step(full_action)

        # Combine gate rewards (sum of all gate rewards)
        # rewards is a dict with "gates" and "barriers" keys
        gate_rewards = rewards.get("gates", np.zeros(self.num_gates))
        combined_reward = float(np.sum(gate_rewards))

        return obs, combined_reward, terminated, truncated, info

    def close(self):
        """Close the base environment."""
        return self.base_env.close()


if __name__ == "__main__":
    """Test the single-agent wrapper."""
    print("=== Testing Single-Agent Quantum Wrapper ===")

    try:
        wrapper = SingleAgentEnvWrapper(training=True)
        print(" Created single-agent wrapper")

        print(f"\nAction space: {wrapper.action_space}")
        print(f"Observation space keys: {list(wrapper.observation_space.keys())}")

        # Test reset
        obs, info = wrapper.reset()
        print(f" Reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Image shape: {obs['image'].shape}")

        # Test step with random action
        action = wrapper.action_space.sample()
        print(f"\nSampled action shape: {action.shape}")

        obs, reward, terminated, truncated, info = wrapper.step(action)
        print(f" Step successful")
        print(f"  Combined reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        wrapper.close()
        print("\n All tests passed!")

    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
