"""
Single-agent environment wrapper for quantum device tuning.

This wrapper modifies the action space to only accept gate voltages,
automatically setting barrier voltages to zero for benchmarking purposes.

Observation format matches swarm's multi-agent wrapper for encoder compatibility:
- 'image': The full multi-channel CSD image (H, W, num_dots-1)
- 'voltage': Concatenated gate and barrier voltages
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
benchmarks_dir = current_dir.parent
src_dir = benchmarks_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


class SingleAgentEnvWrapper(gym.Env):
    """
    Wrapper for QuantumDeviceEnv that simplifies to single-agent control.

    - Action space: Only gate voltages (barriers are always set to zero)
    - Observation space: Dict with 'image' and 'voltage' keys (matches swarm encoder interface)
    - Rewards: Combined gate rewards (sum)
    """

    def __init__(self, training=True, config_path="env_config.yaml"):
        """
        Initialize single-agent wrapper.

        Args:
            training: Whether in training mode
            config_path: Path to environment config yaml file
        """
        super().__init__()
        self.base_env = QuantumDeviceEnv(training=training, config_path=config_path)

        self.num_gates = self.base_env.num_dots
        self.num_barriers = self.base_env.num_dots - 1
        self.num_channels = self.num_gates - 1  # N-1 CSD scans

        # Simplified action space: only gate voltages
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_gates,),
            dtype=np.float32,
        )

        # Observation space matching swarm encoder interface: {image, voltage}
        base_image_space = self.base_env.observation_space["image"]
        self.observation_space = spaces.Dict({
            'image': base_image_space,
            'voltage': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_gates,),  # Only gate voltages (barriers always zero)
                dtype=np.float32,
            ),
        })

    def _convert_observation(self, base_obs):
        """
        Convert base env observation to swarm-compatible format.

        Args:
            base_obs: Dict with 'image', 'obs_gate_voltages', 'obs_barrier_voltages'

        Returns:
            Dict with 'image' and 'voltage' keys
        """
        return {
            'image': base_obs['image'].astype(np.float32),
            'voltage': base_obs['obs_gate_voltages'].astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        """
        Reset environment and return initial observation.

        Returns:
            Tuple of (observation, info)
        """
        base_obs, info = self.base_env.reset(seed=seed, options=options)
        return self._convert_observation(base_obs), info

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
        base_obs, rewards, terminated, truncated, info = self.base_env.step(full_action)

        # Convert observation to swarm-compatible format
        obs = self._convert_observation(base_obs)

        # Combine gate rewards (sum of all gate rewards)
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
        print(f"Observation space: {wrapper.observation_space}")

        # Test reset
        obs, info = wrapper.reset()
        print(f" Reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Image shape: {obs['image'].shape}")
        print(f"  Voltage shape: {obs['voltage'].shape}")

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
