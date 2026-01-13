"""
Dreamer wrapper for QuantumDeviceEnv.

This wrapper converts float32 image observations (0.0-1.0) to uint8 (0-255)
to match Dreamer's encoder expectations.
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


class DreamerEnvWrapper(gym.Wrapper):
    """
    Wrapper that converts QuantumDeviceEnv observations to Dreamer-compatible format.

    Key conversions:
    - Converts float32 images (0.0-1.0) to uint8 (0-255)
    - Updates observation space to reflect uint8 dtype
    """

    def __init__(self, training=True, config_path="env_config.yaml", **kwargs):
        """
        Initialize Dreamer wrapper around QuantumDeviceEnv.

        Args:
            training: Whether in training mode
            config_path: Path to environment config yaml
            **kwargs: Additional kwargs passed to base environment
        """
        base_env = QuantumDeviceEnv(training=training, config_path=config_path)
        super().__init__(base_env)

        # Override observation space to use uint8 for images
        original_obs_space = self.env.observation_space

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=original_obs_space["image"].shape,
                dtype=np.uint8,
            ),
            "obs_gate_voltages": original_obs_space["obs_gate_voltages"],
            "obs_barrier_voltages": original_obs_space["obs_barrier_voltages"],
        })

    def _convert_observation(self, obs):
        """
        Convert float32 image observation to uint8.

        Args:
            obs: Dictionary observation from base environment

        Returns:
            Converted observation with uint8 images
        """
        converted_obs = obs.copy()

        # Convert image from float32 [0.0, 1.0] to uint8 [0, 255]
        float_image = obs["image"]
        # Clip to ensure values are in [0.0, 1.0] range
        float_image = np.clip(float_image, 0.0, 1.0)
        # Scale to [0, 255] and convert to uint8
        uint8_image = (float_image * 255).astype(np.uint8)

        converted_obs["image"] = uint8_image

        return converted_obs

    def reset(self, **kwargs):
        """Reset environment and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_observation(obs), info

    def step(self, action):
        """Step environment and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated, info


if __name__ == "__main__":
    """Test the Dreamer wrapper."""
    print("=== Testing Dreamer Env Wrapper ===")

    try:
        # Create wrapped environment
        env = DreamerEnvWrapper(training=True)
        print("✓ Created Dreamer wrapper")

        # Check observation space
        print(f"\nObservation space:")
        print(f"  Image space: {env.observation_space['image']}")
        print(f"  Image dtype: {env.observation_space['image'].dtype}")
        print(f"  Gate voltages space: {env.observation_space['obs_gate_voltages']}")
        print(f"  Barrier voltages space: {env.observation_space['obs_barrier_voltages']}")

        # Test reset
        obs, info = env.reset()
        print(f"\n✓ Reset successful")
        print(f"  Image shape: {obs['image'].shape}")
        print(f"  Image dtype: {obs['image'].dtype}")
        print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
        print(f"  Gate voltages shape: {obs['obs_gate_voltages'].shape}")
        print(f"  Barrier voltages shape: {obs['obs_barrier_voltages'].shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n✓ Step successful")
        print(f"  Image dtype: {obs['image'].dtype}")
        print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
        print(f"  Reward type: {type(reward)}")

        env.close()
        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
