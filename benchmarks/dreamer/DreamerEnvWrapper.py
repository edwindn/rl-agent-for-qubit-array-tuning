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
import json
from datetime import datetime

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

    def __init__(self, training=True, config_path="env_config.yaml", logging=False, log_file="dreamer_env_log.jsonl", **kwargs):
        """
        Initialize Dreamer wrapper around QuantumDeviceEnv.

        Args:
            training: Whether in training mode
            config_path: Path to environment config yaml
            logging: Whether to log observation and action statistics
            log_file: Path to log file for statistics
            **kwargs: Additional kwargs passed to base environment
        """
        base_env = QuantumDeviceEnv(training=training, config_path=config_path)
        super().__init__(base_env)

        # Override observation space to use uint8 for images
        original_obs_space = self.env.observation_space
        original_action_space = self.env.action_space

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=original_obs_space["image"].shape,
                dtype=np.uint8,
            ),
            "obs_voltages": original_obs_space["obs_gate_voltages"],
        })

        self.action_space = original_action_space["action_gate_voltages"]

        # Logging setup
        self.logging = logging
        self.log_file = log_file
        self.step_count = 0
        if self.logging:
            # Append to log file (main.py clears it at start of run)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "env_init",
                    "timestamp": datetime.now().isoformat(),
                    "observation_space": str(self.observation_space),
                    "action_space": str(self.action_space)
                }) + '\n')

    def _log_statistics(self, data, data_type, space):
        """
        Log statistics about observation or action data.

        Args:
            data: Dictionary or array of data to log
            data_type: Type of data ("observation" or "action")
            space: The corresponding space (for bounds checking)
        """
        if not self.logging:
            return

        log_entry = {
            "event": data_type,
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count
        }

        if isinstance(data, dict):
            # Handle dictionary observations
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    stats = {
                        "mean": float(np.mean(value)),
                        "std": float(np.std(value)),
                        "min": float(np.min(value)),
                        "max": float(np.max(value)),
                        "has_nan": bool(np.isnan(value).any()),
                        "has_inf": bool(np.isinf(value).any()),
                        "shape": value.shape
                    }

                    # Check if values are within space bounds
                    if isinstance(space, spaces.Dict) and key in space.spaces:
                        key_space = space.spaces[key]
                        if isinstance(key_space, spaces.Box):
                            out_of_bounds_low = np.sum(value < key_space.low)
                            out_of_bounds_high = np.sum(value > key_space.high)
                            stats["out_of_bounds_low"] = int(out_of_bounds_low)
                            stats["out_of_bounds_high"] = int(out_of_bounds_high)

                    log_entry[key] = stats
        elif isinstance(data, np.ndarray):
            # Handle array actions
            stats = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "has_nan": bool(np.isnan(data).any()),
                "has_inf": bool(np.isinf(data).any()),
                "shape": data.shape
            }

            # Check if values are within space bounds
            if isinstance(space, spaces.Box):
                out_of_bounds_low = np.sum(data < space.low)
                out_of_bounds_high = np.sum(data > space.high)
                stats["out_of_bounds_low"] = int(out_of_bounds_low)
                stats["out_of_bounds_high"] = int(out_of_bounds_high)

            log_entry["data"] = stats

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

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
        obs = {
            "image": obs["image"],
            "obs_voltages": obs["obs_gate_voltages"]
        }
        converted_obs = self._convert_observation(obs)

        # Log observation statistics
        self._log_statistics(converted_obs, "observation", self.observation_space)

        return converted_obs, info

    def step(self, action):
        """Step environment and convert observation."""
        # Log action statistics
        self._log_statistics(action, "action", self.action_space)

        action = {
            "action_gate_voltages": action,
            "action_barrier_voltages": [0.0] * (len(action) - 1)
        }
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = {
            "image": obs["image"],
            "obs_voltages": obs["obs_gate_voltages"]
        }
        converted_obs = self._convert_observation(obs)

        # Log observation statistics
        self._log_statistics(converted_obs, "observation", self.observation_space)

        self.step_count += 1

        return converted_obs, reward, terminated, truncated, info


if __name__ == "__main__":
    """Test the Dreamer wrapper."""
    print("=== Testing Dreamer Env Wrapper ===")

    try:
        # Create wrapped environment
        wrapper = DreamerEnvWrapper(training=True)
        print("✓ Created Dreamer wrapper")

        # Check observation space
        print(f"\nObservation space:")
        print(f"  Image space: {wrapper.observation_space['image']}")
        print(f"  Image dtype: {wrapper.observation_space['image'].dtype}")
        print(f"  Voltage space: {wrapper.observation_space['obs_voltages']}")

        # Test reset
        obs, info = wrapper.reset()
        print(f"\n✓ Reset successful")
        print(f"  Image shape: {obs['image'].shape}")
        print(f"  Image dtype: {obs['image'].dtype}")
        print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
        print(f"  Gate voltages shape: {obs['obs_voltages'].shape}")

        # Test step
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        print(f"\n✓ Step successful")
        print(f"  Image dtype: {obs['image'].dtype}")
        print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
        print(f"  Reward type: {type(reward)}")

        wrapper.close()
        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
