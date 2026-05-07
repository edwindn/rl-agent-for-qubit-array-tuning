"""
Single-agent environment wrapper for quantum device tuning.

This wrapper provides a simplified interface for RL training.
Supports optional barrier voltage control (enabled by default).

Observation format matches swarm's multi-agent wrapper for encoder compatibility:
- 'image': The full multi-channel CSD image (H, W, num_dots-1)
- 'voltage': Concatenated gate and barrier voltages (when use_barriers=True)
"""

import sys
import glob
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
benchmarks_dir = current_dir.parent
src_dir = benchmarks_dir.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from qadapt.environment.env import QuantumDeviceEnv


class SingleAgentEnvWrapper(gym.Env):
    """
    Wrapper for QuantumDeviceEnv that simplifies to single-agent control.

    - Action space: Gate voltages + barrier voltages (when use_barriers=True)
    - Observation space: Dict with 'image' and 'voltage' keys (matches swarm encoder interface)
    - Rewards: Combined gate + barrier rewards (sum)
    """

    def __init__(
        self,
        training=True,
        config_path="env_config.yaml",
        num_dots_override=None,
        use_barriers=True,
        distance_data_dir=None,
        capacitance_model_checkpoint=None,
    ):
        """
        Initialize single-agent wrapper.

        Args:
            training: Whether in training mode
            config_path: Path to environment config yaml file
            num_dots_override: If provided, overrides num_dots from config
            use_barriers: Whether to include barrier voltages in action/observation space
            distance_data_dir: Optional path to directory for saving distance data
            capacitance_model_checkpoint: Optional path to mobilenet weights, required
                when env_config.capacitance_model.update_method == 'kalman'
        """
        super().__init__()
        self.base_env = QuantumDeviceEnv(
            training=training,
            config_path=config_path,
            num_dots=num_dots_override,
            capacitance_model_checkpoint=capacitance_model_checkpoint,
        )

        self.use_barriers = use_barriers
        self.distance_data_dir = distance_data_dir
        self.distance_history = None
        self.num_gates = self.base_env.num_dots
        self.num_barriers = self.base_env.num_dots - 1
        self.num_channels = self.num_gates - 1  # N-1 CSD scans
        self.num_actions = self.num_gates + (self.num_barriers if use_barriers else 0)

        self.plunger_ids = [f"plunger_{i}" for i in range(self.num_gates)]
        self.barrier_ids = [f"barrier_{i}" for i in range(self.num_barriers)]

        if self.distance_data_dir is not None:
            distance_data_path = Path(self.distance_data_dir)
            for agent_id in self.plunger_ids + self.barrier_ids:
                agent_folder = distance_data_path / agent_id
                agent_folder.mkdir(parents=True, exist_ok=True, mode=0o777)
                try:
                    import os
                    os.chmod(agent_folder, 0o777)
                except Exception:
                    pass

        # Action space: gate voltages + barrier voltages (when enabled)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Observation space matching swarm encoder interface: {image, voltage}
        base_image_space = self.base_env.observation_space["image"]
        self.observation_space = spaces.Dict({
            'image': base_image_space,
            'voltage': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_actions,),  # Gates + barriers when use_barriers=True
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
        voltages = [base_obs['obs_gate_voltages']]
        if self.use_barriers:
            voltages.append(base_obs['obs_barrier_voltages'])
        return {
            'image': base_obs['image'].astype(np.float32),
            'voltage': np.concatenate(voltages).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        """
        Reset environment and return initial observation.

        Returns:
            Tuple of (observation, info)
        """
        base_obs, info = self.base_env.reset(seed=seed, options=options)

        if self.distance_history is not None:
            self._save_agent_histories(self.distance_history)

        if self.distance_data_dir is not None:
            self.distance_history = {agent_id: [] for agent_id in self.plunger_ids + self.barrier_ids}
        return self._convert_observation(base_obs), info

    def step(self, action):
        """
        Step environment with gate (and optionally barrier) actions.

        Args:
            action: Array of voltages (num_gates,) or (num_gates + num_barriers,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to expected format
        action = np.array(action).flatten().astype(np.float32)

        if len(action) != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} actions, got {len(action)}"
            )

        # Split action into gate and barrier components
        if self.use_barriers:
            gate_action = action[:self.num_gates]
            barrier_action = action[self.num_gates:]
        else:
            gate_action = action
            barrier_action = np.zeros(self.num_barriers, dtype=np.float32)

        full_action = {
            "action_gate_voltages": gate_action,
            "action_barrier_voltages": barrier_action,
        }

        # Step base environment
        base_obs, rewards, terminated, truncated, info = self.base_env.step(full_action)

        # Convert observation to swarm-compatible format
        obs = self._convert_observation(base_obs)

        # Combine gate and barrier rewards
        gate_rewards = rewards.get("gates", np.zeros(self.num_gates))
        barrier_rewards = rewards.get("barriers", np.zeros(self.num_barriers)) if self.use_barriers else 0
        combined_reward = float(np.sum(gate_rewards) + np.sum(barrier_rewards))

        if self.distance_history is not None:
            device_state_info = info.get("current_device_state", None)
            if device_state_info:
                for idx, agent_id in enumerate(self.plunger_ids):
                    ground_truth = device_state_info["gate_ground_truth"][idx]
                    current_voltage = device_state_info["current_gate_voltages"][idx]
                    self.distance_history[agent_id].append(current_voltage - ground_truth)

                if self.use_barriers:
                    for idx, agent_id in enumerate(self.barrier_ids):
                        ground_truth = device_state_info["barrier_ground_truth"][idx]
                        current_voltage = device_state_info["current_barrier_voltages"][idx]
                        self.distance_history[agent_id].append(current_voltage - ground_truth)

            if (terminated or truncated) and self.distance_history is not None:
                self._save_agent_histories(self.distance_history)
                self.distance_history = {agent_id: [] for agent_id in self.plunger_ids + self.barrier_ids}

        return obs, combined_reward, terminated, truncated, info

    def close(self):
        """Close the base environment."""
        return self.base_env.close()

    def _save_agent_histories(self, history: dict) -> None:
        distance_data_path = Path(self.distance_data_dir)
        for agent_id, dists in history.items():
            dists = np.array(dists)
            agent_folder = distance_data_path / agent_id
            existing_files = glob.glob(str(agent_folder / "*.npy"))

            if len(existing_files) == 0:
                next_count = 1
            else:
                counts = []
                for filepath in existing_files:
                    filename = Path(filepath).stem
                    count_str = filename.split('_')[0]
                    counts.append(int(count_str))
                next_count = max(counts) + 1

            random_suffix = random.randint(0, 999999)
            filename = f"{next_count:04d}_{random_suffix:06d}.npy"
            filepath = agent_folder / filename
            np.save(filepath, dists)


if __name__ == "__main__":
    """Test the single-agent wrapper."""
    print("=== Testing Single-Agent Quantum Wrapper ===")

    for use_barriers in [True, False]:
        print(f"\n--- Testing with use_barriers={use_barriers} ---")
        try:
            wrapper = SingleAgentEnvWrapper(training=True, use_barriers=use_barriers)
            print(f"Created single-agent wrapper (use_barriers={use_barriers})")

            print(f"\nAction space: {wrapper.action_space}")
            print(f"Observation space: {wrapper.observation_space}")

            # Test reset
            obs, info = wrapper.reset()
            print("Reset successful")
            print(f"  Observation keys: {list(obs.keys())}")
            print(f"  Image shape: {obs['image'].shape}")
            print(f"  Voltage shape: {obs['voltage'].shape}")

            # Test step with random action
            action = wrapper.action_space.sample()
            print(f"\nSampled action shape: {action.shape}")

            obs, reward, terminated, truncated, info = wrapper.step(action)
            print("Step successful")
            print(f"  Combined reward: {reward}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

            wrapper.close()
            print(f"\nAll tests passed for use_barriers={use_barriers}!")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
