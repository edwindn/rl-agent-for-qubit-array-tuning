"""
DreamerV3-compatible wrapper for QuantumDeviceEnv.

Wraps the new src/swarm/environment/env.py environment for use with DreamerV3.
Supports N plungers and N-1 barriers.

Key conversions:
- Image: float32 [0,1] -> uint8 [0,255]
- Observations: Dict -> flattened (image + voltages)
- Actions: single Box -> split to gates/barriers
- Reward: dict -> scalar sum
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


class DreamerEnvWrapper(gym.Env):
    """
    Gymnasium wrapper that adapts QuantumDeviceEnv for DreamerV3.

    Supports N plungers and N-1 barriers for arbitrary array sizes.

    Conversions:
    - Image: float32 [0,1] -> uint8 [0,255]
    - Observations: concat gate + barrier voltages into single 'voltages' array
    - Actions: single flat Box -> split to gate/barrier dicts
    - Reward: dict with gates/barriers -> scalar sum
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    _global_iteration = 0

    def __init__(self, num_dots=2, use_barriers=True, max_steps=50, seed=None, resolution=96, eval=False, **kwargs):
        """
        Initialize DreamerV3 wrapper.

        Args:
            num_dots: Number of quantum dots (N plungers, N-1 barriers)
            use_barriers: Whether to include barrier voltage control
            max_steps: Maximum steps per episode
            seed: Random seed
            resolution: Image resolution (must be divisible by 16 for encoder pooling)
            **kwargs: Additional kwargs passed to base environment
        """
        super().__init__()

        self.num_dots = num_dots
        self.use_barriers = use_barriers
        self.max_steps = max_steps
        self._seed = seed
        self.eval = eval
        self.iteration = 0
        self._step_in_iteration = 0
        self._plunger_history = []
        self._barrier_history = []
        self._dreamer_dir = Path("/home/edn/rl-agent-for-qubit-array-tuning/benchmarks/dreamer")
        self._local_plot_dir = None
        if self.eval:
            self._local_plot_dir = self._dreamer_dir / "eval_distance_plots"
            self._local_plot_dir.mkdir(parents=True, exist_ok=True)
            if not self._local_plot_dir.exists() or not self._local_plot_dir.is_dir():
                raise RuntimeError(f"Eval plot directory is invalid: {self._local_plot_dir}")

        # Create underlying environment with correct parameters
        # (must pass to constructor since reset() is called during __init__)
        self._env = QuantumDeviceEnv(
            training=True,
            num_dots=num_dots,
            use_barriers=use_barriers,
        )

        # Override config values not in constructor
        self._env.max_steps = max_steps
        self._env.resolution = resolution  # Override for DreamerV3 encoder compatibility

        # Calculate dimensions
        self.n_plungers = num_dots
        self.n_barriers = num_dots - 1 if use_barriers else 0
        self.n_actions = self.n_plungers + self.n_barriers
        self.n_channels = num_dots - 1  # CSD scans between adjacent dot pairs

        resolution = self._env.resolution

        # Define DreamerV3-compatible observation space
        # image: uint8 for CNN encoder
        # voltages: float32 for MLP encoder (concatenated plunger + barrier)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(resolution, resolution, self.n_channels),
                dtype=np.uint8
            ),
            "voltages": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_actions,),
                dtype=np.float32
            ),
        })

        # Single flat action space for DreamerV3
        # First n_plungers values = gate voltages
        # Remaining n_barriers values = barrier voltages
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset environment and convert observation."""
        obs, info = self._env.reset(seed=seed or self._seed, options=options)
        converted_obs = self._convert_observation(obs)
        if self.eval:
            DreamerEnvWrapper._global_iteration += 1
            self.iteration = DreamerEnvWrapper._global_iteration
            self._step_in_iteration = 0
            self._plunger_history = []
            self._barrier_history = []
            
        return converted_obs, info

    def step(self, action):
        """
        Step environment with DreamerV3-format action.

        Args:
            action: Flat array of shape (n_plungers + n_barriers,)
                   First n_plungers values control gates
                   Remaining values control barriers

        Returns:
            observation, reward (scalar), terminated, truncated, info
        """
        # Split flat action into gate and barrier components
        action = np.asarray(action, dtype=np.float32)
        env_action = {
            "action_gate_voltages": action[:self.n_plungers],
            "action_barrier_voltages": action[self.n_plungers:] if self.use_barriers else np.zeros(self.n_barriers, dtype=np.float32),
        }

        # Step underlying environment
        obs, reward_dict, terminated, truncated, info = self._env.step(env_action)

        # Convert observation to DreamerV3 format
        converted_obs = self._convert_observation(obs)

        # Aggregate reward: sum of all gate and barrier rewards
        reward = np.sum(reward_dict["gates"]) + np.sum(reward_dict["barriers"])

        if self.eval:
            self._save_distances(env_action)
            if terminated or truncated:
                self._log_distance_plots()

        return converted_obs, float(reward), terminated, truncated, info

    def _save_distances(self, action):
        device_state = self._env.device_state

        plunger_ground_truth = np.asarray(device_state["gate_ground_truth"])
        barrier_ground_truth = np.asarray(device_state["barrier_ground_truth"])

        # Use device_state voltages (already rescaled to physical ranges).
        if "current_gate_voltages" in device_state and "current_barrier_voltages" in device_state:
            current_gate_voltages = np.asarray(device_state["current_gate_voltages"])
            current_barrier_voltages = np.asarray(device_state["current_barrier_voltages"])
        else:
            # Fallback: rescale normalized action values to physical ranges.
            current_gate_voltages = self._env._rescale_gate_voltages(
                np.asarray(action["action_gate_voltages"], dtype=np.float32)
            )
            current_barrier_voltages = self._env._rescale_barrier_voltages(
                np.asarray(action["action_barrier_voltages"], dtype=np.float32)
            )

        if plunger_ground_truth.size and current_gate_voltages.size:
            plunger_distances = np.abs(plunger_ground_truth - current_gate_voltages)
        else:
            raise RuntimeError("plunger_ground_truth or current_gate_voltages not found")

        if barrier_ground_truth.size and current_barrier_voltages.size:
            barrier_distances = np.abs(barrier_ground_truth - current_barrier_voltages)
        else:
            raise RuntimeError("barrier_ground_truth or current_barrier_voltages not found")

        self._plunger_history.append(plunger_distances)
        self._barrier_history.append(barrier_distances)
        self._step_in_iteration += 1

    def _log_distance_plots(self):
        if not self.eval:
            return
        if self._local_plot_dir is None:
            raise RuntimeError("Eval plot directory is not configured.")
        if not self._local_plot_dir.exists() or not self._local_plot_dir.is_dir():
            raise RuntimeError(f"Eval plot directory does not exist: {self._local_plot_dir}")

        import matplotlib.pyplot as plt

        if not self._plunger_history:
            raise RuntimeError("No plunger distance history available for plotting.")
        if self.use_barriers and not self._barrier_history:
            raise RuntimeError("No barrier distance history available for plotting.")

        if self._plunger_history:
            plunger_array = np.stack(self._plunger_history, axis=0)
            steps = np.arange(1, plunger_array.shape[0] + 1)

            fig, ax = plt.subplots(figsize=(10, 6))
            for idx in range(plunger_array.shape[1]):
                ax.plot(steps, plunger_array[:, idx], label=f"plunger_{idx}", alpha=0.7)
            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title("Plunger Agent Distances")
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f"plunger_distances_iter_{self.iteration:04d}.png"
            output_path = self._local_plot_dir / filename
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            if not output_path.exists():
                raise RuntimeError(f"Failed to create plunger distance plot: {output_path}")

        if self._barrier_history:
            barrier_array = np.stack(self._barrier_history, axis=0)
            steps = np.arange(1, barrier_array.shape[0] + 1)

            fig, ax = plt.subplots(figsize=(10, 6))
            for idx in range(barrier_array.shape[1]):
                ax.plot(steps, barrier_array[:, idx], label=f"barrier_{idx}", alpha=0.7)
            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title("Barrier Agent Distances")
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = f"barrier_distances_iter_{self.iteration:04d}.png"
            output_path = self._local_plot_dir / filename
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            if not output_path.exists():
                raise RuntimeError(f"Failed to create barrier distance plot: {output_path}")

    def _convert_observation(self, obs):
        """
        Convert QuantumDeviceEnv observation to DreamerV3 format.

        Args:
            obs: Dict with 'image' (float32), 'obs_gate_voltages', 'obs_barrier_voltages'

        Returns:
            Dict with 'image' (uint8) and 'voltages' (concatenated float32)
        """
        # Convert image from float32 [0,1] to uint8 [0,255]
        float_image = np.clip(obs["image"], 0.0, 1.0)
        uint8_image = (float_image * 255).astype(np.uint8)

        # Concatenate gate and barrier voltages (clip to handle float precision)
        voltages = np.concatenate([
            obs["obs_gate_voltages"],
            obs["obs_barrier_voltages"]
        ]).astype(np.float32)
        voltages = np.clip(voltages, -1.0, 1.0)

        return {
            "image": uint8_image,
            "voltages": voltages,
        }

    def render(self):
        """Render the environment."""
        if hasattr(self._env, '_render_frame'):
            obs = self._env.array._get_obs(
                self._env.device_state["current_gate_voltages"],
                self._env.device_state["current_barrier_voltages"]
            )
            return obs["image"][:, :, 0]
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self._env, '_cleanup'):
            self._env._cleanup()

    @property
    def device_state(self):
        """Access underlying device state for evaluation."""
        return self._env.device_state


def make_dreamer_env(num_dots=2, use_barriers=True, max_steps=50, seed=None, resolution=96, eval=False):
    """
    Factory function to create DreamerV3-compatible environment.

    Args:
        num_dots: Number of quantum dots
        use_barriers: Whether to control barriers
        max_steps: Maximum steps per episode
        seed: Random seed
        resolution: Image resolution (must be divisible by 16 for encoder pooling)
        eval: Whether this environment is used for evaluation

    Returns:
        DreamerEnvWrapper instance
    """
    return DreamerEnvWrapper(
        num_dots=num_dots,
        use_barriers=use_barriers,
        max_steps=max_steps,
        seed=seed,
        resolution=resolution,
        eval=eval,
    )


if __name__ == "__main__":
    """Test the DreamerV3 wrapper."""
    print("=== Testing DreamerV3 Env Wrapper ===\n")

    try:
        # Test with 2 dots
        print("Testing with 2 dots...")
        env = make_dreamer_env(num_dots=2, use_barriers=True)

        print(f"Observation space:")
        print(f"  Image: {env.observation_space['image']}")
        print(f"  Voltages: {env.observation_space['voltages']}")
        print(f"Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"\nReset successful:")
        print(f"  Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
        print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
        print(f"  Voltages shape: {obs['voltages'].shape}")

        # Test step
        action = env.action_space.sample()
        print(f"\nAction shape: {action.shape}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful:")
        print(f"  Reward: {reward} (type: {type(reward).__name__})")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        env.close()
        print("\n✓ 2-dot test passed!")

        # Test with 4 dots
        print("\n" + "="*50)
        print("Testing with 4 dots...")
        env4 = make_dreamer_env(num_dots=4, use_barriers=True)

        print(f"Observation space:")
        print(f"  Image: {env4.observation_space['image']}")  # (res, res, 3) channels
        print(f"  Voltages: {env4.observation_space['voltages']}")  # 4 plungers + 3 barriers = 7
        print(f"Action space: {env4.action_space}")  # 7 actions

        obs, _ = env4.reset(seed=42)
        action = env4.action_space.sample()
        obs, reward, _, _, _ = env4.step(action)
        print(f"Step: reward={reward:.4f}")

        env4.close()
        print("✓ 4-dot test passed!")

        print("\n=== All tests passed! ===")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
