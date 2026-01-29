"""
Single-agent wrapper that drives the multi-agent environment with ground-truth actions.

All plungers and barriers are set to their ground-truth voltages except plunger_0,
which is provided by the caller as the single-agent action.
"""

from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


class SingleAgentWrapper(gym.Env):
    """
    Single-agent view over MultiAgentEnvWrapper.

    - Action: plunger_0 voltage (normalized to [-1, 1])
    - All other plungers/barriers: driven to ground-truth voltages each step
    - Observation: plunger_0 observation from MultiAgentEnvWrapper
    """

    def __init__(
        self,
        training: bool = True,
        return_voltage: bool = False,
        gif_config: dict = None,
        distance_data_dir: str = None,
        env_config_path: str = None,
        capacitance_model_checkpoint: str = None,
        is_collecting_data: bool = False,
        deterministic: bool = False,
        base_env_class: Optional[type] = None,
    ):
        """
        Initialize the single-agent wrapper.

        Args:
            training: Whether in training mode
            return_voltage: If True, returns dict observation with image and voltage
            gif_config: Configuration for GIF capture
            distance_data_dir: Path to directory for saving distance data
            env_config_path: Optional path to custom env config file
            capacitance_model_checkpoint: Path to capacitance model weights checkpoint
            is_collecting_data: Whether to collect extra data in the environment
            deterministic: If True, use deterministic simple environment
            base_env_class: Optional base env class override (takes precedence over deterministic)
        """
        if base_env_class is None and deterministic:
            from swarm.algo_ablations.simple_env import QuantumDeviceEnv as DeterministicEnv
            base_env_class = DeterministicEnv

        if distance_data_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            distance_data_dir = (
                Path(__file__).parent / "data" / "distance_rollouts" / timestamp
            )
        else:
            distance_data_dir = Path(distance_data_dir)

        distance_data_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        try:
            distance_data_dir.chmod(0o777)
        except OSError:
            pass

        self.multi_env = MultiAgentEnvWrapper(
            training=training,
            return_voltage=return_voltage,
            gif_config=gif_config,
            distance_data_dir=str(distance_data_dir),
            env_config_path=env_config_path,
            capacitance_model_checkpoint=capacitance_model_checkpoint,
            is_collecting_data=is_collecting_data,
            base_env_class=base_env_class,
        )

        self.agent_id = "plunger_0"
        self.action_space = self.multi_env.action_spaces[self.agent_id]
        self.observation_space = self.multi_env.observation_spaces[self.agent_id]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict]:
        """Reset the environment and return plunger_0 observation."""
        obs, infos = self.multi_env.reset(seed=seed, options=options)
        return obs[self.agent_id], infos.get(self.agent_id, {})

    def step(self, action: Any):
        """Step the environment with plunger_0 action and ground-truth actions for all others."""
        agent_actions = self._build_agent_actions(action)
        obs, rewards, terminated, truncated, infos = self.multi_env.step(agent_actions)
        return (
            obs[self.agent_id],
            rewards[self.agent_id],
            terminated["__all__"],
            truncated["__all__"],
            infos.get(self.agent_id, {}),
        )

    def close(self):
        """Close the underlying environment."""
        if hasattr(self.multi_env, "close"):
            return self.multi_env.close()
        return None

    def _build_agent_actions(self, plunger_action: Any) -> Dict[str, np.ndarray]:
        """Build a full multi-agent action dict from a single plunger_0 action."""
        base_env = self.multi_env.base_env
        device_state = getattr(base_env, "device_state", None)
        if not device_state:
            raise RuntimeError("Base environment device_state not initialized. Call reset() first.")

        gate_ground_truth = device_state.get("gate_ground_truth")
        barrier_ground_truth = device_state.get("barrier_ground_truth")
        current_gate = device_state.get("current_gate_voltages")
        current_barrier = device_state.get("current_barrier_voltages")

        if gate_ground_truth is None or barrier_ground_truth is None:
            raise RuntimeError("Ground truth voltages not available in device_state.")

        agent_actions: Dict[str, np.ndarray] = {}

        # Normalize the provided plunger_0 action to scalar float
        plunger_action_value = self._to_scalar(plunger_action)
        agent_actions["plunger_0"] = np.array([plunger_action_value], dtype=np.float32)

        # Other plungers: drive to ground-truth voltages
        for agent_id in self.multi_env.gate_agent_ids:
            if agent_id == "plunger_0":
                continue
            idx = int(agent_id.split("_")[1])
            target = float(gate_ground_truth[idx])
            current = float(current_gate[idx]) if current_gate is not None else None
            normalized = self._normalize_gate_voltage(target, current, base_env, idx)
            agent_actions[agent_id] = np.array([normalized], dtype=np.float32)

        # Barriers: drive to ground-truth voltages
        for agent_id in self.multi_env.barrier_agent_ids:
            idx = int(agent_id.split("_")[1])
            target = float(barrier_ground_truth[idx])
            normalized = self._normalize_barrier_voltage(target, base_env)
            agent_actions[agent_id] = np.array([normalized], dtype=np.float32)

        return agent_actions

    @staticmethod
    def _to_scalar(value: Any) -> float:
        """Convert scalar/array input to float."""
        if isinstance(value, (np.ndarray, list, tuple)):
            return float(np.array(value).flatten()[0])
        return float(value)

    @staticmethod
    def _normalize_gate_voltage(
        target_voltage: float, current_voltage: Optional[float], base_env, idx: int
    ) -> float:
        """Normalize a gate voltage to [-1, 1] based on env configuration."""
        if getattr(base_env, "use_deltas", False):
            if current_voltage is None:
                raise RuntimeError("Current gate voltage required for delta-based control.")
            delta = target_voltage - current_voltage
            delta = np.clip(delta, base_env.plunger_delta_min[idx], base_env.plunger_delta_max[idx])
            normalized = (delta - base_env.plunger_delta_min[idx]) / (
                base_env.plunger_delta_max[idx] - base_env.plunger_delta_min[idx]
            )
            return float(normalized * 2 - 1)

        normalized = (target_voltage - base_env.plunger_min[idx]) / (
            base_env.plunger_max[idx] - base_env.plunger_min[idx]
        )
        return float(normalized * 2 - 1)

    @staticmethod
    def _normalize_barrier_voltage(target_voltage: float, base_env) -> float:
        """Normalize a barrier voltage to [-1, 1] based on env configuration."""
        normalized = (target_voltage - base_env.barrier_min) / (base_env.barrier_max - base_env.barrier_min)
        return float(normalized * 2 - 1)
