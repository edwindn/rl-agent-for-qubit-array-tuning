"""
Fixed plunger wrapper for multi-agent environment.

This wrapper extends MultiAgentEnvWrapper to support scenarios where plunger gates
are fixed and only barrier agents are active.
"""

import sys
from typing import Dict
from pathlib import Path
import numpy as np

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


class FixedPlungerWrapper(MultiAgentEnvWrapper):
    """
    Wrapper that extends MultiAgentEnvWrapper with fixed plunger functionality.

    This class has the exact same constructor and APIs as MultiAgentEnvWrapper,
    but can be extended to support fixed plunger gate scenarios.
    """

    def __init__(
        self,
        training: bool = True,
        return_voltage: bool = False,
        gif_config: dict = None,
        distance_data_dir: str = None,
        env_config_path: str = None,
    ):
        """
        Initialize fixed plunger wrapper.

        Args:
            training: Whether in training mode
            return_voltage: If True, returns dict observation with image and voltage.
                          If False, returns only the image array.
            gif_config: Configuration for GIF capture
            distance_data_dir: Path to directory for saving distance data (if enabled)
            env_config_path: Optional path to custom env config file (defaults to env_config.yaml)
        """
        # Call parent constructor with exact same parameters
        super().__init__(
            training=training,
            return_voltage=return_voltage,
            gif_config=gif_config,
            distance_data_dir=distance_data_dir,
            env_config_path=env_config_path,
        )

    
    def step(self, agent_actions: Dict[str, np.ndarray]):
        """
        Step environment with individual agent actions.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        assert len(agent_actions) == len(
            self.all_agent_ids
        ), "Agent actions must match the number of agents"
        assert all(
            agent_id in self.all_agent_ids for agent_id in agent_actions.keys()
        ), "Unknown agent IDs in actions"

        # print("[ENV DEBUG] env.step called")
        # Combine agent actions into global action
        global_action = self._combine_agent_actions(agent_actions)

        # Replace plunger voltages with ground truth from device_state
        # Ground truth is in physical voltage range, need to inverse-rescale to [-1, 1]
        plunger_ground_truth = self.base_env.device_state["gate_ground_truth"]

        # Inverse rescale: physical voltages -> [-1, 1] range
        # Reverse of: obs = (obs + 1) / 2 * (max - min) + min
        plunger_min = self.base_env.plunger_min
        plunger_max = self.base_env.plunger_max
        normalized_plunger = (plunger_ground_truth - plunger_min) / (plunger_max - plunger_min)  # [0, 1]
        normalized_plunger = normalized_plunger * 2 - 1  # [-1, 1]

        global_action["action_gate_voltages"] = normalized_plunger

        # Step the base environment with fixed plungers and actual barrier actions
        global_obs, global_rewards, terminated, truncated, info = self.base_env.step(global_action)

        # Get device state info first (needed for image capture)
        device_state_info = info.get("current_device_state", None)

        # Convert to multi-agent format
        agent_observations = {}
        for agent_id in self.all_agent_ids:
            agent_obs = self._extract_agent_observation(global_obs, agent_id, device_state_info)
            # Always return single observation (RLlib handles temporal sequences via ConnectorV2)
            agent_observations[agent_id] = agent_obs

        agent_rewards = self._distribute_rewards(global_rewards)

        # Multi-agent termination/truncation (all agents have same status)
        agent_terminated = dict.fromkeys(self.all_agent_ids, terminated)
        agent_terminated["__all__"] = terminated  # Required by MultiAgentEnv

        agent_truncated = dict.fromkeys(self.all_agent_ids, truncated)
        agent_truncated["__all__"] = truncated  # Required by MultiAgentEnv

        # Save distance history when episode ends
        if (terminated or truncated) and self.distance_history is not None:
            self._save_agent_histories(self.distance_history)
            # Clear history after saving
            self.distance_history = {_id: [] for _id in self.all_agent_ids}

        # Create per-agent info dict (MultiAgentEnv requirement)

        if not device_state_info:
            agent_infos = {agent_id: {} for agent_id in self.all_agent_ids}
        else:
            try:
                agent_infos = {}
                plunger_ids = [f"plunger_{i}" for i in range(self.num_gates)]
                barrier_ids = [f"barrier_{i}" for i in range(self.num_barriers)]

                for idx, agent_id in enumerate(plunger_ids):
                    ground_truth = device_state_info["gate_ground_truth"][idx]
                    current_voltage = device_state_info["current_gate_voltages"][idx]

                    agent_infos[agent_id] = {
                        "ground_truth": ground_truth,
                        "current_voltage": current_voltage,
                    }

                    if self.distance_data_dir is not None:
                        distance_val = current_voltage - ground_truth
                        self.distance_history[agent_id].append(distance_val)

                for idx, agent_id in enumerate(barrier_ids):
                    ground_truth = device_state_info["barrier_ground_truth"][idx]
                    current_voltage = device_state_info["current_barrier_voltages"][idx]

                    agent_infos[agent_id] = {
                        "ground_truth": ground_truth,
                        "current_voltage": current_voltage,
                    }

                    if self.distance_data_dir is not None:
                        distance_val = current_voltage - ground_truth
                        self.distance_history[agent_id].append(distance_val)

            except Exception as e:
                raise RuntimeError(f"Error creating multi-agent info: {e}")
                agent_infos = dict.fromkeys(self.all_agent_ids, {
                    "error": f"Error creating multi-agent info: {e}"
                })
                
        return (
            agent_observations,
            agent_rewards,
            agent_terminated,
            agent_truncated,
            agent_infos,
        )


if __name__ == "__main__":
    """Test the fixed plunger wrapper."""
    print("=== Testing Fixed Plunger Wrapper ===")

    try:
        # Test image-only mode
        print("\n--- Testing image-only mode ---")
        wrapper = FixedPlungerWrapper(training=True, return_voltage=False)
        print(" Created fixed plunger wrapper (image-only)")

        print(f"Agent IDs: {wrapper.get_agent_ids()}")

        # Test reset
        obs, info = wrapper.reset()
        print(f" Reset successful - got observations for {len(obs)} agents")

        # Test step with random actions
        actions = {}
        for agent_id in wrapper.get_agent_ids():
            actions[agent_id] = wrapper.action_spaces[agent_id].sample()

        obs, rewards, terminated, truncated, info = wrapper.step(actions)
        print(f" Step successful - got {len(rewards)} agent rewards")
        wrapper.close()

        # Test dict mode
        print("\n--- Testing dict observation mode ---")
        wrapper = FixedPlungerWrapper(training=True, return_voltage=True)
        print(" Created fixed plunger wrapper (dict mode)")

        obs, info = wrapper.reset()
        print(f" Reset successful - got observations for {len(obs)} agents")

        actions = {}
        for agent_id in wrapper.get_agent_ids():
            actions[agent_id] = wrapper.action_spaces[agent_id].sample()

        obs, rewards, terminated, truncated, info = wrapper.step(actions)
        print(f" Step successful - got {len(rewards)} agent rewards")

        wrapper.close()
        print("\n All tests passed!")

    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
