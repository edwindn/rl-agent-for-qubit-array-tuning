"""
Wrapper to present single-agent environment as a multi-agent environment.

This is required because RLlib's single-agent PPO code path has different behavior
than the multi-agent code path, and the multi-agent path works correctly for our
continuous action space setup.
"""

from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from single_agent_sac.utils.env_wrapper import SingleAgentEnvWrapper


class SingleAsMultiAgentWrapper(MultiAgentEnv):
    """Wrap single-agent env to look like a multi-agent env with one agent.

    This wrapper is a workaround for RLlib single-agent PPO not learning
    correctly with our continuous action environment. By presenting the
    environment as multi-agent with a single agent, we use RLlib's
    multi-agent code path which works correctly.
    """

    def __init__(self, config=None):
        super().__init__()

        # Extract config parameters
        if config is None:
            config = {}

        config_path = config.get("config_path")
        training = config.get("training", True)
        distance_data_dir = config.get("distance_data_dir")
        num_dots_override = config.get("num_dots_override")
        capacitance_model_checkpoint = config.get("capacitance_model_checkpoint")

        # Create the underlying single-agent environment
        self.base_env = SingleAgentEnvWrapper(
            training=training,
            config_path=config_path,
            distance_data_dir=distance_data_dir,
            num_dots_override=num_dots_override,
            capacitance_model_checkpoint=capacitance_model_checkpoint,
        )

        # Define single "agent" named "agent_0"
        self._agent_ids = {"agent_0"}
        self.agents = self._agent_ids.copy()
        self.possible_agents = list(self._agent_ids)

        # Wrap spaces as multi-agent Dict spaces
        self._obs_space = spaces.Dict({
            "agent_0": self.base_env.observation_space
        })
        self._action_space = spaces.Dict({
            "agent_0": self.base_env.action_space
        })

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return {"agent_0": obs}, {"agent_0": info}

    def step(self, action_dict):
        # Extract single agent's action
        action = action_dict["agent_0"]

        # Step the underlying environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Wrap outputs as multi-agent dicts
        obs_dict = {"agent_0": obs}
        reward_dict = {"agent_0": reward}
        terminated_dict = {"agent_0": terminated, "__all__": terminated}
        truncated_dict = {"agent_0": truncated, "__all__": truncated}
        info_dict = {"agent_0": info}

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def get_agent_ids(self):
        return self._agent_ids

    def close(self):
        return self.base_env.close()
