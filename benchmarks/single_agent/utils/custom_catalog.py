"""
Custom catalog for single-agent RL training with quantum device networks.

Reuses swarm's encoder and head configurations with voltage_dim set to num_gates.
"""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.configs import PolicyHeadConfig, ValueHeadConfig


class CustomSingleAgentCatalog(PPOCatalog):
    """Custom catalog for single-agent quantum device neural network components.

    Uses voltage_dim=num_gates so heads accept all gate voltages at once.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        # Get voltage dimension from observation space
        if isinstance(observation_space, gym.spaces.Dict):
            self.voltage_dim = observation_space["voltage"].shape[0]
        else:
            self.voltage_dim = 1

    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        """Build encoder configuration using swarm's common encoder builder."""
        # Get the image observation space shape for encoder
        if isinstance(observation_space, gym.spaces.Dict):
            image_space = observation_space["image"]
            encoder_obs_space = image_space
        else:
            encoder_obs_space = observation_space

        return build_encoder_config(encoder_obs_space, model_config_dict)

    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        """Build policy head for single-agent with voltage_dim=num_gates."""
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,  # mean and log std for all gates
            log_std_bounds=self._model_config_dict.get("log_std_bounds", [-10, 2]),
            voltage_dim=self.voltage_dim,  # Accept all gate voltages
        )

        return config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        """Build value head for single-agent with voltage_dim=num_gates."""
        value_config = self._model_config_dict["value_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            use_attention=value_config["use_attention"],
            voltage_dim=self.voltage_dim,  # Accept all gate voltages
        )

        return config.build(framework=framework)
