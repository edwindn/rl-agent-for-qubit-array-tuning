"""PPO-specific RLlib integration."""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.configs import PolicyHeadConfig, ValueHeadConfig


class CustomPPOCatalog(PPOCatalog):
    """Custom catalog for PPO with image-based observations."""

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

    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        return build_encoder_config(observation_space, model_config_dict)

    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,  # mean and log std
            log_std_bounds=self._model_config_dict["log_std_bounds"],
        )

        return config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        value_config = self._model_config_dict["value_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            use_attention=value_config["use_attention"],
        )

        return config.build(framework=framework)
