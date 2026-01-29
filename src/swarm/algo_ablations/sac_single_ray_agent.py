"""
Single-agent SAC RLModule spec and catalog using Swarm's IMPALA + heads.
"""

from typing import Dict

import gymnasium as gym
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module.multi_rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.algorithms.sac import CustomSACTorchRLModule
from swarm.voltage_model.configs import PolicyHeadConfig, QValueHeadConfig


DEFAULT_SAC_SINGLE_AGENT_MODEL_CONFIG: Dict = {
    "backbone": {
        "type": "IMPALA",
        "feature_size": 256,
        "adaptive_pooling": True,
        "num_res_blocks": 2,
        "memory_layer": None,
        # Unused for IMPALA but kept for parity with config format.
        "mobilenet_version": "small",
        "load_pretrained": False,
        "freeze_backbone": False,
        "lstm": {
            "hidden_dim": 256,
            "num_layers": 1,
            "max_seq_len": 1,
            "store_voltages": False,
            "voltage_hidden_dim": 16,
        },
        "transformer": {
            "latent_size": 128,
            "num_attention_heads": 4,
            "num_layers": 1,
            "max_seq_len": 10,
            "feedforward_dim": None,
            "dropout": 0.1,
            "pooling_mode": "mean",
            "use_ctlpe": True,
            "add_pos_embeddings": False,
        },
    },
    "policy_head": {
        "hidden_layers": [64, 64, 32],
        "activation": "relu",
        "use_attention": False,
    },
    "value_head": {
        "hidden_layers": [64, 64, 32],
        "activation": "relu",
        "use_attention": False,
    },
    "log_std_bounds": [-5, 2],
    "twin_q": True,
}


class SingleAgentSACCatalog(SACCatalog):
    """SAC catalog that supports Dict observations with image + voltage."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: dict):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        if isinstance(observation_space, gym.spaces.Dict):
            self.voltage_dim = observation_space["voltage"].shape[0]
        else:
            self.voltage_dim = 1

    @override(SACCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if isinstance(observation_space, gym.spaces.Dict):
            encoder_obs_space = observation_space["image"]
        else:
            encoder_obs_space = observation_space
        return build_encoder_config(encoder_obs_space, model_config_dict)

    @override(SACCatalog)
    def build_pi_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,
            log_std_bounds=self._model_config_dict.get("log_std_bounds", [-5, 2]),
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        value_config = self._model_config_dict["value_head"]
        encoder_output_dim = get_head_input_dim(self._model_config_dict)
        action_dim = self.action_space.shape[0]

        config = QValueHeadConfig(
            input_dims=(encoder_output_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            action_dim=action_dim,
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_encoder(self, framework: str = "torch"):
        return self.build_encoder(framework=framework)


def build_single_agent_sac_module_spec(
    observation_space: gym.Space,
    action_space: gym.Space,
    model_config: Dict = None,
) -> RLModuleSpec:
    """Create RLModuleSpec for single-agent SAC using Swarm's IMPALA encoder."""
    if model_config is None:
        model_config = DEFAULT_SAC_SINGLE_AGENT_MODEL_CONFIG

    return RLModuleSpec(
        module_class=CustomSACTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
        catalog_class=SingleAgentSACCatalog,
        inference_only=False,
    )
