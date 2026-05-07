"""
Factory for creating RLlib module specifications for the single-agent benchmark.
"""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module.multi_rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from qadapt.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from qadapt.voltage_model.configs import PolicyHeadConfig, ValueHeadConfig


class CustomSingleAgentCatalog(PPOCatalog):
    """Single-agent catalog that uses the full voltage vector."""

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
        if isinstance(observation_space, gym.spaces.Dict):
            encoder_obs_space = observation_space["image"]
        else:
            encoder_obs_space = observation_space
        return build_encoder_config(encoder_obs_space, model_config_dict)

    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,
            log_std_bounds=self._model_config_dict.get("log_std_bounds", [-10, 2]),
            voltage_dim=self.voltage_dim,
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
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)


def create_rl_module_spec(env_config: dict, algo: str = "ppo", config: dict = None, train_barriers: bool = True) -> RLModuleSpec:
    """Create a single-agent RLModuleSpec based on env_config."""
    import numpy as np
    from gymnasium import spaces

    algo = algo.lower()
    if algo != "ppo":
        raise NotImplementedError("Single-agent benchmark currently supports PPO only.")

    resolution = env_config["simulator"]["resolution"]
    num_dots = env_config["simulator"]["num_dots"]
    num_channels = num_dots - 1

    num_barriers = num_dots - 1 if train_barriers else 0
    num_actions = num_dots + num_barriers

    image_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(resolution, resolution, num_channels),
        dtype=np.float32,
    )
    voltage_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(num_actions,),
        dtype=np.float32,
    )

    observation_space = spaces.Dict({
        "image": image_space,
        "voltage": voltage_space,
    })

    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(num_actions,),
        dtype=np.float32,
    )

    model_config = config if isinstance(config, dict) else {}
    if "backbone" in model_config:
        backbone = model_config["backbone"]
        memory_layer = backbone.get("memory_layer")
        if memory_layer == "lstm":
            model_config["max_seq_len"] = backbone["lstm"]["max_seq_len"]
        elif memory_layer == "transformer":
            model_config["max_seq_len"] = backbone["transformer"]["max_seq_len"]

    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
        catalog_class=CustomSingleAgentCatalog,
        inference_only=False,
    )
