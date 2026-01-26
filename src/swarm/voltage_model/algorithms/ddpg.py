"""DDPG-specific RLlib integration."""

from typing import Any, Dict

import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder, Model
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.configs import PolicyHeadConfig, QValueHeadConfig

try:
    from ray.rllib.algorithms.ddpg.ddpg_catalog import DDPGCatalog
    from ray.rllib.algorithms.ddpg.torch.default_ddpg_torch_rl_module import (
        DefaultDDPGTorchRLModule,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "Ray's DDPG catalog/module are missing. Copy the DDPG algorithm files "
        "from an older Ray release into this environment (or install a Ray build "
        "with DDPG) before importing the DDPG RLModule."
    ) from exc

BaseCatalog = DDPGCatalog
BaseModule = DefaultDDPGTorchRLModule



class CustomDDPGCatalog(BaseCatalog):
    """DDPG catalog that reuses the shared encoder/head configuration."""

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

    @override(BaseCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        return build_encoder_config(observation_space, model_config_dict)

    @override(BaseCatalog)
    def build_actor_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0],
        )
        return config.build(framework=framework)

    @override(BaseCatalog)
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

    @override(BaseCatalog)
    def build_qf_encoder(self, framework: str = "torch"):
        return self.build_encoder(framework=framework)


class CustomDDPGTorchRLModule(BaseModule):
    """DDPG torch module that handles shared encoder output."""

    framework: str = "torch"

    @override(BaseModule)
    def _qf_forward_train_helper(
        self,
        batch: Dict[str, Any],
        encoder: Encoder,
        head: Model,
        squeeze: bool = True,
    ) -> torch.Tensor:
        if isinstance(self.action_space, gym.spaces.Box):
            obs_encoded = encoder(batch)

            if isinstance(obs_encoded, dict) and ENCODER_OUT in obs_encoded:
                obs_encoded = obs_encoded[ENCODER_OUT]

            if isinstance(obs_encoded, dict):
                if "image_features" in obs_encoded and "voltage" in obs_encoded:
                    image_features = obs_encoded["image_features"]
                    voltage = obs_encoded["voltage"]
                else:
                    raise RuntimeError(
                        f"Unexpected encoder output keys: {list(obs_encoded.keys())}"
                    )
            else:
                raise RuntimeError(
                    "DDPG encoder output must be a dict with "
                    "'image_features' and 'voltage'."
                )

            actions = batch[Columns.ACTIONS]
            qf_input = {
                "image_features": image_features,
                "voltage": voltage,
                "action": actions,
            }

            qf_output = head(qf_input)
            if squeeze:
                qf_output = qf_output.squeeze(-1)
            return qf_output
        else:
            qf_batch = {Columns.OBS: batch[Columns.OBS]}
            encoder_out = encoder(qf_batch)[ENCODER_OUT]

            if isinstance(encoder_out, dict) and "image_features" in encoder_out:
                encoder_out = encoder_out["image_features"]

            qf_output = head(encoder_out)
            if squeeze:
                qf_output = qf_output.squeeze(-1)
            return qf_output
