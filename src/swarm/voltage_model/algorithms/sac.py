"""SAC-specific RLlib integration."""

from typing import Any, Dict

import gymnasium as gym
import torch

from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import Encoder, Model, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.configs import PolicyHeadConfig, QValueHeadConfig


class CustomSACCatalog(SACCatalog):
    """Custom catalog for SAC with image-based observations.

    Ray's default SACCatalog only supports 1D observations. This catalog
    enables image observations by using custom CNN encoders.
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

    @override(SACCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        return build_encoder_config(observation_space, model_config_dict)

    @override(SACCatalog)
    def build_pi_head(self, framework: str = "torch"):
        """Build policy head - same as PPO (outputs mean + log_std)."""
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,  # mean and log std
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        """Build Q-function head.

        Input dimension = encoder output + action dimension
        (action concatenation happens in CustomSACTorchRLModule._qf_forward_train_helper)
        """
        value_config = self._model_config_dict["value_head"]
        encoder_output_dim = get_head_input_dim(self._model_config_dict)
        action_dim = self.action_space.shape[0]

        # Q-function takes concatenated [encoded_obs, action] as input
        input_dim = encoder_output_dim + action_dim

        config = QValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_encoder(self, framework: str = "torch"):
        """Build Q-function encoder.

        Returns the same encoder architecture as the policy encoder.
        Action concatenation is handled by CustomSACTorchRLModule, not here.
        """
        return self.build_encoder(framework=framework)


class CustomSACTorchRLModule(DefaultSACTorchRLModule):
    """Custom SAC RL module that handles image observations.

    Ray's default SAC module doesn't support image observations for Q-functions.
    This module overrides the Q-function forward pass to:
    1. Encode observations through the CNN encoder
    2. Concatenate encoded observations with actions
    3. Pass through Q-function head
    """

    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(DefaultSACTorchRLModule)
    def _qf_forward_train_helper(
        self,
        batch: Dict[str, Any],
        encoder: Encoder,
        head: Model,
        squeeze: bool = True
    ) -> torch.Tensor:
        """Forward pass for Q-function during training.

        Handles image observation spaces that RLlib's default SAC doesn't support.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            # Encode observations
            obs_encoded = encoder(batch)

            # Extract ENCODER_OUT from dict
            if isinstance(obs_encoded, dict) and ENCODER_OUT in obs_encoded:
                obs_encoded = obs_encoded[ENCODER_OUT]

            # Handle CNN backbone output format: {"image_features": tensor, "voltage": tensor}
            # We only need image_features for Q-function (voltage is handled separately)
            if isinstance(obs_encoded, dict):
                if "image_features" in obs_encoded:
                    obs_tensor = obs_encoded["image_features"]
                else:
                    raise ValueError(f"Unexpected encoder output structure: {list(obs_encoded.keys())}")
            else:
                # Already a tensor (e.g., from LSTM/Transformer)
                obs_tensor = obs_encoded

            # Get actions from batch
            actions = batch[Columns.ACTIONS]

            # Concatenate encoded observations with actions
            qf_input = torch.concat((obs_tensor, actions), dim=-1)

            # Q-function forward pass
            qf_out = head(qf_input)

            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out

        else:
            # Discrete action spaces - Q outputs values for each action
            qf_batch = {Columns.OBS: batch[Columns.OBS]}
            qf_encoder_outs = encoder(qf_batch)
            encoder_out = qf_encoder_outs[ENCODER_OUT]

            # Handle dict output format
            if isinstance(encoder_out, dict) and "image_features" in encoder_out:
                encoder_out = encoder_out["image_features"]

            qf_out = head(encoder_out)

            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out
