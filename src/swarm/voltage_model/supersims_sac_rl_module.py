import gymnasium as gym
import torch
from typing import Dict, Any

from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import Model, Encoder, ENCODER_OUT
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.utils.annotations import override


class CustomSACTorchRLModule(DefaultSACTorchRLModule):
    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(DefaultSACTorchRLModule)
    def _qf_forward_train_helper(
        self, batch: Dict[str, Any], encoder: Encoder, head: Model, squeeze: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass for the Q-function during training.
        Handles image observation spaces that RLlib SAC does not support yet.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            # Encode the observations using encoder
            obs_encoded = encoder(batch)
            if isinstance(obs_encoded, dict) and ENCODER_OUT in obs_encoded:
                  obs_encoded = obs_encoded[ENCODER_OUT]

            actions = batch[Columns.ACTIONS]

            qf_input = torch.concat((obs_encoded, actions), dim=-1)

            qf_out = head(qf_input)

            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out

        else:
            # For discrete action spaces, we don't need to include the actions
            # in the batch, as the Q function outputs the Q-values for each action
            qf_batch = {Columns.OBS: batch[Columns.OBS]}

        # Encoder forward pass.
        qf_encoder_outs = encoder(qf_batch)

        # Q head forward pass.
        # (B,latent_size) -> (B, 1|action_dim)
        qf_out = head(qf_encoder_outs[ENCODER_OUT])
        if squeeze:
            # Squeeze the last dimension if it is 1.
            qf_out = qf_out.squeeze(-1)
        return qf_out