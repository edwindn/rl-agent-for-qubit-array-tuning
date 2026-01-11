"""TD3-specific RLlib integration.

TD3 (Twin Delayed DDPG) key differences from SAC:
- Deterministic policy (outputs action directly, not distribution parameters)
- Action noise for exploration (Gaussian noise injection)
- Target policy smoothing (add clipped noise to target actions)
- Delayed policy updates (handled in TD3Learner, not here)
"""

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
from swarm.voltage_model.configs import DeterministicPolicyHeadConfig, QValueHeadConfig


class CustomTD3Catalog(SACCatalog):
    """Custom catalog for TD3 with image-based observations.

    Key difference from SAC: policy head outputs action_dim (not action_dim * 2).
    TD3 uses deterministic policy, not stochastic.
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
        """Build DETERMINISTIC policy head - outputs action directly (not mean+std)."""
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = DeterministicPolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config.get("use_attention", False),
            output_layer_dim=self.action_space.shape[0],  # Just action_dim, NOT *2
        )
        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        """Build Q-function head - same as SAC."""
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
        """Build Q-function encoder - same as SAC."""
        return self.build_encoder(framework=framework)


class CustomTD3TorchRLModule(DefaultSACTorchRLModule):
    """Custom TD3 RL module that handles image observations.

    Extends SAC's module but overrides:
    - Policy output (deterministic action instead of mean+log_std)
    - Exploration (adds Gaussian noise)
    - Target computation (adds clipped noise to target actions)
    """

    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TD3 exploration noise parameters (configurable via model_config)
        self.exploration_noise_std = self.model_config.get("exploration_noise_std", 0.1)
        self.target_noise_std = self.model_config.get("target_noise_std", 0.2)
        self.target_noise_clip = self.model_config.get("target_noise_clip", 0.5)

    @override(DefaultSACTorchRLModule)
    def _forward_inference(self, batch: Dict) -> Dict[str, Any]:
        """Deterministic action for inference (no noise)."""
        output = {}

        # Encode observations
        pi_encoder_outs = self.pi_encoder(batch)

        if isinstance(pi_encoder_outs, dict) and ENCODER_OUT in pi_encoder_outs:
            encoder_out = pi_encoder_outs[ENCODER_OUT]
        else:
            encoder_out = pi_encoder_outs

        # Get deterministic action from policy head
        action = self.pi(encoder_out)

        # For TD3, action_dist_inputs IS the action (no sampling needed)
        output[Columns.ACTION_DIST_INPUTS] = action
        return output

    @override(DefaultSACTorchRLModule)
    def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """Add exploration noise during exploration."""
        output = self._forward_inference(batch)

        # Add Gaussian noise for exploration
        action = output[Columns.ACTION_DIST_INPUTS]
        noise = torch.randn_like(action) * self.exploration_noise_std
        noisy_action = action + noise

        # Clip to action space bounds [-1, 1]
        noisy_action = torch.clamp(noisy_action, -1.0, 1.0)
        output[Columns.ACTION_DIST_INPUTS] = noisy_action

        return output

    @override(DefaultSACTorchRLModule)
    def _forward_train(self, batch: Dict) -> Dict[str, Any]:
        """Forward pass for training - compute Q-values and policy actions."""
        output = {}

        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # Current Q-values with actual actions from replay buffer
        output[Columns.ACTION_DIST_INPUTS] = batch[Columns.ACTIONS]
        output["qf_preds"] = self._qf_forward_train_helper(
            batch, self.qf_encoder, self.qf
        )
        if self.twin_q:
            output["qf_twin_preds"] = self._qf_forward_train_helper(
                batch, self.qf_twin_encoder, self.qf_twin
            )

        # Current policy action (for policy gradient)
        pi_encoder_outs = self.pi_encoder(batch_curr)
        if isinstance(pi_encoder_outs, dict) and ENCODER_OUT in pi_encoder_outs:
            encoder_out = pi_encoder_outs[ENCODER_OUT]
        else:
            encoder_out = pi_encoder_outs

        current_action = self.pi(encoder_out)
        output["actions_curr"] = current_action

        # Q-value of current state with policy action (for actor loss)
        q_batch_curr = {Columns.OBS: batch[Columns.OBS], Columns.ACTIONS: current_action}
        output["q_curr"] = self._qf_forward_train_helper(
            q_batch_curr, self.qf_encoder, self.qf
        )

        # Target Q-values with target policy + noise (target policy smoothing)
        output["q_target_next"] = self._compute_target_q(batch_next)

        return output

    def _compute_target_q(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute target Q-value with target policy smoothing.

        This is a key TD3 trick: add clipped noise to target actions.
        """
        # Get target action
        target_pi_encoder_outs = self.target_pi_encoder(batch)
        if isinstance(target_pi_encoder_outs, dict) and ENCODER_OUT in target_pi_encoder_outs:
            target_encoder_out = target_pi_encoder_outs[ENCODER_OUT]
        else:
            target_encoder_out = target_pi_encoder_outs

        target_action = self.target_pi(target_encoder_out)

        # Add clipped noise (target policy smoothing - key TD3 trick)
        noise = torch.randn_like(target_action) * self.target_noise_std
        noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
        target_action = torch.clamp(target_action + noise, -1.0, 1.0)

        # Compute target Q-values with noisy target action
        batch_with_action = {Columns.OBS: batch[Columns.OBS], Columns.ACTIONS: target_action}

        target_qf = self._qf_forward_train_helper(
            batch_with_action, self.target_qf_encoder, self.target_qf
        )

        if self.twin_q:
            target_qf_twin = self._qf_forward_train_helper(
                batch_with_action, self.target_qf_twin_encoder, self.target_qf_twin
            )
            # Take minimum of twin Q-networks (clipped double Q-learning)
            return torch.min(target_qf, target_qf_twin).detach()

        return target_qf.detach()

    @override(DefaultSACTorchRLModule)
    def _qf_forward_train_helper(
        self,
        batch: Dict[str, Any],
        encoder: Encoder,
        head: Model,
        squeeze: bool = True
    ) -> torch.Tensor:
        """Q-function forward pass - handles image observations.

        Same implementation as CustomSACTorchRLModule._qf_forward_train_helper
        """
        if isinstance(self.action_space, gym.spaces.Box):
            # Encode observations
            obs_encoded = encoder(batch)

            # Extract ENCODER_OUT from dict
            if isinstance(obs_encoded, dict) and ENCODER_OUT in obs_encoded:
                obs_encoded = obs_encoded[ENCODER_OUT]

            # Handle CNN backbone output format: {"image_features": tensor, "voltage": tensor}
            if isinstance(obs_encoded, dict):
                if "image_features" in obs_encoded:
                    obs_tensor = obs_encoded["image_features"]
                else:
                    raise ValueError(f"Unexpected encoder output structure: {list(obs_encoded.keys())}")
            else:
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

            if isinstance(encoder_out, dict) and "image_features" in encoder_out:
                encoder_out = encoder_out["image_features"]

            qf_out = head(encoder_out)

            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out
