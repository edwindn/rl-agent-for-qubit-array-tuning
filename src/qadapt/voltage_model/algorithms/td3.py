"""TD3-specific RLlib integration.

TD3 (Twin Delayed DDPG) key differences from SAC:
- Deterministic policy (no stochastic sampling)
- Exploration via Gaussian noise injection
- Target policy smoothing (clipped noise added to target actions)
- Delayed policy updates (update actor less frequently than critic)
- No entropy/alpha optimization
"""

from typing import Any, Dict

import gymnasium as gym
import torch

from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import (
    DefaultSACTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.utils.annotations import override

from qadapt.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from qadapt.voltage_model.configs import DeterministicPolicyHeadConfig, QValueHeadConfig


class CustomTD3Catalog(SACCatalog):
    """Custom catalog for TD3 with image-based observations.

    Uses deterministic policy head instead of stochastic.
    Reuses Q-value heads from SAC (twin Q-networks).
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
    ):
        """Build encoder config using shared infrastructure."""
        return build_encoder_config(observation_space, model_config_dict)

    @override(SACCatalog)
    def build_pi_head(self, framework: str = "torch"):
        """Build deterministic policy head (outputs action directly).

        Unlike SAC which outputs mean + log_std (2 * action_dim),
        TD3 outputs action directly (action_dim) with tanh activation.
        """
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = DeterministicPolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config.get("use_attention", False),
            output_layer_dim=self.action_space.shape[0],  # Just action_dim (not 2x)
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        """Build Q-function head (same as SAC).

        Input dimension = encoder output (image_features).
        Action and voltage are passed separately and handled inside QValueHead.
        """
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
        """Build Q-function encoder (same architecture as policy encoder)."""
        return self.build_encoder(framework=framework)


class CustomTD3TorchRLModule(DefaultSACTorchRLModule):
    """TD3 RL module with deterministic policy and exploration noise.

    Key differences from SAC:
    1. _forward_inference: Returns deterministic action (no sampling)
    2. _forward_exploration: Adds Gaussian noise to deterministic action
    3. _forward_train: No log_prob computation, adds target policy smoothing
    """

    framework: str = "torch"

    @override(DefaultSACTorchRLModule)
    def setup(self):
        """Setup TD3 module components."""
        self.twin_q = self.model_config.get("twin_q", True)

        # Get TD3-specific params from model_config
        # These must be set in setup(), not __init__, because setup() is called
        # during super().__init__() before __init__ completes
        self.exploration_noise = self.model_config.get("exploration_noise", 0.1)
        self.policy_noise = self.model_config.get("policy_noise", 0.2)
        self.noise_clip = self.model_config.get("noise_clip", 0.5)

        # Policy encoder
        self.pi_encoder = self.catalog.build_encoder(framework=self.framework)

        if not self.inference_only or self.framework != "torch":
            # Q-network encoders
            self.qf_encoder = self.catalog.build_qf_encoder(framework=self.framework)
            if self.twin_q:
                self.qf_twin_encoder = self.catalog.build_qf_encoder(
                    framework=self.framework
                )

        # Build heads
        self.pi = self.catalog.build_pi_head(framework=self.framework)

        if not self.inference_only or self.framework != "torch":
            self.qf = self.catalog.build_qf_head(framework=self.framework)
            if self.twin_q:
                self.qf_twin = self.catalog.build_qf_head(framework=self.framework)

    @override(DefaultSACTorchRLModule)
    def _forward_inference(self, batch: Dict) -> Dict[str, Any]:
        """Deterministic action selection (no noise).

        Returns Columns.ACTIONS directly to bypass RLlib's action distribution
        sampling, since TD3 uses a deterministic policy.
        """
        output = {}
        pi_encoder_outs = self.pi_encoder(batch)

        # Extract encoder output
        encoder_out = pi_encoder_outs[ENCODER_OUT]

        # Deterministic policy outputs action directly (with tanh)
        # Return ACTIONS directly to bypass distribution sampling in GetActions connector
        output[Columns.ACTIONS] = self.pi(encoder_out)
        return output

    @override(DefaultSACTorchRLModule)
    def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """Add exploration noise to deterministic action.

        Returns Columns.ACTIONS directly to bypass RLlib's action distribution
        sampling, since TD3 uses deterministic policy + additive noise.
        """
        output = self._forward_inference(batch)

        # Add Gaussian exploration noise
        action = output[Columns.ACTIONS]
        noise = torch.randn_like(action) * self.exploration_noise
        noisy_action = (action + noise).clamp(-1.0, 1.0)

        output[Columns.ACTIONS] = noisy_action
        return output

    @override(DefaultSACTorchRLModule)
    def _forward_train(self, batch: Dict) -> Dict[str, Any]:
        """Forward pass for training - computes Q-values and target values."""
        output = {}

        batch_curr = {Columns.OBS: batch[Columns.OBS]}
        batch_next = {Columns.OBS: batch[Columns.NEXT_OBS]}

        # Encode current and next observations
        pi_encoder_outs = self.pi_encoder(batch_curr)
        pi_encoder_next_outs = self.pi_encoder(batch_next)

        # Q-values for current state-action pairs (from batch)
        batch_curr_with_actions = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: batch[Columns.ACTIONS],
        }
        output["qf_preds"] = self._qf_forward_train_helper(
            batch_curr_with_actions, self.qf_encoder, self.qf
        )
        if self.twin_q:
            output["qf_twin_preds"] = self._qf_forward_train_helper(
                batch_curr_with_actions, self.qf_twin_encoder, self.qf_twin
            )

        # Current policy action (for actor loss)
        action_curr = self.pi(pi_encoder_outs[ENCODER_OUT])
        output["action_curr"] = action_curr

        # Q-value for current policy action (detach Q-network gradients for actor update)
        q_batch_curr = {
            Columns.OBS: batch[Columns.OBS],
            Columns.ACTIONS: action_curr,
        }

        # Temporarily disable gradients for Q-networks during actor loss computation
        all_params = list(self.qf.parameters()) + list(self.qf_encoder.parameters())
        if self.twin_q:
            all_params += list(self.qf_twin.parameters()) + list(
                self.qf_twin_encoder.parameters()
            )

        for param in all_params:
            param.requires_grad = False

        output["q_curr"] = self._qf_forward_train_helper(
            q_batch_curr, self.qf_encoder, self.qf
        )

        for param in all_params:
            param.requires_grad = True

        # Target policy action with clipped noise (for target Q computation)
        with torch.no_grad():
            action_next = self.pi(pi_encoder_next_outs[ENCODER_OUT])
            noise = (torch.randn_like(action_next) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action_next_smoothed = (action_next + noise).clamp(-1.0, 1.0)

        output["action_next_smoothed"] = action_next_smoothed

        # Target Q-values
        q_batch_next = {
            Columns.OBS: batch[Columns.NEXT_OBS],
            Columns.ACTIONS: action_next_smoothed,
        }
        output["q_target_next"] = self.forward_target(q_batch_next).detach()

        return output

    @override(DefaultSACTorchRLModule)
    def _qf_forward_train_helper(
        self, batch, encoder, head, squeeze: bool = True
    ) -> torch.Tensor:
        """Forward pass for Q-function (handles image observations).

        This is the same pattern as SAC - encodes observations and passes
        encoded features + action to the Q-head.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            obs_encoded = encoder(batch)

            if isinstance(obs_encoded, dict) and ENCODER_OUT in obs_encoded:
                obs_encoded = obs_encoded[ENCODER_OUT]

            # Handle CNN backbone output format: {"image_features": tensor, "voltage": tensor}
            if isinstance(obs_encoded, dict):
                if "image_features" in obs_encoded and "voltage" in obs_encoded:
                    image_features = obs_encoded["image_features"]
                    voltage = obs_encoded["voltage"]
                else:
                    raise ValueError(
                        f"Unexpected encoder output structure: {list(obs_encoded.keys())}"
                    )
            else:
                raise ValueError(
                    "Expected encoder output to be dict with image_features and voltage"
                )

            actions = batch[Columns.ACTIONS]
            qf_input = {
                "image_features": image_features,
                "voltage": voltage,
                "action": actions,
            }

            qf_out = head(qf_input)

            if squeeze:
                qf_out = qf_out.squeeze(-1)
            return qf_out

        else:
            raise ValueError("TD3 only supports continuous (Box) action spaces")
