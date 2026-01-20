"""
Custom catalog for single-agent RL training with quantum device networks.
"""

import sys
from pathlib import Path
import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

# Add parent directory to path to import networks
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from networks import InputEncoderConfig, PolicyHeadConfig, ValueHeadConfig


class CustomSingleAgentCatalog(PPOCatalog):
    """Custom catalog for single-agent quantum device neural network components."""

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
        """
        Build encoder configuration for single-agent training.

        For now, uses simple InputEncoder from networks.py.
        Future: Add support for memory layers (LSTM/Transformer).
        """
        encoder_config = model_config_dict.get("encoder", {})

        # Get the image observation space shape
        # observation_space is Dict with {"image": Box(...), "obs_gate_voltages": ..., "obs_barrier_voltages": ...}
        if isinstance(observation_space, gym.spaces.Dict):
            image_space = observation_space["image"]
            input_dims = image_space.shape
        else:
            input_dims = observation_space.shape

        # Create InputEncoder config
        config = InputEncoderConfig(
            input_dims=input_dims,
            # num_input_scans, feature_size, cnn_activation loaded from config.yaml
        )

        return config

    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        """Build policy head for single-agent."""
        encoder_config = self._model_config_dict.get("encoder", {})

        # Input dimension is the encoder output size
        input_dim = encoder_config.get("feature_size", 256)

        # PolicyHead config loaded from config.yaml
        # output_layer_dim set to action_space.shape[0] * 2 for mean and log_std
        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            output_layer_dim=self.action_space.shape[0] * 2,  # mean and log std
        )

        return config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        """Build value head for single-agent."""
        encoder_config = self._model_config_dict.get("encoder", {})

        # Input dimension is the encoder output size
        input_dim = encoder_config.get("feature_size", 256)

        # ValueHead config loaded from config.yaml
        config = ValueHeadConfig(
            input_dims=(input_dim,),
        )

        return config.build(framework=framework)
