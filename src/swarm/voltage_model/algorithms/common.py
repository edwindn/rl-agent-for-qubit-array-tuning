"""Shared encoder configuration builders for all algorithms."""

import gymnasium as gym
from ray.rllib.core.models.configs import ModelConfig

from swarm.voltage_model.configs import (
    SimpleCNNConfig,
    IMPALAConfig,
    MobileNetConfig,
    TransformerConfig,
    LSTMConfig,
)


def _build_backbone_config(observation_space: gym.Space, backbone_config: dict) -> ModelConfig:
    """Build CNN backbone config (SimpleCNN, IMPALA, or MobileNet)."""
    backbone_type = backbone_config["type"]

    if backbone_type == "IMPALA":
        return IMPALAConfig(
            input_dims=observation_space.shape,
            cnn_activation="relu",
            conv_layers=backbone_config.get("conv_layers"),
            feature_size=backbone_config["feature_size"],
            adaptive_pooling=backbone_config["adaptive_pooling"],
            num_res_blocks=backbone_config["num_res_blocks"],
        )
    elif backbone_type == "SimpleCNN":
        return SimpleCNNConfig(
            input_dims=observation_space.shape,
            cnn_activation="relu",
            conv_layers=backbone_config.get("conv_layers"),
            feature_size=backbone_config["feature_size"],
            adaptive_pooling=backbone_config["adaptive_pooling"],
        )
    elif backbone_type == "MobileNet":
        return MobileNetConfig(
            input_dims=observation_space.shape,
            mobilenet_version=backbone_config["mobilenet_version"],
            feature_size=backbone_config["feature_size"],
            freeze_backbone=backbone_config["freeze_backbone"],
        )
    else:
        raise ValueError(
            f"Unsupported backbone type: {backbone_type}. "
            "Supported types: 'SimpleCNN', 'IMPALA', 'MobileNet'"
        )


def build_encoder_config(observation_space: gym.Space, model_config_dict: dict) -> ModelConfig:
    """
    Build encoder config for PPO and SAC catalogs.

    Supports SimpleCNN/IMPALA/MobileNet backbones with optional LSTM/Transformer memory layers.

    Args:
        observation_space: The observation space (typically image-based).
        model_config_dict: Model configuration dictionary containing backbone and memory layer settings.

    Returns:
        ModelConfig for the encoder.
    """
    backbone_config = model_config_dict["backbone"]
    memory_layer = backbone_config["memory_layer"]

    if memory_layer == "transformer":
        transformer_config = backbone_config["transformer"]
        tokenizer_config = _build_backbone_config(observation_space, backbone_config)

        return TransformerConfig(
            input_dims=tokenizer_config.output_dims,
            tokenizer_config=tokenizer_config,
            latent_size=transformer_config["latent_size"],
            num_attention_heads=transformer_config["num_attention_heads"],
            num_layers=transformer_config["num_layers"],
            feedforward_dim=transformer_config.get("feedforward_dim"),
            dropout=transformer_config["dropout"],
            voltage_num_frequencies=transformer_config.get("voltage_num_frequencies", 8),
            voltage_learnable_frequencies=transformer_config.get("voltage_learnable_frequencies", True),
            pooling_mode=transformer_config.get("pooling_mode", "attention"),
        )

    elif memory_layer == "lstm":
        lstm_config = backbone_config["lstm"]
        tokenizer_config = _build_backbone_config(observation_space, backbone_config)

        return LSTMConfig(
            input_dims=tokenizer_config.output_dims,
            tokenizer_config=tokenizer_config,
            hidden_dim=lstm_config["hidden_dim"],
            num_layers=lstm_config["num_layers"],
            max_seq_len=lstm_config["max_seq_len"],
            store_voltages=lstm_config["store_voltages"],
            voltage_hidden_dim=lstm_config["voltage_hidden_dim"],
        )

    # No memory layer - just backbone
    return _build_backbone_config(observation_space, backbone_config)


def get_head_input_dim(model_config_dict: dict) -> int:
    """
    Get the input dimension for policy/value/Q-function heads.

    Args:
        model_config_dict: Model configuration dictionary.

    Returns:
        Input dimension for heads based on backbone and memory layer configuration.
    """
    backbone_config = model_config_dict["backbone"]
    memory_layer = backbone_config["memory_layer"]

    if memory_layer == "lstm":
        return backbone_config["lstm"]["hidden_dim"]
    elif memory_layer == "transformer":
        return backbone_config["transformer"]["latent_size"]
    else:
        return backbone_config["feature_size"]
