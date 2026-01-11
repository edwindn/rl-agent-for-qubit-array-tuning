"""Configuration classes for temporal memory models."""

from dataclasses import dataclass
from typing import Optional

from ray.rllib.core.models.configs import CNNEncoderConfig, ModelConfig, RecurrentEncoderConfig


@dataclass
class LSTMConfig(RecurrentEncoderConfig):
    """LSTM recurrent encoder configuration.

    Wraps a CNN tokenizer and adds temporal processing through LSTM layers.

    Args:
        tokenizer_config: Configuration object for the CNN backbone
        hidden_dim: Size of LSTM hidden and cell states
        num_layers: Number of stacked LSTM layers
        max_seq_len: Maximum sequence length for padding/truncation
        store_voltages: Whether to store voltage information
        voltage_hidden_dim: Dimension for voltage embedding
    """

    tokenizer_config: Optional[CNNEncoderConfig] = None
    hidden_dim: int = 256
    voltage_hidden_dim: int = 16
    num_layers: int = 1
    max_seq_len: int = 50
    store_voltages: bool = True
    use_bias: bool = True

    def __post_init__(self):
        if self.tokenizer_config is None:
            raise ValueError("tokenizer_config must be provided")
        self.input_dims = self.tokenizer_config.output_dims

    @property
    def output_dims(self):
        return (self.hidden_dim,)

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.memory import LSTM
        return LSTM(self)


@dataclass
class TransformerConfig(ModelConfig):
    """Transformer encoder configuration.

    Wraps a CNN tokenizer and adds spatial attention through self-attention.

    Args:
        tokenizer_config: Configuration object for the CNN backbone
        latent_size: Output feature dimension
        num_attention_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        feedforward_dim: Hidden dimension of feedforward network
        dropout: Dropout probability
        pooling_mode: How to pool outputs ("mean" or "max")
        use_ctlpe: Whether to use CTLPE positional embedding
        use_pos_embeddings: Whether to use sinusoidal positional embeddings
    """

    tokenizer_config: Optional[CNNEncoderConfig] = None
    latent_size: int = 256
    num_attention_heads: int = 4
    num_layers: int = 2
    max_seq_len: int = 20
    feedforward_dim: Optional[int] = None
    dropout: float = 0.1
    pooling_mode: str = "mean"
    use_ctlpe: bool = False
    use_pos_embeddings: bool = False

    def __post_init__(self):
        if self.tokenizer_config is None:
            raise ValueError("tokenizer_config must be provided")

        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.latent_size

        if self.pooling_mode not in ["mean", "max"]:
            raise ValueError(f"pooling_mode must be 'mean' or 'max', got {self.pooling_mode}")

    @property
    def output_dims(self):
        return (self.latent_size,)

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.memory import Transformer
        return Transformer(self)
