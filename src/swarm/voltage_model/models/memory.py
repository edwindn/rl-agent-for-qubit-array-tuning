"""
Temporal memory models for sequential decision making.

Contains:
- LSTM: Recurrent encoder with CNN tokenizer
- Transformer: Attention-based encoder with CNN tokenizer
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from ray.rllib.core.models.base import ENCODER_OUT, Encoder, tokenize
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

# Import custom transformer (non-causal, encoder-only)
from swarm.voltage_model.models.transformer import TransformerEncoder, TransformerEncoderLayer

if TYPE_CHECKING:
    from swarm.voltage_model.configs import LSTMConfig, TransformerConfig


# =============================================================================
# LSTM
# =============================================================================

class LSTM(TorchModel, Encoder):
    """LSTM encoder for sequential decision making.

    Architecture:
    1. CNN Tokenizer converts images to feature vectors
    2. Multi-layer LSTM processes sequence of tokens
    3. Returns output compressed state with history

    State shape: batch-first (B, num_layers, hidden_dim)
    """

    def __init__(self, config: "LSTMConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.config = config

        # Build CNN tokenizer backbone
        self.tokenizer = config.tokenizer_config.build()

        lstm_input_dims = self.tokenizer.output_dims
        assert len(lstm_input_dims) == 1, "CNN tokenizer should return a flat tensor"
        lstm_input_dim = lstm_input_dims[0]

        if self.config.store_voltages:
            lstm_input_dim += self.config.voltage_hidden_dim

        self.lstm = nn.LSTM(
            lstm_input_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            bias=config.use_bias,
        )

        if self.config.store_voltages:
            self.voltage_encoder = nn.Linear(1, config.voltage_hidden_dim)
        else:
            self.voltage_encoder = None

        self._output_dims = (config.hidden_dim,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def get_initial_state(self):
        """Return initial LSTM states (Ray pattern).

        States are returned batch-first for consistency with input format.
        Shape: (num_layers, hidden_dim) per batch element.
        """
        return {
            "h": torch.zeros(self.config.num_layers, self.config.hidden_dim),
            "c": torch.zeros(self.config.num_layers, self.config.hidden_dim),
        }

    def _forward(self, inputs, **kwargs):
        obs = inputs[Columns.OBS]

        # Embed image through CNN
        image_for_tokenizer = {Columns.OBS: obs["image"]}
        image_out = tokenize(self.tokenizer, image_for_tokenizer, framework="torch")

        if isinstance(obs, dict) and "voltage" in obs:
            if self.config.store_voltages:
                voltage = obs["voltage"]
            else:
                voltage = None
        else:
            if self.config.store_voltages:
                raise ValueError(f"LSTM config attempting to store voltage, but none found in observation: {inputs}")
            voltage = None

        if voltage is not None:
            voltage_shape = voltage.shape
            voltage_folded = voltage.reshape(-1, voltage_shape[-1])  # (B*T, 1)
            voltage_encoded = self.voltage_encoder(voltage_folded)  # (B*T, voltage_hidden_dim)
            voltage_out = voltage_encoded.reshape(voltage_shape[0], voltage_shape[1], -1)  # (B, T, voltage_hidden_dim)
        else:
            voltage_out = None

        prev_hidden_states = inputs[Columns.STATE_IN]

        out = torch.cat((image_out, voltage_out), dim=-1) if voltage_out is not None else image_out

        # RLlib stores states as (batch, num_layers, hidden_dim), but LSTM expects (num_layers, batch, hidden_dim)
        h = prev_hidden_states["h"].transpose(0, 1)
        c = prev_hidden_states["c"].transpose(0, 1)

        out, next_hidden_states = self.lstm(out, (h, c))

        # Swap dimensions back for RLlib
        next_hidden_states = {
            "h": next_hidden_states[0].transpose(0, 1),
            "c": next_hidden_states[1].transpose(0, 1)
        }

        outputs = {}
        outputs[ENCODER_OUT] = out
        outputs[Columns.STATE_OUT] = next_hidden_states

        return outputs


# =============================================================================
# Transformer
# =============================================================================

class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class CTLPEPositionalEncoding(nn.Module):
    """Continuous Time Linear Positional Embedding.

    From "CTLPE: Continuous Time Linear Positional Embedding for Irregular Time Series"
    Uses voltage values as continuous "time" to create positional embeddings.
    Formula: p(v) = slope * v + bias
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.slope = nn.Parameter(torch.randn(d_model))
        self.bias = nn.Parameter(torch.randn(d_model))

    def forward(self, x, voltages):
        """
        Args:
            x: Token embeddings (batch, seq_len, d_model)
            voltages: Voltage values (batch, seq_len)

        Returns:
            Token embeddings with CTLPE positional encoding added
        """
        pos_emb = voltages.unsqueeze(-1) * self.slope + self.bias
        return x + pos_emb


class Transformer(TorchModel, Encoder):
    """Transformer encoder for spatial attention in quantum device control.

    Architecture:
    1. CNN Tokenizer converts images to spatial feature tokens
    2. Linear projection to transformer dimension
    3. Positional encoding added to tokens
    4. Multi-layer transformer encoder (self-attention + FFN)
    5. Pooling across spatial dimension
    """

    def __init__(self, config: "TransformerConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Build CNN tokenizer backbone
        self.tokenizer = config.tokenizer_config.build()
        tokenizer_output_dim = self.tokenizer.output_dims[0]

        # Linear projection from tokenizer features to transformer dimension
        self.token_projection = nn.Linear(tokenizer_output_dim, config.latent_size)

        # Positional encoding
        if config.use_ctlpe:
            self.pos_encoder = CTLPEPositionalEncoding(config.latent_size)
        else:
            self.pos_encoder = None

        # Sinusoidal positional embeddings
        if config.use_pos_embeddings:
            sinusoids = self._get_sinusoids(config.max_seq_len, config.latent_size)
            self.register_buffer('sinusoidal_embeddings', sinusoids)

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=config.latent_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.latent_size)
        )

        self._output_dims = (config.latent_size,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _get_sinusoids(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(np.log(10000.0) / d_model)
        )

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]

        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError(f"Transformer expects dict input with 'image' key, got {type(inputs)}")

        images = inputs["image"]
        voltages = inputs["voltage"]
        attention_mask = inputs["attention_mask"]

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(voltages, np.ndarray):
            voltages = torch.from_numpy(voltages)
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask)

        # Move tensors to model's device
        device = next(self.parameters()).device
        images = images.to(device)
        voltages = voltages.to(device)
        attention_mask = attention_mask.to(device)

        if images.dim() == 5:  # (B, T, H, W, C)
            batch_size, seq_len, h, w, c = images.shape
        elif images.dim() == 4:  # (T, H, W, C)
            seq_len, h, w, c = images.shape
            batch_size = 1
            images = images.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")

        if voltages.dim() == 1:
            voltages = voltages.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Process each frame through CNN
        token_list = []
        for t in range(seq_len):
            frame_t = images[:, t, :, :, :]
            tokenizer_out = self.tokenizer._forward(frame_t)
            tokens_t = tokenizer_out[ENCODER_OUT]
            token_list.append(tokens_t)

        tokens = torch.stack(token_list, dim=1)

        # Project tokens to transformer dimension
        tokens = self.token_projection(tokens)

        # Add positional encodings
        if self.config.use_ctlpe:
            tokens = self.pos_encoder(tokens, voltages)

        if self.config.use_pos_embeddings:
            tokens = tokens + self.sinusoidal_embeddings[:seq_len, :].unsqueeze(0)

        # Apply transformer encoder with attention mask
        attention_mask = attention_mask.bool().to(tokens.device)
        transformed = self.transformer(tokens, src_key_padding_mask=attention_mask)

        # Pool across sequence dimension
        if self.config.pooling_mode == "mean":
            output = transformed.mean(dim=1)
        elif self.config.pooling_mode == "max":
            output = transformed.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling_mode: {self.config.pooling_mode}")

        return {ENCODER_OUT: output}
