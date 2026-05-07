"""
Temporal memory models for sequential decision making.

Contains:
- LSTM: Recurrent encoder with CNN tokenizer
- Transformer: Set-based attention encoder with CNN tokenizer
"""

import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ray.rllib.core.models.base import ENCODER_OUT, Encoder, tokenize
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

# Import custom transformer (non-causal, encoder-only)
from qadapt.voltage_model.models.transformer import TransformerEncoder, TransformerEncoderLayer

if TYPE_CHECKING:
    from qadapt.voltage_model.configs import LSTMConfig, TransformerConfig


# =============================================================================
# Voltage Encoding
# =============================================================================

class FourierFeatures(nn.Module):
    """Fourier feature encoding for continuous values.

    Encodes continuous values (like voltage) using sinusoidal features
    at multiple frequencies, providing a smooth representation that
    neural networks can easily interpolate.

    Args:
        num_frequencies: Number of frequency bands
        learnable: If True, frequencies are learned; else fixed exponential
    """

    def __init__(self, num_frequencies: int = 8, learnable: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        if learnable:
            # Initialize with small random values for stable training
            self.frequencies = nn.Parameter(torch.randn(num_frequencies) * 0.1)
        else:
            # Fixed exponential frequencies like in NeRF
            self.register_buffer('frequencies', 2.0 ** torch.arange(num_frequencies).float())
        self.output_dim = num_frequencies * 2  # sin + cos for each frequency

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T) or (B, T, 1)

        Returns:
            Fourier features of shape (B, T, num_frequencies * 2)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        # Compute frequencies: x * freq * pi
        freqs = x * self.frequencies * math.pi  # (B, T, num_freq)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


# =============================================================================
# Attention Pooling
# =============================================================================

class AttentionPooling(nn.Module):
    """Aggregate a set of tokens into a single vector via attention.

    Uses a learnable query to attend over all tokens, producing a
    weighted combination that serves as the set representation.

    Args:
        d_model: Dimension of input tokens
        num_heads: Number of attention heads
    """

    def __init__(self, d_model: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, tokens, key_padding_mask=None):
        """
        Args:
            tokens: Input tokens of shape (B, T, D)
            key_padding_mask: Boolean mask where True = padding (B, T)

        Returns:
            Pooled output of shape (B, D)
        """
        B = tokens.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        output, _ = self.attention(query, tokens, tokens, key_padding_mask=key_padding_mask)
        return output.squeeze(1)  # (B, D)


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

        # Extract image and voltage from obs dict
        # obs is expected to be a dict with 'image' and 'voltage' keys
        # image shape: (B, T, H, W, C)
        images = obs["image"]
        voltages = obs["voltage"]

        # Get batch and sequence dimensions
        if images.dim() == 5:  # (B, T, H, W, C)
            batch_size, seq_len, h, w, c = images.shape
        elif images.dim() == 4:  # (T, H, W, C) - add batch dim
            seq_len, h, w, c = images.shape
            batch_size = 1
            images = images.unsqueeze(0)
            voltages = voltages.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}")

        # Process each timestep through the CNN tokenizer
        image_feature_list = []
        for t in range(seq_len):
            frame_t = images[:, t, :, :, :]  # (B, H, W, C)
            voltage_t = voltages[:, t:t+1]   # (B, 1)

            tokenizer_input = {
                "image": frame_t,
                "voltage": voltage_t,
            }
            tokenizer_out = self.tokenizer._forward(tokenizer_input)
            image_feat_t = tokenizer_out[ENCODER_OUT]["image_features"]
            image_feature_list.append(image_feat_t)

        # Stack features: (B, T, feature_dim)
        image_out = torch.stack(image_feature_list, dim=1)

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

        lstm_out, next_hidden_states = self.lstm(out, (h, c))

        # Swap dimensions back for RLlib
        next_hidden_states = {
            "h": next_hidden_states[0].transpose(0, 1),
            "c": next_hidden_states[1].transpose(0, 1)
        }

        # Return dict structure matching SimpleCNN output format
        # Policy/value heads expect {"image_features": ..., "voltage": ...}
        encoder_out = {
            "image_features": lstm_out,
            "voltage": voltages,  # Pass through original voltages for heads
        }

        outputs = {}
        outputs[ENCODER_OUT] = encoder_out
        outputs[Columns.STATE_OUT] = next_hidden_states

        return outputs


# =============================================================================
# Transformer
# =============================================================================

class Transformer(TorchModel, Encoder):
    """Transformer encoder for set-based attention over voltage-image observations.

    Treats observations as an UNORDERED SET of (voltage, image) pairs.
    Uses self-attention to learn relationships between observations at
    different voltage points, then pools via attention to produce output.

    Architecture:
    1. CNN Tokenizer converts images to feature vectors
    2. Fourier features encode voltage as equal partner
    3. Fused (image, voltage) tokens projected to transformer dimension
    4. Self-attention over all tokens (NO positional encoding)
    5. Attention pooling aggregates set into single output vector
    """

    def __init__(self, config: "TransformerConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Build CNN tokenizer backbone
        self.tokenizer = config.tokenizer_config.build()
        tokenizer_output_dim = self.tokenizer.output_dims[0]

        # Voltage encoder using Fourier features
        self.voltage_encoder = FourierFeatures(
            num_frequencies=config.voltage_num_frequencies,
            learnable=config.voltage_learnable_frequencies,
        )
        voltage_feature_dim = self.voltage_encoder.output_dim

        # Fuse image + voltage features, then project to transformer dimension
        fused_dim = tokenizer_output_dim + voltage_feature_dim
        self.token_projection = nn.Linear(fused_dim, config.latent_size)

        # Transformer encoder layers (self-attention over the set)
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

        # Pooling mechanism
        if config.pooling_mode == "attention":
            self.attention_pool = AttentionPooling(
                d_model=config.latent_size,
                num_heads=1,
            )
        else:
            self.attention_pool = None

        self._output_dims = (config.latent_size,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

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

        # Encode images through CNN tokenizer
        # Backbone expects dict with 'image' and 'voltage' keys
        # Backbone returns {ENCODER_OUT: {"image_features": tensor, "voltage": tensor}}
        image_feature_list = []
        for t in range(seq_len):
            frame_t = images[:, t, :, :, :]  # (B, H, W, C)
            voltage_t = voltages[:, t:t+1]   # (B, 1) - keep dim for backbone

            tokenizer_input = {
                "image": frame_t,
                "voltage": voltage_t,
            }
            tokenizer_out = self.tokenizer._forward(tokenizer_input)

            # Extract image features from backbone output dict
            image_feat_t = tokenizer_out[ENCODER_OUT]["image_features"]
            image_feature_list.append(image_feat_t)

        image_features = torch.stack(image_feature_list, dim=1)  # (B, T, D_img)

        # Encode voltages through Fourier features
        voltage_features = self.voltage_encoder(voltages)  # (B, T, D_voltage)

        # Fuse image + voltage as equal partners and project
        combined = torch.cat([image_features, voltage_features], dim=-1)  # (B, T, D_img + D_voltage)
        tokens = self.token_projection(combined)  # (B, T, latent_size)

        # Self-attention over the set (NO positional encoding - it's a set!)
        mask = attention_mask.bool().to(tokens.device)
        tokens = self.transformer(tokens, src_key_padding_mask=mask)

        # Pool to single output vector
        if self.config.pooling_mode == "attention":
            output = self.attention_pool(tokens, key_padding_mask=mask)
        elif self.config.pooling_mode == "mean":
            # Mask out padding for mean
            mask_expanded = (~mask).unsqueeze(-1).float()  # (B, T, 1), True for valid
            output = (tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.config.pooling_mode == "max":
            # Mask out padding for max
            tokens_masked = tokens.masked_fill(mask.unsqueeze(-1), float('-inf'))
            output = tokens_masked.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling_mode: {self.config.pooling_mode}")

        # Return in format expected by heads (dict with "image_features" and "voltage")
        # Note: voltage info is already fused into output, but heads expect this format
        return {ENCODER_OUT: {
            "image_features": output,
            "voltage": voltages[:, -1:],  # Last voltage for compatibility with heads
        }}
