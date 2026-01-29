"""
Policy and value heads for quantum device RL agents.

Contains:
- PolicyHead: Outputs action distribution parameters (mean + log_std)
- ValueHead: Outputs scalar value estimate (for PPO)
- QValueHead: Q-function head for SAC (takes [encoder_features, action] tensor)
"""

from typing import TYPE_CHECKING, Tuple

from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

if TYPE_CHECKING:
    from swarm.voltage_model.configs import (
        PolicyHeadConfig,
        ValueHeadConfig,
        QValueHeadConfig,
        DeterministicPolicyHeadConfig,
    )


# =============================================================================
# Policy Head
# =============================================================================

class PolicyHead(TorchModel):
    """Policy head for quantum device control with optional attention."""

    def __init__(self, config: "PolicyHeadConfig"):
        super().__init__(config)

        self.config = config

        voltage_embedding_dim = 16  # Embedding dim per voltage input
        voltage_dim = getattr(config, 'voltage_dim', 1)  # Number of voltage inputs

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.voltage_layer = nn.Linear(voltage_dim, voltage_embedding_dim)

        self.final_layer = nn.Linear(in_dim + voltage_embedding_dim, config.output_layer_dim)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )

        self._output_dims = (config.output_layer_dim,)

        # Store log_std bounds for clamping (replicate Ray's TorchMLPHead implementation)
        # Ray uses symmetric clamping but we support asymmetric bounds
        if config.log_std_bounds is not None:
            self.log_std_min = torch.Tensor([config.log_std_bounds[0]])
            self.log_std_max = torch.Tensor([config.log_std_bounds[1]])
            # Register buffers to handle device mapping (same as Ray does)
            self.register_buffer("log_std_min_const", self.log_std_min)
            self.register_buffer("log_std_max_const", self.log_std_max)
            self.clip_log_std = True
        else:
            self.clip_log_std = False

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        voltage = inputs["voltage"]
        inputs = inputs["image_features"]

        if self.config.use_attention:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)

        x = self.mlp(inputs)

        voltage_features = self.voltage_layer(voltage)

        # Handle both 2D (B, features) and 3D (B, T, features) inputs
        # For 2D: concat on dim=1, for 3D: concat on dim=2 (feature dimension)
        concat_dim = -1  # Use -1 to always concat on last dimension (features)
        x = torch.cat((x, voltage_features), dim=concat_dim)

        # Apply log_std clipping if enabled (replicate Ray's TorchMLPHead behavior)
        # See: ray/rllib/core/models/torch/heads.py TorchMLPHead._forward()
        if self.clip_log_std:
            # Forward pass
            output = self.final_layer(x)
            # Split into means and log_stds (output_dim is 2*action_dim)
            means, log_stds = torch.chunk(output, chunks=2, dim=-1)
            # Clip the log standard deviations
            log_stds = torch.clamp(
                log_stds, self.log_std_min_const, self.log_std_max_const
            )
            return torch.cat((means, log_stds), dim=-1)
        else:
            # No clipping - just return raw output
            return self.final_layer(x)


# =============================================================================
# Value Head
# =============================================================================

class ValueHead(TorchModel):
    """Value head for quantum device RL with optional attention."""

    def __init__(self, config: "ValueHeadConfig"):
        super().__init__(config)

        self.config = config

        voltage_embedding_dim = 16  # Embedding dim per voltage input
        voltage_dim = getattr(config, 'voltage_dim', 1)  # Number of voltage inputs

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.voltage_layer = nn.Linear(voltage_dim, voltage_embedding_dim)

        self.final_layer = nn.Linear(in_dim + voltage_embedding_dim, 1)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )

        self._output_dims = (1,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        voltage = inputs["voltage"]
        inputs = inputs["image_features"]

        if self.config.use_attention:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)

        x = self.mlp(inputs)

        voltage_features = self.voltage_layer(voltage)

        # Handle both 2D (B, features) and 3D (B, T, features) inputs
        # For 2D: concat on dim=1, for 3D: concat on dim=2 (feature dimension)
        concat_dim = -1  # Use -1 to always concat on last dimension (features)
        x = torch.cat((x, voltage_features), dim=concat_dim)

        return self.final_layer(x)


# =============================================================================
# Q-Value Head (for SAC)
# =============================================================================

class QValueHead(TorchModel):
    """Q-function head for SAC.

    Takes dict input with image_features, voltage, and action.
    Processes voltage the same way as ValueHead for consistency.
    """

    def __init__(self, config: "QValueHeadConfig"):
        super().__init__(config)

        self.config = config

        voltage_embedding_dim = 16  # VOLTAGE DIM HARDCODED FOR NOW (same as other heads)
        voltage_dim = getattr(config, "voltage_dim", 1)

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.voltage_layer = nn.Linear(voltage_dim, voltage_embedding_dim)

        # Final layer: mlp_output + voltage_embedding + action
        self.final_layer = nn.Linear(in_dim + voltage_embedding_dim + config.action_dim, 1)

        self._output_dims = (1,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        image_features = inputs["image_features"]
        voltage = inputs["voltage"]
        action = inputs["action"]

        x = self.mlp(image_features)

        voltage_features = self.voltage_layer(voltage)
        x = torch.cat((x, voltage_features, action), dim=-1)

        return self.final_layer(x)


# =============================================================================
# Deterministic Policy Head (for TD3)
# =============================================================================

class DeterministicPolicyHead(TorchModel):
    """Deterministic policy head for TD3.

    Outputs action directly with tanh activation (bounded to [-1, 1]).
    Unlike PolicyHead, does NOT output log_std - just the action.
    """

    def __init__(self, config: "DeterministicPolicyHeadConfig"):
        super().__init__(config)

        self.config = config

        voltage_embedding_dim = 16  # VOLTAGE DIM HARDCODED FOR NOW

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.voltage_layer = nn.Linear(1, voltage_embedding_dim)

        self.final_layer = nn.Linear(in_dim + voltage_embedding_dim, config.output_layer_dim)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )

        self._output_dims = (config.output_layer_dim,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        voltage = inputs["voltage"]
        inputs = inputs["image_features"]

        if self.config.use_attention:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)

        x = self.mlp(inputs)

        voltage_features = self.voltage_layer(voltage)
        x = torch.cat((x, voltage_features), dim=1)

        # Apply tanh to bound action to [-1, 1]
        return torch.tanh(self.final_layer(x))
