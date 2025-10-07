"""Custom neural network components for quantum device RL agents."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path
import numpy as np

from ray.rllib.core.models.base import ENCODER_OUT, Encoder, tokenize
from ray.rllib.core.models.configs import CNNEncoderConfig, MLPHeadConfig, ModelConfig
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.columns import Columns
from ray.rllib.models.utils import get_initializer_fn
from ray.rllib.utils.framework import try_import_torch
import tree

torch, nn = try_import_torch()
import torch.nn.functional as F
import torchvision.models as models #for MobileNet


# Import custom transformer (non-causal, encoder-only)
# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))
from swarm.voltage_model.transformer import TransformerEncoder, TransformerEncoderLayer



@dataclass
class SimpleCNNConfig(CNNEncoderConfig):
    """CNN configuration for quantum charge stability diagrams with clean YAML interface."""
    
    conv_layers: Optional[List[Dict]] = None
    feature_size: int = 256
    adaptive_pooling: bool = True
    
    def __post_init__(self):
        if self.conv_layers:
            self.cnn_filter_specifiers = [
                [layer["channels"], [layer["kernel"], layer["kernel"]], layer["stride"]]
                for layer in self.conv_layers
            ]
        else:
            self.cnn_filter_specifiers = [
                [16, [4, 4], 2],
                [32, [3, 3], 2], 
                [64, [3, 3], 1],
            ]
    
    @property
    def output_dims(self):
        return (self.feature_size,)
    
    def build(self, framework: str = "torch") -> "SimpleCNN":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return SimpleCNN(self)


class SimpleCNN(TorchModel, Encoder):
    """CNN encoder for quantum charge stability diagrams."""
    
    def __init__(self, config: SimpleCNNConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        self.config = config
        
        cnn_layers = []
        in_channels = config.input_dims[-1]
        
        for out_channels, kernel_size, stride in config.cnn_filter_specifiers:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
                nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            ])
            in_channels = out_channels
        
        if config.adaptive_pooling:
            cnn_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        self._calculate_cnn_output_size()
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, config.feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )
        
        self._output_dims = (config.feature_size,)
    
    def _calculate_cnn_output_size(self):
        h, w, c = self.config.input_dims
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self._cnn_output_size = cnn_output.shape[1]
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims
    
    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]
        
        if isinstance(inputs, dict):
            if "image" in inputs:
                x = inputs["image"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            x = inputs

        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)
        
        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)
        
        return {ENCODER_OUT: output_features}


@dataclass
class PolicyHeadConfig(MLPHeadConfig):
    """Policy head configuration for quantum device control."""
    
    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    
    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 128]
        
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
    
    def build(self, framework: str = "torch") -> "PolicyHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return PolicyHead(self)


class PolicyHead(TorchModel):
    """Policy head for quantum device control with optional attention."""
    
    def __init__(self, config: PolicyHeadConfig):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims
        
        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.output_layer_dim))
        
        self.mlp = nn.Sequential(*layers)
        
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
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)
        
        x = self.mlp(inputs)
        return F.tanh(x)

@dataclass
class IMPALAConfig(CNNEncoderConfig):
    """IMPALA CNN configuration with ResNet blocks for quantum charge stability diagrams."""
    
    conv_layers: Optional[List[Dict]] = None
    feature_size: int = 256
    adaptive_pooling: bool = True
    num_res_blocks: int = 2
    
    def __post_init__(self):
        if self.conv_layers:
            self.cnn_filter_specifiers = [
                [layer["channels"], [layer["kernel"], layer["kernel"]], layer["stride"]]
                for layer in self.conv_layers
            ]
        else:
            # IMPALA default architecture
            self.cnn_filter_specifiers = [
                [16, [8, 8], 4],
                [32, [4, 4], 2], 
                [32, [3, 3], 1],
            ]
    
    @property
    def output_dims(self):
        return (self.feature_size,)
    
    def build(self, framework: str = "torch") -> "IMPALA":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return IMPALA(self)


class ResNetBlock(nn.Module):
    """ResNet block for IMPALA CNN."""
    
    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class IMPALA(TorchModel, Encoder):
    """IMPALA CNN encoder with ResNet blocks for quantum charge stability diagrams."""
    
    def __init__(self, config: IMPALAConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        self.config = config
        
        # Build initial conv layers
        cnn_layers = []
        in_channels = config.input_dims[-1]
        
        for i, (out_channels, kernel_size, stride) in enumerate(config.cnn_filter_specifiers):
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.MaxPool2d(stride, stride) if stride > 1 else nn.Identity(),
                nn.ReLU()
            ])
            

            for _ in range(config.num_res_blocks):
                cnn_layers.append(ResNetBlock(out_channels, config.cnn_activation))
            
            in_channels = out_channels
        
        if config.adaptive_pooling:
            cnn_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        self._calculate_cnn_output_size()
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, config.feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )
        
        self._output_dims = (config.feature_size,)
    
    def _calculate_cnn_output_size(self):
        h, w, c = self.config.input_dims
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self._cnn_output_size = cnn_output.shape[1]
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims
    
    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]
        
        if isinstance(inputs, dict):
            if "image" in inputs:
                x = inputs["image"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            x = inputs
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)
        
        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)
        
        return {ENCODER_OUT: output_features}


@dataclass
class ValueHeadConfig(MLPHeadConfig):
    """Value head configuration for quantum device RL."""
    
    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    
    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 64]
        
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = 1
    
    def build(self, framework: str = "torch") -> "ValueHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return ValueHead(self)


class ValueHead(TorchModel):
    """Value head for quantum device RL with optional attention mechanism."""
    
    def __init__(self, config: ValueHeadConfig):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims
        
        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
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
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)
        
        return self.mlp(inputs)


@dataclass
class MobileNetConfig(CNNEncoderConfig):
    """MobileNet configuration for quantum charge stability diagrams with pretrained backbone."""

    mobilenet_version: str = "small"  # "small" or "large"
    feature_size: int = 256
    freeze_backbone: bool = False
    load_pretrained: bool = True

    def __post_init__(self):
        # Set the feature dimensions based on MobileNet version
        if self.mobilenet_version == "small":
            self._backbone_feature_dim = 576
        elif self.mobilenet_version == "large":
            self._backbone_feature_dim = 960
        else:
            raise ValueError(f"Unsupported MobileNet version: {self.mobilenet_version}. Use 'small' or 'large'.")

    @property
    def output_dims(self):
        return (self.feature_size,)

    def build(self, framework: str = "torch") -> "MobileNet":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        if models is None:
            raise ImportError("torchvision is required for MobileNet backbone")
        return MobileNet(self)


class MobileNet(TorchModel, Encoder):
    """MobileNet encoder for quantum charge stability diagrams using pretrained backbone."""

    def __init__(self, config: MobileNetConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Load MobileNet backbone (with or without pretrained weights)
        if config.mobilenet_version == "small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if config.load_pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            feature_dim = 576
        elif config.mobilenet_version == "large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if config.load_pretrained else None
            self.backbone = models.mobilenet_v3_large(weights=weights)
            feature_dim = 960
        else:
            raise ValueError(f"Unsupported MobileNet version: {config.mobilenet_version}")

        # Modify first conv layer to accept the correct number of input channels
        input_channels = config.input_dims[-1]
        original_conv1 = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # Initialize new conv1 weights
        if config.load_pretrained:
            with torch.no_grad():
                if input_channels <= 3:
                    # Use subset of original weights if we have fewer channels
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv1.weight[:, :input_channels, :, :].clone()
                    )
                else:
                    # Repeat channels if we need more than 3
                    weight = original_conv1.weight
                    repeats = (input_channels + 2) // 3  # Ceiling division
                    repeated_weight = weight.repeat(1, repeats, 1, 1)
                    self.backbone.features[0][0].weight = nn.Parameter(
                        repeated_weight[:, :input_channels, :, :].clone()
                    )

        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add final projection layer to match desired feature size
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, config.feature_size),
            nn.ReLU(),
        )

        self._output_dims = (config.feature_size,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]

        if isinstance(inputs, dict):
            if "image" in inputs:
                x = inputs["image"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            x = inputs

        if x.dim() == 3:
            x = x.unsqueeze(0)

        # MobileNet expects channel-first format (B, C, H, W)
        if x.shape[-1] <= 8:  # Assume last dim is channels if small
            x = x.permute(0, 3, 1, 2)

        # Extract features using MobileNet backbone
        backbone_features = self.backbone(x)

        # Project to desired feature size
        output_features = self.projection(backbone_features)

        return {ENCODER_OUT: output_features}



@dataclass
class LSTMConfig(RecurrentEncoderConfig):
    """LSTM recurrent encoder configuration for quantum charge stability diagrams.

    Similar to TransformerConfig, wraps a CNN tokenizer and adds temporal processing
    through LSTM layers for sequential decision making.

    Args:
        tokenizer_config: Configuration object for the CNN backbone
        cell_size: Size of LSTM hidden and cell states (hidden_dim in Ray's config)
        num_layers: Number of stacked LSTM layers
        max_seq_len: Maximum sequence length for padding/truncation
        store_voltages: Whether to store voltage information (custom feature)
        batch_major: Whether input is batch-first (B, T, ...) or time-first (T, B, ...)
        use_bias: Whether to use bias in LSTM layers
        hidden_weights_initializer: Initializer for LSTM weights
        hidden_weights_initializer_config: Config dict for weight initializer
        hidden_bias_initializer: Initializer for LSTM biases
        hidden_bias_initializer_config: Config dict for bias initializer
    """

    tokenizer_config: Optional[CNNEncoderConfig] = None
    hidden_dim: int = 256
    voltage_hidden_dim: int = 16
    num_layers: int = 1
    max_seq_len: int = 50
    store_voltages: bool = True
    batch_first: bool = True
    use_bias: bool = True

    # hidden_weights_initializer: Optional[Union[str, Callable]] = None
    # hidden_weights_initializer_config: Optional[Dict] = None
    # hidden_bias_initializer: Optional[Union[str, Callable]] = None
    # hidden_bias_initializer_config: Optional[Dict] = None

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
        return LSTM(self)

class LSTM(TorchModel, Encoder):
    """
    1. CNN Tokenizer converts images to feature vectors
    2. Multi-layer LSTM processes sequence of tokens
    3. Returns output compressed state with history

    -State shape: batch-first (B, num_layers, hidden_dim)
    """

    def __init__(self, config: LSTMConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.config = config

        # Build CNN tokenizer backbone
        self.tokenizer = config.tokenizer_config.build()

        lstm_input_dims = self.tokenizer.output_dims
        assert len(lstm_input_dims) == 1, "CNN tokenizer should return a flat tensor"
        lstm_input_dim = lstm_input_dims[0]

        # Get initializer functions
        # lstm_weights_initializer = get_initializer_fn(
        #     config.hidden_weights_initializer, framework="torch"
        # )
        # lstm_bias_initializer = get_initializer_fn(
        #     config.hidden_bias_initializer, framework="torch"
        # )

        self.lstm = nn.LSTM(
            lstm_input_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=config.batch_first,
            bias=config.use_bias,
        )

        if self.config.store_voltage:
            self.voltage_encoder = nn.Linear(1, config.voltage_hidden_dim)

        # Initialize LSTM layer weights and biases
        # for layer in self.lstm.all_weights:
        #     if lstm_weights_initializer:
        #         lstm_weights_initializer(
        #             layer[0], **config.hidden_weights_initializer_config or {}
        #         )
        #         lstm_weights_initializer(
        #             layer[1], **config.hidden_weights_initializer_config or {}
        #         )
        #     if lstm_bias_initializer:
        #         lstm_bias_initializer(
        #             layer[2], **config.hidden_bias_initializer_config or {}
        #         )
        #         lstm_bias_initializer(
        #             layer[3], **config.hidden_bias_initializer_config or {}
        #         )

        self._output_dims = (config.cell_size,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def get_initial_state(self):
        """Return initial LSTM states (Ray pattern).

        States are returned batch-first for consistency with input format.
        Shape: (num_layers, hidden_dim) per batch element.
        """
        return {
            "h": torch.zeros(self.config.num_layers, self.config.cell_size),
            "c": torch.zeros(self.config.num_layers, self.config.cell_size),
        }

    def _forward(self, inputs, **kwargs):
        """Forward pass through LSTM encoder (following Ray's TorchLSTMEncoder pattern).

        Args:
            inputs: Dict with Columns.OBS key containing observation dict with "image" key,
                   OR direct observation dict with "image" key.
                   This is a SINGLE observation, not a sequence.

        Returns:
            Dict with ENCODER_OUT and STATE_OUT keys.
        """

        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]

        # Embed image through CNN
        image_out = self.tokenizer._forward(inputs)[ENCODER_OUT]
        
        if isinstance(inputs, dict) and "voltage" in inputs:
            if self.config.store_voltage:
                voltage = inputs["voltage"]# might need to reshape / unsqueeze
            else:
                voltage = None
        else:
            if self.config.store_voltage:
                raise ValueError(f"LSTM config attempting to store voltage, but none found in observation: {inputs}")
            voltage = None

        if voltage is not None:
            voltage_out = self.voltage_encoder(voltage)
        else:
            voltage_out = None


        # Get states from kwargs (Ray pattern)
        if Columns.STATE_IN in kwargs:
            states_in = kwargs[Columns.STATE_IN]
        else:
            # Initialize states if not provided
            batch_size = out.shape[0] if out.dim() > 1 else 1
            states_in = self.get_initial_state()
            # Repeat across batch
            states_in = tree.map_structure(
                lambda s: s.unsqueeze(0).repeat(batch_size, 1, 1), states_in
            )

        # Ensure proper shape for LSTM input
        if out.dim() == 2:
            dim_to_unsqueeze = 1 if self.config.batch_first else 0
            out = out.unsqueeze(dim_to_unsqueeze)
            # (batch, features) -> (batch, 1, features) if batch first
            # (batch, features) -> (1, batch, features) if seq first

        # Push through LSTM
        # states_in are always (num_layers, batch, hidden_dim)
        # states_out will also be (num_layers, batch, hidden_dim)
        out, states_out = self.lstm(out, (states_in["h"], states_in["c"]))
        states_out = {"h": states_out[0], "c": states_out[1]}

        # Remove sequence dimension from output
        if self.config.batch_first and out.shape[1] == 1:
            out = out.squeeze(1)  # (batch, 1, features) -> (batch, features)
        elif not self.config.batch_first and out.shape[0] == 1:
            out = out.squeeze(0)  # (1, batch, features) -> (batch, features)

        # Insert them into the output dict
        outputs = {}
        outputs[ENCODER_OUT] = out
        outputs[Columns.STATE_OUT] = states_out

        return outputs


@dataclass
class TransformerConfig(ModelConfig):
    """Transformer encoder configuration for quantum charge stability diagrams.

    Wraps a CNN tokenizer (SimpleCNN, IMPALA, or MobileNet) and adds
    spatial attention through self-attention mechanisms.

    Args:
        tokenizer_config: Configuration object for the CNN backbone
        latent_size: Output feature dimension
        num_attention_heads: Number of attention heads for multi-head attention
        num_layers: Number of transformer encoder layers
        feedforward_dim: Hidden dimension of feedforward network (default: 4 * latent_size)
        dropout: Dropout probability
        pooling_mode: How to pool transformer outputs ("mean" or "max")
        use_ctlpe: Whether to use CTLPE (Continuous Time Linear Positional Embedding)
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

    def build(self, framework: str = "torch") -> "Transformer":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return Transformer(self)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding for transformer."""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pos_embedding[:, :x.size(1), :]


class CTLPEPositionalEncoding(nn.Module):
    """Continuous Time Linear Positional Embedding.

    From "CTLPE: Continuous Time Linear Positional Embedding for Irregular Time Series"
    (https://arxiv.org/abs/2409.20092)

    Uses voltage values as continuous "time" to create positional embeddings.
    Formula: p(v) = slope * v + bias
    where slope and bias are learnable per-dimension parameters.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Learnable slope and bias for each embedding dimension
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
        # Compute positional embedding: p(v) = slope * v + bias
        # Broadcasting: (batch, seq_len, 1) * (d_model,) + (d_model,)
        pos_emb = voltages.unsqueeze(-1) * self.slope + self.bias  # (batch, seq_len, d_model)
        return x + pos_emb


class Transformer(TorchModel, Encoder):
    """Transformer encoder for spatial attention in quantum device control.

    Architecture:
    1. CNN Tokenizer converts images to spatial feature tokens
    2. Linear projection to transformer dimension
    3. Positional encoding added to tokens
    4. Multi-layer transformer encoder (self-attention + FFN)
    5. Pooling across spatial dimension
    6. Output features for policy/value heads
    """

    def __init__(self, config: TransformerConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Build CNN tokenizer backbone
        self.tokenizer = config.tokenizer_config.build()
        tokenizer_output_dim = self.tokenizer.output_dims[0]

        # Linear projection from tokenizer features to transformer dimension
        self.token_projection = nn.Linear(tokenizer_output_dim, config.latent_size)

        # Positional encoding - CTLPE for voltage-based embeddings
        if config.use_ctlpe:
            # CTLPE uses voltage values directly, no need for max_seq_len
            self.pos_encoder = CTLPEPositionalEncoding(config.latent_size)
        else:
            self.pos_encoder = None

        # Sinusoidal positional embeddings
        if config.use_pos_embeddings:
            # Generate sinusoidal embeddings and register as buffer (not trainable)
            sinusoids = self._get_sinusoids(config.max_seq_len, config.latent_size)
            self.register_buffer('sinusoidal_embeddings', sinusoids)

        # Transformer encoder layers (using custom non-causal implementation from transformer.py)
        encoder_layer = TransformerEncoderLayer(
            d_model=config.latent_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better stability
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
        """Generate sinusoidal positional embeddings.

        Args:
            seq_len: Maximum sequence length
            d_model: Model dimension (embedding size)

        Returns:
            Tensor of shape (seq_len, d_model) with sinusoidal positional embeddings
        """
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)

        # Compute the div_term: 1 / (10000^(2i/d_model)) for i in [0, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(np.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Initialize positional encoding tensor
        pe = torch.zeros(seq_len, d_model)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _pad_or_truncate(self, inputs):
        num_frames = self.config.max_seq_len
        if len(inputs) >= num_frames:
            # False = don't ignore, True = padding
            attn_mask = torch.zeros(num_frames, dtype=torch.bool)
            return attn_mask, inputs[-num_frames:]
        else:
            needed = num_frames - len(inputs)
            attn_mask = torch.tensor([False]*len(inputs) + [True]*needed, dtype=torch.bool)
            padding = [None] * needed
            return attn_mask, inputs + padding

    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]

        # Debug: Print input format to understand frame stacking shape (will be removed after testing)
        if isinstance(inputs, dict) and "image" in inputs:
            img_shape = inputs['image'].shape if hasattr(inputs['image'], 'shape') else type(inputs['image'])
            volt_shape = inputs['voltage'].shape if hasattr(inputs['voltage'], 'shape') else type(inputs['voltage'])
            print(f"[Transformer Debug] Image shape: {img_shape}, Voltage shape: {volt_shape}")

        # Handle frame-stacked observations from RLlib's FrameStackingEnvToModule
        # Expected format: {image: (B, T, H, W, C), voltage: (B, T, 1)} where T = num_frames
        if not isinstance(inputs, dict) or "image" not in inputs:
            raise ValueError(f"Transformer expects dict input with 'image' key, got {type(inputs)}")

        images = inputs["image"]
        voltages = inputs["voltage"]

        # Convert to torch tensors if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(voltages, np.ndarray):
            voltages = torch.from_numpy(voltages)

        # Check if we have a time dimension (frame stacking enabled)
        if images.dim() == 5:  # (B, T, H, W, C) - frame stacked
            batch_size, seq_len, h, w, c = images.shape
            # Reshape to (B*T, H, W, C) for CNN processing
            images = images.reshape(batch_size * seq_len, h, w, c)
            voltages = voltages.reshape(batch_size * seq_len, -1)

            # No attention mask needed - all frames are valid
            device = images.device
            attention_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

        elif images.dim() == 4:  # (B, H, W, C) - no frame stacking (shouldn't happen with frame stacking enabled)
            # Add sequence dimension for consistency
            batch_size = images.shape[0]
            seq_len = 1
            images = images.reshape(batch_size * seq_len, *images.shape[1:])
            voltages = voltages.reshape(batch_size * seq_len, -1)
            device = images.device
            attention_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        else:
            raise ValueError(f"Unexpected image tensor shape: {images.shape}. Expected (B, H, W, C) or (B, T, H, W, C)")

        # Get tokenizer features
        tokenizer_out = self.tokenizer._forward(images)
        tokens = tokenizer_out[ENCODER_OUT]  # (B*T, feature_dim)

        # Reshape tokens back to (B, T, feature_dim)
        tokens = tokens.view(batch_size, seq_len, -1)

        # Project tokens to transformer dimension
        tokens = self.token_projection(tokens)  # (B, T, latent_size)

        # Add positional encodings
        if self.config.use_ctlpe:
            # CTLPE requires voltage values for positional encoding
            # Reshape voltages back to (B, T) for CTLPE
            voltages_2d = voltages.view(batch_size, seq_len)
            tokens = self.pos_encoder(tokens, voltages_2d)

        if self.config.use_pos_embeddings:
            # Add sinusoidal positional embeddings
            tokens = tokens + self.sinusoidal_embeddings[:seq_len, :].unsqueeze(0)

        # Apply transformer encoder
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)  # (B, T)
        transformed = self.transformer(tokens, src_key_padding_mask=attention_mask)  # (B, T, latent_size)

        # Pool across sequence dimension
        if self.config.pooling_mode == "mean":
            output = transformed.mean(dim=1)
        elif self.config.pooling_mode == "max":
            output = transformed.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling_mode: {self.config.pooling_mode}")

        return {ENCODER_OUT: output}


if __name__ == "__main__":
    """Print parameter counts for all network configurations."""
    import yaml
    from pathlib import Path
    
    def count_parameters(model):
        """Count trainable parameters in a PyTorch model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load training config
    config_path = Path(__file__).parent.parent / "training" / "training_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        neural_configs = config.get('neural_networks', {})
        
        print("\n" + "="*60)
        print("NETWORK PARAMETER COUNTS FROM CONFIGURATION")
        print("="*60)
        
        total_params = 0
        
        for policy_name, policy_config in neural_configs.items():
            print(f"\n{policy_name.upper()}:")
            policy_total = 0
            
            # Create backbone
            backbone_config = policy_config.get('backbone', {})
            backbone_type = backbone_config.get('type', 'SimpleCNN')
            
            # Typical input dimensions for charge stability diagrams
            input_dims = (64, 64, 1)  # height, width, channels
            
            if backbone_type == 'SimpleCNN':
                config_obj = SimpleCNNConfig(
                    input_dims=input_dims,
                    conv_layers=backbone_config.get('conv_layers'),
                    feature_size=backbone_config.get('feature_size', 256),
                    adaptive_pooling=backbone_config.get('adaptive_pooling', True),
                    cnn_activation="relu"
                )
                backbone = config_obj.build()
                
            elif backbone_type == 'IMPALA':
                config_obj = IMPALAConfig(
                    input_dims=input_dims,
                    conv_layers=backbone_config.get('conv_layers'),
                    feature_size=backbone_config.get('feature_size', 256),
                    adaptive_pooling=backbone_config.get('adaptive_pooling', True),
                    num_res_blocks=backbone_config.get('num_res_blocks', 2),
                    cnn_activation="relu"
                )
                backbone = config_obj.build()
            elif backbone_type == 'MobileNet':
                config_obj = MobileNetConfig(
                    input_dims=input_dims,
                    mobilenet_version=backbone_config.get('mobilenet_version', 'small'),
                    feature_size=backbone_config.get('feature_size', 256),
                    freeze_backbone=backbone_config.get('freeze_backbone', False),
                    load_pretrained=backbone_config.get('load_pretrained', True)
                )
                backbone = config_obj.build()
            else:
                print(f"  Unknown backbone type: {backbone_type}")
                continue
            
            backbone_params = count_parameters(backbone)
            print(f"  Backbone ({backbone_type}): {backbone_params:,} parameters")
            policy_total += backbone_params
            
            # Create policy head
            policy_head_config = policy_config.get('policy_head', {})
            policy_head_obj = PolicyHeadConfig(
                input_dims=(backbone_config.get('feature_size', 256),),
                output_layer_dim=2,  # mean + log_std for continuous actions
                hidden_layers=policy_head_config.get('hidden_layers', [128, 128]),
                activation=policy_head_config.get('activation', 'relu'),
                use_attention=policy_head_config.get('use_attention', False)
            )
            policy_head = policy_head_obj.build()
            policy_head_params = count_parameters(policy_head)
            print(f"  Policy Head: {policy_head_params:,} parameters")
            policy_total += policy_head_params
            
            # Create value head
            value_head_config = policy_config.get('value_head', {})
            value_head_obj = ValueHeadConfig(
                input_dims=(backbone_config.get('feature_size', 256),),
                hidden_layers=value_head_config.get('hidden_layers', [128, 64]),
                activation=value_head_config.get('activation', 'relu'),
                use_attention=value_head_config.get('use_attention', False)
            )
            value_head = value_head_obj.build()
            value_head_params = count_parameters(value_head)
            print(f"  Value Head: {value_head_params:,} parameters")
            policy_total += value_head_params
            
            print(f"  {policy_name} Total: {policy_total:,} parameters")
            total_params += policy_total
        
        print(f"\nGRAND TOTAL: {total_params:,} parameters")
        print("="*60)

        # Always print transformer parameter count from config
        print("\n" + "="*60)
        print("TRANSFORMER PARAMETER COUNT")
        print("="*60)

        input_dims = (64, 64, 1)

        # Look for transformer config in any policy
        transformer_found = False
        for policy_name, policy_config in neural_configs.items():
            backbone_config = policy_config.get('backbone', {})
            if backbone_config.get('memory_layer') == 'transformer':
                transformer_found = True
                transformer_dict = backbone_config.get('transformer', {})

                # Get backbone config for tokenizer
                backbone_type = backbone_config.get('type', 'SimpleCNN')

                # Create tokenizer config based on backbone type
                if backbone_type == 'SimpleCNN':
                    tokenizer_cfg = SimpleCNNConfig(
                        input_dims=input_dims,
                        conv_layers=backbone_config.get('conv_layers'),
                        feature_size=backbone_config.get('feature_size', 256),
                        adaptive_pooling=backbone_config.get('adaptive_pooling', True),
                        cnn_activation="relu"
                    )
                elif backbone_type == 'IMPALA':
                    tokenizer_cfg = IMPALAConfig(
                        input_dims=input_dims,
                        conv_layers=backbone_config.get('conv_layers'),
                        feature_size=backbone_config.get('feature_size', 256),
                        adaptive_pooling=backbone_config.get('adaptive_pooling', True),
                        num_res_blocks=backbone_config.get('num_res_blocks', 2),
                        cnn_activation="relu"
                    )
                elif backbone_type == 'MobileNet':
                    tokenizer_cfg = MobileNetConfig(
                        input_dims=input_dims,
                        mobilenet_version=backbone_config.get('mobilenet_version', 'small'),
                        feature_size=backbone_config.get('feature_size', 256),
                        freeze_backbone=backbone_config.get('freeze_backbone', False),
                        load_pretrained=backbone_config.get('load_pretrained', True)
                    )
                else:
                    print(f"Unknown backbone type for transformer tokenizer: {backbone_type}")
                    continue

                # Create transformer config from yaml
                transformer_config = TransformerConfig(
                    input_dims=input_dims,
                    tokenizer_config=tokenizer_cfg,
                    latent_size=transformer_dict.get('latent_size', 256),
                    num_attention_heads=transformer_dict.get('num_attention_heads', 4),
                    num_layers=transformer_dict.get('num_layers', 2),
                    max_seq_len=transformer_dict.get('max_seq_len', 20),
                    feedforward_dim=transformer_dict.get('feedforward_dim'),
                    dropout=transformer_dict.get('dropout', 0.1),
                    pooling_mode=transformer_dict.get('pooling_mode', 'mean'),
                    use_ctlpe=transformer_dict.get('use_ctlpe', False),
                    use_pos_embeddings=transformer_dict.get('add_pos_embeddings', False)
                )

                transformer = transformer_config.build()
                transformer_params = count_parameters(transformer)

                latent_size = transformer_dict.get('latent_size', 256)
                ffn_dim = transformer_dict.get('feedforward_dim') or (4 * latent_size)

                print(f"\nTransformer for {policy_name}:")
                print(f"  Total parameters: {transformer_params:,}")
                print(f"  Configuration:")
                print(f"    - Tokenizer: {backbone_type} ({backbone_config.get('feature_size', 256)} features)")
                print(f"    - Latent size: {latent_size}")
                print(f"    - Attention heads: {transformer_dict.get('num_attention_heads', 4)}")
                print(f"    - Layers: {transformer_dict.get('num_layers', 2)}")
                print(f"    - Max sequence length: {transformer_dict.get('max_seq_len', 20)}")
                print(f"    - Feedforward dim: {ffn_dim}")
                print(f"    - Use CTLPE: {transformer_dict.get('use_ctlpe', False)}")
                print(f"    - Use positional embeddings: {transformer_dict.get('add_pos_embeddings', False)}")
                break

        if not transformer_found:
            print("\nNo transformer configuration found in training_config.yaml")

        print("="*60)
        
    except Exception as e:
        print(f"Error calculating parameter counts: {e}")
        print("Make sure training_config.yaml exists and is properly formatted.")