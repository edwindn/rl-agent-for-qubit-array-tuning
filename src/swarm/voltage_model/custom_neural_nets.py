"""Custom neural network components for quantum device RL agents."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import CNNEncoderConfig, MLPHeadConfig, ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import torch.nn.functional as F

# Import torchvision models for MobileNet
try:
    import torchvision.models as models
except ImportError:
    models = None


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

        # Load pretrained MobileNet backbone
        if config.mobilenet_version == "small":
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            feature_dim = 576
        elif config.mobilenet_version == "large":
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
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
class TransformerEncoderConfig(CNNEncoderConfig):
    """Transformer encoder configuration for quantum charge stability diagrams.

    This is a placeholder for future transformer-based memory implementation.
    The transformer will wrap a CNN tokenizer (SimpleCNN, IMPALA, or MobileNet)
    and add temporal modeling through self-attention mechanisms.
    """

    tokenizer_config: None
    latent_size: int = 256
    num_attention_heads: int = 4
    num_layers: int = 1
    max_seq_len: int = 50
    dropout: float = 0.1

    assert tokenizer_config is not None

    @property
    def output_dims(self):
        return (self.latent_size,)

    def build(self, framework: str = "torch") -> "TransformerEncoder":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return TransformerEncoder(self)


class TransformerEncoder(TorchModel, Encoder):
    """Transformer encoder for temporal modeling in quantum device control.

    - Accept tokenized CNN features as input
    - Apply multi-head self-attention across temporal sequence
    - Optionally incorporate previous actions and rewards
    - Output contextualized features for policy/value heads
    """

    def __init__(self, config: TransformerEncoderConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # TODO: Implement transformer architecture
        # - Positional encoding
        # - Multi-head self-attention layers
        # - Feed-forward networks
        # - Layer normalization
        # - Residual connections

        raise NotImplementedError(
            "TransformerEncoder is not yet implemented. "
            "Use memory_layer='lstm' or memory_layer=null in config."
        )

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return (self.config.latent_size,)

    def _forward(self, inputs, **kwargs):
        # TODO: Implement forward pass
        raise NotImplementedError("TransformerEncoder forward pass not implemented")


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
                    freeze_backbone=backbone_config.get('freeze_backbone', False)
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
        
    except Exception as e:
        print(f"Error calculating parameter counts: {e}")
        print("Make sure training_config.yaml exists and is properly formatted.")