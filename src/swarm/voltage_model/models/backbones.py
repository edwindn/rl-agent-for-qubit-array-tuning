"""
CNN backbone encoders for quantum charge stability diagrams.

Contains:
- SimpleCNN: Basic CNN encoder
- IMPALA: CNN with ResNet blocks
- MobileNet: Pretrained MobileNetV3 backbone
"""

from typing import TYPE_CHECKING, Tuple

from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import torchvision.models as models

if TYPE_CHECKING:
    from swarm.voltage_model.configs import SimpleCNNConfig, IMPALAConfig, MobileNetConfig


# =============================================================================
# SimpleCNN
# =============================================================================

class SimpleCNN(TorchModel, Encoder):
    """CNN encoder based on DQN Nature paper (Mnih et al., 2015).

    Architecture:
    - Conv1: 32 filters, 8x8 kernel, stride 4, ReLU (no padding)
    - Conv2: 64 filters, 4x4 kernel, stride 2, ReLU (no padding)
    - Conv3: 64 filters, 3x3 kernel, stride 1, ReLU (no padding)
    - FC: 512 units, ReLU
    """

    def __init__(self, config: "SimpleCNNConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        cnn_layers = []
        in_channels = config.input_dims[-1]

        # DQN Nature paper uses valid convolutions (no padding)
        for out_channels, kernel_size, stride in config.cnn_filter_specifiers:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
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
                voltage = inputs["voltage"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            raise ValueError("For using voltage memory we need return_voltage to be enabled.")

        if voltage.dim() == 1:
            voltage = voltage.unsqueeze(0)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)

        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)

        out = {
            "image_features": output_features,
            "voltage": voltage,
        }

        return {ENCODER_OUT: out}


# =============================================================================
# IMPALA (with ResNet blocks)
# =============================================================================

class ResNetBlock(nn.Module):
    """
    Canonical IMPALA residual block:
      y = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
    This matches the common reproduction of the IMPALA-CNN residual unit.
    """

    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.relu = nn.ReLU(inplace=True) if activation == "relu" else nn.Tanh()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class IMPALA(TorchModel, Encoder):
    """
    IMPALA CNN encoder with canonical conv sequences.

    Each sequence: Conv3x3 -> MaxPool(stride) -> ResidualBlocks -> ReLU
    This follows the canonical IMPALA-CNN architecture while maintaining Ray RLlib compatibility.
    """

    def __init__(self, config: "IMPALAConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        cnn_layers = []
        in_channels = config.input_dims[-1]

        for i, (out_channels, kernel_size, stride) in enumerate(config.cnn_filter_specifiers):
            # Canonical IMPALA conv sequence: Conv -> MaxPool -> ResBlocks -> ReLU
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

            if stride > 1:
                cnn_layers.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

            for _ in range(config.num_res_blocks):
                cnn_layers.append(ResNetBlock(out_channels, config.cnn_activation))

            cnn_layers.append(nn.ReLU(inplace=True))

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
                voltage = inputs["voltage"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            raise ValueError("For using voltage memory we need return_voltage to be enabled.")

        if voltage.dim() == 1:
            voltage = voltage.unsqueeze(0)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)

        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)

        out = {
            "image_features": output_features,
            "voltage": voltage,
        }

        return {ENCODER_OUT: out}


# =============================================================================
# MobileNet
# =============================================================================

class MobileNet(TorchModel, Encoder):
    """MobileNet encoder using pretrained MobileNetV3 backbone."""

    def __init__(self, config: "MobileNetConfig"):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Load MobileNet backbone
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

        # Modify first conv layer to accept correct number of input channels
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
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv1.weight[:, :input_channels, :, :].clone()
                    )
                else:
                    weight = original_conv1.weight
                    repeats = (input_channels + 2) // 3
                    repeated_weight = weight.repeat(1, repeats, 1, 1)
                    self.backbone.features[0][0].weight = nn.Parameter(
                        repeated_weight[:, :input_channels, :, :].clone()
                    )

        # Remove final classification layer
        self.backbone.classifier = nn.Identity()

        # Freeze backbone if requested
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Final projection layer
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
                voltage = inputs["voltage"]
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            raise ValueError("For using voltage memory we need return_voltage to be enabled.")

        if voltage.dim() == 1:
            voltage = voltage.unsqueeze(0)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)

        backbone_features = self.backbone(x)
        output_features = self.projection(backbone_features)

        out = {
            "image_features": output_features,
            "voltage": voltage,
        }

        return {ENCODER_OUT: out}
