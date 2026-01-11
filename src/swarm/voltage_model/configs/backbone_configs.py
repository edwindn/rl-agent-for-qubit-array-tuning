"""Configuration classes for CNN backbone encoders."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ray.rllib.core.models.configs import CNNEncoderConfig


@dataclass
class SimpleCNNConfig(CNNEncoderConfig):
    """CNN configuration for quantum charge stability diagrams."""

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

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.backbones import SimpleCNN
        return SimpleCNN(self)


@dataclass
class IMPALAConfig(CNNEncoderConfig):
    """IMPALA CNN configuration with ResNet blocks."""

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
            self.cnn_filter_specifiers = [
                [16, [8, 8], 4],
                [32, [4, 4], 2],
                [32, [3, 3], 1],
            ]

    @property
    def output_dims(self):
        return (self.feature_size,)

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.backbones import IMPALA
        return IMPALA(self)


@dataclass
class MobileNetConfig(CNNEncoderConfig):
    """MobileNet configuration with pretrained backbone."""

    mobilenet_version: str = "small"  # "small" or "large"
    feature_size: int = 256
    freeze_backbone: bool = False
    load_pretrained: bool = True

    def __post_init__(self):
        if self.mobilenet_version == "small":
            self._backbone_feature_dim = 576
        elif self.mobilenet_version == "large":
            self._backbone_feature_dim = 960
        else:
            raise ValueError(f"Unsupported MobileNet version: {self.mobilenet_version}. Use 'small' or 'large'.")

    @property
    def output_dims(self):
        return (self.feature_size,)

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.backbones import MobileNet
        return MobileNet(self)
