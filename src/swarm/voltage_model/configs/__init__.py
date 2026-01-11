"""
Configuration classes for neural network models.

Each config is a dataclass with a .build() method that returns the corresponding model.
"""

from .backbone_configs import (
    SimpleCNNConfig,
    IMPALAConfig,
    MobileNetConfig,
)

from .memory_configs import (
    LSTMConfig,
    TransformerConfig,
)

from .head_configs import (
    PolicyHeadConfig,
    ValueHeadConfig,
)

__all__ = [
    # Backbones
    "SimpleCNNConfig",
    "IMPALAConfig",
    "MobileNetConfig",
    # Memory
    "LSTMConfig",
    "TransformerConfig",
    # Heads
    "PolicyHeadConfig",
    "ValueHeadConfig",
]
