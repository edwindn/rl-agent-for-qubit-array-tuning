"""
Neural network models (PyTorch nn.Modules) for quantum device RL agents.

Models:
- Backbones: CNN encoders (SimpleCNN, IMPALA, MobileNet)
- Memory: Temporal models (LSTM, Transformer)
- Heads: Policy and value heads

For configuration classes, see swarm.voltage_model.configs
"""

from .backbones import (
    SimpleCNN,
    IMPALA,
    ResNetBlock,
    MobileNet,
)

from .memory import (
    LSTM,
    Transformer,
    LearnedPositionalEncoding,
    CTLPEPositionalEncoding,
)

from .heads import (
    PolicyHead,
    ValueHead,
    QValueHead,
)

__all__ = [
    # Backbones
    "SimpleCNN",
    "IMPALA",
    "ResNetBlock",
    "MobileNet",
    # Memory
    "LSTM",
    "Transformer",
    "LearnedPositionalEncoding",
    "CTLPEPositionalEncoding",
    # Heads
    "PolicyHead",
    "ValueHead",
    "QValueHead",
]
