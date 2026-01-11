"""
Voltage model components for quantum device RL agents.

Structure:
- models/      : PyTorch nn.Module classes (SimpleCNN, IMPALA, LSTM, etc.)
- configs/     : Configuration dataclasses (SimpleCNNConfig, etc.)
- algorithms/  : RLlib integration (PPO, SAC, TD3 catalogs)
- factory.py   : Entry point for creating RL modules
"""

from .factory import create_rl_module_spec
from .algorithms import CustomPPOCatalog, CustomSACCatalog, CustomSACTorchRLModule
from .models.transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = [
    "create_rl_module_spec",
    "CustomPPOCatalog",
    "CustomSACCatalog",
    "CustomSACTorchRLModule",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]
