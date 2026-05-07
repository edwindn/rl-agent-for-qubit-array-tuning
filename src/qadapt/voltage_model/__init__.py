"""
Voltage model components for quantum device RL agents.

Structure:
- models/                : PyTorch nn.Module classes (SimpleCNN, IMPALA, LSTM, etc.)
- configs/               : Configuration dataclasses (SimpleCNNConfig, etc.)
- algorithms/            : RLlib integration (PPO, SAC, TD3 catalogs)
- factory.py             : Entry point for dot-tuning RLModule specs

SuperSims-specific RLlib glue (catalog, neural nets, factory, eval) lives in the
sibling qadapt_for_supersim package.
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
