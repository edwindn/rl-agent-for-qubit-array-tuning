"""
Voltage model components for quantum device RL agents.

Structure:
- models/                : PyTorch nn.Module classes (SimpleCNN, IMPALA, LSTM, etc.)
- configs/               : Configuration dataclasses (SimpleCNNConfig, etc.)
- algorithms/            : RLlib integration (PPO, SAC, TD3 catalogs)
- factory.py             : Entry point for dot-tuning RLModule specs
- supersims_factory.py   : Entry point for SuperSims All-XY env RLModule specs
- supersims_*.py         : SuperSims-only RLlib glue (catalog, neural nets, sac module)
"""

from .factory import create_rl_module_spec
from .supersims_factory import create_rl_module_spec_supersims
from .algorithms import CustomPPOCatalog, CustomSACCatalog, CustomSACTorchRLModule
from .models.transformer import TransformerEncoder, TransformerEncoderLayer

__all__ = [
    "create_rl_module_spec",
    "create_rl_module_spec_supersims",
    "CustomPPOCatalog",
    "CustomSACCatalog",
    "CustomSACTorchRLModule",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]
