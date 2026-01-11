"""
RLlib algorithm integration.

Each algorithm (PPO, SAC, TD3) has its own module with:
- Catalog class (tells RLlib how to build models)
- Optional custom RL module

Common utilities are in common.py
"""

from .common import build_encoder_config, get_head_input_dim
from .ppo import CustomPPOCatalog
from .sac import CustomSACCatalog, CustomSACTorchRLModule
from .td3 import CustomTD3Catalog, CustomTD3TorchRLModule

__all__ = [
    # Common
    "build_encoder_config",
    "get_head_input_dim",
    # PPO
    "CustomPPOCatalog",
    # SAC
    "CustomSACCatalog",
    "CustomSACTorchRLModule",
    # TD3
    "CustomTD3Catalog",
    "CustomTD3TorchRLModule",
]
