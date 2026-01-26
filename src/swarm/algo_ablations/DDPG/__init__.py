"""Local DDPG implementation (SimpleQ-based) for the ablation scripts."""

from .ddpg import DDPG, DDPGConfig

__all__ = [
    "DDPG",
    "DDPGConfig",
]
