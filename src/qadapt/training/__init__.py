"""
Training components for quantum device RL agents.

This package provides training utilities, logging, and policy mapping
functionality for multi-agent reinforcement learning.
"""

from .train import main as train_main

__all__ = ["train_main"]