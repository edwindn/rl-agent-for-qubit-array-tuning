"""
VoltageAgent module for quantum device voltage control.

This module provides swappable neural network implementations for RL-based
voltage control of quantum devices. Different trainer implementations can
be easily switched based on configuration.
"""

# from .ppo_trainer import PPOTrainer, MultiModalPPOModel, create_ppo_config  # Missing file
from .ppo_trainer_recurrent import RecurrentPPOTrainer, SingleAgentRecurrentPPOModel, create_recurrent_ppo_config
from .ppo_config import get_default_ppo_config, get_ppo_config

__all__ = [
    # 'PPOTrainer',  # Missing file
    # 'MultiModalPPOModel',  # Missing file
    # 'create_ppo_config',  # Missing file
    'RecurrentPPOTrainer',
    'SingleAgentRecurrentPPOModel',
    'create_recurrent_ppo_config',
    'get_default_ppo_config',
    'get_ppo_config',
    'get_trainer_class'
]


def get_trainer_class(trainer_type: str = 'ppo'):
    """
    Get trainer class based on type string.
    
    This enables easy switching between different neural network implementations
    by changing configuration values.
    
    Args:
        trainer_type: Type of trainer ('ppo', 'ppo_v2', 'sac', etc.)
        
    Returns:
        Trainer class
        
    Raises:
        ValueError: If trainer_type is not supported
    """
    trainer_map = {
        # 'ppo': PPOTrainer,  # Missing file
        'recurrent_ppo': RecurrentPPOTrainer,
        # Future implementations:
        # 'ppo_v2': PPOTrainerV2,
        # 'sac': SACTrainer,
        # 'td3': TD3Trainer,
    }
    
    if trainer_type not in trainer_map:
        available = ', '.join(trainer_map.keys())
        raise ValueError(f"Unknown trainer type '{trainer_type}'. Available: {available}")
    
    return trainer_map[trainer_type]