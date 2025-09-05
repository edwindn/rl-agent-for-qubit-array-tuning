"""
PPO configuration and hyperparameters for voltage control agents.
Contains default settings that can be overridden from training infrastructure.
"""

from typing import Dict, Any


def get_default_ppo_config() -> Dict[str, Any]:
    """
    Get default PPO configuration for voltage control agents.
    
    Returns:
        Dictionary with PPO hyperparameters and model configuration
    """
    return {
        "lr": 3e-4,
        "lr_schedule": None,
        
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        
        "num_sgd_iter": 10,
        "gamma": 0.99,
        "lambda": 0.95,
        
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "use_lstm": False,
            "max_seq_len": 20,
            "lstm_cell_size": 256,
            "lstm_use_prev_action": False,
            "lstm_use_prev_reward": False,
            
            "conv_filters": None,
            "conv_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": "relu"
        }
    }


def get_ppo_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get PPO configuration with optional overrides.
    
    Args:
        overrides: Dictionary of configuration values to override
        
    Returns:
        Complete PPO configuration dictionary
    """
    config = get_default_ppo_config()
    
    if overrides:
        # Deep merge overrides
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config, overrides)
    
    return config