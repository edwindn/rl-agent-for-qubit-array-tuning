"""Shared pytest fixtures for TD3 and other RL algorithm tests."""

import pytest
import torch
import numpy as np
from gymnasium import spaces


@pytest.fixture
def simple_obs_space():
    """Simple image observation space for testing (64x64x2 channels)."""
    return spaces.Box(low=0.0, high=1.0, shape=(64, 64, 2), dtype=np.float32)


@pytest.fixture
def barrier_obs_space():
    """Barrier agent observation space (64x64x1 channel)."""
    return spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32)


@pytest.fixture
def simple_action_space():
    """Simple continuous action space for testing."""
    return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


@pytest.fixture
def sample_batch(simple_obs_space, simple_action_space):
    """Sample batch for testing forward passes."""
    batch_size = 32
    obs_shape = simple_obs_space.shape
    action_shape = simple_action_space.shape

    return {
        "obs": torch.rand(batch_size, *obs_shape),
        "actions": torch.rand(batch_size, *action_shape) * 2 - 1,
        "rewards": torch.rand(batch_size),
        "terminateds": torch.zeros(batch_size),
        "next_obs": torch.rand(batch_size, *obs_shape),
    }


@pytest.fixture
def model_config_dict():
    """Standard model config for testing with SimpleCNN backbone."""
    return {
        "backbone": {
            "type": "SimpleCNN",
            "feature_size": 64,
            "adaptive_pooling": True,
            "memory_layer": None,
        },
        "policy_head": {
            "hidden_layers": [32, 32],
            "activation": "relu",
            "use_attention": False,
        },
        "value_head": {
            "hidden_layers": [32, 32],
            "activation": "relu",
        },
        "twin_q": True,
        "free_log_std": False,
        "log_std_bounds": [-10, 2],
    }


@pytest.fixture
def td3_hyperparams():
    """TD3-specific hyperparameters for testing."""
    return {
        "exploration_noise": 0.1,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_frequency": 2,
        "actor_lr": 0.0003,
        "critic_lr": 0.0003,
        "tau": 0.005,
        "gamma": 0.99,
    }


@pytest.fixture
def env_config():
    """Minimal environment config for testing."""
    return {
        "simulator": {
            "resolution": 64,
            "num_dots": 3,
        }
    }


@pytest.fixture
def neural_networks_config(model_config_dict):
    """Neural networks config for both plunger and barrier policies."""
    return {
        "plunger_policy": model_config_dict.copy(),
        "barrier_policy": {
            **model_config_dict,
            "backbone": {
                "type": "SimpleCNN",
                "feature_size": 64,
                "adaptive_pooling": True,
                "memory_layer": None,
            },
        },
    }
