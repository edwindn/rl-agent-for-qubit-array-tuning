"""Unit tests for TD3Config."""

import pytest


class TestTD3Config:
    """Tests for TD3 configuration."""

    def test_default_td3_params(self):
        """TD3 should have correct default parameters."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        assert config.exploration_noise == 0.1
        assert config.policy_noise == 0.2
        assert config.noise_clip == 0.5
        assert config.policy_frequency == 2

    def test_no_alpha_optimization(self):
        """TD3 should not optimize alpha (entropy coefficient)."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        # Alpha-related params should be None or unused
        assert config.alpha_lr is None

    def test_training_method_sets_params(self):
        """training() method should update TD3 parameters."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        config.training(
            exploration_noise=0.2,
            policy_noise=0.3,
            noise_clip=0.4,
            policy_frequency=3,
        )

        assert config.exploration_noise == 0.2
        assert config.policy_noise == 0.3
        assert config.noise_clip == 0.4
        assert config.policy_frequency == 3

    def test_inherits_sac_params(self):
        """TD3 should inherit SAC's replay buffer and Q-network params."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        # These are inherited from SAC
        assert hasattr(config, "twin_q")
        assert hasattr(config, "tau")
        assert hasattr(config, "actor_lr")
        assert hasattr(config, "critic_lr")
        assert hasattr(config, "replay_buffer_config")

    def test_training_method_returns_self(self):
        """training() should return self for method chaining."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()
        result = config.training(exploration_noise=0.15)

        assert result is config

    def test_training_method_accepts_sac_params(self):
        """training() should accept SAC parent class parameters."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        # These params are from SACConfig
        config.training(
            actor_lr=0.0001,
            critic_lr=0.0002,
            tau=0.01,
            gamma=0.95,
        )

        assert config.actor_lr == 0.0001
        assert config.critic_lr == 0.0002
        assert config.tau == 0.01
        assert config.gamma == 0.95
