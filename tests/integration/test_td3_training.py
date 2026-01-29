"""Integration tests for TD3 training pipeline."""

import pytest
import torch
from ray.rllib.core.columns import Columns


@pytest.mark.integration
class TestTD3ModuleIntegration:
    """Integration tests for TD3 module building and forward passes."""

    def test_td3_module_builds_successfully(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """TD3 module should build without errors."""
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from swarm.voltage_model.algorithms.td3 import (
            CustomTD3TorchRLModule,
            CustomTD3Catalog,
        )

        spec = RLModuleSpec(
            module_class=CustomTD3TorchRLModule,
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config=model_config_dict,
            catalog_class=CustomTD3Catalog,
        )

        module = spec.build()

        assert module is not None
        assert hasattr(module, "pi")
        assert hasattr(module, "qf")
        assert hasattr(module, "pi_encoder")
        assert hasattr(module, "qf_encoder")

    def test_td3_forward_backward_pass(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """TD3 should complete forward and backward passes without errors."""
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from swarm.voltage_model.algorithms.td3 import (
            CustomTD3TorchRLModule,
            CustomTD3Catalog,
        )

        spec = RLModuleSpec(
            module_class=CustomTD3TorchRLModule,
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config=model_config_dict,
            catalog_class=CustomTD3Catalog,
        )
        module = spec.build()
        module.make_target_networks()

        # Create batch
        batch_size = 8
        batch = {
            Columns.OBS: {
                "image": torch.rand(batch_size, *simple_obs_space.shape),
                "voltage": torch.rand(batch_size, 1),
            },
            Columns.NEXT_OBS: {
                "image": torch.rand(batch_size, *simple_obs_space.shape),
                "voltage": torch.rand(batch_size, 1),
            },
            Columns.ACTIONS: torch.rand(batch_size, *simple_action_space.shape) * 2 - 1,
            Columns.REWARDS: torch.rand(batch_size),
            Columns.TERMINATEDS: torch.zeros(batch_size),
        }

        # Forward pass
        output = module._forward_train(batch)

        # Verify gradients can flow
        loss = output["qf_preds"].mean() - output["q_curr"].mean()
        loss.backward()

        # Check gradients exist
        for param in module.qf.parameters():
            if param.requires_grad:
                assert param.grad is not None


@pytest.mark.integration
class TestTD3FactoryIntegration:
    """Integration tests for TD3 with the factory."""

    def test_create_multi_agent_module_spec(self, env_config, neural_networks_config):
        """Factory should create MultiRLModuleSpec for plunger and barrier policies."""
        from swarm.voltage_model.factory import create_rl_module_spec

        spec = create_rl_module_spec(
            env_config=env_config,
            algo="td3",
            config=neural_networks_config,
        )

        assert "plunger_policy" in spec.rl_module_specs
        assert "barrier_policy" in spec.rl_module_specs

    def test_factory_builds_td3_modules(self, env_config, neural_networks_config):
        """Factory-created specs should build valid TD3 modules."""
        from swarm.voltage_model.factory import create_rl_module_spec

        spec = create_rl_module_spec(
            env_config=env_config,
            algo="td3",
            config=neural_networks_config,
        )

        # Build the plunger policy module
        plunger_spec = spec.rl_module_specs["plunger_policy"]
        plunger_module = plunger_spec.build()

        assert plunger_module is not None
        assert hasattr(plunger_module, "pi")
        assert hasattr(plunger_module, "qf")


@pytest.mark.integration
class TestTD3ConfigIntegration:
    """Integration tests for TD3 config."""

    def test_td3_config_creates_learner_class(self):
        """TD3Config should specify TD3TorchLearner as default."""
        from swarm.training.utils.td3_config import TD3Config
        from swarm.training.utils.td3_learner import TD3TorchLearner

        config = TD3Config()
        learner_class = config.get_default_learner_class()

        assert learner_class == TD3TorchLearner

    def test_td3_config_training_params(self):
        """TD3Config should accept all TD3-specific training params."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()
        config.training(
            exploration_noise=0.15,
            policy_noise=0.25,
            noise_clip=0.4,
            policy_frequency=3,
            actor_lr=0.0001,
            critic_lr=0.0002,
            tau=0.01,
            gamma=0.95,
        )

        assert config.exploration_noise == 0.15
        assert config.policy_noise == 0.25
        assert config.noise_clip == 0.4
        assert config.policy_frequency == 3
        assert config.actor_lr == 0.0001
        assert config.critic_lr == 0.0002
        assert config.tau == 0.01
        assert config.gamma == 0.95
