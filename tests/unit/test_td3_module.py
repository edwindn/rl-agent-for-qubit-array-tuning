"""Unit tests for CustomTD3TorchRLModule."""

import pytest
import torch
from ray.rllib.core.columns import Columns


class TestCustomTD3TorchRLModule:
    """Tests for TD3 RL module forward passes."""

    @pytest.fixture
    def td3_module(self, simple_obs_space, simple_action_space, model_config_dict):
        """Create TD3 module for testing."""
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
        module.make_target_networks()  # Create target networks
        return module

    @pytest.fixture
    def sample_obs_batch(self, simple_obs_space):
        """Create sample observation batch in correct format."""
        batch_size = 8
        return {
            Columns.OBS: {
                "image": torch.rand(batch_size, *simple_obs_space.shape),
                "voltage": torch.rand(batch_size, 1),
            }
        }

    @pytest.fixture
    def sample_train_batch(self, simple_obs_space, simple_action_space):
        """Create sample training batch."""
        batch_size = 8
        obs_shape = simple_obs_space.shape
        action_shape = simple_action_space.shape

        return {
            Columns.OBS: {
                "image": torch.rand(batch_size, *obs_shape),
                "voltage": torch.rand(batch_size, 1),
            },
            Columns.NEXT_OBS: {
                "image": torch.rand(batch_size, *obs_shape),
                "voltage": torch.rand(batch_size, 1),
            },
            Columns.ACTIONS: torch.rand(batch_size, *action_shape) * 2 - 1,
            Columns.REWARDS: torch.rand(batch_size),
            Columns.TERMINATEDS: torch.zeros(batch_size),
        }

    def test_forward_inference_deterministic(self, td3_module, sample_obs_batch):
        """Inference should return deterministic action directly (bypassing distribution)."""
        with torch.no_grad():
            output = td3_module._forward_inference(sample_obs_batch)

        # TD3 returns ACTIONS directly, not ACTION_DIST_INPUTS
        assert Columns.ACTIONS in output
        actions = output[Columns.ACTIONS]

        # Actions should be bounded [-1, 1] (tanh output)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_forward_inference_same_input_same_output(
        self, td3_module, sample_obs_batch
    ):
        """Same input should produce same output (deterministic)."""
        with torch.no_grad():
            output1 = td3_module._forward_inference(sample_obs_batch)
            output2 = td3_module._forward_inference(sample_obs_batch)

        assert torch.allclose(
            output1[Columns.ACTIONS],
            output2[Columns.ACTIONS],
        )

    def test_forward_exploration_adds_noise(self, td3_module, sample_obs_batch):
        """Exploration should add noise to deterministic action."""
        with torch.no_grad():
            inference_output = td3_module._forward_inference(sample_obs_batch)
            exploration_output = td3_module._forward_exploration(sample_obs_batch)

        # Exploration should differ from inference (due to noise)
        # Note: Very small chance they're equal by random chance
        assert not torch.allclose(
            inference_output[Columns.ACTIONS],
            exploration_output[Columns.ACTIONS],
            atol=1e-6,
        )

    def test_forward_exploration_clipped_actions(self, td3_module, sample_obs_batch):
        """Exploration actions should still be bounded after noise."""
        with torch.no_grad():
            output = td3_module._forward_exploration(sample_obs_batch)

        # TD3 returns ACTIONS directly
        actions = output[Columns.ACTIONS]

        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_forward_train_returns_required_outputs(
        self, td3_module, sample_train_batch
    ):
        """Training forward should return Q-values and actions."""
        output = td3_module._forward_train(sample_train_batch)

        # Check required outputs
        assert "qf_preds" in output
        assert "q_curr" in output
        assert "action_curr" in output
        assert "q_target_next" in output
        assert "action_next_smoothed" in output

        # Check twin Q if enabled
        if td3_module.twin_q:
            assert "qf_twin_preds" in output

    def test_forward_train_output_shapes(self, td3_module, sample_train_batch):
        """Training outputs should have correct shapes."""
        batch_size = sample_train_batch[Columns.OBS]["image"].shape[0]
        output = td3_module._forward_train(sample_train_batch)

        assert output["qf_preds"].shape == (batch_size,)
        assert output["q_curr"].shape == (batch_size,)
        assert output["action_curr"].shape == (batch_size, 1)
        assert output["q_target_next"].shape == (batch_size,)
        assert output["action_next_smoothed"].shape == (batch_size, 1)

    def test_target_action_smoothing(self, td3_module, sample_train_batch):
        """Target actions should have noise added (policy smoothing)."""
        # Run training forward multiple times and check that smoothed actions vary
        outputs = []
        for _ in range(5):
            output = td3_module._forward_train(sample_train_batch)
            outputs.append(output["action_next_smoothed"].clone())

        # Check that at least some smoothed actions differ between runs
        # (due to random noise in target policy smoothing)
        all_same = all(
            torch.allclose(outputs[0], o, atol=1e-6) for o in outputs[1:]
        )
        assert not all_same, "Target action smoothing should add noise"

    def test_target_actions_clipped(self, td3_module, sample_train_batch):
        """Smoothed target actions should be clipped to [-1, 1]."""
        output = td3_module._forward_train(sample_train_batch)

        actions = output["action_next_smoothed"]
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_module_has_twin_q_networks(self, td3_module):
        """Module should have twin Q-networks when twin_q=True."""
        assert hasattr(td3_module, "qf")
        assert hasattr(td3_module, "qf_encoder")

        if td3_module.twin_q:
            assert hasattr(td3_module, "qf_twin")
            assert hasattr(td3_module, "qf_twin_encoder")

    def test_module_has_target_networks(self, td3_module):
        """Module should have target networks after make_target_networks()."""
        assert hasattr(td3_module, "target_qf")
        assert hasattr(td3_module, "target_qf_encoder")

        if td3_module.twin_q:
            assert hasattr(td3_module, "target_qf_twin")
            assert hasattr(td3_module, "target_qf_twin_encoder")

    def test_exploration_noise_configurable(
        self, simple_obs_space, simple_action_space
    ):
        """Exploration noise should be configurable via model_config."""
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from swarm.voltage_model.algorithms.td3 import (
            CustomTD3TorchRLModule,
            CustomTD3Catalog,
        )

        config = {
            "backbone": {
                "type": "SimpleCNN",
                "feature_size": 64,
                "adaptive_pooling": True,
                "memory_layer": None,
            },
            "policy_head": {
                "hidden_layers": [32],
                "activation": "relu",
                "use_attention": False,
            },
            "value_head": {
                "hidden_layers": [32],
                "activation": "relu",
            },
            "twin_q": True,
            "exploration_noise": 0.5,  # Custom value
            "policy_noise": 0.3,
            "noise_clip": 0.4,
        }

        spec = RLModuleSpec(
            module_class=CustomTD3TorchRLModule,
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config=config,
            catalog_class=CustomTD3Catalog,
        )
        module = spec.build()

        assert module.exploration_noise == 0.5
        assert module.policy_noise == 0.3
        assert module.noise_clip == 0.4
