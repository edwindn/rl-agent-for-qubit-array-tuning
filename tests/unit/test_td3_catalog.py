"""Unit tests for CustomTD3Catalog."""

import pytest
import torch


class TestCustomTD3Catalog:
    """Tests for TD3 catalog building correct components."""

    def test_build_pi_head_returns_deterministic(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """Policy head should be deterministic, not stochastic."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog
        from swarm.voltage_model.models.heads import DeterministicPolicyHead

        catalog = CustomTD3Catalog(
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config_dict=model_config_dict,
        )

        pi_head = catalog.build_pi_head(framework="torch")

        assert isinstance(pi_head, DeterministicPolicyHead)

    def test_pi_head_output_dim_is_action_dim(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """Policy head output should be action_dim (not 2*action_dim like SAC)."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog

        catalog = CustomTD3Catalog(
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config_dict=model_config_dict,
        )

        pi_head = catalog.build_pi_head(framework="torch")

        # Output dim should be action_dim (1), not 2*action_dim (2) like SAC
        assert pi_head.output_dims == (simple_action_space.shape[0],)

    def test_build_qf_head_returns_qvalue_head(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """Q-function head should be QValueHead (same as SAC)."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog
        from swarm.voltage_model.models.heads import QValueHead

        catalog = CustomTD3Catalog(
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config_dict=model_config_dict,
        )

        qf_head = catalog.build_qf_head(framework="torch")

        assert isinstance(qf_head, QValueHead)
        assert qf_head.output_dims == (1,)

    def test_build_encoder_creates_cnn(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """Encoder should be CNN-based for image observations."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog

        catalog = CustomTD3Catalog(
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config_dict=model_config_dict,
        )

        encoder = catalog.build_encoder(framework="torch")

        assert encoder is not None
        # Test forward pass works - encoder expects {"obs": {"image": ..., "voltage": ...}}
        batch = {
            "obs": {
                "image": torch.rand(1, *simple_obs_space.shape),
                "voltage": torch.rand(1, 1),
            }
        }
        output = encoder(batch)
        assert output is not None

    def test_qf_encoder_same_architecture_as_pi_encoder(
        self, simple_obs_space, simple_action_space, model_config_dict
    ):
        """QF encoder should have same architecture as policy encoder."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog

        catalog = CustomTD3Catalog(
            observation_space=simple_obs_space,
            action_space=simple_action_space,
            model_config_dict=model_config_dict,
        )

        pi_encoder = catalog.build_encoder(framework="torch")
        qf_encoder = catalog.build_qf_encoder(framework="torch")

        # Should have same number of parameters (same architecture)
        pi_params = sum(p.numel() for p in pi_encoder.parameters())
        qf_params = sum(p.numel() for p in qf_encoder.parameters())
        assert pi_params == qf_params

    def test_catalog_with_different_backbones(
        self, simple_obs_space, simple_action_space
    ):
        """Catalog should work with different backbone types."""
        from swarm.voltage_model.algorithms.td3 import CustomTD3Catalog

        backbone_configs = [
            {
                "type": "SimpleCNN",
                "feature_size": 64,
                "adaptive_pooling": True,
                "memory_layer": None,
            },
            {
                "type": "IMPALA",
                "feature_size": 64,
                "adaptive_pooling": True,
                "memory_layer": None,
                "num_res_blocks": 1,  # Required for IMPALA
            },
        ]

        for backbone_config in backbone_configs:
            config = {
                "backbone": backbone_config,
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
            }

            catalog = CustomTD3Catalog(
                observation_space=simple_obs_space,
                action_space=simple_action_space,
                model_config_dict=config,
            )

            encoder = catalog.build_encoder(framework="torch")
            pi_head = catalog.build_pi_head(framework="torch")
            qf_head = catalog.build_qf_head(framework="torch")

            assert encoder is not None
            assert pi_head is not None
            assert qf_head is not None
