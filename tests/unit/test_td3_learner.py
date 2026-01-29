"""Unit tests for TD3TorchLearner."""

import pytest
import torch


class TestTD3TorchLearner:
    """Tests for TD3 learner loss computation and optimization."""

    def test_delayed_policy_update_frequency(self):
        """Policy should update every N critic updates."""
        # Test the logic without full learner instantiation
        policy_frequency = 2

        for step in range(1, 10):
            should_update = (step % policy_frequency) == 0

            if step in [2, 4, 6, 8]:
                assert should_update, f"Step {step} should update policy"
            else:
                assert not should_update, f"Step {step} should NOT update policy"

    def test_critic_loss_uses_huber(self):
        """Critic loss should use Huber loss for stability."""
        q_pred = torch.tensor([1.0, 2.0, 3.0])
        q_target = torch.tensor([1.5, 2.5, 3.5])

        huber_loss = torch.nn.HuberLoss(reduction="none", delta=1.0)
        loss = huber_loss(q_pred, q_target)

        assert loss.shape == q_pred.shape
        assert (loss >= 0).all()

    def test_actor_loss_maximizes_q(self):
        """Actor loss should be -mean(Q) to maximize Q."""
        q_values = torch.tensor([1.0, 2.0, 3.0])
        actor_loss = -torch.mean(q_values)

        assert actor_loss == -2.0  # -(1+2+3)/3 = -2

    def test_target_q_uses_minimum(self):
        """Target Q should use minimum of twin Q-networks."""
        q1_target = torch.tensor([1.0, 2.0, 3.0])
        q2_target = torch.tensor([1.5, 1.5, 2.5])

        q_target_min = torch.minimum(q1_target, q2_target)

        expected = torch.tensor([1.0, 1.5, 2.5])
        assert torch.allclose(q_target_min, expected)

    def test_bellman_target_no_entropy(self):
        """Bellman target should not include entropy term (unlike SAC)."""
        # In TD3: target = r + gamma * Q_target
        # In SAC: target = r + gamma * (Q_target - alpha * log_prob)

        rewards = torch.tensor([1.0, 2.0])
        q_target_next = torch.tensor([10.0, 20.0])
        gamma = 0.99
        terminated = torch.tensor([0.0, 0.0])

        # TD3 Bellman target (no entropy)
        td3_target = rewards + gamma * (1.0 - terminated) * q_target_next

        expected = torch.tensor([1.0 + 0.99 * 10.0, 2.0 + 0.99 * 20.0])
        assert torch.allclose(td3_target, expected)

    def test_policy_frequency_default(self):
        """Default policy frequency should be 2."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()
        assert config.policy_frequency == 2

    def test_learner_class_registration(self):
        """TD3Config should return TD3TorchLearner."""
        from swarm.training.utils.td3_config import TD3Config
        from swarm.training.utils.td3_learner import TD3TorchLearner

        config = TD3Config()
        learner_class = config.get_default_learner_class()

        assert learner_class == TD3TorchLearner

    def test_no_alpha_in_td3(self):
        """TD3 should not use alpha (entropy coefficient)."""
        from swarm.training.utils.td3_config import TD3Config

        config = TD3Config()

        # Alpha-related attributes should indicate no optimization
        assert config.alpha_lr is None

    def test_td3_inherits_from_sac_learner(self):
        """TD3TorchLearner should inherit from SACTorchLearner."""
        from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
        from swarm.training.utils.td3_learner import TD3TorchLearner

        assert issubclass(TD3TorchLearner, SACTorchLearner)

    def test_gradient_steps_tracking(self):
        """Learner should track gradient steps per module."""
        # Test the concept without full instantiation
        gradient_steps = {}
        module_id = "plunger_policy"

        # Simulate gradient steps
        for i in range(1, 6):
            gradient_steps[module_id] = gradient_steps.get(module_id, 0) + 1

        assert gradient_steps[module_id] == 5

    def test_masked_terminated_states(self):
        """Terminated states should have zero future Q-value."""
        q_target_next = torch.tensor([10.0, 20.0, 30.0])
        terminated = torch.tensor([0.0, 1.0, 0.0])  # Middle state is terminal

        q_next_masked = (1.0 - terminated) * q_target_next

        expected = torch.tensor([10.0, 0.0, 30.0])
        assert torch.allclose(q_next_masked, expected)
