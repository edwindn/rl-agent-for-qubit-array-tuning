"""
TD3 Learner with delayed policy updates.

TD3 key differences from SAC implemented here:
1. No entropy coefficient (alpha) - remove entropy loss
2. Delayed policy updates - update actor every N critic updates
3. Simple actor loss: -Q(s, pi(s)) without entropy term
"""

from collections import defaultdict
from typing import Any, Dict

import torch
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID


class TD3TorchLearner(SACTorchLearner):
    """TD3 Learner implementing delayed policy updates.

    Extends SAC learner but:
    - Removes entropy regularization (no alpha)
    - Implements delayed policy updates (policy_delay)
    - Uses deterministic policy gradient
    """

    @override(SACTorchLearner)
    def build(self) -> None:
        super().build()

        # Track update counts for delayed policy updates per module
        self._critic_update_counts: Dict[ModuleID, int] = defaultdict(int)

    @override(SACTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config,
        batch: Dict[str, Any],
        fwd_out: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute TD3 loss.

        Key differences from SAC:
        - No entropy term in actor loss
        - Delayed actor updates
        - No alpha (entropy coefficient) optimization
        """
        # Get hyperparameters
        gamma = config.gamma
        policy_delay = getattr(config, "policy_delay", 2)

        # Get batch data
        rewards = batch[Columns.REWARDS]
        dones = batch[Columns.TERMINATEDS]

        # ====== Critic Loss ======
        # TD target: r + gamma * (1 - done) * Q_target(s', a')
        # a' comes from target policy with smoothing noise (computed in RLModule)
        q_target_next = fwd_out["q_target_next"]
        td_target = rewards + gamma * (1.0 - dones.float()) * q_target_next

        # Current Q-values from replay buffer actions
        qf_preds = fwd_out["qf_preds"]
        critic_loss = torch.nn.functional.mse_loss(qf_preds, td_target.detach())

        # Twin Q loss if enabled
        if "qf_twin_preds" in fwd_out:
            twin_loss = torch.nn.functional.mse_loss(
                fwd_out["qf_twin_preds"], td_target.detach()
            )
            critic_loss = critic_loss + twin_loss

            self.metrics.log_dict(
                {
                    "qf_twin_loss": twin_loss.item(),
                },
                key=module_id,
                window=1,
            )

        # Log critic metrics
        td_error = torch.abs(qf_preds - td_target)
        self.metrics.log_dict(
            {
                "qf_loss": critic_loss.item(),
                "td_error_mean": td_error.mean().item(),
                "q_values_mean": qf_preds.mean().item(),
            },
            key=module_id,
            window=1,
        )

        # Increment critic update count
        self._critic_update_counts[module_id] += 1

        # ====== Actor Loss (Delayed) ======
        total_loss = critic_loss

        if self._critic_update_counts[module_id] % policy_delay == 0:
            # Actor loss: maximize Q(s, pi(s)) = minimize -Q(s, pi(s))
            # Note: No entropy term in TD3 (unlike SAC)
            q_curr = fwd_out["q_curr"]
            actor_loss = -q_curr.mean()

            total_loss = total_loss + actor_loss

            self.metrics.log_dict(
                {
                    "actor_loss": actor_loss.item(),
                    "policy_update": 1.0,  # Flag that policy was updated
                },
                key=module_id,
                window=1,
            )
        else:
            # Log that policy was not updated this step
            self.metrics.log_dict(
                {
                    "policy_update": 0.0,
                },
                key=module_id,
                window=1,
            )

        return total_loss
