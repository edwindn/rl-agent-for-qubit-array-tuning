#!/usr/bin/env python3
"""
Custom SAC learner that supports reward scaling.
SAC is sensitive to reward magnitude - this allows configurable scaling.
"""
from typing import Any, Dict

from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType


class SACLearnerWithRewardScaling(SACTorchLearner):
    """SAC Learner that scales rewards before loss computation.

    Set reward_scale in training config:
        training:
            reward_scale: 10.0
    """

    @override(SACTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        """Compute loss with optional reward scaling."""

        # Get reward_scale from config (default 1.0)
        reward_scale = getattr(config, 'reward_scale', 1.0)

        # Scale rewards if needed
        if reward_scale != 1.0 and Columns.REWARDS in batch:
            batch[Columns.REWARDS] = batch[Columns.REWARDS] * reward_scale

        # Call parent method for actual loss computation
        return super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
