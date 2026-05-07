#!/usr/bin/env python3
"""
Custom PPO learner that logs value function prediction statistics and gradient norms.
"""
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override


class PPOLearnerWithValueStats(PPOTorchLearner):
    @override(PPOTorchLearner)
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        """
        Compute loss and log value function statistics for the single policy.
        """
        # Call parent method to get base loss computation (unchanged behavior)
        total_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        
        if config.use_critic:
            if hasattr(self.module, "__getitem__"):
                module = self.module[module_id].unwrapped()
            else:
                module = self.module.unwrapped() if hasattr(self.module, "unwrapped") else self.module
            
            # Get value function predictions (same computation as in parent)
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )
            
            # Compute and log mean and variance for the single policy
            with torch.no_grad():  # Don't track gradients for logging
                self.metrics.log_dict({
                    "vf_predictions_mean": torch.mean(value_fn_out),
                    "vf_predictions_variance": torch.var(value_fn_out),
                }, key=module_id, window=1)
        
        return total_loss
