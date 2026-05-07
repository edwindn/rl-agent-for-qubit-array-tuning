#!/usr/bin/env python3
"""
Custom PPO learner that logs value function prediction statistics and gradient norms.
"""
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override


class PPOLearnerWithValueStats(PPOTorchLearner):
    @override(PPOTorchLearner)
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        """
        Compute loss and log value function statistics for plunger agent only.
        """
        # Call parent method to get base loss computation (unchanged behavior)
        total_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        
        # Only log value function stats for plunger policy
        if config.use_critic and "plunger" in module_id:
            module = self.module[module_id].unwrapped()
            
            # Get value function predictions (same computation as in parent)
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )
            
            # Compute and log only mean and variance for plunger agent
            with torch.no_grad():  # Don't track gradients for logging
                self.metrics.log_dict({
                    "vf_predictions_mean": torch.mean(value_fn_out),
                    "vf_predictions_variance": torch.var(value_fn_out),
                }, key=module_id, window=1)
        
        return total_loss