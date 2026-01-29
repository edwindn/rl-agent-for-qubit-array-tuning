#!/usr/bin/env python3
"""
Custom PPO learner with comprehensive debugging logging.

Logs:
- Policy outputs (mean, std) to track if policy is changing
- Observation sanity checks
- Loss components breakdown
- Gradient norms per component
- Value predictions vs actual returns
"""
import torch
import numpy as np
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override


class PPOLearnerWithValueStats(PPOTorchLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_log_counter = 0
        self._debug_log_interval = 10  # Log detailed debug info every N batches

    @override(PPOTorchLearner)
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        """
        Compute loss and log comprehensive debugging statistics.
        """
        self._debug_log_counter += 1
        should_log_debug = (self._debug_log_counter % self._debug_log_interval == 1)

        # Get module reference
        if hasattr(self.module, "__getitem__"):
            module = self.module[module_id].unwrapped()
        else:
            module = self.module.unwrapped() if hasattr(self.module, "unwrapped") else self.module

        with torch.no_grad():
            # === 1. LOG OBSERVATION SANITY ===
            if should_log_debug and "obs" in batch:
                obs = batch["obs"]
                if isinstance(obs, dict):
                    if "image" in obs:
                        img = obs["image"]
                        self.metrics.log_dict({
                            "debug/obs_image_min": img.min().item(),
                            "debug/obs_image_max": img.max().item(),
                            "debug/obs_image_mean": img.mean().item(),
                        }, key=module_id, window=1)
                    if "voltage" in obs:
                        volt = obs["voltage"]
                        self.metrics.log_dict({
                            "debug/obs_voltage_min": volt.min().item(),
                            "debug/obs_voltage_max": volt.max().item(),
                            "debug/obs_voltage_mean": volt.mean().item(),
                        }, key=module_id, window=1)

            # === 2. LOG POLICY OUTPUTS (mean, std) ===
            action_dist_inputs = fwd_out.get(Columns.ACTION_DIST_INPUTS)
            if action_dist_inputs is not None:
                # For Gaussian: first half is mean, second half is log_std
                action_dim = action_dist_inputs.shape[-1] // 2
                means = action_dist_inputs[..., :action_dim]
                log_stds = action_dist_inputs[..., action_dim:]
                stds = log_stds.exp()

                self.metrics.log_dict({
                    "policy/action_mean_mean": means.mean().item(),
                    "policy/action_mean_std": means.std().item(),
                    "policy/action_mean_min": means.min().item(),
                    "policy/action_mean_max": means.max().item(),
                    "policy/action_std_mean": stds.mean().item(),
                    "policy/action_std_min": stds.min().item(),
                    "policy/action_std_max": stds.max().item(),
                    "policy/log_std_mean": log_stds.mean().item(),
                }, key=module_id, window=1)

            # === 3. LOG ACTUAL ACTIONS TAKEN ===
            if Columns.ACTIONS in batch:
                actions = batch[Columns.ACTIONS]
                self.metrics.log_dict({
                    "actions/mean": actions.mean().item(),
                    "actions/std": actions.std().item(),
                    "actions/min": actions.min().item(),
                    "actions/max": actions.max().item(),
                }, key=module_id, window=1)

            # === 4. LOG REWARDS ===
            if Columns.REWARDS in batch:
                rewards = batch[Columns.REWARDS]
                self.metrics.log_dict({
                    "rewards/mean": rewards.mean().item(),
                    "rewards/std": rewards.std().item(),
                    "rewards/min": rewards.min().item(),
                    "rewards/max": rewards.max().item(),
                }, key=module_id, window=1)

            # === 5. LOG VALUE PREDICTIONS VS RETURNS ===
            if config.use_critic:
                value_fn_out = module.compute_values(
                    batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
                )
                self.metrics.log_dict({
                    "vf_predictions_mean": value_fn_out.mean().item(),
                    "vf_predictions_std": value_fn_out.std().item(),
                    "vf_predictions_min": value_fn_out.min().item(),
                    "vf_predictions_max": value_fn_out.max().item(),
                }, key=module_id, window=1)

                # Compare to actual returns if available
                if Columns.VALUE_TARGETS in batch:
                    value_targets = batch[Columns.VALUE_TARGETS]
                    value_error = (value_fn_out.squeeze() - value_targets).abs()
                    self.metrics.log_dict({
                        "vf/target_mean": value_targets.mean().item(),
                        "vf/prediction_error_mean": value_error.mean().item(),
                    }, key=module_id, window=1)

        # === COMPUTE LOSS (parent method) ===
        total_loss = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )

        return total_loss

    @override(PPOTorchLearner)
    def _update(self, batch, **kwargs):
        """Override to log gradient norms after backward pass."""
        result = super()._update(batch, **kwargs)

        # Log gradient norms after update
        if self._debug_log_counter % self._debug_log_interval == 0:
            self._log_gradient_norms()

        return result

    def _log_gradient_norms(self):
        """Log gradient norms for different components."""
        if not hasattr(self, "module"):
            return

        # Get the module
        if hasattr(self.module, "__getitem__"):
            # Multi-agent case
            for module_id in self.module.keys():
                module = self.module[module_id].unwrapped()
                self._log_module_gradients(module, module_id)
        else:
            # Single-agent case
            module = self.module.unwrapped() if hasattr(self.module, "unwrapped") else self.module
            self._log_module_gradients(module, "default_policy")

    def _log_module_gradients(self, module, module_id):
        """Log gradient norms for a single module."""
        grad_norms = {
            "encoder": [],
            "pi": [],
            "vf": [],
        }

        for name, param in module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if "encoder" in name:
                    grad_norms["encoder"].append(grad_norm)
                elif "pi" in name:
                    grad_norms["pi"].append(grad_norm)
                elif "vf" in name:
                    grad_norms["vf"].append(grad_norm)

        log_dict = {}
        for component, norms in grad_norms.items():
            if norms:
                log_dict[f"grad_norm/{component}_mean"] = np.mean(norms)
                log_dict[f"grad_norm/{component}_max"] = np.max(norms)

        if log_dict:
            self.metrics.log_dict(log_dict, key=module_id, window=1)
