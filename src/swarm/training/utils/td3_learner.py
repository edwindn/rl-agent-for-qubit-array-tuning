"""TD3 Torch Learner.

Key differences from SAC:
1. No alpha/entropy optimization
2. Delayed policy updates (every N critic updates)
3. Target policy smoothing in Q-target computation
4. Take minimum of twin Q-networks for target
"""

from typing import Any, Dict

import torch
from ray.rllib.algorithms.sac.torch.sac_torch_learner import SACTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType


# Loss keys (reuse some from SAC)
POLICY_LOSS_KEY = "policy_loss"
QF_LOSS_KEY = "qf_loss"
QF_TWIN_LOSS_KEY = "qf_twin_loss"
TD_ERROR_KEY = "td_error"
TD_ERROR_MEAN_KEY = "td_error_mean"
QF_MEAN_KEY = "qf_mean"
QF_MAX_KEY = "qf_max"
QF_MIN_KEY = "qf_min"


class TD3TorchLearner(SACTorchLearner):
    """TD3 Learner with delayed policy updates.

    Overrides SAC loss computation to:
    1. Remove alpha/entropy terms
    2. Use minimum of twin Q-networks for target
    3. Delay policy updates by policy_frequency
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track gradient steps for delayed policy updates
        self._gradient_steps: Dict[ModuleID, int] = {}
        # Store losses for compute_gradients
        self._temp_losses: Dict[tuple, torch.Tensor] = {}

    @override(SACTorchLearner)
    def build(self) -> None:
        super().build()
        # Initialize step counters per module
        for module_id in self._module._rl_modules.keys():
            self._gradient_steps[module_id] = 0

    @override(SACTorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config=None
    ) -> None:
        """Configure optimizers - no alpha optimizer for TD3."""
        module = self._module[module_id]

        # Get parameters helper
        def get_params(model):
            return list(model.parameters())

        # Critic optimizer (Q-network 1)
        params_critic = get_params(module.qf_encoder) + get_params(module.qf)
        optim_critic = torch.optim.Adam(params_critic, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="qf",
            optimizer=optim_critic,
            params=params_critic,
            lr_or_lr_schedule=config.critic_lr,
        )

        # Twin critic optimizer (if enabled)
        if config.twin_q:
            params_twin = get_params(module.qf_twin_encoder) + get_params(
                module.qf_twin
            )
            optim_twin = torch.optim.Adam(params_twin, eps=1e-7)
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="qf_twin",
                optimizer=optim_twin,
                params=params_twin,
                lr_or_lr_schedule=config.critic_lr,
            )

        # Actor optimizer
        params_actor = get_params(module.pi_encoder) + get_params(module.pi)
        optim_actor = torch.optim.Adam(params_actor, eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="policy",
            optimizer=optim_actor,
            params=params_actor,
            lr_or_lr_schedule=config.actor_lr,
        )
        # Note: No alpha optimizer in TD3

    @override(SACTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        """Compute TD3 loss (critic + delayed actor)."""

        # Increment gradient step counter
        self._gradient_steps[module_id] = self._gradient_steps.get(module_id, 0) + 1

        # Get Q-values for current actions (from replay buffer)
        q_selected = fwd_out["qf_preds"]
        if config.twin_q:
            q_twin_selected = fwd_out["qf_twin_preds"]

        # Compute target Q-value using minimum of twin Q-networks
        # Target actions already have smoothing noise applied in forward_train
        q_target_next = fwd_out["q_target_next"]

        # Mask terminated states
        q_next_masked = (1.0 - batch[Columns.TERMINATEDS].float()) * q_target_next

        # Bellman target (no entropy term in TD3)
        n_step = batch.get("n_step", 1)
        if isinstance(n_step, torch.Tensor):
            n_step = n_step.float().mean().item()
        q_selected_target = (
            batch[Columns.REWARDS] + (config.gamma**n_step) * q_next_masked
        ).detach()

        # TD error for prioritized replay
        td_error = torch.abs(q_selected - q_selected_target)
        if config.twin_q:
            td_error = td_error + torch.abs(q_twin_selected - q_selected_target)
            td_error = td_error * 0.5

        # Critic loss (Huber loss for stability)
        weights = batch.get("weights", torch.ones_like(q_selected))
        critic_loss = torch.mean(
            weights
            * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                q_selected, q_selected_target
            )
        )

        if config.twin_q:
            critic_twin_loss = torch.mean(
                weights
                * torch.nn.HuberLoss(reduction="none", delta=1.0)(
                    q_twin_selected, q_selected_target
                )
            )
        else:
            critic_twin_loss = torch.tensor(0.0, device=q_selected.device)

        # Delayed policy update
        policy_frequency = getattr(config, "policy_frequency", 2)
        update_policy = (self._gradient_steps[module_id] % policy_frequency) == 0

        if update_policy:
            # Actor loss: maximize Q(s, pi(s)) -> minimize -Q(s, pi(s))
            actor_loss = -torch.mean(fwd_out["q_curr"])
        else:
            # Zero actor loss when not updating
            actor_loss = torch.tensor(
                0.0, device=q_selected.device, requires_grad=False
            )

        # Total loss
        total_loss = critic_loss + critic_twin_loss
        if update_policy:
            total_loss = total_loss + actor_loss

        # Log metrics
        self.metrics.log_value(
            key=(module_id, TD_ERROR_KEY),
            value=td_error,
            reduce=None,
            clear_on_reduce=True,
        )

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: actor_loss.detach() if update_policy else actor_loss,
                QF_LOSS_KEY: critic_loss.detach(),
                QF_MEAN_KEY: torch.mean(fwd_out["q_curr"]).detach(),
                QF_MAX_KEY: torch.max(fwd_out["q_curr"]).detach(),
                QF_MIN_KEY: torch.min(fwd_out["q_curr"]).detach(),
                TD_ERROR_MEAN_KEY: torch.mean(td_error).detach(),
                "policy_update": float(update_policy),
                "gradient_steps": float(self._gradient_steps[module_id]),
            },
            key=module_id,
            window=1,
        )

        # Store losses for compute_gradients
        self._temp_losses[(module_id, "qf_loss")] = critic_loss
        self._temp_losses[(module_id, "policy_loss")] = actor_loss

        if config.twin_q:
            self.metrics.log_value(
                key=(module_id, QF_TWIN_LOSS_KEY),
                value=critic_twin_loss.detach(),
                window=1,
            )
            self._temp_losses[(module_id, "qf_twin_loss")] = critic_twin_loss

        return total_loss

    @override(SACTorchLearner)
    def compute_gradients(
        self, loss_per_module: Dict[ModuleID, TensorType], **kwargs
    ) -> Dict[str, Any]:
        """Compute gradients - skip policy gradients when not updating."""
        grads = {}

        for module_id in set(loss_per_module.keys()) - {"__all__"}:
            config = self.config.get_config_for_module(module_id)
            policy_frequency = getattr(config, "policy_frequency", 2)
            update_policy = (self._gradient_steps[module_id] % policy_frequency) == 0

            for optim_name, optim in self.get_optimizers_for_module(module_id):
                # Skip policy optimizer when not updating
                if optim_name == "policy" and not update_policy:
                    continue

                optim.zero_grad(set_to_none=True)

                loss_key = f"{optim_name}_loss"
                loss_tensor = self._temp_losses.pop((module_id, loss_key), None)

                if loss_tensor is not None and loss_tensor.requires_grad:
                    loss_tensor.backward(retain_graph=True)

                    # Collect gradients
                    param_dict = self.filter_param_dict_for_optimizer(
                        self._params, optim
                    )
                    for pid, p in param_dict.items():
                        if p.grad is not None:
                            grads[pid] = p.grad

        # Clear any remaining temp losses
        self._temp_losses.clear()
        return grads
