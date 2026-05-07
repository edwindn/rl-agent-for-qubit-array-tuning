"""
GroupedMAC — custom multi-agent controller that holds one agent network per
group (plunger, barrier) with shared parameters within each group. Subclasses
CQMixMAC so the vendor's FACMAC learner, target MAC deepcopy, and continuous-
action paths all work unchanged.

Key behaviours:
  * `_build_agents` instantiates an `nn.ModuleDict` keyed by group name.
  * `forward` reshapes flat buffer obs -> (B*N_group, C, H, W) per group,
    slices the zero-pad channel for barriers, runs each group's agent, and
    stitches outputs back into canonical `(B, n_agents, n_actions)` order.
  * `select_actions` mirrors CQMixMAC's "mlp-like" path with gaussian noise
    plus action-space clamping.
  * `save_models` / `load_models` write one checkpoint per group so the eval
    adapter (task 5) can load actor-only weights without loading the mixer.

Shape invariant:
  `self.hidden_states` keeps the parent's layout — `(batch, n_agents, hidden_dim)` —
  so `select_actions`'s `self.hidden_states[bs]` slicing works unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces

from controllers.cqmix_controller import CQMixMAC


class GroupedMAC(CQMixMAC):

    def __init__(self, scheme: dict, groups: dict, args: Any) -> None:
        self.agent_groups: dict[str, list[int]] = args.agent_groups
        self.group_obs_shapes: dict[str, tuple[int, int, int]] = args.group_obs_shapes
        self.group_names = list(self.agent_groups.keys())
        self.agent_modules_cfg: dict[str, str] = args.agent_modules
        super().__init__(scheme, groups, args)

        sample = next(iter(self.group_obs_shapes.values()))
        _, self.obs_H, self.obs_W = sample
        self.obs_channels_padded = max(c for c, _, _ in self.group_obs_shapes.values())

    def _build_agents(self, input_shape: int) -> None:
        from agents import REGISTRY as agent_REGISTRY
        modules = {}
        for group in self.group_names:
            cls = agent_REGISTRY[self.agent_modules_cfg[group]]
            modules[group] = cls(self.group_obs_shapes[group], self.args)
        self.agents = nn.ModuleDict(modules)
        self.agent = self.agents  # back-compat alias for any parent-class touch-point

    def _reshape_group_obs(self, flat_group_obs: th.Tensor, group: str) -> th.Tensor:
        """flat (B*N, 2*H*W) -> (B*N, C_native, H, W). Strips pad for barriers."""
        native_C = self.group_obs_shapes[group][0]
        full = flat_group_obs.view(-1, self.obs_channels_padded, self.obs_H, self.obs_W)
        return full[:, :native_C].contiguous()

    def forward(
        self,
        ep_batch,
        t: int,
        actions: th.Tensor | None = None,
        hidden_states: th.Tensor | None = None,
        select_actions: bool = False,
        test_mode: bool = False,
    ) -> dict[str, th.Tensor]:
        bs = ep_batch.batch_size
        obs = ep_batch["obs"][:, t]  # (bs, n_agents, flat)

        action_out = th.zeros(
            bs, self.n_agents, self.args.n_actions,
            device=obs.device, dtype=obs.dtype,
        )
        new_hidden = self.hidden_states.clone() if self.hidden_states is not None else None

        for group, idxs in self.agent_groups.items():
            if not idxs:
                continue
            agent = self.agents[group]
            group_obs = obs[:, idxs, :]                      # (bs, N_g, flat)
            group_obs_flat = group_obs.reshape(bs * len(idxs), -1)
            group_inputs = self._reshape_group_obs(group_obs_flat, group)

            if new_hidden is not None:
                hidden_group = new_hidden[:, idxs, :].reshape(bs * len(idxs), -1)
            else:
                hidden_group = agent.init_hidden().expand(bs * len(idxs), -1).contiguous()

            ret = agent(group_inputs, hidden_group, actions=None)
            group_actions = ret["actions"].view(bs, len(idxs), self.args.n_actions)
            action_out[:, idxs, :] = group_actions

            if new_hidden is not None:
                new_hidden[:, idxs, :] = ret["hidden_state"].view(bs, len(idxs), -1)

        if select_actions:
            self.hidden_states = new_hidden
            return {"actions": action_out, "hidden_state": new_hidden}

        self.hidden_states = new_hidden
        return action_out, actions

    def select_actions(
        self,
        ep_batch,
        t_ep: int,
        t_env: int | None,
        bs: slice = slice(None),
        test_mode: bool = False,
        past_actions=None,
        critic=None,
        target_mac: bool = False,
        explore_agent_ids=None,
        **kwargs,
    ) -> th.Tensor:
        chosen_actions = self.forward(
            ep_batch[bs], t_ep,
            hidden_states=self.hidden_states[bs] if self.hidden_states is not None else None,
            test_mode=test_mode, select_actions=True,
        )["actions"]
        chosen_actions = chosen_actions.view(
            ep_batch[bs].batch_size, self.n_agents, self.args.n_actions
        ).detach()

        if not test_mode:
            exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
            if exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env is None or t_env >= start_steps:
                    noise = th.zeros_like(chosen_actions).normal_()
                    chosen_actions = chosen_actions + act_noise * noise
                else:
                    chosen_actions = th.from_numpy(np.array([
                        [self.args.action_spaces[i].sample() for i in range(self.n_agents)]
                        for _ in range(ep_batch[bs].batch_size)
                    ])).float().to(device=ep_batch.device)

        if all(isinstance(a, spaces.Box) for a in self.args.action_spaces):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    lo = np.asscalar(self.args.action_spaces[_aid].low[_actid])
                    hi = np.asscalar(self.args.action_spaces[_aid].high[_actid])
                    chosen_actions[:, _aid, _actid].clamp_(lo, hi)
        return chosen_actions

    def init_hidden(self, batch_size: int) -> None:
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = th.device("cpu")
        self.hidden_states = th.zeros(
            batch_size, self.n_agents, self.args.rnn_hidden_dim, device=device,
        )

    def parameters(self):
        for a in self.agents.values():
            yield from a.parameters()

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.agents.named_parameters(prefix=prefix, recurse=recurse)

    def load_state(self, other_mac: "GroupedMAC") -> None:
        for name in self.agents:
            self.agents[name].load_state_dict(other_mac.agents[name].state_dict())

    def load_state_from_state_dict(self, state_dict: dict) -> None:
        for name, agent in self.agents.items():
            prefixed = {
                k[len(name) + 1:]: v for k, v in state_dict.items()
                if k.startswith(f"{name}.")
            }
            agent.load_state_dict(prefixed)

    def cuda(self, device: str = "cuda") -> None:
        for agent in self.agents.values():
            agent.cuda(device=device)

    def share(self) -> None:
        for agent in self.agents.values():
            agent.share_memory()

    def save_models(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        for name, agent in self.agents.items():
            th.save(agent.state_dict(), f"{path}/agent_{name}.th")

    def load_models(self, path: str) -> None:
        for name, agent in self.agents.items():
            agent.load_state_dict(
                th.load(f"{path}/agent_{name}.th", map_location=lambda s, _: s)
            )

    def _get_input_shape(self, scheme: dict) -> int:
        return scheme["obs"]["vshape"]

    def _build_inputs(self, batch, t, target_mac: bool = False, last_target_action=None):
        return batch["obs"][:, t]
