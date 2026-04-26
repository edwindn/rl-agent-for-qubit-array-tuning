"""
PyMARLEnvWrapper — adapts MultiAgentEnvWrapper (Ray RLlib MultiAgentEnv) to
PyMARL's flat-list MultiAgentEnv interface used by FACMAC.

Canonical agent ordering is plungers then barriers:
    index 0 ... num_dots-1           -> plunger_{i}
    index num_dots ... 2*num_dots-2  -> barrier_{j}

Obs storage is padded+channels-first per agent so the PyMARL episode buffer can
pack all agents into a single uniform tensor:
    plunger obs : (H, W, 2) -> permute -> (2, H, W) -> flatten -> (2*H*W,)
    barrier obs : (H, W, 1) -> permute -> (1, H, W) -> pad channel -> (2, H, W) -> flatten
GroupedMAC slices channel [0:1] back out for barriers before the CNN forward.

Team reward = sum of per-agent rewards (QMIX convention). Info carries
`episode_limit` so the runner can distinguish truncation from termination.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gym.spaces import Box

_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _THIS_DIR / "vendor"
_PROJECT_SRC = _THIS_DIR.parent.parent / "src"
for _p in (_VENDOR_DIR, _PROJECT_SRC):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from envs.multiagentenv import MultiAgentEnv  # vendor
from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


class PyMARLEnvWrapper(MultiAgentEnv):

    def __init__(self, batch_size: int | None = None, **kwargs: Any) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        args = self.args  # namedtuple from super().__init__

        env_config_path = getattr(args, "env_config_path", None)
        capacitance_model_checkpoint = getattr(args, "capacitance_model_checkpoint", None)
        training = getattr(args, "training", True)

        self._inner = MultiAgentEnvWrapper(
            training=training,
            return_voltage=False,
            env_config_path=env_config_path,
            capacitance_model_checkpoint=capacitance_model_checkpoint,
        )

        self.num_dots = self._inner.num_gates
        self.num_barriers = self._inner.num_barriers
        self.n_agents = self.num_dots + self.num_barriers
        self.episode_limit = self._inner.base_env.max_steps

        self.plunger_ids = list(self._inner.gate_agent_ids)
        self.barrier_ids = list(self._inner.barrier_agent_ids)
        self.ordered_agent_ids = self.plunger_ids + self.barrier_ids

        self.agent_groups = {
            "plunger": list(range(self.num_dots)),
            "barrier": list(range(self.num_dots, self.n_agents)),
        }

        sample_space = self._inner.observation_spaces[self.plunger_ids[0]]
        self.obs_H, self.obs_W, _ = sample_space.shape
        self.obs_channels_padded = 2
        self.flat_obs_size = self.obs_channels_padded * self.obs_H * self.obs_W

        self.obs_shapes = {
            "plunger": (2, self.obs_H, self.obs_W),
            "barrier": (1, self.obs_H, self.obs_W),
        }

        low = self._inner.action_spaces[self.plunger_ids[0]].low[0]
        high = self._inner.action_spaces[self.plunger_ids[0]].high[0]
        self._action_space_box = Box(low=low, high=high, shape=(1,), dtype=np.float32)
        self.action_spaces = [self._action_space_box for _ in range(self.n_agents)]

        self._latest_global_obs = None
        self._last_per_agent_rewards: dict[str, float] = {}
        self._steps_in_episode = 0
        self._episode_plunger_return = 0.0
        self._episode_barrier_return = 0.0

    def reset(self) -> None:
        global_obs_dict, _ = self._inner.reset()
        self._latest_global_obs = global_obs_dict
        self._steps_in_episode = 0
        self._episode_plunger_return = 0.0
        self._episode_barrier_return = 0.0

    def step(self, actions: "th.Tensor | np.ndarray") -> tuple[float, bool, dict]:
        if isinstance(actions, th.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        actions_np = actions_np.reshape(self.n_agents, -1)

        agent_actions = {
            agent_id: actions_np[i].astype(np.float32)
            for i, agent_id in enumerate(self.ordered_agent_ids)
        }

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self._inner.step(agent_actions)
        self._latest_global_obs = obs_dict
        self._steps_in_episode += 1

        team_reward = float(sum(reward_dict[a] for a in self.ordered_agent_ids))
        terminated = bool(term_dict["__all__"])
        truncated = bool(trunc_dict["__all__"])
        done = terminated or truncated

        # Vendor's episode_runner sums everything in info as numeric stats, so info
        # must be scalars-only. Full per-agent breakdown is stashed on the wrapper
        # for diagnostic access.
        self._last_per_agent_rewards = {a: float(reward_dict[a]) for a in self.ordered_agent_ids}

        # Per-group running returns — final-step values end up as cur_stats[k] and
        # are divided by n_episodes in runner._log, i.e. logged as the per-episode
        # mean team reward for plungers / barriers separately.
        step_plunger = sum(self._last_per_agent_rewards[a] for a in self.plunger_ids)
        step_barrier = sum(self._last_per_agent_rewards[a] for a in self.barrier_ids)
        self._episode_plunger_return += step_plunger
        self._episode_barrier_return += step_barrier

        n_p = max(1, len(self.plunger_ids))
        n_b = max(1, len(self.barrier_ids))
        info: dict = {
            "episode_limit": bool(truncated and not terminated),
            "plunger_return": float(self._episode_plunger_return),
            "barrier_return": float(self._episode_barrier_return),
            "plunger_return_avg": float(self._episode_plunger_return / n_p),
            "barrier_return_avg": float(self._episode_barrier_return / n_b),
        }
        return team_reward, done, info

    def _agent_obs_channels_first(self, agent_id: str) -> np.ndarray:
        """Returns the agent's local obs as a channels-first (C, H, W) array, padded to 2 channels."""
        agent_image = self._latest_global_obs[agent_id]  # (H, W, C_native)
        channels_first = np.transpose(agent_image, (2, 0, 1))  # (C_native, H, W)
        if channels_first.shape[0] == 1:
            pad = np.zeros_like(channels_first)
            channels_first = np.concatenate([channels_first, pad], axis=0)
        return channels_first.astype(np.float32)

    def get_obs(self) -> list[np.ndarray]:
        return [
            self._agent_obs_channels_first(a).flatten()
            for a in self.ordered_agent_ids
        ]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        return self._agent_obs_channels_first(self.ordered_agent_ids[agent_id]).flatten()

    def get_obs_size(self) -> int:
        return self.flat_obs_size

    def get_state(self) -> np.ndarray:
        per_agent = [self._agent_obs_channels_first(a).flatten() for a in self.ordered_agent_ids]
        return np.concatenate(per_agent).astype(np.float32)

    def get_state_size(self) -> int:
        return self.n_agents * self.flat_obs_size

    def get_avail_actions(self) -> list[np.ndarray]:
        return [np.ones(1, dtype=np.int64) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        return np.ones(1, dtype=np.int64)

    def get_total_actions(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        info = super().get_env_info()
        info["action_spaces"] = self.action_spaces
        info["actions_dtype"] = np.float32
        info["normalise_actions"] = False
        info["agent_groups"] = self.agent_groups
        info["obs_shapes"] = self.obs_shapes
        info["agent_ids"] = self.ordered_agent_ids
        return info

    def get_stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    def render(self) -> None:
        pass
