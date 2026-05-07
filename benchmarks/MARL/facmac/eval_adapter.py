"""
eval_adapter.load_policy — reconstructs a deterministic FACMAC policy from
per-group actor checkpoints (`agent_plunger.th`, `agent_barrier.th`) and returns
a callable `policy(obs_dict) -> action_dict` usable by any rollout runner that
consumes MultiAgentEnvWrapper's dict format.

Mixer + critic weights are NEVER loaded — they live on the learner and aren't
needed at inference. This is what makes size-agnostic eval work: train on
4-dot, load the same actor weights into a policy built for 2/6/8-dot.

Only runs on CPU by default. If you want GPU, pass device="cuda".
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch as th
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _THIS_DIR / "vendor"
_PROJECT_SRC = _THIS_DIR.parent.parent / "src"
for _p in (_VENDOR_DIR, _PROJECT_SRC, _THIS_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from agents.plunger_cnn import PlungerCNNAgent
from agents.barrier_cnn import BarrierCNNAgent


def _load_env_config(env_config_path: Path) -> dict:
    with env_config_path.open() as f:
        return yaml.safe_load(f)


def _build_args(rnn_hidden_dim: int = 64) -> SimpleNamespace:
    return SimpleNamespace(rnn_hidden_dim=rnn_hidden_dim, n_actions=1)


def _agent_obs_channels_first(agent_image: np.ndarray) -> np.ndarray:
    """Matches env_wrapper._agent_obs_channels_first: transpose + zero-pad barriers to 2 channels."""
    cf = np.transpose(agent_image, (2, 0, 1))
    if cf.shape[0] == 1:
        pad = np.zeros_like(cf)
        cf = np.concatenate([cf, pad], axis=0)
    return cf.astype(np.float32)


def load_policy(
    checkpoint_dir: Path,
    env_config_path: Path,
    num_dots: int | None = None,
    rnn_hidden_dim: int = 64,
    device: str = "cpu",
) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """
    Args:
        checkpoint_dir: path like results/models/<run>/<step>/ containing
                        agent_plunger.th and agent_barrier.th.
        env_config_path: env_config yaml — used to determine resolution (H=W).
        num_dots: optional override. If None, uses env_config.simulator.num_dots.
                  Setting num_dots != training num_dots enables size transfer.
        rnn_hidden_dim: must match what training used (64 in our configs).
        device: "cpu" or "cuda" / "cuda:N".

    Returns:
        policy(obs_dict) -> action_dict
            obs_dict keys:  "plunger_0", ..., "plunger_{N-1}",
                            "barrier_0", ..., "barrier_{N-2}"
            obs values:      np.ndarray of shape (H, W, C_native)
                             plunger C=2, barrier C=1
            return values:   np.ndarray of shape (1,) — voltage in [-1, 1]

    The policy is deterministic (no exploration noise). Weights are loaded once
    at construction time; the returned callable is cheap to invoke per step.
    """
    checkpoint_dir = Path(checkpoint_dir)
    env_config_path = Path(env_config_path)

    env_cfg = _load_env_config(env_config_path)
    H = env_cfg["simulator"]["resolution"]
    W = H
    inferred_num_dots = env_cfg["simulator"]["num_dots"]
    N = num_dots if num_dots is not None else inferred_num_dots

    args = _build_args(rnn_hidden_dim=rnn_hidden_dim)

    plunger = PlungerCNNAgent(obs_shape=(2, H, W), args=args)
    barrier = BarrierCNNAgent(obs_shape=(1, H, W), args=args)

    plunger_ckpt = checkpoint_dir / "agent_plunger.th"
    barrier_ckpt = checkpoint_dir / "agent_barrier.th"
    if not plunger_ckpt.exists() or not barrier_ckpt.exists():
        raise FileNotFoundError(
            f"Expected {plunger_ckpt.name} and {barrier_ckpt.name} in {checkpoint_dir}"
        )
    plunger.load_state_dict(th.load(plunger_ckpt, map_location=device))
    barrier.load_state_dict(th.load(barrier_ckpt, map_location=device))

    plunger.to(device).eval()
    barrier.to(device).eval()

    plunger_ids = [f"plunger_{i}" for i in range(N)]
    barrier_ids = [f"barrier_{j}" for j in range(N - 1)]

    @th.no_grad()
    def policy(obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        action_dict: dict[str, np.ndarray] = {}

        if plunger_ids:
            imgs = np.stack(
                [_agent_obs_channels_first(obs_dict[a])[:2] for a in plunger_ids],
                axis=0,
            )
            x = th.from_numpy(imgs).to(device)
            hidden = plunger.init_hidden().expand(x.shape[0], -1)
            acts = plunger(x, hidden)["actions"].cpu().numpy()
            for i, agent_id in enumerate(plunger_ids):
                action_dict[agent_id] = acts[i].astype(np.float32)

        if barrier_ids:
            imgs = np.stack(
                [_agent_obs_channels_first(obs_dict[a])[:1] for a in barrier_ids],
                axis=0,
            )
            x = th.from_numpy(imgs).to(device)
            hidden = barrier.init_hidden().expand(x.shape[0], -1)
            acts = barrier(x, hidden)["actions"].cpu().numpy()
            for j, agent_id in enumerate(barrier_ids):
                action_dict[agent_id] = acts[j].astype(np.float32)

        return action_dict

    return policy
