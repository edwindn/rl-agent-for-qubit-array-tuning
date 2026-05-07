"""
Task 2 diagnostics — data-rich validation of PyMARLEnvWrapper.

Produces four artifacts under benchmarks/MARL/facmac/diagnostics/task_2/:

    1. env_info.txt         — get_env_info() dict + agent_groups printout
    2. obs_routing.png      — global scan vs each agent's local obs, labeled
    3. reward_trace.txt     — scripted-action reward parity check (FACMAC vs RLlib wrapper)
    4. state_reconstruction.png — global state vector reshaped back to per-agent tiles

Usage:
    cd benchmarks/MARL/facmac
    uv run --extra facmac python tests/task_2_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent.parent
_PROJECT_SRC = _BENCH_DIR.parent.parent / "src"
for _p in (_BENCH_DIR, _PROJECT_SRC):
    sys.path.insert(0, str(_p))

from env_wrapper import PyMARLEnvWrapper

OUT_DIR = _BENCH_DIR / "diagnostics" / "task_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env() -> PyMARLEnvWrapper:
    checkpoint = _PROJECT_SRC / "qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    env_args = {
        "env_config_path": None,
        "capacitance_model_checkpoint": str(checkpoint),
        "training": True,
        "seed": 12345,
    }
    return PyMARLEnvWrapper(env_args=env_args)


def dump_env_info(env: PyMARLEnvWrapper) -> None:
    info = env.get_env_info()
    lines = ["=== PyMARLEnvWrapper.get_env_info() ==="]
    for k, v in info.items():
        if k == "action_spaces":
            lines.append(f"  action_spaces: list[Box] x {len(v)}  (low={v[0].low[0]}, high={v[0].high[0]})")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=== Derived quantities ===")
    lines.append(f"  n_agents              = {env.n_agents}  ({env.num_dots} plungers + {env.num_barriers} barriers)")
    lines.append(f"  episode_limit         = {env.episode_limit}")
    lines.append(f"  obs_H, obs_W          = {env.obs_H}, {env.obs_W}")
    lines.append(f"  flat_obs_size         = {env.flat_obs_size}  (channels_padded={env.obs_channels_padded})")
    lines.append(f"  state_size            = {env.get_state_size()}")
    lines.append(f"  plunger agent ids     = {env.plunger_ids}")
    lines.append(f"  barrier agent ids     = {env.barrier_ids}")
    lines.append(f"  agent_groups          = {env.agent_groups}")

    out = OUT_DIR / "env_info.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  [written] {out}")


def plot_obs_routing(env: PyMARLEnvWrapper) -> None:
    """
    Three-row figure:
      Row 1: reconstructed global scan, one panel per global channel.
             Barrier i natively sees global channel i, so we recover the raw
             global scan from the barriers' own obs.
      Row 2: each plunger's 2-channel local obs tiled side-by-side.
             Label shows the channel pair it should have been assigned.
      Row 3: each barrier's obs (real channel + zero pad).
             Label shows max |value| of the pad region — must be ~0.

    Also runs consistency assertions and includes them in the title.
    """
    env.reset()
    obs_list = env.get_obs()
    chan_map = env._inner.agent_channel_map

    n_plungers = env.num_dots
    n_barriers = env.num_barriers
    n_channels_global = env.num_dots - 1

    reconstructed_global = np.zeros(
        (env.obs_H, env.obs_W, n_channels_global), dtype=np.float32
    )
    for j, agent_id in enumerate(env.barrier_ids):
        flat_idx = n_plungers + j
        barrier_cf = obs_list[flat_idx].reshape(2, env.obs_H, env.obs_W)
        reconstructed_global[:, :, j] = barrier_cf[0]

    assertions = []
    plunger_0_cf = obs_list[0].reshape(2, env.obs_H, env.obs_W)
    delta_p0 = float(np.max(np.abs(plunger_0_cf[0] - plunger_0_cf[1])))
    assertions.append(f"plunger_0 channels identical? max|diff|={delta_p0:.2e}")

    last_p = n_plungers - 1
    plunger_last_cf = obs_list[last_p].reshape(2, env.obs_H, env.obs_W)
    delta_pL = float(np.max(np.abs(plunger_last_cf[0] - plunger_last_cf[1])))
    assertions.append(f"plunger_{last_p} channels identical (both transposed)? max|diff|={delta_pL:.2e}")

    pad_max_overall = 0.0
    for j in range(n_barriers):
        flat_idx = n_plungers + j
        b_cf = obs_list[flat_idx].reshape(2, env.obs_H, env.obs_W)
        pad_max_overall = max(pad_max_overall, float(np.max(np.abs(b_cf[1]))))
    assertions.append(f"barrier pad region all zero? max|pad|={pad_max_overall:.2e}")

    max_row_width = max(n_channels_global, n_plungers, n_barriers)
    fig = plt.figure(figsize=(3.0 * max_row_width + 2, 11))
    gs = fig.add_gridspec(3, max_row_width)

    for c in range(n_channels_global):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(reconstructed_global[:, :, c], cmap="viridis", origin="lower")
        ax.set_title(f"global chan {c}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.005, 0.83, "GLOBAL SCAN\n(reconstructed\nfrom barriers)", fontsize=10, weight="bold", ha="left")

    for i, agent_id in enumerate(env.plunger_ids):
        obs_cf = obs_list[i].reshape(2, env.obs_H, env.obs_W)
        ax = fig.add_subplot(gs[1, i])
        tile = np.concatenate([obs_cf[0], obs_cf[1]], axis=1)
        ax.imshow(tile, cmap="viridis", origin="lower")
        ax.set_title(f"{agent_id}\nchans={chan_map[agent_id]}", fontsize=9)
        ax.axvline(env.obs_W, color="red", linewidth=0.7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.005, 0.50, "PLUNGERS\n(2-channel:\nleft|right)", fontsize=10, weight="bold", ha="left")

    for j, agent_id in enumerate(env.barrier_ids):
        flat_idx = n_plungers + j
        obs_cf = obs_list[flat_idx].reshape(2, env.obs_H, env.obs_W)
        ax = fig.add_subplot(gs[2, j])
        tile = np.concatenate([obs_cf[0], obs_cf[1]], axis=1)
        ax.imshow(tile, cmap="viridis", origin="lower")
        pad_max = float(np.max(np.abs(obs_cf[1])))
        ax.set_title(f"{agent_id}  chan={chan_map[agent_id][0]}\npad max|v|={pad_max:.1e}", fontsize=8)
        ax.axvline(env.obs_W, color="red", linewidth=0.7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.005, 0.17, "BARRIERS\n(real+pad)", fontsize=10, weight="bold", ha="left")

    fig.suptitle(
        "Task 2 obs routing — each agent's local obs.\n"
        + "  |  ".join(assertions),
        fontsize=10,
    )
    plt.tight_layout(rect=(0.10, 0, 1, 0.94))
    out = OUT_DIR / "obs_routing.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [written] {out}")
    for a in assertions:
        print(f"    - {a}")


def reward_parity_check(env: PyMARLEnvWrapper) -> None:
    """
    Per step, compares team_reward returned by step() against the sum of per-agent
    rewards exposed via info["per_agent_rewards"]. That invariant must hold exactly.

    Also prints per-group decomposition (plunger sum, barrier sum) so you can eyeball
    whether signs/magnitudes move sensibly as the scripted actions change.
    """
    rng = np.random.default_rng(7)
    env.reset()

    lines = ["=== Scripted-action reward trace ==="]
    lines.append(
        f"{'step':>4} | {'action':<20s} | {'plunger_sum':>12s} | {'barrier_sum':>12s} | {'team_reward':>12s} | {'sum(per_agent)':>14s} | {'abs diff':>10s}"
    )
    lines.append("-" * 104)

    scripted = [
        ("zeros",     np.zeros((env.n_agents, 1), dtype=np.float32)),
        ("all +0.5",  np.full((env.n_agents, 1), 0.5, dtype=np.float32)),
        ("all -0.5",  np.full((env.n_agents, 1), -0.5, dtype=np.float32)),
        ("rand 1",    rng.uniform(-0.3, 0.3, size=(env.n_agents, 1)).astype(np.float32)),
        ("rand 2",    rng.uniform(-0.3, 0.3, size=(env.n_agents, 1)).astype(np.float32)),
        ("rand 3",    rng.uniform(-0.3, 0.3, size=(env.n_agents, 1)).astype(np.float32)),
    ]

    max_diff = 0.0
    for step, (desc, action_vec) in enumerate(scripted):
        team_reward, done, info = env.step(action_vec)
        per_agent = env._last_per_agent_rewards
        plunger_sum = sum(per_agent[a] for a in env.plunger_ids)
        barrier_sum = sum(per_agent[a] for a in env.barrier_ids)
        per_agent_sum = sum(per_agent.values())
        diff = abs(team_reward - per_agent_sum)
        max_diff = max(max_diff, diff)
        lines.append(
            f"{step:>4} | {desc:<20s} | {plunger_sum:>+12.6f} | {barrier_sum:>+12.6f} | {team_reward:>+12.6f} | {per_agent_sum:>+12.6f} | {diff:>10.2e}"
        )
        if done:
            lines.append(f"     (episode ended at step {step})")
            break

    lines.append("")
    lines.append(f"Max |team_reward - sum(per_agent)| across scripted steps = {max_diff:.2e}")
    lines.append("Expectation: this should be 0 exactly (same float summation).")

    env.reset()
    r, done, info = env.step(np.zeros((env.n_agents, 1), dtype=np.float32))
    lines.append("")
    lines.append("=== Return-type check ===")
    lines.append(f"  reward type         : {type(r).__name__}  value={r:+.6f}")
    lines.append(f"  done type           : {type(done).__name__}   value={done}")
    lines.append(f"  info['episode_limit']: {info['episode_limit']}  (type {type(info['episode_limit']).__name__})")
    lines.append(f"  info keys           : {sorted(info.keys())[:8]}")

    out = OUT_DIR / "reward_trace.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  [written] {out}")


def plot_state_reconstruction(env: PyMARLEnvWrapper) -> None:
    env.reset()
    state_flat = env.get_state()

    expected_size = env.n_agents * env.flat_obs_size
    assert state_flat.shape == (expected_size,), f"state shape {state_flat.shape} != ({expected_size},)"

    per_agent = state_flat.reshape(env.n_agents, 2, env.obs_H, env.obs_W)

    fig, axes = plt.subplots(
        2, env.n_agents,
        figsize=(2.0 * env.n_agents, 4.2),
        squeeze=False,
    )
    for i, agent_id in enumerate(env.ordered_agent_ids):
        for c in range(2):
            ax = axes[c, i]
            ax.imshow(per_agent[i, c], cmap="viridis", origin="lower")
            if c == 0:
                ax.set_title(agent_id, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(f"chan {c}", fontsize=8)

    fig.suptitle(
        f"Global state reconstruction — flat vector of size {expected_size} sliced back into per-agent tiles.\n"
        "Barrier columns (rightmost) should have zero channel-1 (bottom row).",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "state_reconstruction.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [written] {out}")


def main() -> None:
    print(f"Writing diagnostics to {OUT_DIR}\n")
    env = make_env()

    print("[1/4] env_info")
    dump_env_info(env)
    print()

    print("[2/4] obs routing figure")
    plot_obs_routing(env)
    print()

    print("[3/4] reward parity check")
    reward_parity_check(env)
    print()

    print("[4/4] state reconstruction figure")
    plot_state_reconstruction(env)
    print()

    print(f"Done. Inspect artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
