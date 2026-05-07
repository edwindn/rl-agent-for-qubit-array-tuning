"""
For a list of detunings spanning the policy's failure modes, render the All-XY
staircase the agent sees AND show the policy mean action it emits.

Picks 7 detunings:
  -50 MHz (max-negative — policy wrong sign)
  -35 MHz (sign-flip region)
  -10 MHz (typical small detuning — policy timid)
   0 MHz (GT)
  +10 MHz (typical small)
  +35 MHz (sign-flip region)
  +50 MHz (max-positive — policy wrong sign)

Outputs a 7x1 grid of staircases (qubit 0 only) with the ideal pattern overlaid,
plus a sidebar with [reward, policy mean, ideal sign-step].

Usage:
  CUDA_VISIBLE_DEVICES=2 uv run python scripts/plot_failure_mode_staircases.py \\
    --ckpt checkpoints_supersims_1d_omegad_freelogstd/iteration_17
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from qadapt.environment.supersims_env import SuperSimsEnv  # noqa: E402
from qadapt.inference.eval_supersims import load_modules_from_checkpoint  # noqa: E402
from all_xy_sequence import ALLXY_GATES, ALLXY_IDEAL  # noqa: E402
from reward import allxy_rewards  # noqa: E402

DETUNINGS_MHZ = [-50, -35, -10, 0, 10, 35, 50]
LABELS = [
    "policy WRONG sign",
    "policy WRONG sign",
    "policy timid (right sign)",
    "GT",
    "policy timid (right sign)",
    "policy WRONG sign",
    "policy WRONG sign",
]


@torch.no_grad()
def policy_mean(module, obs_batch):
    from ray.rllib.core.columns import Columns
    out = module._forward({"obs": obs_batch})
    logits = out[Columns.ACTION_DIST_INPUTS]
    half = logits.shape[-1] // 2
    return logits[:, :half].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints_supersims_1d_omegad_freelogstd/iteration_17")
    ap.add_argument("--config", type=str, default="supersims_env_config_1d_omegad.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qubit", type=int, default=0)
    args = ap.parse_args()

    ckpt = (_REPO / args.ckpt).resolve()
    print(f"Loading checkpoint: {ckpt}")
    split, modules = load_modules_from_checkpoint(ckpt)
    omegad_pi = modules["omegad_policy"]

    env = SuperSimsEnv(config_path=args.config)
    env.reset(seed=args.seed)
    print(f"Env reset (seed={args.seed}). Sampled t_g={float(env._t_g):.2f} ns,  "
          f"omega_01[q{args.qubit}]={float(env._params[args.qubit, 0])/(2*np.pi)*1000:.1f} MHz/2π")

    # For each test detuning, simulate the staircase, query the policy, save.
    rows = []
    for d_MHz in DETUNINGS_MHZ:
        # Reset and override omega_d
        env.reset(seed=args.seed)
        d_radns = d_MHz / 1000.0 * 2 * np.pi
        new_omega_d = env._params[:, 0] + d_radns
        env._params = env._params.at[:, 1].set(new_omega_d)
        P1 = env._run_sim()
        rewards, _ = allxy_rewards(P1)
        obs = env._make_obs(P1)
        obs_batch = {
            "staircase": torch.from_numpy(obs["staircase"].astype(np.float32)),
            "params":    torch.from_numpy(obs["params"].astype(np.float32)),
        }
        mean_action = policy_mean(omegad_pi, obs_batch).flatten()
        rows.append({
            "d_MHz": d_MHz,
            "P1_q":  np.asarray(P1)[args.qubit],            # (21,)
            "reward_q": float(np.asarray(rewards)[args.qubit]),
            "policy_mean_q": float(mean_action[args.qubit]),
            "norm_detuning": float(obs["params"][args.qubit, 1]),
        })

    # ---- Plot: 7 staircases stacked vertically, ideal overlaid in green. ---- #
    fig, axes = plt.subplots(len(DETUNINGS_MHZ), 1, figsize=(13, 2.4 * len(DETUNINGS_MHZ)),
                             sharex=True)
    target = np.array(ALLXY_IDEAL)
    x = np.arange(21)
    gate_labels = [f"({g[0]},{g[1]})" for g in ALLXY_GATES]

    for ax, row, label in zip(axes, rows, LABELS):
        ax.step(x, row["P1_q"], where="post", lw=2, color="C0", label="agent's staircase")
        ax.step(x, target, where="post", lw=2, color="green", ls="--", label="ideal target")
        ax.fill_between(x, target, row["P1_q"], step="post", alpha=0.15, color="red")
        # Right-side annotation: reward + policy action + ideal
        ideal_action = -np.sign(row["norm_detuning"])
        text = (
            f"detuning = {row['d_MHz']:+d} MHz   ({label})\n"
            f"  reward = {row['reward_q']:.3f}\n"
            f"  policy mean = {row['policy_mean_q']:+.3f}    "
            f"(ideal = {ideal_action:+.0f})\n"
            f"  norm obs  = {row['norm_detuning']:+.2f}"
        )
        ax.text(1.02, 0.5, text, transform=ax.transAxes, va="center",
                family="monospace", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="lavender", ec="grey"))
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f"P(|1>) q{args.qubit}", fontsize=9)
        ax.grid(True, alpha=0.3)
        # First subplot only: legend
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(gate_labels, rotation=60, ha="right", fontsize=7)
    axes[-1].set_xlabel("All-XY gate sequence")

    plt.suptitle(f"All-XY staircase at 7 detunings ({ckpt.name}, qubit {args.qubit})\n"
                 "agent's response to each: see right-side annotation", y=1.005)
    out_path = _REPO / "plots_supersims_diagnostic" / f"staircase_failure_modes_{ckpt.name}_q{args.qubit}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # Also print the table
    print(f"\nSummary:")
    print(f"  {'d_MHz':>6}  {'reward':>7}  {'policy_mean':>11}  {'ideal':>5}  {'norm_obs':>9}  comment")
    for row, label in zip(rows, LABELS):
        ideal_action = -np.sign(row["norm_detuning"])
        print(f"  {row['d_MHz']:>+6d}  {row['reward_q']:>7.3f}  "
              f"{row['policy_mean_q']:>+11.4f}  {ideal_action:>+5.0f}  "
              f"{row['norm_detuning']:>+9.3f}  {label}")


if __name__ == "__main__":
    main()
