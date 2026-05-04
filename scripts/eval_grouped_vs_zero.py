"""
Eval a grouped-policy SuperSims checkpoint vs a zero-action baseline across
multiple seeds. Answers two questions:

  1. What's the realistic reward ceiling under hardware/cross-talk noise?
     Greedy rollouts from the trained policy give a lower bound on it.
  2. How much better is the policy vs doing nothing? The zero-action baseline
     emits 0.0 every step (no parameter movement from random init).

Outputs:
  - eval_grouped_vs_zero_reward.png  — per-step mean-across-qubits reward,
    one trace per seed, faded; bold mean across seeds for greedy and zero.
  - terminal: final/mean reward stats for both, per-step climb shape.

Usage:
  CUDA_VISIBLE_DEVICES=7 uv run python scripts/eval_grouped_vs_zero.py \
      --checkpoint checkpoints_supersims_grouped/iteration_16 \
      --n-seeds 5 --out plots_supersims_diagnostic/eval_grouped_iter16.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from swarm.inference.eval_supersims import (  # noqa: E402
    load_modules_from_checkpoint,
    greedy_action,
)
from swarm.environment.supersims_env import SuperSimsEnv  # noqa: E402


def run_episode(env, policy_split, modules, seed, mode="greedy"):
    """Returns (n_steps+1, n_qubits) per-step per-qubit rewards."""
    obs, info = env.reset(seed=seed)
    rewards = [info["per_qubit_rewards"].copy()]
    for _ in range(env.max_steps):
        if mode == "greedy":
            action = greedy_action(policy_split, modules, obs["staircase"], obs["params"])
        elif mode == "zero":
            n_qubits = obs["params"].shape[0]
            action = np.zeros((n_qubits, 5), dtype=np.float32)
        else:
            raise ValueError(mode)
        obs, _, terminated, _, info = env.step(action)
        rewards.append(info["per_qubit_rewards"].copy())
        if terminated:
            break
    return np.asarray(rewards)


def _render(G: np.ndarray, Z: np.ndarray, out: Path, n_seeds: int, ckpt_name: str):
    """Plot the convergence curve from precomputed arrays. Saves PNG + SVG."""
    # Shaded ±std band — benchmarks/results convergence-plot style for paper.
    # Match: DejaVu Sans, compact figsize, palette colours, no title/grid,
    # spines top/right hidden, legend frameless.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(G.shape[1])
    g_mean, g_std = G.mean(axis=0), G.std(axis=0)
    z_mean, z_std = Z.mean(axis=0), Z.std(axis=0)

    QADAPT_C = "#3369c6"   # blue from benchmarks palette
    ZERO_C   = "#303030"   # near-black axis colour
    CEIL_C   = "#309385"   # teal from benchmarks palette

    ax.fill_between(x, g_mean - g_std, g_mean + g_std, color=QADAPT_C, alpha=0.22, lw=0)
    ax.fill_between(x, z_mean - z_std, z_mean + z_std, color=ZERO_C,   alpha=0.18, lw=0)
    ax.plot(x, g_mean, color=QADAPT_C, lw=1.6, label="QAdapt (greedy)")
    ax.plot(x, z_mean, color=ZERO_C,   lw=1.4, ls="--", label="Zero-action")
    ax.axhline(0.998, color=CEIL_C, lw=1.0, ls=":", alpha=0.85, label="GT ceiling")

    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, G.shape[1] - 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#303030")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#303030", width=0.8)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderpad=0.2)
    plt.tight_layout()

    out_png = out.with_suffix(".png")
    out_svg = out.with_suffix(".svg")
    out_npz = out.with_suffix(".npz")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    np.savez(out_npz, greedy=G, zero=Z, n_seeds=n_seeds, checkpoint=ckpt_name)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_svg}")
    print(f"Wrote {out_npz}  (raw seed arrays for re-styling)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a grouped checkpoint dir (required unless --from-npz).")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=str, default="plots_supersims_diagnostic/eval_grouped_vs_zero.png")
    ap.add_argument("--from-npz", default=None,
                    help="Skip eval; replot from a .npz saved by a previous run.")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_npz:
        npz = np.load(args.from_npz, allow_pickle=True)
        G = npz["greedy"]
        Z = npz["zero"]
        n_seeds = int(npz["n_seeds"])
        ckpt_name = str(npz["checkpoint"])
        print(f"Replotting from {args.from_npz}: n_seeds={n_seeds}, ckpt={ckpt_name}")
        _render(G, Z, out, n_seeds, Path(ckpt_name).name)
        return

    if args.checkpoint is None:
        ap.error("--checkpoint is required unless --from-npz is given")
    ckpt = Path(args.checkpoint).resolve()
    print(f"Checkpoint: {ckpt}")
    print(f"Seeds:      {args.n_seeds}")

    print("\nLoading policy...")
    policy_split, modules = load_modules_from_checkpoint(ckpt)
    print(f"  policy_split={policy_split}")

    print("\nBuilding env (JIT warmup)...")
    env = SuperSimsEnv()

    greedy_runs, zero_runs = [], []
    for s in range(args.n_seeds):
        print(f"\n[seed {s}] greedy...")
        g = run_episode(env, policy_split, modules, seed=s, mode="greedy")
        print(f"  step0 mean={g[0].mean():.4f}  final mean={g[-1].mean():.4f}  "
              f"episode mean={g.mean():.4f}")
        greedy_runs.append(g)

        print(f"[seed {s}] zero-action...")
        z = run_episode(env, policy_split, modules, seed=s, mode="zero")
        print(f"  step0 mean={z[0].mean():.4f}  final mean={z[-1].mean():.4f}  "
              f"episode mean={z.mean():.4f}")
        zero_runs.append(z)

    # Aggregate: per-step mean across qubits, per seed.
    G = np.stack([r.mean(axis=1) for r in greedy_runs], axis=0)  # (n_seeds, n_steps+1)
    Z = np.stack([r.mean(axis=1) for r in zero_runs], axis=0)

    print("\n=== Summary across seeds (mean per-step per-qubit reward) ===")
    print(f"  Greedy: step0 mean={G[:, 0].mean():.4f}  "
          f"final mean={G[:, -1].mean():.4f}  episode mean={G.mean():.4f}")
    print(f"  Zero:   step0 mean={Z[:, 0].mean():.4f}  "
          f"final mean={Z[:, -1].mean():.4f}  episode mean={Z.mean():.4f}")
    print(f"  Lift   final = {G[:, -1].mean() - Z[:, -1].mean():+.4f}")
    print(f"  Lift   episode-mean = {G.mean() - Z.mean():+.4f}")

    _render(G, Z, out, args.n_seeds, ckpt.name)


if __name__ == "__main__":
    main()
