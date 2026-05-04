"""
Eval a grouped-policy SuperSims checkpoint vs a random-action baseline across
multiple seeds. Answers two questions:

  1. What's the realistic reward ceiling under hardware/cross-talk noise?
     Greedy rollouts from the trained policy give a lower bound on it.
  2. How much better is the policy vs random control? The random baseline
     samples actions uniformly from the action space [-1, 1]^5 each step.

Outputs:
  - eval_grouped_vs_random.png  — per-step mean-across-qubits reward,
    mean ± std band across seeds for both greedy and random.
  - terminal: final/mean reward stats for both, per-step climb shape.

Usage:
  CUDA_VISIBLE_DEVICES=7 uv run python scripts/eval_grouped_vs_random.py \
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


def run_episode(env, policy_split, modules, seed, mode="greedy", rng=None):
    """Returns (n_steps+1, n_qubits) per-step per-qubit rewards."""
    obs, info = env.reset(seed=seed)
    rewards = [info["per_qubit_rewards"].copy()]
    for _ in range(env.max_steps):
        if mode == "greedy":
            action = greedy_action(policy_split, modules, obs["staircase"], obs["params"])
        elif mode == "random":
            n_qubits = obs["params"].shape[0]
            action = rng.uniform(-1.0, 1.0, size=(n_qubits, 5)).astype(np.float32)
        else:
            raise ValueError(mode)
        obs, _, terminated, _, info = env.step(action)
        rewards.append(info["per_qubit_rewards"].copy())
        if terminated:
            break
    return np.asarray(rewards)


def _render(G: np.ndarray, R: np.ndarray, out: Path, n_seeds: int, ckpt_name: str):
    """Plot the convergence curve from precomputed arrays. Saves PNG + SVG.

    G and R are (n_seeds, n_steps+1) arrays of per-step mean-across-qubits
    reward. We plot the cumulative best ("closest the policy got, ever") along
    the step axis so the curve reflects "score we'd record if we kept the best
    configuration seen so far" — calibration is non-destructive: there's no
    cost to rolling back to an earlier-seen better point.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(G.shape[1])

    G_best = np.maximum.accumulate(G, axis=1)
    R_best = np.maximum.accumulate(R, axis=1)

    g_mean, g_std = G_best.mean(axis=0), G_best.std(axis=0)
    r_mean, r_std = R_best.mean(axis=0), R_best.std(axis=0)

    QADAPT_C = "#3369c6"   # blue from benchmarks palette
    RANDOM_C = "#303030"   # near-black axis colour

    ax.fill_between(x, g_mean - g_std, g_mean + g_std, color=QADAPT_C, alpha=0.22, lw=0)
    ax.fill_between(x, r_mean - r_std, r_mean + r_std, color=RANDOM_C, alpha=0.18, lw=0)
    ax.plot(x, g_mean, color=QADAPT_C, lw=1.6)
    ax.plot(x, r_mean, color=RANDOM_C, lw=1.4, ls="--")

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
    plt.tight_layout()

    out_png = out.with_suffix(".png")
    out_svg = out.with_suffix(".svg")
    out_npz = out.with_suffix(".npz")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    np.savez(out_npz, greedy=G, random=R, n_seeds=n_seeds, checkpoint=ckpt_name)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_svg}")
    print(f"Wrote {out_npz}  (raw seed arrays for re-styling)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a grouped checkpoint dir (required unless --from-npz).")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", type=str, default="plots_supersims_diagnostic/eval_grouped_vs_random.png")
    ap.add_argument("--from-npz", default=None,
                    help="Skip eval; replot from a .npz saved by a previous run.")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_npz:
        npz = np.load(args.from_npz, allow_pickle=True)
        G = npz["greedy"]
        R = npz["random"]
        n_seeds = int(npz["n_seeds"])
        ckpt_name = str(npz["checkpoint"])
        print(f"Replotting from {args.from_npz}: n_seeds={n_seeds}, ckpt={ckpt_name}")
        _render(G, R, out, n_seeds, Path(ckpt_name).name)
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

    greedy_runs, random_runs = [], []
    for s in range(args.n_seeds):
        print(f"\n[seed {s}] greedy...")
        g = run_episode(env, policy_split, modules, seed=s, mode="greedy")
        print(f"  step0 mean={g[0].mean():.4f}  final mean={g[-1].mean():.4f}  "
              f"episode mean={g.mean():.4f}")
        greedy_runs.append(g)

        print(f"[seed {s}] random-action...")
        rng = np.random.default_rng(seed=s)
        r = run_episode(env, policy_split, modules, seed=s, mode="random", rng=rng)
        print(f"  step0 mean={r[0].mean():.4f}  final mean={r[-1].mean():.4f}  "
              f"episode mean={r.mean():.4f}")
        random_runs.append(r)

    # Aggregate: per-step mean across qubits, per seed.
    G = np.stack([r.mean(axis=1) for r in greedy_runs], axis=0)  # (n_seeds, n_steps+1)
    R = np.stack([r.mean(axis=1) for r in random_runs], axis=0)

    print("\n=== Summary across seeds (mean per-step per-qubit reward) ===")
    print(f"  Greedy: step0 mean={G[:, 0].mean():.4f}  "
          f"final mean={G[:, -1].mean():.4f}  episode mean={G.mean():.4f}")
    print(f"  Random: step0 mean={R[:, 0].mean():.4f}  "
          f"final mean={R[:, -1].mean():.4f}  episode mean={R.mean():.4f}")
    print(f"  Lift   final = {G[:, -1].mean() - R[:, -1].mean():+.4f}")
    print(f"  Lift   episode-mean = {G.mean() - R.mean():+.4f}")

    _render(G, R, out, args.n_seeds, ckpt.name)


if __name__ == "__main__":
    main()
