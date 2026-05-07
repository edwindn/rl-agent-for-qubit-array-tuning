"""Appendix SuperSims convergence: multi-N curves (4 panels, one per N).

For each N ∈ {2, 4, 6, 8}, plot the cumulative-best per-step mean-across-qubits
reward across seeds, mean ± std band, with a uniform-random-action baseline.
Mirrors the styling of scripts/eval_grouped_vs_random.py:_render so the existing
N=4 figure (eval_grouped_vs_random_iter28.png) remains the reference.

Cumulative-best convention: at each step t, score = max over t'<=t of the
per-step mean-across-qubits reward. Justification: calibration is non-destructive
— there's no cost to rolling back to a better-seen earlier point.

Inputs:  plots_supersims_diagnostic/staircase_scan_N{N}.npz
Outputs: plots_supersims_diagnostic/appendix_supersim_convergence.{png,svg}
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

DEFAULT_N_VALUES = [2, 4, 6, 8]
_REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _REPO / "plots_supersims_diagnostic"
DEFAULT_OUT = DEFAULT_DATA_DIR / "appendix_supersim_convergence"


def _load(data_dir: Path, N: int):
    p = data_dir / f"staircase_scan_N{N}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    # reward_*: (n_seeds, n_steps+1, n_q) — average across qubits per step
    g_per_step = d["reward_greedy"].mean(axis=2)   # (n_seeds, n_steps+1)
    r_per_step = d["reward_random"].mean(axis=2)
    return {
        "G": np.maximum.accumulate(g_per_step, axis=1),
        "R": np.maximum.accumulate(r_per_step, axis=1),
        "n_seeds": int(d["n_seeds"]),
        "n_q": int(d["n_qubits"]),
    }


def render(data_dir: Path, out_stem: Path, n_values=DEFAULT_N_VALUES):
    # Match the benchmark script: rely on matplotlib's default font.family /
    # font.sans-serif resolution (no explicit family override). Only the size
    # is set so the appendix figure is readable.
    plt.rcParams.update({
        "font.size": 14,
    })

    # Match the benchmarks paper figure: methods map to gouldian indices
    # COLORS = [_cmap[int(i)] for i in np.array([0.0, 0.17, 0.34, 0.47, 0.65, 0.85]) * 256]
    # 'random' -> COLORS[2] (idx 87, blue), 'ppo' (== QADAPT) -> COLORS[5] (idx 217, gold).
    import colorcet as cc
    GREEDY_C = cc.gouldian[int(0.85 * 256)]   # QADAPT — same hue as benchmarks plot
    RANDOM_C = cc.gouldian[int(0.34 * 256)]   # random — same hue as benchmarks plot

    runs = {N: _load(data_dir, N) for N in n_values}
    available = [N for N, r in runs.items() if r is not None]
    if not available:
        print("ERROR: no scan npzs found, nothing to plot.")
        return
    print(f"Plotting for N values: {available}")

    fig, axes = plt.subplots(1, len(available),
                              figsize=(3.5 * len(available), 3.0),
                              sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, N in zip(axes, available):
        d = runs[N]
        G, R = d["G"], d["R"]
        x = np.arange(G.shape[1])
        g_mean, g_std = G.mean(axis=0), G.std(axis=0)
        r_mean, r_std = R.mean(axis=0), R.std(axis=0)

        ax.fill_between(x, g_mean - g_std, g_mean + g_std,
                        color=GREEDY_C, alpha=0.22, lw=0)
        ax.fill_between(x, r_mean - r_std, r_mean + r_std,
                        color=RANDOM_C, alpha=0.18, lw=0)
        ax.plot(x, g_mean, color=GREEDY_C, lw=1.6, label="trained policy")
        ax.plot(x, r_mean, color=RANDOM_C, lw=1.4, ls="--", label="random")

        ax.set_xlim(0, G.shape[1] - 1)
        ax.set_xlabel("Time Step ($t$)", fontsize=14)
        ax.set_title(f"{N} qubits", fontsize=14)
        ax.set_yticks([0.5, 0.7, 0.9])
        # All four spines visible, thick (1.6).
        for spine in ("top", "bottom", "left", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#303030")
            ax.spines[spine].set_linewidth(1.6)
        # Ticks on all four sides, pointing inward, longer.
        ax.tick_params(axis="y", which="both", left=True, right=True,
                       direction="in", length=8, width=1.2, colors="#303030")
        ax.tick_params(axis="x", which="both", top=True, bottom=True,
                       direction="in", length=8, width=1.2, colors="#303030")
        ax.grid(axis="y", alpha=0.18)

    axes[0].set_ylabel("Score", fontsize=16)
    axes[0].set_ylim(0.5, 1.0)

    plt.tight_layout()

    out_png = out_stem.with_suffix(".png")
    out_svg = out_stem.with_suffix(".svg")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-stem", default=str(DEFAULT_OUT))
    ap.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    args = ap.parse_args()
    render(Path(args.data_dir), Path(args.out_stem), args.n_values)


if __name__ == "__main__":
    main()
