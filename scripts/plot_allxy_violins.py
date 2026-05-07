"""Appendix SuperSims violin: All-XY staircase, start vs end, across N_QUBITS.

Two stacked panels (top = start, bottom = end). x-axis = 21 All-XY gate pairs.
Within each gate pair, 4 violins (one per N ∈ {2, 4, 6, 8}, dodged horizontally),
pooling 100 seeds × N qubits of P(|1>) values per violin. Ideal staircase
{0×5, 0.5×12, 1×4} overlaid as a dashed reference.

Inputs: paper_plots/data/staircase_scan_N{N}.npz for each N.
Outputs: paper_plots/appendix_supersim_violin.{png,svg}.

Usage:
  uv run python scripts/plot_allxy_violins.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SuperSims"))

# Pull the canonical 21-pair labels + ideal staircase from the SuperSims source.
from all_xy_sequence import ALLXY_GATES, ALLXY_IDEAL  # noqa: E402

DEFAULT_N_VALUES = [2, 4, 6, 8]
DEFAULT_DATA_DIR = _REPO / "paper_plots" / "data"
DEFAULT_OUT = _REPO / "paper_plots" / "appendix_supersim_violin"


def _load_scans(data_dir: Path, n_values):
    """Returns dict[N] -> {"start": (S, n_q, 21), "end": (S, n_q, 21), "n_seeds": int}.

    Drops any N for which the npz isn't yet on disk so the plot can be generated
    iteratively as data lands.
    """
    out = {}
    for N in n_values:
        path = data_dir / f"staircase_scan_N{N}.npz"
        if not path.exists():
            print(f"[skip] N={N}: {path} not found")
            continue
        npz = np.load(path, allow_pickle=True)
        sc = npz["staircase_greedy"]  # (n_seeds, n_steps+1, n_q, n_allxy)
        # Sanity: env stores P(|1>) in [0, 1] after our normalisation reverse.
        out[N] = {
            "start": sc[:, 0, :, :].astype(np.float32),    # (S, n_q, 21)
            "end":   sc[:, -1, :, :].astype(np.float32),   # (S, n_q, 21)
            "n_seeds": int(npz["n_seeds"]),
            "n_q": int(npz["n_qubits"]),
        }
        print(f"[load] N={N}: {sc.shape[0]} seeds × {sc.shape[2]} qubits "
              f"= {sc.shape[0] * sc.shape[2]} samples per violin")
    return out


def _violin_data_per_pair(arr_S_Q_21):
    """arr: (n_seeds, n_q, 21) → list of length 21, each element is (n_seeds*n_q,)."""
    S, Q, P = arr_S_Q_21.shape
    flat = arr_S_Q_21.reshape(S * Q, P)
    return [flat[:, k] for k in range(P)]


def _plot_panel(ax, scans, key, n_values, colors, base_x, dodge):
    """Draw violins for one panel (start or end). Returns lengths of legend handles."""
    for j, N in enumerate(n_values):
        if N not in scans:
            continue
        data = _violin_data_per_pair(scans[N][key])  # 21 distributions
        positions = base_x + (j - (len(n_values) - 1) / 2) * dodge
        parts = ax.violinplot(
            data, positions=positions, widths=dodge * 0.95,
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(colors[j])
            body.set_edgecolor("none")
            body.set_alpha(0.75)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#222")
            parts["cmedians"].set_linewidth(0.6)


def render(data_dir: Path, out_stem: Path, n_values=DEFAULT_N_VALUES):
    scans = _load_scans(data_dir, n_values)
    if not scans:
        print("ERROR: no scans found, nothing to plot.")
        return
    actual_Ns = [N for N in n_values if N in scans]

    # Color palette: gouldian (from colorcet), matching the benchmarks paper figure.
    import colorcet as cc
    _cmap = cc.gouldian
    _idx = (np.linspace(0.0, 0.85, len(n_values)) * 256).astype(int)
    colors = [_cmap[i] for i in _idx]

    n_pairs = 21
    base_x = np.arange(n_pairs, dtype=float)
    dodge = 0.18  # horizontal spacing between violins within a gate pair

    _GATE_LATEX = {
        "I":    r"$I$",
        "Xpi":  r"$X_\pi$",
        "Ypi":  r"$Y_\pi$",
        "Xpi2": r"$X_{\pi/2}$",
        "Ypi2": r"$Y_{\pi/2}$",
    }
    seq_labels = [f"({_GATE_LATEX[g1]}, {_GATE_LATEX[g2]})" for g1, g2 in ALLXY_GATES]
    ideal = np.asarray(ALLXY_IDEAL)

    # Match the benchmark script: rely on matplotlib's default font.family /
    # font.sans-serif resolution (no explicit family override). Only the size
    # is set so the appendix figure is readable.
    plt.rcParams.update({
        "font.size": 14,
    })

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.0), sharex=True)
    titles = [
        ("start", "Random initialisation (step 0)"),
        ("end", "After 20 RL steps"),
    ]
    for ax, (key, title) in zip(axes, titles):
        # Ideal staircase reference.
        ax.step(base_x, ideal, where="mid", color="black", lw=1.2,
                ls="--", alpha=0.55, label="ideal", zorder=3)
        _plot_panel(ax, scans, key, n_values, colors, base_x, dodge)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(r"$P(|1\rangle)$", fontsize=14)
        ax.set_title(title, fontsize=14, loc="left", pad=4)
        ax.grid(axis="y", alpha=0.25)
        # All four spines visible, thick (consistent with convergence plot).
        for spine in ("top", "bottom", "left", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#303030")
            ax.spines[spine].set_linewidth(1.6)
        # Ticks pointing inward, on all four sides, longer.
        ax.tick_params(axis="y", which="both", left=True, right=True,
                       direction="in", length=8, width=1.2, colors="#303030")
        ax.tick_params(axis="x", which="both", top=True, bottom=True,
                       direction="in", length=8, width=1.2, colors="#303030")

    # X-axis: gate-pair labels on the bottom panel only.
    axes[-1].set_xticks(base_x)
    axes[-1].set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=11)
    axes[-1].set_xlabel("All-XY gate pair", fontsize=14)

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
