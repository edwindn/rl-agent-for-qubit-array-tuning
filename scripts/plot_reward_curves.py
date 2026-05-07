#!/usr/bin/env python3
"""
Render the QADAPT-family training reward curves appendix figure.

Pulls per-run histories from wandb (chained runs stitched on the iteration axis)
and emits a single appendix figure:

  qadapt (473), nature_cnn (484), lstm (520), transformer (555),
  gamma_nonzero (511), w_o_virtualization=IPPO (496), MAPPO (648),
  single_agent_ppo (57)
  x = epoch (full extent of each run), y = episode_return_mean

Output: paper_plots/training_reward_curves_appendix.{png,svg}.

Usage:
  uv run python scripts/plot_reward_curves.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

# QADAPT-family runs. Some methods chain multiple wandb runs (resumed training);
# we stitch them on the `iteration` axis. Format: (project, [run_numbers...], label, y_key).
QADAPT_FAMILY = [
    ("rl_agents_for_tuning/RLModel", [473],            "QADAPT",                    "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [479, 484],       "Nature CNN backbone",       "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [477, 509, 520],  "LSTM memory",               "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [555],            "Transformer memory",        "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [511],            r"$\gamma > 0$",             "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [478, 482, 496],  "IPPO",                      "episode_return_mean"),
    ("rl_agents_for_tuning/RLModel", [647, 648],       "MAPPO",                     "episode_return_mean"),
    ("rl_agents_for_tuning/SingleAgentBenchmark", [57], "single-agent PPO",         "episode_return_mean"),
]

def _resolve_run(api, project: str, run_number: int):
    """For RLModel-style: find run by display_name suffix '-<number>'."""
    runs = list(api.runs(project, per_page=500))
    suffix = f"-{run_number}"
    hits = [r for r in runs if (r.display_name or "").endswith(suffix)]
    if not hits:
        return None
    return hits[0]


def _scan_history(run, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in run.scan_history(keys=[x_key, y_key]):
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def _stitch_chained_runs(api, project: str, run_numbers: list[int],
                          y_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull chained (resumed) runs in order, concatenate into a single curve.

    Each segment's iteration counter may restart at 0 — we offset by the
    cumulative max-iteration so the resulting x-axis is monotonically increasing.
    """
    seg_x: list[np.ndarray] = []
    seg_y: list[np.ndarray] = []
    offset = 0.0
    for num in run_numbers:
        run = _resolve_run(api, project, num)
        if run is None:
            print(f"    [skip seg] -{num}: not found")
            continue
        x, y = _scan_history(run, "iteration", y_key)
        if len(x) == 0:
            x, y = _scan_history(run, "_step", y_key)
        if len(x) == 0:
            print(f"    [skip seg] -{num}: no history rows")
            continue
        order = np.argsort(x); x, y = x[order], y[order]
        # If this segment's iteration counter restarted at <= offset, shift up.
        if len(seg_x) > 0 and float(x[0]) <= offset:
            shift = offset - float(x[0]) + 1.0
            x = x + shift
        seg_x.append(x); seg_y.append(y)
        offset = float(x.max())
        print(f"    [seg] -{num}: {len(x)} pts, iters {x.min():.0f}->{x.max():.0f}")
    if not seg_x:
        return np.empty(0), np.empty(0)
    return np.concatenate(seg_x), np.concatenate(seg_y)


def render_qadapt_family(api, out_stem: Path, max_iter: int | None = None) -> None:
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    import colorcet as cc
    palette_idx = (np.linspace(0.0, 0.95, len(QADAPT_FAMILY)) * 256).astype(int)
    colors = [cc.gouldian[int(i)] for i in palette_idx]

    plotted = 0
    global_max_x = 0.0
    for (project, run_nums, label, y_key), color in zip(QADAPT_FAMILY, colors):
        print(f"  [pull] {label}: runs {run_nums}")
        x, y = _stitch_chained_runs(api, project, run_nums, y_key)
        if len(x) == 0:
            print(f"  [skip] {label}: no data")
            continue
        if max_iter is not None:
            mask = x <= max_iter
            x, y = x[mask], y[mask]
        if len(x) < 2:
            continue
        ax.plot(x, y, color=color, lw=1.6, label=label, alpha=0.9)
        plotted += 1
        global_max_x = max(global_max_x, float(x.max()))
        print(f"  [plot] {label}: {len(x)} stitched points (max iter {x.max():.0f})")

    ax.set_xlim(0, max_iter if max_iter is not None else global_max_x)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Episode return")
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#303030")
        ax.spines[spine].set_linewidth(1.6)
    ax.tick_params(axis="both", which="both", direction="in", length=8, width=1.2,
                   colors="#303030", top=True, bottom=True, left=True, right=True)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", fontsize=10, frameon=False, ncol=2)

    fig.tight_layout()
    out_png = out_stem.with_suffix(".png")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png} (+ .svg)  — {plotted} runs plotted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "paper_plots")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="Optional cap on iterations (default: full extent of each run)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    print("== Training reward curves (QADAPT family) ==")
    render_qadapt_family(api, args.out_dir / "training_reward_curves_appendix",
                         max_iter=args.max_iter)


if __name__ == "__main__":
    main()
