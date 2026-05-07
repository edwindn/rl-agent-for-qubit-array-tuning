#!/usr/bin/env python3
"""
Parse rescue_modal/*.log files and plot test_return_mean and critic_grad_norm
trajectories for each variant. Produces:
  - <out>/test_return_curves.png + .svg
  - <out>/critic_grad_curves.png + .svg

Usage:
  uv run python scripts/rescue_plot_curves.py --logs /tmp/claude_runs/rescue_modal --out /tmp/eval_results
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_HEADER = re.compile(r"Recent Stats \| t_env:\s+(\d+)")
KEY_VAL = re.compile(r"([a-z_]+):\s+([-+]?\d*\.?\d+)")

KEY_OF_INTEREST = ("test_return_mean", "critic_grad_norm", "agent_grad_norm",
                   "action_norm_sq", "test_action_norms_mean")

# colour scheme — algo families share a hue
COLORS = {
    "maddpg_M1_td3":           "#1f77b4",
    "maddpg_M2_initbias":      "#aec7e8",
    "maddpg_M3_lowcriticlr":   "#7fb1d3",
    "maddpg_M6_antizero":      "#0a3d62",
    "maddpg_M6b_strongantizero": "#000080",
    "facmac_F1_lowcriticlr":   "#d62728",
    "facmac_F2_vdn":           "#2ca02c",
    "facmac_F2b_nomixer":      "#98df8a",
    "facmac_F3_slowtau":       "#ff7f0e",
    "facmac_F4_rewardnorm":    "#9467bd",
}


def parse_log(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Walks the log, emits {key: [(t_env, value), ...]} for keys of interest."""
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in KEY_OF_INTEREST}
    cur_t = None
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        h = LOG_HEADER.search(line)
        if h:
            cur_t = int(h.group(1))
            continue
        if cur_t is None:
            continue
        # parse all "key: value" pairs on this line
        for m in KEY_VAL.finditer(line):
            k, v = m.group(1), m.group(2)
            if k in KEY_OF_INTEREST:
                try:
                    series[k].append((cur_t, float(v)))
                except ValueError:
                    pass
    return series


def plot_metric(metric_key: str, all_series: dict[str, dict[str, list]], out_base: Path,
                title: str, ylabel: str, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, series in sorted(all_series.items()):
        pts = series.get(metric_key, [])
        if not pts:
            continue
        # dedupe (same t_env emitted from multiple stat blocks)
        seen = {}
        for t, v in pts:
            seen[t] = v
        ts = sorted(seen)
        vs = [seen[t] for t in ts]
        ax.plot(ts, vs, label=variant, color=COLORS.get(variant, "gray"), lw=1.4)

    ax.set_xlabel("env step (t_env)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=160)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)
    print(f"wrote {out_base}.png + .svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("/tmp/claude_runs/rescue_modal"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/eval_results"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_series: dict[str, dict] = {}
    for log in sorted(args.logs.glob("*.log")):
        if log.name.endswith(".episode-runner-attempt"):
            continue
        variant = log.stem
        series = parse_log(log)
        if any(series.values()):
            all_series[variant] = series
            n_pts = sum(len(v) for v in series.values())
            print(f"  {variant}: {n_pts} points across {len([v for v in series.values() if v])} keys")

    if not all_series:
        print("no data found")
        return

    plot_metric("test_return_mean", all_series, args.out / "test_return_curves",
                title="test_return_mean vs t_env (rescue campaign)",
                ylabel="test_return_mean")
    plot_metric("critic_grad_norm", all_series, args.out / "critic_grad_curves",
                title="critic_grad_norm (pre-clip) vs t_env",
                ylabel="critic_grad_norm", log_y=True)
    plot_metric("agent_grad_norm", all_series, args.out / "agent_grad_curves",
                title="agent_grad_norm vs t_env",
                ylabel="agent_grad_norm", log_y=True)


if __name__ == "__main__":
    main()
