#!/usr/bin/env python3
"""
Convert convergence_table.json (output of ablation_metrics.py) into the
appendix-ready markdown table for the rescue campaign.

Input layout:
{
  "2.0": {"run_facmac_F2_vdn": {...metrics...}, ...},
  "5.0": {...},
  "10.0": {...}
}

Output (markdown):

| Variant | Algo | Hypothesis tested | conv@2V (%) | conv@5V (%) | conv@10V (%) | mean steps@5V |
|...      | ...  | ...               | ...         | ...         | ...          | ...           |

Usage:
  uv run python scripts/rescue_appendix_table.py \\
    --input /tmp/eval_results/convergence_table.json \\
    --output /tmp/eval_results/appendix_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ALGO_DESCRIPTIONS = {
    "run_facmac_F1_lowcriticlr":   ("FACMAC", "F1",  "critic_lr 3e-4 → 5e-5"),
    "run_facmac_F2_vdn":           ("FACMAC", "F2",  "QMIX → VDN sum-mixer"),
    "run_facmac_F2b_nomixer":      ("FACMAC", "F2b", "no mixer (independent DDPG)"),
    "run_facmac_F3_slowtau":       ("FACMAC", "F3",  "Polyak τ 5e-3 → 1e-3"),
    "run_facmac_F4_rewardnorm":    ("FACMAC", "F4",  "running mean/std reward norm."),
    "run_maddpg_M1_td3":           ("MADDPG", "M1",  "TD3 (twin critics + delayed actor)"),
    "run_maddpg_M2_initbias":      ("MADDPG", "M2",  "actor final-layer bias U(±0.1)"),
    "run_maddpg_M3_lowcriticlr":   ("MADDPG", "M3",  "critic_lr 3e-4 → 1e-4"),
    "run_maddpg_M6_antizero":      ("MADDPG", "M6",  "−λ‖π‖² actor-loss term, λ=0.05"),
    "run_maddpg_M6b_strongantizero": ("MADDPG", "M6b", "as M6, λ=0.5 (10× stronger)"),
    "run_random_baseline":         ("baseline", "—", "uniform-random policy"),
}

RADIUS_ORDER = ["2.0", "5.0", "10.0"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())

    # collect run names that appear at any radius
    runs = sorted({r for radius in RADIUS_ORDER for r in data.get(radius, {})})

    rows = []
    for run in runs:
        algo, vid, descr = ALGO_DESCRIPTIONS.get(run, ("?", run, "?"))
        cells = [vid, algo, descr]
        for radius in RADIUS_ORDER:
            metrics = data.get(radius, {}).get(run, {})
            pct = metrics.get("percent_converged")
            cells.append(f"{pct:.1f}" if pct is not None else "—")
        # mean steps at 5V
        m5 = data.get("5.0", {}).get(run, {})
        steps = m5.get("mean_steps_to_converge")
        if steps is None or (isinstance(steps, float) and steps != steps):  # NaN
            cells.append("—")
        else:
            cells.append(f"{steps:.1f}")
        rows.append(cells)

    headers = ["ID", "Algo", "Change", "conv@2V (%)", "conv@5V (%)", "conv@10V (%)", "mean steps@5V"]
    sep = ["---"] * len(headers)

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    md = [fmt(headers), fmt(sep)] + [fmt(r) for r in rows]
    args.output.write_text("\n".join(md) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
