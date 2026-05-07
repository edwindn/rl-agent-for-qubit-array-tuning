#!/usr/bin/env python3
"""
Aggregate ablation results into the final paper table.

Walks the per-algo collected_data dirs (named like
'<timestamp>_<algo>' under the configured output_root), computes 2/5/10V
convergence metrics via the existing ablation_metrics pipeline, and prints a
merged table + saves a JSON keyed by radius -> algo_name -> metrics.

Each algo's result is the LATEST run output for that algo
(max-timestamp prefix), so re-running an algo automatically supersedes
previous numbers.

Usage:
  uv run python compute_table.py
  uv run python compute_table.py --json out.json --markdown out.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ABLATION_DIR.parent))
from ablation_metrics import _compute_run_metrics  # type: ignore  # noqa: E402

CONFIG_PATH = ABLATION_DIR / "ablation_config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _latest_data_dir_per_algo(output_root: Path, algo_names: list[str]) -> dict:
    """For each algo name, find the most recent <timestamp>_<algo> dir."""
    result = {}
    if not output_root.exists():
        return result
    for d in output_root.iterdir():
        if not d.is_dir():
            continue
        # name like '20260504_153012_qadapt'
        parts = d.name.split("_", 2)
        if len(parts) < 3:
            continue
        algo = parts[2]
        if algo not in algo_names:
            continue
        # Keep the lexicographically largest (newest timestamp prefix)
        prev = result.get(algo)
        if prev is None or d.name > prev.name:
            result[algo] = d
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--radii", default=None,
                    help="Comma-separated. Defaults to ablation_config defaults.radii.")
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    length = args.length or defaults["episode_length"]
    radii = (
        [float(x) for x in args.radii.split(",")] if args.radii
        else [float(x) for x in defaults["radii"]]
    )
    output_root = Path(defaults["output_root"]).resolve()

    algos = cfg["algos"]
    data_dirs = _latest_data_dir_per_algo(output_root, list(algos))

    print(f"Length: {length}, Radii: {radii}")
    print(f"Output root: {output_root}")
    print(f"Algos with data: {sorted(data_dirs)} / {sorted(algos)}")
    missing = sorted(set(algos) - set(data_dirs))
    if missing:
        print(f"  MISSING: {missing}")

    # Compute per (radius, algo)
    table: dict[str, dict[str, dict]] = defaultdict(dict)
    for algo_name, data_dir in sorted(data_dirs.items()):
        for radius in radii:
            row = _compute_run_metrics(data_dir, radius, length)
            if row is None:
                print(f"  [{algo_name} @ r={radius}] no data (episodes shorter than length?)")
                continue
            row.pop("run", None)
            table[str(radius)][algo_name] = row

    # Print human-readable table
    print()
    for radius in radii:
        rkey = str(radius)
        print(f"=== radius = {radius} V ===")
        rows = table.get(rkey, {})
        for algo in sorted(rows):
            r = rows[algo]
            ms = r["mean_steps_to_converge"]
            ms_s = f"{ms:.2f}" if ms == ms else "nan"  # nan check
            print(
                f"  {algo:24s} "
                f"%conv={r['percent_converged']:5.1f}  "
                f"steps={ms_s:>6s}  "
                f"avg|d|={r['avg_abs_distance']:6.2f}  "
                f"({int(r['episodes_converged'])}/{int(r['episodes_total'])})"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2, sort_keys=True))
        print(f"\nWrote {args.json}")

    if args.markdown:
        lines = []
        for radius in radii:
            rkey = str(radius)
            lines.append(f"### Convergence radius = {radius} V\n")
            lines.append("| Algo | % conv | mean steps | avg \\|dist\\| | episodes |")
            lines.append("|---|---|---|---|---|")
            rows = table.get(rkey, {})
            for algo in sorted(rows):
                r = rows[algo]
                ms = r["mean_steps_to_converge"]
                ms_s = f"{ms:.2f}" if ms == ms else "—"
                lines.append(
                    f"| {algo} | {r['percent_converged']:.1f}% | {ms_s} | "
                    f"{r['avg_abs_distance']:.2f} | "
                    f"{int(r['episodes_converged'])}/{int(r['episodes_total'])} |"
                )
            lines.append("")
        Path(args.markdown).write_text("\n".join(lines))
        print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
