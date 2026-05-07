#!/usr/bin/env python3
"""
Generate a summary table from Nelder-Mead hyperparameter ablation results.

Usage:
    python make_table.py [--dir results/nelder_mead_ablation]
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_result(path: Path) -> dict:
    """Load a benchmark result JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from benchmark result."""
    trials = data.get("trials", [])

    successful = [t for t in trials if t["success"]]
    success_rate = len(successful) / len(trials) if trials else 0

    # Mean scans for successful trials only
    if successful:
        mean_scans_success = np.mean([t["num_scans"] for t in successful])
        std_scans_success = np.std([t["num_scans"] for t in successful])
    else:
        mean_scans_success = float('nan')
        std_scans_success = float('nan')

    # Mean scans for all trials
    mean_scans_all = np.mean([t["num_scans"] for t in trials]) if trials else float('nan')

    return {
        "success_rate": success_rate,
        "mean_scans_success": mean_scans_success,
        "std_scans_success": std_scans_success,
        "mean_scans_all": mean_scans_all,
        "n_trials": len(trials),
        "n_success": len(successful),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ablation table")
    parser.add_argument("--dir", type=str,
                        default="benchmarks/results/nelder_mead_ablation",
                        help="Directory containing result JSONs")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown",
                        help="Output format")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found")
        return

    # Load all results
    results = []
    for path in sorted(results_dir.glob("*.json")):
        data = load_result(path)
        metrics = extract_metrics(data)

        # Extract hyperparams from data or filename
        results.append({
            "file": path.name,
            "simplex_plunger": data.get("simplex_step_plunger", "?"),
            "simplex_barrier": data.get("simplex_step_barrier", "?"),
            "xatol": data.get("xatol", "?"),
            "fatol": data.get("fatol", "?"),
            **metrics,
        })

    if not results:
        print("No results found")
        return

    # Sort by success rate (descending), then by mean scans (ascending)
    results.sort(key=lambda x: (-x["success_rate"], x["mean_scans_success"] if not np.isnan(x["mean_scans_success"]) else float('inf')))

    # Output table
    if args.format == "markdown":
        print("| Simplex (P) | Simplex (B) | xatol | fatol | Success Rate | Mean Scans (success) | Std |")
        print("|-------------|-------------|-------|-------|--------------|----------------------|-----|")
        for r in results:
            scans_str = f"{r['mean_scans_success']:.1f}" if not np.isnan(r['mean_scans_success']) else "N/A"
            std_str = f"{r['std_scans_success']:.1f}" if not np.isnan(r['std_scans_success']) else "N/A"
            print(f"| {r['simplex_plunger']} | {r['simplex_barrier']} | {r['xatol']} | {r['fatol']} | {r['success_rate']*100:.1f}% | {scans_str} | {std_str} |")
    else:
        print("simplex_plunger,simplex_barrier,xatol,fatol,success_rate,mean_scans_success,std_scans_success")
        for r in results:
            print(f"{r['simplex_plunger']},{r['simplex_barrier']},{r['xatol']},{r['fatol']},{r['success_rate']:.3f},{r['mean_scans_success']:.1f},{r['std_scans_success']:.1f}")


if __name__ == "__main__":
    main()
