#!/usr/bin/env python3
"""
Generate a summary table from benchmark results in any directory.

Usage:
    python make_table.py results/final_2dot
    python make_table.py results/nelder_mead_ablation --format csv
    python make_table.py results/final_4dot --sort scans
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_result(path: Path) -> dict:
    """Load a benchmark result JSON file."""
    with open(path) as f:
        return json.load(f)


def find_first_convergence_step(trial: dict, num_plungers: int, num_barriers: int, threshold: float) -> int | None:
    """
    Find the first scan where all voltages are within threshold of optimal.

    Uses sum/num_gates < threshold as approximation (average distance < threshold).

    Returns:
        Scan number of first convergence, or None if never converged.
    """
    scan_numbers = trial.get("scan_numbers", [])
    plunger_dists = trial.get("plunger_distance_history", [])
    barrier_dists = trial.get("barrier_distance_history", [])

    if not scan_numbers or not plunger_dists:
        return None

    for i, scan in enumerate(scan_numbers):
        # Check if average distance per gate is within threshold
        avg_plunger = plunger_dists[i] / num_plungers if num_plungers > 0 else 0
        avg_barrier = barrier_dists[i] / num_barriers if num_barriers > 0 else 0

        if avg_plunger < threshold and avg_barrier < threshold:
            return scan

    return None


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from benchmark result."""
    trials = data.get("trials", [])
    num_dots = data.get("num_dots", 2)
    threshold = data.get("success_threshold", 0.5)

    num_plungers = num_dots
    num_barriers = num_dots - 1

    n_trials = len(trials)

    # Find first convergence step for each trial
    convergence_steps = []
    for t in trials:
        step = find_first_convergence_step(t, num_plungers, num_barriers, threshold)
        if step is not None:
            convergence_steps.append(step)

    n_success = len(convergence_steps)
    pct_converged = (n_success / n_trials * 100) if n_trials else 0

    # Mean steps to convergence (converged trials only)
    if convergence_steps:
        mean_steps = np.mean(convergence_steps)
        std_steps = np.std(convergence_steps)
    else:
        mean_steps = float('nan')
        std_steps = float('nan')

    return {
        "pct_converged": pct_converged,
        "mean_steps": mean_steps,
        "std_steps": std_steps,
        "n_trials": n_trials,
        "n_success": n_success,
    }


def extract_info(data: dict, path: Path) -> dict:
    """Extract method info and hyperparameters from result."""
    info = {
        "file": path.name,
        "method": data.get("method", "unknown"),
        "mode": data.get("mode", "unknown"),
        "num_dots": data.get("num_dots", "?"),
        "max_iterations": data.get("max_iterations", "?"),
        "success_threshold": data.get("success_threshold", "?"),
    }

    # Add method-specific hyperparameters if present
    optional_fields = [
        # Nelder-Mead
        "simplex_step_plunger", "simplex_step_barrier",
        "xatol", "fatol",
        # L-BFGS
        "ftol", "gtol", "maxcor",
        # Basin hopping
        "bh_T", "bh_stepsize", "bh_niter",
        # Pairwise mode (shared)
        "cap_per_plunger", "cap_per_barrier",
        "threshold_per_plunger", "threshold_per_barrier",
        "max_sweeps",
    ]
    for field in optional_fields:
        if field in data and data[field] is not None:
            info[field] = data[field]

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary table from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python make_table.py results/final_2dot
    python make_table.py results/nelder_mead_ablation --format csv
        """
    )
    parser.add_argument("dir", type=str, help="Directory containing result JSONs")
    parser.add_argument("--format", choices=["markdown", "csv", "simple"], default="simple",
                        help="Output format (default: simple)")
    parser.add_argument("--sort", choices=["converged", "steps", "name"], default="converged",
                        help="Sort by: converged %%, mean steps, or filename (default: converged)")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found")
        return 1

    # Load all results
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = load_result(path)
            metrics = extract_metrics(data)
            info = extract_info(data, path)
            results.append({**info, **metrics})
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    if not results:
        print("No results found")
        return 1

    # Sort results
    if args.sort == "converged":
        results.sort(key=lambda x: (-x["pct_converged"], x["mean_steps"] if not np.isnan(x["mean_steps"]) else float('inf')))
    elif args.sort == "steps":
        results.sort(key=lambda x: x["mean_steps"] if not np.isnan(x["mean_steps"]) else float('inf'))
    else:
        results.sort(key=lambda x: x["file"])

    # Detect if this is an ablation (multiple hyperparameter configs)
    has_hyperparams = any(k in results[0] for k in ["simplex_step_plunger", "xatol", "ftol", "gtol", "maxcor", "bh_T", "bh_stepsize"])

    # Output table
    if args.format == "simple":
        print(f"\n{'='*70}")
        print(f"Results from: {results_dir}")
        print(f"{'='*70}\n")

        for r in results:
            steps_str = f"{r['mean_steps']:.1f}" if not np.isnan(r['mean_steps']) else "N/A"
            std_str = f"±{r['std_steps']:.1f}" if not np.isnan(r['std_steps']) else ""

            print(f"{r['method']} ({r['mode']}) - {r['num_dots']} dots")
            print(f"  % Converged:    {r['pct_converged']:.1f}% ({r['n_success']}/{r['n_trials']})")
            print(f"  Mean Steps:     {steps_str} {std_str}")

            # Print hyperparams if present
            hp_strs = []
            # Nelder-Mead params
            if "simplex_step_plunger" in r:
                hp_strs.append(f"simplex_p={r['simplex_step_plunger']}")
            if "simplex_step_barrier" in r:
                hp_strs.append(f"simplex_b={r['simplex_step_barrier']}")
            if "xatol" in r:
                hp_strs.append(f"xatol={r['xatol']}")
            # L-BFGS params
            if "ftol" in r:
                hp_strs.append(f"ftol={r['ftol']}")
            if "gtol" in r:
                hp_strs.append(f"gtol={r['gtol']}")
            if "maxcor" in r:
                hp_strs.append(f"maxcor={r['maxcor']}")
            # Basin hopping params
            if "bh_T" in r:
                hp_strs.append(f"T={r['bh_T']}")
            if "bh_stepsize" in r:
                hp_strs.append(f"step={r['bh_stepsize']}")
            if "bh_niter" in r:
                hp_strs.append(f"niter={r['bh_niter']}")
            if hp_strs:
                print(f"  Params:         {', '.join(hp_strs)}")
            print()

    elif args.format == "markdown":
        if has_hyperparams:
            print("| Method | Dots | Params | % Converged | Mean Steps | Std |")
            print("|--------|------|--------|-------------|------------|-----|")
            for r in results:
                steps_str = f"{r['mean_steps']:.1f}" if not np.isnan(r['mean_steps']) else "N/A"
                std_str = f"{r['std_steps']:.1f}" if not np.isnan(r['std_steps']) else "N/A"
                params = []
                # Nelder-Mead
                if "simplex_step_plunger" in r:
                    params.append(f"sp={r['simplex_step_plunger']}")
                if "simplex_step_barrier" in r:
                    params.append(f"sb={r['simplex_step_barrier']}")
                if "xatol" in r:
                    params.append(f"tol={r['xatol']}")
                # L-BFGS
                if "ftol" in r:
                    params.append(f"ftol={r['ftol']}")
                if "gtol" in r:
                    params.append(f"gtol={r['gtol']}")
                if "maxcor" in r:
                    params.append(f"mc={r['maxcor']}")
                # Basin hopping
                if "bh_T" in r:
                    params.append(f"T={r['bh_T']}")
                if "bh_stepsize" in r:
                    params.append(f"step={r['bh_stepsize']}")
                if "bh_niter" in r:
                    params.append(f"niter={r['bh_niter']}")
                params_str = ", ".join(params) if params else "-"
                print(f"| {r['method']} | {r['num_dots']} | {params_str} | {r['pct_converged']:.1f}% | {steps_str} | {std_str} |")
        else:
            print("| Method | Mode | Dots | % Converged | Mean Steps | Std |")
            print("|--------|------|------|-------------|------------|-----|")
            for r in results:
                steps_str = f"{r['mean_steps']:.1f}" if not np.isnan(r['mean_steps']) else "N/A"
                std_str = f"{r['std_steps']:.1f}" if not np.isnan(r['std_steps']) else "N/A"
                print(f"| {r['method']} | {r['mode']} | {r['num_dots']} | {r['pct_converged']:.1f}% | {steps_str} | {std_str} |")

    else:  # csv
        headers = ["method", "mode", "num_dots", "pct_converged", "n_success", "n_trials", "mean_steps", "std_steps"]
        if has_hyperparams:
            # Add all possible hyperparams, they'll be empty if not present
            headers.extend(["simplex_step_plunger", "simplex_step_barrier", "xatol", "fatol", "ftol", "gtol", "maxcor", "bh_T", "bh_stepsize", "bh_niter"])
        print(",".join(headers))
        for r in results:
            row = [str(r.get(h, "")) for h in headers]
            print(",".join(row))

    return 0


if __name__ == "__main__":
    exit(main())
