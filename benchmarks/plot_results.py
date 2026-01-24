"""
Plot benchmark comparison across methods and array sizes.

Two main plots:
1. Scans to threshold vs num_dots (for converged trials)
2. Convergence curves: normalized convergence score vs scans

Usage:
    uv run python plot_results.py --dir results/final_results
    uv run python plot_results.py --dir results/final_results --plot scans
    uv run python plot_results.py --dir results/final_results --plot convergence --num-dots 4
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from env_init import get_voltage_ranges_from_config


def cumulative_min(data: np.ndarray) -> np.ndarray:
    """Compute cumulative minimum (best seen so far)."""
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = min(result[i - 1], data[i])
    return result


def get_plunger_voltage_range() -> float:
    """
    Load the plunger voltage range from centralized env config.

    Returns the midpoint of the full_plunger_range_width config,
    which represents the maximum possible distance from ground truth.
    """
    plunger_range, _ = get_voltage_ranges_from_config()
    return plunger_range


def load_all_results(results_dir: Path) -> list:
    """Load all JSON result files."""
    results = []
    for path in results_dir.glob("*.json"):
        with open(path) as f:
            results.append(json.load(f))
    return results


def obj_to_mean_abs_dist(obj: float, num_gates: int) -> float:
    """Convert objective (sum of squared distances) to mean absolute distance."""
    return np.sqrt(np.abs(obj)) / np.sqrt(num_gates)


def get_scans_to_threshold(history: list, num_gates: int, threshold: float) -> int | None:
    """
    Find the first scan where mean abs distance drops below threshold.
    Returns None if threshold was never reached.
    """
    for i, obj in enumerate(history):
        dist = obj_to_mean_abs_dist(obj, num_gates)
        if dist <= threshold:
            return i + 1  # 1-indexed scan count
    return None


def plot_scans_to_threshold(results_dir: Path, output_path: Path = None, threshold: float = 0.5):
    """
    Plot 1: X = num_dots, Y = mean scans to reach threshold.

    Only includes trials that eventually converged (reached threshold).
    Shows mean with std error bars.
    """
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    # Group by method
    methods = defaultdict(lambda: defaultdict(list))

    for r in results:
        method = r["method"]
        num_dots = r["num_dots"]
        use_barriers = r.get("use_barriers", True)
        num_gates = num_dots + (num_dots - 1 if use_barriers else 0)

        for trial in r.get("trials", []):
            history = trial.get("global_objective_history", [])
            if not history:
                continue

            scans = get_scans_to_threshold(history, num_gates, threshold)
            if scans is not None:
                methods[method][num_dots].append(scans)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'nelder_mead': 'C0', 'lbfgs': 'C1', 'random': 'C2', 'bayesian': 'C3'}
    markers = {'nelder_mead': 'o', 'lbfgs': 's', 'random': '^', 'bayesian': 'D'}

    all_dots = set()
    for method, data in sorted(methods.items()):
        dots = sorted(data.keys())
        all_dots.update(dots)

        means = []
        stds = []
        valid_dots = []

        for d in dots:
            scans_list = data[d]
            if scans_list:
                valid_dots.append(d)
                means.append(np.mean(scans_list))
                stds.append(np.std(scans_list))

        if not valid_dots:
            continue

        color = colors.get(method, 'gray')
        marker = markers.get(method, 'o')

        ax.errorbar(
            valid_dots, means,
            yerr=stds,
            marker=marker, capsize=5, capthick=2,
            label=method, color=color,
            linewidth=2, markersize=8
        )

    ax.set_xlabel("Number of Dots", fontsize=12)
    ax.set_ylabel(f"Scans to Reach {threshold}V Threshold", fontsize=12)
    ax.set_title("Scans to Convergence (Converged Trials Only)", fontsize=14)
    ax.legend(loc="upper left", title="Method")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(all_dots))

    ax.annotate("Error bars: ±1 std\nPoints: mean",
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # Print summary
    print(f"\nScans to {threshold}V threshold (mean ± std, converged/total):")
    print("-" * 70)
    for method, data in sorted(methods.items()):
        print(f"\n{method}:")
        for num_dots in sorted(data.keys()):
            scans_list = data[num_dots]
            if scans_list:
                mean = np.mean(scans_list)
                std = np.std(scans_list)
                # Count total trials for this method/dots
                total = sum(1 for r in results
                           if r["method"] == method and r["num_dots"] == num_dots
                           for _ in r.get("trials", []))
                print(f"  {num_dots} dots: {mean:.0f} ± {std:.0f}, {len(scans_list)}/{total} converged")
            else:
                print(f"  {num_dots} dots: no trials converged")


def interpolate_to_scans(scan_numbers: list, values: list, target_scans: np.ndarray) -> np.ndarray:
    """
    Interpolate values at given scan_numbers to target_scans grid.

    Uses linear interpolation between recorded points.
    Extends last value for scans beyond the recorded range.
    """
    if not scan_numbers or not values:
        return np.full(len(target_scans), np.nan)

    scan_numbers = np.array(scan_numbers)
    values = np.array(values)

    # Handle case where data starts after target_scans[0]
    # Fill with first value for early scans
    result = np.interp(target_scans, scan_numbers, values)

    return result


def plot_convergence_curves(
    results_dir: Path,
    output_path: Path = None,
    max_scans: int = 2000,
    num_dots_filter: int = None,
    threshold: float = 0.5,
):
    """
    Plot 2: X = scans, Y = normalized convergence score.

    Shows convergence curves for all methods, optionally filtered by num_dots.
    Uses cumulative minimum (best seen so far) with median + IQR.

    Y-axis normalized using:
    - max_distance = (plunger_range * num_plungers) + (barrier_range * num_barriers)
    - 0 = max distance (worst)
    - 1 = zero distance (converged)

    Uses new distance tracking fields (scan_numbers, plunger_distance_history,
    barrier_distance_history) with interpolation for consistent x-axis.
    Falls back to legacy global_objective_history if new fields not available.
    """
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    # Filter by num_dots if specified
    if num_dots_filter:
        results = [r for r in results if r.get("num_dots") == num_dots_filter]
        if not results:
            print(f"No results found for {num_dots_filter} dots")
            return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'nelder_mead': 'C0', 'lbfgs': 'C1', 'random': 'C2', 'bayesian': 'C3', 'dreamerv3': 'C4'}

    plotted = []
    target_scans = np.arange(max_scans + 1)  # Common x-axis grid

    for result in sorted(results, key=lambda r: (r["method"], r.get("num_dots", 0))):
        method = result["method"]
        num_dots = result.get("num_dots", 2)
        use_barriers = result.get("use_barriers", True)
        num_plungers = num_dots
        num_barriers = num_dots - 1 if use_barriers else 0

        trials = result.get("trials", [])
        if not trials:
            print(f"Skipping {method} {num_dots}d: no trials")
            continue

        # Check if new distance fields are available
        first_trial = trials[0]
        has_new_fields = (
            "plunger_distance_history" in first_trial
            and "barrier_distance_history" in first_trial
            and "plunger_range" in first_trial
            and "barrier_range" in first_trial
        )

        if has_new_fields:
            # Use new distance tracking fields
            plunger_range = first_trial["plunger_range"]
            barrier_range = first_trial["barrier_range"]
            max_distance = plunger_range * num_plungers + barrier_range * num_barriers

            print(f"{method} {num_dots}d: Using new distance fields")
            print(f"  plunger_range={plunger_range}V, barrier_range={barrier_range}V")
            print(f"  max_distance={max_distance:.1f}V ({num_plungers} plungers, {num_barriers} barriers)")

            interpolated_scores = []
            for trial in trials:
                scan_nums = trial.get("scan_numbers", [])
                plunger_dists = trial.get("plunger_distance_history", [])
                barrier_dists = trial.get("barrier_distance_history", [])

                if not scan_nums or not plunger_dists:
                    continue

                # Total distance at each recorded point
                total_dists = [p + b for p, b in zip(plunger_dists, barrier_dists)]

                # Interpolate to target scans
                interp_dists = interpolate_to_scans(scan_nums, total_dists, target_scans)

                # Compute cumulative minimum (best seen so far)
                interp_dists = cumulative_min(interp_dists)

                # Normalize: 0 = max_distance (worst), 1 = 0 (converged)
                scores = 1.0 - (interp_dists / max_distance)
                scores = np.clip(scores, 0, 1)
                interpolated_scores.append(scores)

            if not interpolated_scores:
                print(f"Skipping {method} {num_dots}d: no valid distance data")
                continue

            arr = np.array(interpolated_scores)

        else:
            # Fall back to legacy global_objective_history
            print(f"{method} {num_dots}d: Using legacy global_objective_history (old format)")
            max_distance = get_plunger_voltage_range()

            histories = [t.get("global_objective_history", []) for t in trials]
            histories = [h for h in histories if h]

            if not histories:
                print(f"Skipping {method} {num_dots}d: no history data")
                continue

            # Pad to max_scans length
            padded = []
            for h in histories:
                if len(h) >= max_scans:
                    padded.append(h[:max_scans + 1])
                else:
                    padded.append(h + [h[-1]] * (max_scans + 1 - len(h)))

            arr = np.array(padded)

            # Convert objective to approximate distance
            arr = np.sqrt(np.abs(arr)) / np.sqrt(num_plungers)

            # Compute cumulative minimum for each trial
            for j in range(arr.shape[0]):
                arr[j] = cumulative_min(arr[j])

            # Normalize
            arr = 1.0 - (arr / max_distance)
            arr = np.clip(arr, 0, 1)

        # Compute statistics across trials
        median = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)

        label = f"{method}" if num_dots_filter else f"{method} ({num_dots}d)"
        color = colors.get(method, f'C{len(plotted)}')

        ax.plot(target_scans, median, color=color, linewidth=2, label=label)
        ax.fill_between(target_scans, q25, q75, alpha=0.2, color=color)
        plotted.append((method, num_dots, arr.shape[0]))

    # Threshold line - use a representative max_distance for display
    # (actual normalization is per-method, so this is approximate)
    max_distance_approx = get_plunger_voltage_range()
    normalized_threshold = 1.0 - (threshold / max_distance_approx)
    ax.axhline(y=normalized_threshold, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=f'Threshold ({threshold}V)')

    ax.set_xlabel("Scan Number", fontsize=12)
    ax.set_ylabel("Convergence Score (0=worst, 1=converged)", fontsize=12)

    title = "Convergence Comparison"
    if num_dots_filter:
        title += f" - {num_dots_filter} Dots"
    ax.set_title(title, fontsize=14)

    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_scans)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # Print info
    print(f"\nPlotted {len(plotted)} results:")
    for method, dots, n_trials in plotted:
        print(f"  {method} {dots}d: {n_trials} trials")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help="Results directory (default: results/final_results)")
    parser.add_argument("--plot", "-p", choices=["scans", "convergence", "both"],
                        default="both", help="Which plot to generate")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (auto-generated if not specified)")
    parser.add_argument("--num-dots", "-n", type=int, default=None,
                        help="Filter convergence plot to specific num_dots")
    parser.add_argument("--max-scans", type=int, default=2000,
                        help="Max scans for convergence plot x-axis")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Success threshold in volts")
    args = parser.parse_args()

    if args.dir:
        results_dir = Path(args.dir)
    else:
        results_dir = Path(__file__).parent / "results" / "final_results"

    if args.plot in ["scans", "both"]:
        output = Path(args.output) if args.output else results_dir / "num_dots_scaling.png"
        plot_scans_to_threshold(results_dir, output, threshold=args.threshold)

    if args.plot in ["convergence", "both"]:
        suffix = f"_{args.num_dots}dots" if args.num_dots else ""
        output = Path(args.output) if args.output else results_dir / f"convergence{suffix}.png"
        plot_convergence_curves(
            results_dir, output,
            max_scans=args.max_scans,
            num_dots_filter=args.num_dots,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
