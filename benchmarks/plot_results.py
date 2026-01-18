"""
Plot benchmark comparison across methods and array sizes.

Reads from results/final/ by default - move curated results there for plotting.

Usage:
    uv run python plot_results.py
    uv run python plot_results.py --output comparison.png
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def cumulative_min(data: np.ndarray) -> np.ndarray:
    """Compute cumulative minimum (best seen so far)."""
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = min(result[i - 1], data[i])
    return result


def load_all_results(results_dir: Path) -> list:
    """Load all JSON result files."""
    results = []
    for path in results_dir.glob("*.json"):
        with open(path) as f:
            results.append(json.load(f))
    return results


def compute_stats(result: dict) -> dict:
    """Compute median and quartiles for a result."""
    iterations = [t["num_iterations"] for t in result["trials"]]
    return {
        "median": np.median(iterations),
        "q25": np.percentile(iterations, 25),
        "q75": np.percentile(iterations, 75),
        "n_trials": len(iterations),
    }


def plot_comparison(results_dir: Path, output_path: Path = None):
    """Create comparison plot of methods vs array size."""
    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    # Group by method and num_dots
    methods = defaultdict(dict)
    for r in results:
        method = r["method"]
        num_dots = r["num_dots"]
        methods[method][num_dots] = compute_stats(r)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for i, (method, data) in enumerate(sorted(methods.items())):
        dots = sorted(data.keys())
        medians = [data[d]["median"] for d in dots]
        q25s = [data[d]["q25"] for d in dots]
        q75s = [data[d]["q75"] for d in dots]

        yerr_lower = [m - q for m, q in zip(medians, q25s)]
        yerr_upper = [q - m for m, q in zip(medians, q75s)]

        ax.errorbar(
            dots, medians,
            yerr=[yerr_lower, yerr_upper],
            marker='o', capsize=5, capthick=2,
            label=method, color=colors[i % len(colors)],
            linewidth=2, markersize=8
        )

    ax.set_xlabel("Number of Dots", fontsize=12)
    ax.set_ylabel("Steps to Convergence", fontsize=12)
    ax.set_title("Benchmark Comparison", fontsize=14)
    ax.legend(loc="upper left", title="Method")
    ax.grid(True, alpha=0.3)

    # Label what error bars represent
    ax.annotate("Error bars: IQR (Q25-Q75)\nPoints: median",
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, color='gray')

    all_dots = set()
    for data in methods.values():
        all_dots.update(data.keys())
    ax.set_xticks(sorted(all_dots))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)

    # Print summary
    print("\nSummary (median [Q25-Q75], n trials):")
    print("-" * 50)
    for method, data in sorted(methods.items()):
        print(f"\n{method}:")
        for num_dots in sorted(data.keys()):
            s = data[num_dots]
            print(f"  {num_dots} dots: {s['median']:.0f} [{s['q25']:.0f}-{s['q75']:.0f}], n={s['n_trials']}")


def plot_convergence(result: dict, output_path: Path = None, max_scans: int = None):
    """
    Plot best mean absolute distance to optimal vs scan step for a single benchmark result.

    Shows cumulative minimum (best seen so far) with median + IQR across trials.

    Args:
        result: Loaded benchmark result dict
        output_path: Where to save (None = show interactively)
        max_scans: Truncate x-axis at this value (None = auto)
    """
    trials = result.get("trials", [])
    if not trials:
        print("No trials found")
        return

    # Extract global_objective_history from each trial
    histories = []
    for t in trials:
        hist = t.get("global_objective_history", [])
        if hist:
            histories.append(hist)

    if not histories:
        print("No global_objective_history found in trials")
        return

    # Number of gates for computing mean distance
    num_dots = result.get("num_dots", 2)
    use_barriers = result.get("use_barriers", True)
    num_gates = num_dots + (num_dots - 1 if use_barriers else 0)

    # Find max length and pad shorter histories with their final value
    max_len = max(len(h) for h in histories)
    if max_scans:
        max_len = min(max_len, max_scans)

    padded = []
    for h in histories:
        if len(h) >= max_len:
            padded.append(h[:max_len])
        else:
            # Pad with final value
            padded.append(h + [h[-1]] * (max_len - len(h)))

    # Convert to array for easy stats
    arr = np.array(padded)  # (n_trials, n_scans)

    # Convert sum of squared distances to mean absolute distance
    # global_obj = sum((V - V_opt)^2), so mean_abs_dist = sqrt(|global_obj|) / sqrt(num_gates)
    arr = np.sqrt(np.abs(arr)) / np.sqrt(num_gates)

    # Compute cumulative minimum for each trial (best seen so far)
    for j in range(arr.shape[0]):
        arr[j] = cumulative_min(arr[j])

    # Compute stats at each scan step
    median = np.median(arr, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)

    scans = np.arange(len(median))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(scans, median, color='blue', linewidth=2, label='Median')
    ax.fill_between(scans, q25, q75, alpha=0.3, color='blue', label='IQR (Q25-Q75)')

    # Success threshold line
    threshold = result.get("success_threshold", 0.5)
    ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, label=f'Threshold ({threshold}V)')

    ax.set_xlabel("Scan Step", fontsize=12)
    ax.set_ylabel("Best Mean Absolute Distance to Optimal (V)", fontsize=12)
    ax.set_title(f"Convergence: {result['method']} ({result['mode']}, {result['num_dots']} dots)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate
    ax.annotate(f"n={len(histories)} trials",
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, color='gray')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_convergence_comparison(results: list, output_path: Path = None, max_scans: int = None):
    """
    Plot best mean absolute distance to optimal vs scan step for multiple methods.

    Shows cumulative minimum (best seen so far) which can only decrease.

    Args:
        results: List of loaded benchmark result dicts
        output_path: Where to save (None = show interactively)
        max_scans: Truncate x-axis at this value (None = auto)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, result in enumerate(results):
        trials = result.get("trials", [])
        histories = [t.get("global_objective_history", []) for t in trials]
        histories = [h for h in histories if h]  # Filter empty

        if not histories:
            continue

        # Number of gates for computing mean distance
        num_dots = result.get("num_dots", 2)
        use_barriers = result.get("use_barriers", True)
        num_gates = num_dots + (num_dots - 1 if use_barriers else 0)

        # Pad to same length
        max_len = max(len(h) for h in histories)
        if max_scans:
            max_len = min(max_len, max_scans)

        padded = []
        for h in histories:
            if len(h) >= max_len:
                padded.append(h[:max_len])
            else:
                padded.append(h + [h[-1]] * (max_len - len(h)))

        arr = np.array(padded)

        # Convert sum of squared distances to mean absolute distance
        # global_obj = sum((V - V_opt)^2), so mean_abs_dist = sqrt(|global_obj|) / sqrt(num_gates)
        arr = np.sqrt(np.abs(arr)) / np.sqrt(num_gates)

        # Compute cumulative minimum for each trial, then take median across trials
        for j in range(arr.shape[0]):
            arr[j] = cumulative_min(arr[j])

        median = np.median(arr, axis=0)
        std = np.std(arr, axis=0)
        scans = np.arange(len(median))

        label = f"{result['method']} ({result['num_dots']}d)"
        color = colors[i % len(colors)]

        ax.plot(scans, median, color=color, linewidth=2, label=label)
        ax.fill_between(scans, median - std / 2, median + std / 2, alpha=0.2, color=color)

    ax.set_xlabel("Scan Step", fontsize=12)
    ax.set_ylabel("Best Mean Absolute Distance to Optimal (V)", fontsize=12)
    ax.set_title("Convergence Comparison (Best Seen So Far)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark comparison")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help="Results directory (default: results/final)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: comparison.png in results dir)")
    parser.add_argument("--convergence", "-c", type=str, default=None,
                        help="Plot convergence for a specific result file")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Max scans for convergence plot x-axis")
    args = parser.parse_args()

    if args.convergence:
        # Plot convergence for specific file
        with open(args.convergence) as f:
            result = json.load(f)
        output = Path(args.output) if args.output else None
        plot_convergence(result, output, max_scans=args.max_scans)
        return

    if args.dir:
        results_dir = Path(args.dir)
    else:
        results_dir = Path(__file__).parent / "results" / "final"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "comparison.png"

    plot_comparison(results_dir, output_path)


if __name__ == "__main__":
    main()
