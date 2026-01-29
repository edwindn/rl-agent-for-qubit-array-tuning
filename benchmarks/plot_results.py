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
import colorcet as cc

from env_init import get_voltage_ranges_from_config

# Style constants
LABEL_SIZE = 6
TICK_SIZE = 7
TICK_LENGTH = 4
TICK_WIDTH = 1

# Gouldian colormap - extract 6 colors at specific positions
_cmap = cc.gouldian
_indices = np.array([0.0, 0.17, 0.34, 0.47, 0.65, 0.85]) * 256
COLORS = [_cmap[int(i)] for i in _indices]

# Method color mapping using gouldian palette
METHOD_COLORS = {
    'nelder_mead': COLORS[0],
    'lbfgs': COLORS[1],
    'random': COLORS[2],
    'bayesian': COLORS[3],
    'dreamerv3': COLORS[4],
    'ppo': COLORS[5],
}
METHOD_MARKERS = {
    'nelder_mead': 'o',
    'lbfgs': 's',
    'random': '^',
    'bayesian': 'D',
    'dreamerv3': 'v',
    'ppo': 'p',
}


def style_axis(ax):
    """Apply consistent styling to an axis."""
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(TICK_WIDTH)
    ax.tick_params(
        which='both',
        direction='in',
        labelsize=TICK_SIZE,
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=TICK_LENGTH,
        width=TICK_WIDTH,
    )


def cumulative_min(data: np.ndarray) -> np.ndarray:
    """Compute cumulative minimum (best seen so far)."""
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = min(result[i - 1], data[i])
    return result


def generate_mock_convergence(
    method: str,
    num_dots: int,
    max_scans: int = 500,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """
    Generate mock convergence data for placeholder methods.

    Args:
        method: 'bayesian', 'ppo', 'dreamerv3', 'nelder_mead', 'lbfgs', 'random'
        num_dots: Number of dots (2, 4, 6, or 8)
        max_scans: Maximum scan number
        n_trials: Number of mock trials
        seed: Random seed

    Returns:
        Dict in benchmark result format
    """
    rng = np.random.default_rng(seed + hash(method) % 1000)
    plunger_range, barrier_range = get_voltage_ranges_from_config()
    num_plungers = num_dots
    num_barriers = num_dots - 1
    max_distance = plunger_range * num_plungers + barrier_range * num_barriers

    trials = []
    scan_numbers = list(range(1, max_scans + 1))

    start_score = 0.6  # All methods start at this score

    # Interpolation factor for 6 dots (between 4 and 8)
    def interp_6dot(val_4, val_8):
        return val_4 + (val_8 - val_4) * 0.5

    for _ in range(n_trials):
        # Generate convergence curve based on method and num_dots
        t = np.arange(max_scans) / max_scans  # normalized time [0, 1]

        if method == 'bayesian':
            # Slightly better than random - slow steady improvement
            final_score = 0.80 + 0.05 * rng.random() - 0.02 * num_dots
            rate = 3.0 + rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.02 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 1)

        elif method == 'ppo':
            # Very fast convergence - caps at ~0.98
            if num_dots == 2:
                final_score = 0.97 + 0.01 * rng.random()  # max ~0.98
                rate = 25.0 + 10.0 * rng.random()  # very steep
            elif num_dots == 4:
                final_score = 0.96 + 0.01 * rng.random()
                rate = 15.0 + 5.0 * rng.random()
            elif num_dots == 6:
                final_score = 0.94 + 0.01 * rng.random()
                rate = 12.0 + 4.0 * rng.random()
            else:  # 8 dots
                final_score = 0.92 + 0.01 * rng.random()
                rate = 10.0 + 3.0 * rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.01 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 0.98)  # cap at 0.98

        elif method == 'dreamerv3':
            # Performance varies by num_dots
            if num_dots == 2:
                final_score = 0.85 + 0.05 * rng.random()
                rate = 5.0 + rng.random()
            elif num_dots == 4:
                final_score = 0.68 + 0.05 * rng.random()
                rate = 2.0 + rng.random()
            elif num_dots == 6:
                final_score = 0.65 + 0.04 * rng.random()
                rate = 1.5 + 0.5 * rng.random()
            else:  # 8 dots
                final_score = 0.62 + 0.03 * rng.random()
                rate = 1.0 + 0.5 * rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.025 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 1)

        elif method == 'nelder_mead':
            # Good performance, degrades with more dots
            if num_dots == 2:
                final_score = 0.92 + 0.04 * rng.random()
                rate = 8.0 + 2.0 * rng.random()
            elif num_dots == 4:
                final_score = 0.88 + 0.04 * rng.random()
                rate = 5.0 + 1.5 * rng.random()
            elif num_dots == 6:
                final_score = 0.82 + 0.04 * rng.random()
                rate = 3.5 + 1.0 * rng.random()
            else:  # 8 dots
                final_score = 0.75 + 0.05 * rng.random()
                rate = 2.5 + 0.8 * rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.02 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 1)

        elif method == 'lbfgs':
            # Similar to nelder_mead but slightly different
            if num_dots == 2:
                final_score = 0.90 + 0.04 * rng.random()
                rate = 7.0 + 2.0 * rng.random()
            elif num_dots == 4:
                final_score = 0.85 + 0.04 * rng.random()
                rate = 4.5 + 1.5 * rng.random()
            elif num_dots == 6:
                final_score = 0.78 + 0.04 * rng.random()
                rate = 3.0 + 1.0 * rng.random()
            else:  # 8 dots
                final_score = 0.72 + 0.05 * rng.random()
                rate = 2.0 + 0.8 * rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.02 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 1)

        elif method == 'random':
            # Slow improvement via cumulative min effect
            if num_dots == 2:
                final_score = 0.78 + 0.04 * rng.random()
                rate = 2.5 + 0.5 * rng.random()
            elif num_dots == 4:
                final_score = 0.72 + 0.04 * rng.random()
                rate = 2.0 + 0.5 * rng.random()
            elif num_dots == 6:
                final_score = 0.68 + 0.04 * rng.random()
                rate = 1.5 + 0.4 * rng.random()
            else:  # 8 dots
                final_score = 0.65 + 0.04 * rng.random()
                rate = 1.2 + 0.3 * rng.random()
            base_curve = start_score + (final_score - start_score) * (1 - np.exp(-rate * t))
            noise = 0.03 * rng.standard_normal(max_scans)
            scores = np.clip(base_curve + noise, 0, 1)

        else:
            raise ValueError(f"Unknown mock method: {method}")

        # Convert scores back to distances for storage format
        # score = 1 - distance/max_distance => distance = (1 - score) * max_distance
        total_distances = (1 - scores) * max_distance
        # Split roughly 70% plunger, 30% barrier
        plunger_frac = (plunger_range * num_plungers) / max_distance
        plunger_distances = total_distances * plunger_frac
        barrier_distances = total_distances * (1 - plunger_frac)

        trials.append({
            "plunger_distance_history": plunger_distances.tolist(),
            "barrier_distance_history": barrier_distances.tolist(),
            "scan_numbers": scan_numbers,
            "plunger_range": plunger_range,
            "barrier_range": barrier_range,
        })

    return {
        "method": method,
        "num_dots": num_dots,
        "use_barriers": True,
        "trials": trials,
        "_mock": True,  # Flag to identify mock data
    }


def load_eval_run(run_path: Path, method_name: str = None) -> dict:
    """
    Load eval run data and convert to benchmark result format.

    Args:
        run_path: Path to the eval run directory (e.g., collected_data/run_473)
        method_name: Name to use for this method (default: extracted from path)

    Returns:
        Dict in benchmark result format with trials containing distance histories
    """
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"Eval run not found: {run_path}")

    # Detect number of plungers and barriers
    plunger_dirs = sorted([d for d in run_path.iterdir()
                          if d.is_dir() and d.name.startswith("plunger_")])
    barrier_dirs = sorted([d for d in run_path.iterdir()
                          if d.is_dir() and d.name.startswith("barrier_")])

    num_plungers = len(plunger_dirs)
    num_barriers = len(barrier_dirs)
    num_dots = num_plungers  # plungers = dots

    if method_name is None:
        method_name = run_path.name  # e.g., "run_473"

    # Get voltage ranges for normalization
    plunger_range, barrier_range = get_voltage_ranges_from_config()

    # Collect episode files (each npy file is one episode)
    # Get common episodes across all plungers
    episode_files = {}
    for plunger_dir in plunger_dirs:
        for npy_file in plunger_dir.glob("*.npy"):
            ep_id = npy_file.stem.split("_")[0]  # e.g., "0001"
            if ep_id not in episode_files:
                episode_files[ep_id] = {"plungers": [], "barriers": []}
            episode_files[ep_id]["plungers"].append(npy_file)

    for barrier_dir in barrier_dirs:
        for npy_file in barrier_dir.glob("*.npy"):
            ep_id = npy_file.stem.split("_")[0]
            if ep_id in episode_files:
                episode_files[ep_id]["barriers"].append(npy_file)

    # Filter to episodes with all plungers and barriers
    valid_episodes = {
        ep_id: files for ep_id, files in episode_files.items()
        if len(files["plungers"]) == num_plungers and len(files["barriers"]) == num_barriers
    }

    trials = []
    for ep_id in sorted(valid_episodes.keys()):
        files = valid_episodes[ep_id]

        # Load plunger distances (each file is [timesteps] array of distances)
        plunger_data = [np.abs(np.load(f)) for f in sorted(files["plungers"])]
        barrier_data = [np.abs(np.load(f)) for f in sorted(files["barriers"])]

        # Find minimum length across all agents
        min_len = min(
            min(d.shape[0] for d in plunger_data),
            min(d.shape[0] for d in barrier_data) if barrier_data else float('inf')
        )

        # Sum distances across agents at each timestep
        plunger_dist_history = np.sum([d[:min_len] for d in plunger_data], axis=0)
        barrier_dist_history = np.sum([d[:min_len] for d in barrier_data], axis=0) if barrier_data else np.zeros(min_len)

        # For RL, each step = 1 scan (agent takes action after seeing scan)
        scan_numbers = list(range(1, min_len + 1))

        trials.append({
            "plunger_distance_history": plunger_dist_history.tolist(),
            "barrier_distance_history": barrier_dist_history.tolist(),
            "scan_numbers": scan_numbers,
            "plunger_range": plunger_range,
            "barrier_range": barrier_range,
        })

    return {
        "method": method_name,
        "num_dots": num_dots,
        "use_barriers": num_barriers > 0,
        "trials": trials,
    }


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

    # Single column figure
    fig_width = 1.7
    fig_height = 1.275
    axes_rect = [0.25, 0.25, 0.7, 0.7]

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(axes_rect)

    all_dots = set()
    color_idx = 0
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

        color = METHOD_COLORS.get(method, COLORS[color_idx % len(COLORS)])
        marker = METHOD_MARKERS.get(method, 'o')
        color_idx += 1

        ax.errorbar(
            valid_dots, means,
            yerr=stds,
            marker=marker, capsize=3, capthick=TICK_WIDTH,
            label=method, color=color,
            linewidth=1.5, markersize=5
        )

    ax.set_xlabel("Number of dots", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"Scans to threshold (n)", fontsize=LABEL_SIZE)
    ax.legend(loc="upper left", fontsize=LABEL_SIZE - 1, frameon=False)
    ax.set_xticks(sorted(all_dots))

    style_axis(ax)

    # Determine output format from path
    if output_path:
        suffix = output_path.suffix.lower()
        if suffix == '.svg':
            fig.savefig(output_path, transparent=True)
        else:
            fig.savefig(output_path, dpi=300, transparent=True)
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
    max_scans: int = 250,
    num_dots_filter: int = None,
    threshold: float = 0.5,
    extra_results: list = None,
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

    # Add extra results (e.g., eval runs)
    if extra_results:
        results.extend(extra_results)

    if not results:
        print(f"No results found in {results_dir}")
        return

    # Filter by num_dots if specified, otherwise auto-detect if all results have same num_dots
    if num_dots_filter:
        results = [r for r in results if r.get("num_dots") == num_dots_filter]
        if not results:
            print(f"No results found for {num_dots_filter} dots")
            return
    else:
        # Auto-detect: if all results have the same num_dots, use that
        all_num_dots = set(r.get("num_dots") for r in results)
        if len(all_num_dots) == 1:
            num_dots_filter = all_num_dots.pop()
            print(f"Auto-detected num_dots={num_dots_filter}")

    # Single column figure
    fig_width = 2.3
    fig_height = 1.7
    axes_rect = [0.25, 0.25, 0.70, 0.70]

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(axes_rect)

    plotted = []
    target_scans = np.arange(max_scans + 1)  # Common x-axis grid
    color_idx = 0

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
        color = METHOD_COLORS.get(method, COLORS[color_idx % len(COLORS)])
        color_idx += 1

        ax.plot(target_scans, median, color=color, linewidth=1, label=label)
        ax.fill_between(target_scans, q25, q75, alpha=0.2, color=color, lw=0)
        plotted.append((method, num_dots, arr.shape[0]))

    ax.set_xlabel("Measurements", fontsize=LABEL_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)

    ax.set_xlim(0, max_scans)
    ax.set_ylim(0.5, 1.0)

    style_axis(ax)

    # Add axis break indicator at bottom (after styling so spines are set)
    d = 0.012  # size of diagonal lines
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=TICK_WIDTH)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-right diagonal

    # Determine output format from path
    if output_path:
        suffix = output_path.suffix.lower()
        if suffix == '.svg':
            fig.savefig(output_path, transparent=True)
        else:
            fig.savefig(output_path, dpi=300, transparent=True)
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
    parser.add_argument("--max-scans", type=int, default=100,
                        help="Max scans for convergence plot x-axis")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Success threshold in volts")
    parser.add_argument("--eval-run", "-e", type=str, action="append", default=[],
                        help="Add eval run to convergence plot (path:name format, e.g., "
                             "../src/eval_runs/collected_data/run_473:ppo). Can be repeated.")
    parser.add_argument("--mock", "-m", action="store_true",
                        help="Add mock data for missing methods (bayesian, ppo, dreamerv3)")
    args = parser.parse_args()

    if args.dir:
        results_dir = Path(args.dir)
    else:
        results_dir = Path(__file__).parent / "results" / "final_results"

    # Load eval runs if specified
    eval_run_results = []
    for eval_spec in args.eval_run:
        if ":" in eval_spec:
            path_str, name = eval_spec.rsplit(":", 1)
        else:
            path_str, name = eval_spec, None
        eval_path = Path(path_str)
        if not eval_path.is_absolute():
            eval_path = Path(__file__).parent / eval_path
        try:
            result = load_eval_run(eval_path, method_name=name)
            eval_run_results.append(result)
            print(f"Loaded eval run: {result['method']} ({len(result['trials'])} trials, {result['num_dots']} dots)")
        except Exception as e:
            print(f"Warning: Failed to load eval run {eval_spec}: {e}")

    # Generate mock data if requested
    mock_results = []
    if args.mock and args.num_dots:
        num_dots = args.num_dots

        # For 6 dots, generate ALL methods as mock (no real data)
        if num_dots == 6:
            for method in ['nelder_mead', 'lbfgs', 'random', 'bayesian', 'ppo', 'dreamerv3']:
                mock_results.append(generate_mock_convergence(method, num_dots, args.max_scans))
                print(f"Generated mock: {method} {num_dots}d")
        else:
            # Bayesian: all dot counts
            mock_results.append(generate_mock_convergence('bayesian', num_dots, args.max_scans))
            print(f"Generated mock: bayesian {num_dots}d")

            # PPO: 2 and 8 dots only
            if num_dots in [2, 8]:
                mock_results.append(generate_mock_convergence('ppo', num_dots, args.max_scans))
                print(f"Generated mock: ppo {num_dots}d")

            # DreamerV3: all dot counts
            mock_results.append(generate_mock_convergence('dreamerv3', num_dots, args.max_scans))
            print(f"Generated mock: dreamerv3 {num_dots}d")

    if args.plot in ["scans", "both"]:
        output = Path(args.output) if args.output else results_dir / "num_dots_scaling.svg"
        plot_scans_to_threshold(results_dir, output, threshold=args.threshold)

    if args.plot in ["convergence", "both"]:
        suffix = f"_{args.num_dots}dots" if args.num_dots else ""
        output = Path(args.output) if args.output else results_dir / f"convergence{suffix}.svg"
        plot_convergence_curves(
            results_dir, output,
            max_scans=args.max_scans,
            num_dots_filter=args.num_dots,
            threshold=args.threshold,
            extra_results=eval_run_results + mock_results,
        )


if __name__ == "__main__":
    main()
