#!/usr/bin/env python3
"""
Compute ablation metrics for plunger agents across eval run folders.

Metrics per run:
  - average absolute distance from ground truth (over all agents, episodes, timesteps)
  - percentage of episodes that converged
  - mean steps to converge (over converged episodes only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


StepId = str
AgentName = str
AgentStepMap = Dict[AgentName, Dict[StepId, Path]]


def _resolve_collected_data_dir(data_dir_arg: str) -> Path:
    base_dir = Path(__file__).resolve().parent / "collected_data"
    candidate = Path(data_dir_arg)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _iter_run_dirs(data_dir: Path) -> List[Path]:
    return sorted(
        (p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("run")),
        key=lambda p: p.name,
    )


def _collect_agent_steps(run_dir: Path, prefix: str) -> AgentStepMap:
    agent_steps: AgentStepMap = {}
    for agent_dir in sorted(run_dir.iterdir()):
        if not agent_dir.is_dir() or not agent_dir.name.startswith(prefix):
            continue
        step_map: Dict[StepId, Path] = {}
        for npy_file in sorted(agent_dir.glob("*.npy")):
            step = npy_file.stem.split("_", 1)[0]
            step_map[step] = npy_file
        if step_map:
            agent_steps[agent_dir.name] = step_map
    return agent_steps


def _sorted_steps(steps: Iterable[StepId]) -> List[StepId]:
    def _key(step: StepId) -> Tuple[int, StepId]:
        return (int(step), step) if step.isdigit() else (10**9, step)

    return sorted(steps, key=_key)


def _load_distances(path: Path) -> Optional[np.ndarray]:
    """Load a per-episode distance trajectory.

    Returns None for empty files (e.g. an algo.evaluate() reset that never
    stepped) so the caller can skip the episode without aborting the run.
    """
    distances = np.load(path)
    distances = np.asarray(distances).squeeze()
    if distances.size == 0:
        return None
    if not np.isfinite(distances).all():
        raise ValueError(f"Distance data contains non-finite values: {path}")
    if distances.ndim != 1:
        distances = distances.ravel()
    return distances


def _compute_run_metrics(run_dir: Path, radius: float, length: int) -> Optional[Dict[str, float]]:
    agent_steps = _collect_agent_steps(run_dir, "plunger_")
    if not agent_steps:
        return None

    step_sets = [set(steps.keys()) for steps in agent_steps.values()]
    common_steps = set.intersection(*step_sets) if step_sets else set()
    if not common_steps:
        return None

    total_abs_sum = 0.0
    total_count = 0
    converged = 0
    steps_to_converge: List[int] = []
    included_episodes = 0

    for step in _sorted_steps(common_steps):
        per_agent_series: List[np.ndarray] = []
        skip_episode = False
        for agent_name, step_map in agent_steps.items():
            path = step_map[step]
            series = _load_distances(path)
            if series is None:
                skip_episode = True
                break
            per_agent_series.append(series)
        if skip_episode:
            continue

        min_len = min(series.size for series in per_agent_series)
        if min_len < length:
            # Skip episodes that are shorter than the requested length.
            continue

        included_episodes += 1
        for series in per_agent_series:
            total_abs_sum += np.abs(series[:length]).sum()
            total_count += int(length)

        converged_step: Optional[int] = None
        for idx in range(length):
            if all(abs(series[idx]) < radius for series in per_agent_series):
                converged_step = idx + 1  # 1-based step index for readability
                break

        if converged_step is not None:
            converged += 1
            steps_to_converge.append(converged_step)

    if total_count == 0:
        return None

    total_episodes = included_episodes
    percent_converged = 100.0 * converged / total_episodes if total_episodes else 0.0
    mean_steps = float(np.mean(steps_to_converge)) if steps_to_converge else float("nan")

    return {
        "run": run_dir.name,
        "avg_abs_distance": total_abs_sum / total_count,
        "percent_converged": percent_converged,
        "mean_steps_to_converge": mean_steps,
        "episodes_total": float(total_episodes),
        "episodes_converged": float(converged),
    }


def _write_json(out_path: Path, payload: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_radii(radius_arg: str) -> List[float]:
    parts = [part.strip() for part in radius_arg.split(",") if part.strip()]
    if not parts:
        raise ValueError("Radius must be a number or a comma-separated list of numbers.")
    radii: List[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"Invalid radius value: {part}") from exc
        radii.append(value)
    return radii


def _plot_results(
    results: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> None:
    if not results:
        return

    radius_keys = sorted(results.keys(), key=lambda r: float(r))
    run_names = sorted({name for run_map in results.values() for name in run_map.keys()})
    if not run_names:
        return

    x = np.arange(len(radius_keys))
    bar_width = 0.8 / max(len(run_names), 1)
    offsets = (np.arange(len(run_names)) - (len(run_names) - 1) / 2.0) * bar_width

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharex=True)
    ax_success, ax_steps = axes

    for idx, run_name in enumerate(run_names):
        success_vals = []
        step_vals = []
        for radius_key in radius_keys:
            run_map = results.get(radius_key, {})
            row = run_map.get(run_name)
            if row is None:
                success_vals.append(np.nan)
                step_vals.append(np.nan)
            else:
                success_vals.append(row["percent_converged"])
                step_vals.append(row["mean_steps_to_converge"])

        ax_success.bar(x + offsets[idx], success_vals, width=bar_width, label=run_name)
        ax_steps.bar(x + offsets[idx], step_vals, width=bar_width, label=run_name)

    ax_success.set_title("Success Rate vs Convergence Radius")
    ax_success.set_ylabel("Success Rate (%)")
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(radius_keys)
    ax_success.grid(True, axis="y", alpha=0.3)

    ax_steps.set_title("Mean Steps to Converge vs Radius")
    ax_steps.set_ylabel("Mean Steps to Converge")
    ax_steps.set_xticks(x)
    ax_steps.set_xticklabels(radius_keys)
    ax_steps.grid(True, axis="y", alpha=0.3)

    ax_steps.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute plunger ablation metrics for eval runs.")
    parser.add_argument(
        "--data-dir",
        default="./",
        help="Collected data directory or run folder path (default: collected_data).",
    )
    parser.add_argument(
        "--radius",
        required=True,
        help="Convergence radius (single value or comma-separated list).",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="Rollout episode length; data is truncated to this length.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON output path. Defaults to ablation_metrics.json next to this script.",
    )
    parser.add_argument(
        "--plot-graphs",
        action="store_true",
        help="Save a side-by-side bar chart of success rate and mean steps to converge.",
    )
    args = parser.parse_args()

    radii = _parse_radii(args.radius)

    data_dir = _resolve_collected_data_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    run_dirs = _iter_run_dirs(data_dir)
    if not run_dirs and data_dir.is_dir() and data_dir.name.startswith("run"):
        run_dirs = [data_dir]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for radius in radii:
        rows: List[Dict[str, float]] = []
        for run_dir in run_dirs:
            metrics = _compute_run_metrics(run_dir, radius, args.length)
            if metrics is None:
                continue
            rows.append(metrics)

        if not rows:
            raise ValueError(
                f"No runs have episodes at least length {args.length} for radius={radius}."
            )

        run_map: Dict[str, Dict[str, float]] = {}
        for row in rows:
            run_name = row.pop("run")
            run_map[run_name] = row
        results[str(radius)] = run_map

    if not results:
        raise ValueError("No run metrics computed (no plunger data found).")

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "ablation_metrics.json"
    _write_json(out_path, results)

    print(f"Saved: {out_path}")
    for radius_key, run_map in results.items():
        print(f"radius={radius_key}")
        for run_name, row in run_map.items():
            mean_steps = row["mean_steps_to_converge"]
            mean_steps_str = f"{mean_steps:.2f}" if np.isfinite(mean_steps) else "nan"
            print(
                f"  {run_name}: avg_abs_distance={row['avg_abs_distance']:.6f}, "
                f"percent_converged={row['percent_converged']:.2f}%, "
                f"mean_steps_to_converge={mean_steps_str}, "
                f"episodes={int(row['episodes_total'])}, "
                f"converged={int(row['episodes_converged'])}"
            )

    if args.plot_graphs:
        plot_path = Path(__file__).resolve().parent / "ablation_results.png"
        _plot_results(results, plot_path)
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
