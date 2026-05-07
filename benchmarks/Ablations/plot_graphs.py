#!/usr/bin/env python3
"""
Analyze distance data stored under collected_data/.
Creates grid plots for plunger and barrier distance time series.
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["text.usetex"] = False


DistanceEntry = Tuple[str, Path]
GroupedEntries = Dict[str, List[DistanceEntry]]
CgdMap = Dict[str, List[float]]


def _resolve_data_dir(data_dir_arg: str) -> Path:
    base_dir = Path(__file__).resolve().parent / "collected_data"
    candidate = Path(data_dir_arg)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _collect_distance_files(data_dir: Path, prefix: str) -> List[DistanceEntry]:
    entries: List[DistanceEntry] = []
    for agent_dir in sorted(data_dir.iterdir()):
        if not agent_dir.is_dir() or not agent_dir.name.startswith(prefix):
            continue
        for npy_file in sorted(agent_dir.glob("*.npy")):
            entries.append((agent_dir.name, npy_file))
    return entries


def _group_by_step(entries: List[DistanceEntry]) -> GroupedEntries:
    grouped: GroupedEntries = defaultdict(list)
    for agent_name, npy_file in entries:
        step = npy_file.stem.split("_", 1)[0]
        grouped[step].append((agent_name, npy_file))
    return dict(sorted(grouped.items()))


def _load_cgd_values(data_dir: Path) -> CgdMap:
    cgd_dir = data_dir / "cgd"
    if not cgd_dir.exists():
        return {}

    cgd_map: CgdMap = {}
    for json_file in sorted(cgd_dir.glob("*.json")):
        step = json_file.stem.split("_", 1)[0]
        with json_file.open("r", encoding="utf-8") as handle:
            matrix = json.load(handle)
        cgd = np.asarray(matrix, dtype=float)
        if cgd.ndim != 2:
            continue
        upper_diag: List[float] = []
        limit = min(cgd.shape[0] - 1, cgd.shape[1] - 1)
        for idx in range(limit):
            upper_diag.append(float(cgd[idx, idx + 1]))
        cgd_map[step] = upper_diag
    return cgd_map


def _plot_distance_grid(
    grouped: GroupedEntries,
    out_path: Path,
    title: str,
    cgd_map: Optional[CgdMap] = None,
) -> None:
    if not grouped:
        raise ValueError(f"No distance data found for {title}")

    total = len(grouped)
    ncols = math.ceil(math.sqrt(total))
    nrows = math.ceil(total / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for idx, (step, step_entries) in enumerate(grouped.items()):
        ax = axes_arr[idx]
        for agent_name, npy_file in sorted(step_entries):
            distances = np.load(npy_file)
            distances = np.asarray(distances).squeeze()
            if distances.size == 0:
                # algo.evaluate() can produce empty .npy artifacts; skip them.
                continue
            if not np.isfinite(distances).all():
                raise ValueError(f"Distance data contains non-finite values: {npy_file}")
            if distances.ndim != 1:
                distances = distances.ravel()

            steps = np.arange(1, len(distances) + 1)
            ax.plot(steps, distances, label=agent_name, alpha=0.85)

        ax.set_title(f"Step {step}")
        ax.set_xlabel("Episode Step")
        ax.set_ylabel("Distance from Ground Truth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        if cgd_map is not None and step in cgd_map:
            cgd_vals = cgd_map[step]
            if cgd_vals:
                formatted = ", ".join(f"{val:.4f}" for val in cgd_vals)
                ax.text(
                    0.5,
                    -0.28,
                    f"cgd[0,1..]: {formatted}",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=8,
                )

    for ax in axes_arr[total:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distance grids for plunger and barrier agents.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Run folder name inside collected_data/ (e.g., 20260127_071239_run_482).",
    )
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    plunger_entries = _collect_distance_files(data_dir, "plunger_")
    barrier_entries = _collect_distance_files(data_dir, "barrier_")

    plunger_grouped = _group_by_step(plunger_entries)
    barrier_grouped = _group_by_step(barrier_entries)
    cgd_map = _load_cgd_values(data_dir)

    plunger_out = data_dir / "plunger_distances_grid.png"
    barrier_out = data_dir / "barrier_distances_grid.png"

    _plot_distance_grid(plunger_grouped, plunger_out, "Plunger Agent Distances", cgd_map=cgd_map)
    _plot_distance_grid(barrier_grouped, barrier_out, "Barrier Agent Distances")

    print(f"Saved: {plunger_out}")
    print(f"Saved: {barrier_out}")


if __name__ == "__main__":
    main()
