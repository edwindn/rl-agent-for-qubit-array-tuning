"""
Scaling plot: measurements and env steps to convergence vs number of dots.

Plunger-only convergence (no barriers) for fair comparison across data sources.
2/4/6/8 dots from benchmarks/results JSON, 10/12 dots from wandb artifacts.

Usage:
    uv run python src/qadapt/capacitance_model/plot_scaling.py --output scaling_paper.svg

Convergence: plateau detection via |diff(avg_plunger_distance_per_dot)| < delta
for `sustained` consecutive steps.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "results"
WANDB_DATA_DIR = PROJECT_ROOT / "data"


def _convergence_from_avg(avg, delta=0.5, sustained=3):
    """Find first step where |diff(avg)| < delta for `sustained` consecutive steps."""
    diffs = np.abs(np.diff(avg))
    for i in range(len(diffs) - sustained + 1):
        if all(diffs[i:i+sustained] < delta):
            return i + 1
    return None


def compute_convergence_benchmark(dots, delta=0.5, sustained=3):
    """Compute convergence from benchmark JSON (2/4/6/8 dots). Plunger + barrier."""
    d = BENCHMARKS_DIR / f"final_{dots}dot"
    ppo = [f for f in os.listdir(d) if 'ppo' in f and f.endswith('.json')]
    if not ppo:
        return None

    with open(d / ppo[0]) as f:
        data = json.load(f)

    total_gates = dots + (dots - 1)

    steps_to_conv = []
    for t in data['trials']:
        pd = np.array(t['plunger_distance_history'])
        bd = np.array(t['barrier_distance_history'])
        avg = (pd + bd) / total_gates  # average per gate (plungers + barriers)
        step = _convergence_from_avg(avg, delta, sustained)
        if step is not None:
            steps_to_conv.append(step)

    if not steps_to_conv:
        return None

    s = np.array(steps_to_conv)
    return {
        'dots': dots,
        'mean_steps': np.mean(s),
        'mean_measurements': np.mean(s) * (dots - 1),
        'n_converged': len(s),
        'n_total': len(data['trials']),
    }


def compute_convergence_wandb(n_plungers, base_dir, delta=0.5, sustained=3):
    """Compute convergence from wandb artifact dirs. Plunger-only."""
    all_steps = []
    total = 0

    for art_dir in sorted(os.listdir(base_dir)):
        art_path = os.path.join(base_dir, art_dir)
        if not os.path.isdir(art_path):
            continue

        plunger_dirs = sorted([d for d in os.listdir(art_path) if d.startswith('plunger_')])
        if not plunger_dirs:
            continue

        # Build episode index -> file mapping per plunger
        plunger_files = {}
        for pd in plunger_dirs:
            plunger_files[pd] = {}
            for f in sorted(os.listdir(os.path.join(art_path, pd))):
                ep_idx = f.split('_')[0]
                plunger_files[pd][ep_idx] = os.path.join(art_path, pd, f)

        ep_indices = sorted(set.intersection(*[set(pf.keys()) for pf in plunger_files.values()]))

        for ep_idx in ep_indices:
            dists = [np.abs(np.load(plunger_files[pd][ep_idx])) for pd in plunger_dirs]
            avg = np.mean(dists, axis=0)  # average across plungers per step
            step = _convergence_from_avg(avg, delta, sustained)
            if step is not None:
                all_steps.append(step)
            total += 1

    if not all_steps:
        return None

    s = np.array(all_steps)
    return {
        'dots': n_plungers,
        'mean_steps': np.mean(s),
        'mean_measurements': np.mean(s) * (n_plungers - 1),
        'n_converged': len(s),
        'n_total': total,
    }


def plot_scaling(save_path):
    label_size = 7
    tick_size = 7
    tick_length = 4
    tick_width = 1

    # Collect results from all sources
    results = []
    for dots in [2, 4, 6, 8]:
        r = compute_convergence_benchmark(dots)
        if r:
            results.append(r)

    r10 = compute_convergence_wandb(10, WANDB_DATA_DIR / "scaling_10dot")
    if r10:
        results.append(r10)

    r12 = compute_convergence_wandb(12, WANDB_DATA_DIR / "scaling_12dot")
    if r12:
        results.append(r12)

    dots = [r['dots'] for r in results]
    steps = [r['mean_steps'] for r in results]
    measurements = [r['mean_measurements'] for r in results]

    fig = plt.figure(figsize=(2.3, 1.625))
    ax1 = fig.add_axes((0.18, 0.19, 0.48, 0.68))

    y_max = max(measurements) * 1.1

    # Right y-axis: time steps (scatter, drawn first)
    ax2 = ax1.twinx()
    color_s = 'C1'
    for i in range(len(dots)):
        marker = 'x' if dots[i] > 8 else 'o'
        lw = 1.0 if dots[i] > 8 else None
        ax2.scatter(dots[i], steps[i], color=color_s, s=16, zorder=3,
                    edgecolors='none' if marker == 'o' else color_s,
                    marker=marker, linewidths=lw)
    ax2.set_ylabel(r'Time Steps ($t$)', fontsize=label_size, color=color_s, rotation=270, labelpad=12)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_ylim(0, 40)

    # Left y-axis: measurements (scatter, on top)
    color_m = 'C0'
    for i in range(len(dots)):
        marker = 'x' if dots[i] > 8 else 'o'
        lw = 1.0 if dots[i] > 8 else None
        ax1.scatter(dots[i], measurements[i], color=color_m, s=16, zorder=5,
                    edgecolors='none' if marker == 'o' else color_m,
                    marker=marker, linewidths=lw)
    ax1.set_xlabel(r'Quantum Dots ($D$)', fontsize=label_size)
    ax1.set_ylabel('Measurements', fontsize=label_size, color=color_m)
    ax1.tick_params(axis='y', labelcolor=color_m)
    ax1.set_xticks(dots)
    ax1.set_xlim(1, max(dots) + 1)
    ax1.set_ylim(0, y_max)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Adjust right spine position
    ax2.spines['right'].set_position(('axes', 1.0))

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(tick_width)
        ax.tick_params(which='both', direction='in', labelsize=tick_size,
                       length=tick_length, width=tick_width)
    ax1.tick_params(top=True)

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format='svg')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format='png')
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")

    for r in results:
        print(f"  {r['dots']} dots: {r['n_converged']}/{r['n_total']} converged, "
              f"mean {r['mean_steps']:.1f} steps / {r['mean_measurements']:.0f} measurements")


def main():
    parser = argparse.ArgumentParser(description="Scaling plot for paper")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "paper_plots" / "capacitance_scaling.svg"))
    args = parser.parse_args()

    plot_scaling(args.output)


if __name__ == "__main__":
    main()
