"""
Paper figures for capacitance model calibration and convergence.

Generates 3 plots:
  1. kalman_calibration_nn.svg   — NN: Kalman estimate error vs Kalman σ
  2. kalman_calibration_nnn.svg  — NNN: same, with tighter y-axis
  3. model_convergence_paper.svg — CNN prediction error over episode steps

Each calibration plot shows one point per timestep (averaged over all episodes),
colored by step in episode (blue=early, yellow=late). The convergence plot shows
mean ± 0.5 std for 1st and 2nd neighbor couplings.

Data generation (1000 episodes, NN=0.7, NNN=0.3 fixed, explore mode):
    # Set qarray_config.yaml: cross_coupling 1: {min: 0.7, max: 0.7}, 2: {min: 0.3, max: 0.3}
    # Requires NNN sign fix in env.py (NNN outputs not negated)
    # Requires prior_variance_nnn=0.03 in env.py Kalman init
    for gpu in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$gpu uv run python src/qadapt/capacitance_model/collect_episode_data.py \
            --checkpoint artifacts/rl_checkpoint_best:v3482 \
            --episodes 125 --explore \
            --output data/episode_data_1000ep_gpu${gpu}.npy &
    done
    # Then merge with: python -c "see collect + merge script below"

Plotting:
    uv run python src/qadapt/capacitance_model/plot_kalman_calibration.py \
        --data data/episode_data_1000ep.npy --output-dir .

Depends on plot_calibration_grid.py for load_episode_data() and reconstruct_kalman_variances().
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import ListedColormap

from plot_capacitance_convergence import load_episode_data, reconstruct_kalman_variances

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "episode_data_1000ep.npy"


def truncated_gouldian(vmin=0.25, vmax=0.9):
    """Return gouldian colormap truncated to [vmin, vmax] range."""
    full_cmap = ListedColormap(cc.gouldian)
    colors = full_cmap(np.linspace(vmin, vmax, 256))
    return ListedColormap(colors)


def plot_kalman_calibration(samples, kalman_vars, filter_name, filter_fn, save_path, ylim=0.4, xticks=None):
    """Single Kalman calibration panel: |est - truth| vs Kalman σ, averaged per step."""
    cmap = truncated_gouldian()
    label_size = 7
    tick_size = 7
    tick_length = 4
    tick_width = 1

    # Build per-element records
    records = []
    for i, s in enumerate(samples):
        est = np.array(s['current_estimate'])
        truth = np.array(s['capacitance'])
        kv = kalman_vars[i]
        step = s.get('step', 0)
        if kv is None:
            continue
        kalman_sigma = np.sqrt(np.maximum(kv, 0))
        for k in range(len(est)):
            is_edge = abs(truth[k]) < 1e-6 and abs(est[k]) < 1e-6
            if is_edge:
                continue
            records.append({
                'kalman_err': abs(est[k] - truth[k]),
                'kalman_sigma': kalman_sigma[k],
                'step': step,
                'is_nn': k == 0,
            })

    subset = [r for r in records if filter_fn(r)]
    all_steps = sorted(set(r['step'] for r in records))
    min_step = min(all_steps)
    max_step = max(all_steps)
    step_range = max_step - min_step if max_step > min_step else 1

    # Average per timestep
    step_groups = defaultdict(list)
    for r in subset:
        step_groups[r['step']].append(r)

    avg_xs, avg_ys, avg_steps = [], [], []
    for step in all_steps:
        grp = step_groups.get(step, [])
        if not grp:
            continue
        avg_xs.append(np.mean([r['kalman_sigma'] for r in grp]))
        avg_ys.append(np.mean([r['kalman_err'] for r in grp]))
        avg_steps.append(step)

    avg_xs = np.array(avg_xs)
    avg_ys = np.array(avg_ys)
    norm_steps = (np.array(avg_steps) - min_step) / step_range

    # Plot
    # Wider tick labels (e.g. 0.05) need more left space
    left_pad = 0.21 if ylim < 0.4 else 0.18
    ax_width = 0.45 if ylim < 0.4 else 0.48
    fig = plt.figure(figsize=(2.3, 1.625))
    ax = fig.add_axes((left_pad, 0.19, ax_width, 0.68))

    colors = cmap(norm_steps)
    ax.plot(avg_xs, avg_ys, '-', color='black', linewidth=0.5, zorder=1)
    ax.scatter(avg_xs, avg_ys, s=17, c=colors, edgecolors='none', zorder=5)

    ax.set_xlim(0, avg_xs.max() * 1.1)
    ax.set_ylim(0, ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(r'$\sigma$', fontsize=label_size)
    ax.set_ylabel(r'$\hat{\mu}_{\mathrm{error}}$', fontsize=label_size)

    # Colorbar — right side, same height as axes
    ax_bottom = 0.19
    ax_height = 0.68
    cax = fig.add_axes((0.73, ax_bottom, 0.04, ax_height))
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['1', '50'], fontsize=tick_size - 1)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cax.tick_params(length=tick_length, width=tick_width, pad=1)
    cbar.set_label(r'Time Step ($t$)', fontsize=label_size - 1, labelpad=5)

    for spine in ax.spines.values():
        spine.set_linewidth(tick_width)
    ax.tick_params(which='both', direction='in', labelsize=tick_size,
                   top=True, right=True, length=tick_length, width=tick_width)

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format='svg')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format='png')
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")


def plot_model_convergence(samples, save_path):
    """Convergence plot: CNN prediction error |est + delta - truth| vs step, NN and NNN."""
    cmap = truncated_gouldian()
    label_size = 7
    tick_size = 7
    tick_length = 4
    tick_width = 1

    step_nn_err = defaultdict(list)
    step_nnn_err = defaultdict(list)

    for s in samples:
        est = np.array(s['current_estimate'])
        delta = np.array(s['model_values'])
        truth = np.array(s['capacitance'])
        step = int(s.get('step', 0))

        # NN = output 0
        step_nn_err[step].append(abs(est[0] + delta[0] - truth[0]))

        # NNN = outputs 1, 2 (skip edges)
        for k in [1, 2]:
            if abs(truth[k]) > 1e-6 or abs(est[k]) > 1e-6:
                step_nnn_err[step].append(abs(est[k] + delta[k] - truth[k]))

    steps_0 = np.array(sorted(step_nn_err.keys()))
    steps = steps_0 + 1  # 1-indexed for display
    nn_mean = np.array([np.mean(step_nn_err[s]) for s in steps_0])
    nn_std = np.array([np.std(step_nn_err[s]) for s in steps_0])
    nnn_mean = np.array([np.mean(step_nnn_err[s]) for s in steps_0])
    nnn_std = np.array([np.std(step_nnn_err[s]) for s in steps_0])

    fig = plt.figure(figsize=(1.625, 1.625))
    ax = fig.add_axes((0.25, 0.22, 0.68, 0.68))

    ax.fill_between(steps, nn_mean - 0.5*nn_std, nn_mean + 0.5*nn_std,
                     alpha=0.15, color='C0', edgecolor='none')
    ax.plot(steps, nn_mean, '-', color='C0', linewidth=1.5, label=r'$1^{\mathrm{st}}$')
    ax.fill_between(steps, nnn_mean - 0.5*nnn_std, nnn_mean + 0.5*nnn_std,
                     alpha=0.15, color='C1', edgecolor='none')
    ax.plot(steps, nnn_mean, '-', color='C1', linewidth=1.5, label=r'$2^{\mathrm{nd}}$')

    ax.set_xlabel(r'Time Step ($t$)', fontsize=label_size)
    ax.set_ylabel(r'$\hat{\mu}_{\mathrm{error}}$', fontsize=label_size)
    ax.set_xlim(1, 50)
    ax.set_xticks([1, 50])
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=label_size - 1, loc='upper right', frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(tick_width)
    ax.tick_params(which='both', direction='in', labelsize=tick_size,
                   top=True, right=True, length=tick_length, width=tick_width)

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format='svg')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format='png')
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")


def main():
    parser = argparse.ArgumentParser(description="Kalman calibration plots for paper")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "paper_plots"),
                        help="Directory to save plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading: {args.data}")
    samples = load_episode_data(args.data)
    print(f"Loaded {len(samples)} samples")

    print("Reconstructing Kalman variances...")
    kalman_vars = reconstruct_kalman_variances(samples)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_kalman_calibration(
        samples, kalman_vars,
        filter_name='NN',
        filter_fn=lambda r: r['is_nn'],
        save_path=output_dir / 'capacitance_kalman_calibration_nn.svg',
        xticks=[0, 0.15, 0.3],
    )
    plot_kalman_calibration(
        samples, kalman_vars,
        filter_name='NNN',
        filter_fn=lambda r: not r['is_nn'],
        save_path=output_dir / 'capacitance_kalman_calibration_nnn.svg',
        ylim=0.2,
        xticks=[0, 0.1, 0.2],
    )

    plot_model_convergence(
        samples,
        save_path=output_dir / 'capacitance_model_convergence.svg',
    )


if __name__ == "__main__":
    main()
