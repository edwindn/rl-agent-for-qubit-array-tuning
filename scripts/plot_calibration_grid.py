"""
Generate calibration grid plots + convergence from episode data.

4 scatter plots × 3 filter modes (All/NN/NNN) + convergence panel.
Scatter plots:
  (1) y=Kalman error, x=CNN σ
  (2) y=Kalman error, x=Kalman σ
  (3) y=CNN prediction error, x=CNN σ
  (4) y=CNN prediction error, x=Kalman σ
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "episode_data.npy"


def load_episode_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if isinstance(data, dict):
        data = [data]
    return list(data)


def reconstruct_kalman_variances(samples, n_dots=4, prior_variance=0.1,
                                  prior_variance_nnn=0.03,
                                  variance_threshold=0.05, log_var_bounds=(-6, 2)):
    """
    Reconstruct per-element Kalman variances from saved episode data.
    Returns a list parallel to samples, each entry a dict mapping output_idx -> kalman_var.
    """
    # Group by episode
    episodes = defaultdict(list)
    for i, s in enumerate(samples):
        episodes[s['episode']].append((i, s))

    # For each episode, simulate Kalman variance per element
    kalman_vars = [None] * len(samples)

    for ep_id, ep_samples in episodes.items():
        ep_samples.sort(key=lambda x: x[1]['step'])

        # Element variances: keyed by (row, col) in upper-triangle form
        elem_var = {}
        for i in range(n_dots - 1):
            elem_var[(i, i+1)] = prior_variance  # NN
        for i in range(n_dots - 2):
            elem_var[(i, i+2)] = prior_variance_nnn  # NNN

        # Map (scan_idx, output_idx) -> element (row, col) in upper-triangle
        def get_elements(scan_idx):
            i = scan_idx
            elems = [(min(i, i+1), max(i, i+1))]  # NN: (i, i+1)
            if i + 2 < n_dots:
                elems.append((min(i, i+2), max(i, i+2)))  # NNN_right
            else:
                elems.append(None)
            if i - 1 >= 0:
                elems.append((min(i+1, i-1), max(i+1, i-1)))  # NNN_left
            else:
                elems.append(None)
            return elems

        # Process in step order. 3 samples per step (scan 0, 1, 2).
        by_step = defaultdict(list)
        for idx, s in ep_samples:
            by_step[s['step']].append((idx, s))

        for step in sorted(by_step.keys()):
            step_samples = by_step[step]

            # Record pre-update variances for this step's samples
            for scan_idx, (idx, s) in enumerate(step_samples[:n_dots-1]):
                elements = get_elements(scan_idx)
                log_vars = np.array(s['model_log_vars'])

                sample_kalman_vars = []
                for k, elem in enumerate(elements):
                    if elem is not None and k < len(log_vars):
                        sample_kalman_vars.append(elem_var.get(elem, prior_variance))
                    else:
                        sample_kalman_vars.append(0.0)  # edge
                kalman_vars[idx] = np.array(sample_kalman_vars)

            # Now apply updates (same order as env)
            for scan_idx, (idx, s) in enumerate(step_samples[:n_dots-1]):
                elements = get_elements(scan_idx)
                log_vars = np.array(s['model_log_vars'])

                for k, elem in enumerate(elements):
                    if elem is None or k >= len(log_vars):
                        continue
                    clamped = np.clip(log_vars[k], *log_var_bounds)
                    R = np.exp(clamped)
                    if R >= variance_threshold:
                        continue  # rejected
                    P = elem_var[elem]
                    K = P / (P + R)
                    elem_var[elem] = (1 - K) * P

    return kalman_vars


def make_plots(samples, kalman_vars, save_path, average_per_step=False):
    cmap = ListedColormap(cc.gouldian)
    label_size = 8
    tick_size = 7
    tick_length = 3
    tick_width = 0.8

    # Extract all data points
    records = []
    for i, s in enumerate(samples):
        est = np.array(s['current_estimate'])
        delta = np.array(s['model_values'])
        truth = np.array(s['capacitance'])
        cnn_sigma = np.exp(0.5 * np.array(s['model_log_vars']))
        kv = kalman_vars[i]
        step = s.get('step', 0)

        if kv is None:
            continue

        kalman_sigma = np.sqrt(np.maximum(kv, 0))

        for k in range(len(est)):
            is_edge = abs(truth[k]) < 1e-6 and abs(est[k]) < 1e-6
            records.append({
                'kalman_err': abs(est[k] - truth[k]),
                'cnn_err': abs(est[k] + delta[k] - truth[k]),
                'cnn_sigma': cnn_sigma[k],
                'kalman_sigma': kalman_sigma[k],
                'step': step,
                'is_nn': k == 0,
                'is_edge': is_edge,
            })

    records = [r for r in records if not r['is_edge']]

    all_steps = sorted(set(r['step'] for r in records))
    max_step = max(all_steps)
    min_step = min(all_steps)
    step_range = max_step - min_step if max_step > min_step else 1

    # Filters
    filters = {
        'All': lambda r: True,
        'NN': lambda r: r['is_nn'],
        'NNN': lambda r: not r['is_nn'],
    }

    # Plot configs: (y_key, x_key, ylabel, xlabel)
    plot_cfgs = [
        ('kalman_err', 'cnn_sigma', r'$|\mathrm{est} - \mathrm{truth}|$', r'CNN $\sigma$'),
        ('kalman_err', 'kalman_sigma', r'$|\mathrm{est} - \mathrm{truth}|$', r'Kalman $\sigma$'),
        ('cnn_err', 'cnn_sigma', r'$|\hat{c} - \mathrm{truth}|$', r'CNN $\sigma$'),
        ('cnn_err', 'kalman_sigma', r'$|\hat{c} - \mathrm{truth}|$', r'Kalman $\sigma$'),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.90, wspace=0.35, hspace=0.4)

    for row, (filter_name, filter_fn) in enumerate(filters.items()):
        subset = [r for r in records if filter_fn(r)]

        for col, (y_key, x_key, ylabel, xlabel) in enumerate(plot_cfgs):
            ax = axes[row, col]

            if average_per_step:
                # Average per timestep: one point per step
                step_groups = defaultdict(list)
                for r in subset:
                    step_groups[r['step']].append(r)

                avg_xs, avg_ys, avg_steps = [], [], []
                for step in all_steps:
                    grp = step_groups.get(step, [])
                    if not grp:
                        continue
                    avg_xs.append(np.mean([r[x_key] for r in grp]))
                    avg_ys.append(np.mean([r[y_key] for r in grp]))
                    avg_steps.append(step)

                avg_xs = np.array(avg_xs)
                avg_ys = np.array(avg_ys)
                norm_steps = (np.array(avg_steps) - min_step) / step_range

                colors = cmap(norm_steps)
                ax.plot(avg_xs, avg_ys, '-', color='gray', linewidth=0.5, alpha=0.5, zorder=1)
                ax.scatter(avg_xs, avg_ys, s=15, c=colors, edgecolors='k',
                          linewidths=0.3, zorder=5)
                max_val = avg_xs.max() * 1.1 if len(avg_xs) > 0 else 0.3
            else:
                # Raw scatter: one point per datapoint
                xs = np.array([r[x_key] for r in subset])
                ys = np.array([r[y_key] for r in subset])
                steps = np.array([r['step'] for r in subset])
                norm_steps = (steps - min_step) / step_range

                colors = cmap(norm_steps)
                ax.scatter(xs, ys, alpha=0.3, s=1, c=colors)
                max_val = np.percentile(xs, 99.5) * 1.1

            # Calibration line
            x_line = np.linspace(0, max_val, 100)
            ax.plot(x_line, x_line * np.sqrt(2 / np.pi), '--', color='k', linewidth=0.8)

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, 0.4)
            ax.set_xlabel(xlabel, fontsize=label_size)
            ax.set_ylabel(ylabel, fontsize=label_size)

            if row == 0:
                titles = ['Kalman err vs CNN σ', 'Kalman err vs Kalman σ',
                          'CNN err vs CNN σ', 'CNN err vs Kalman σ']
                ax.set_title(titles[col], fontsize=label_size)

            if col == 0:
                ax.annotate(filter_name, xy=(-0.35, 0.5), xycoords='axes fraction',
                           fontsize=10, fontweight='bold', va='center', rotation=90)

            for spine in ax.spines.values():
                spine.set_linewidth(tick_width)
            ax.tick_params(which='both', direction='in', labelsize=tick_size,
                          top=True, right=True, length=tick_length, width=tick_width)

    # Colorbar
    cax = fig.add_axes((0.35, 0.94, 0.3, 0.015))
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Early', 'Late'])
    cbar.set_label('Step in episode', fontsize=label_size, labelpad=-12)

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format="svg")
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format="png")
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")


def make_convergence_plot(samples, kalman_vars, save_path):
    """Convergence: Kalman estimate error and Kalman σ over steps, NN vs NNN."""
    label_size = 9
    tick_size = 8

    step_nn_err = defaultdict(list)
    step_nnn_err = defaultdict(list)
    step_nn_kvar = defaultdict(list)
    step_nnn_kvar = defaultdict(list)

    for i, s in enumerate(samples):
        est = np.array(s['current_estimate'])
        truth = np.array(s['capacitance'])
        step = int(s.get('step', 0))
        kv = kalman_vars[i]
        if kv is None:
            continue

        # NN = output 0
        step_nn_err[step].append(abs(est[0] - truth[0]))
        step_nn_kvar[step].append(np.sqrt(max(kv[0], 0)))

        # NNN = outputs 1, 2 (skip edges)
        for k in [1, 2]:
            if abs(truth[k]) > 1e-6 or abs(est[k]) > 1e-6:
                step_nnn_err[step].append(abs(est[k] - truth[k]))
                step_nnn_kvar[step].append(np.sqrt(max(kv[k], 0)))

    steps = sorted(step_nn_err.keys())
    steps = np.array(steps)

    nn_err_mean = np.array([np.mean(step_nn_err[s]) for s in steps])
    nn_err_std = np.array([np.std(step_nn_err[s]) for s in steps])
    nnn_err_mean = np.array([np.mean(step_nnn_err[s]) for s in steps])
    nnn_err_std = np.array([np.std(step_nnn_err[s]) for s in steps])

    nn_kvar_mean = np.array([np.mean(step_nn_kvar[s]) for s in steps])
    nnn_kvar_mean = np.array([np.mean(step_nnn_kvar[s]) for s in steps])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.90, wspace=0.3)

    # Left: estimate error
    ax1.fill_between(steps, nn_err_mean - nn_err_std, nn_err_mean + nn_err_std, alpha=0.15, color='C0')
    ax1.plot(steps, nn_err_mean, '-', color='C0', linewidth=1.5, label='NN')
    ax1.fill_between(steps, nnn_err_mean - nnn_err_std, nnn_err_mean + nnn_err_std, alpha=0.15, color='C1')
    ax1.plot(steps, nnn_err_mean, '-', color='C1', linewidth=1.5, label='NNN')
    ax1.axhline(0.05, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('Step', fontsize=label_size)
    ax1.set_ylabel(r'$|\mathrm{estimate} - \mathrm{truth}|$', fontsize=label_size)
    ax1.set_title('Kalman estimate error', fontsize=label_size)
    ax1.legend(fontsize=label_size - 1)
    ax1.set_ylim(0, None)

    # Right: Kalman σ over time
    ax2.plot(steps, nn_kvar_mean, '-', color='C0', linewidth=1.5, label='NN')
    ax2.plot(steps, nnn_kvar_mean, '-', color='C1', linewidth=1.5, label='NNN')
    ax2.set_xlabel('Step', fontsize=label_size)
    ax2.set_ylabel(r'Kalman $\sigma$', fontsize=label_size)
    ax2.set_title('Kalman uncertainty', fontsize=label_size)
    ax2.legend(fontsize=label_size - 1)
    ax2.set_ylim(0, None)

    for ax in (ax1, ax2):
        ax.tick_params(which='both', direction='in', labelsize=tick_size, top=True, right=True)

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format="svg")
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format="png")
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "paper_plots" / "capacitance_calibration_grid.svg"))
    parser.add_argument("--convergence-output", type=str,
                        default=str(PROJECT_ROOT / "paper_plots" / "capacitance_convergence.svg"))
    parser.add_argument("--average", action="store_true",
                        help="Average per timestep (one point per step) instead of raw scatter")
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    samples = load_episode_data(args.data)
    print(f"Loaded {len(samples)} samples")

    print("Reconstructing Kalman variances...")
    kalman_vars = reconstruct_kalman_variances(samples)

    make_plots(samples, kalman_vars, args.output, average_per_step=True)
    make_convergence_plot(samples, kalman_vars, args.convergence_output)


if __name__ == "__main__":
    main()
