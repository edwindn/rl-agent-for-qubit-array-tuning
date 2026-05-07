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
                        default=str(PROJECT_ROOT / "paper_plots" / "capacitance_convergence.svg"))
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    samples = load_episode_data(args.data)
    print(f"Loaded {len(samples)} samples")

    print("Reconstructing Kalman variances...")
    kalman_vars = reconstruct_kalman_variances(samples)

    make_convergence_plot(samples, kalman_vars, args.output)


if __name__ == "__main__":
    main()
