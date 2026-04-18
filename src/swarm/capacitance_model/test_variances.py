"""
Calibration and convergence plots for capacitance model uncertainty estimates.

Loads episode data from collect_episode_data.py and produces a 2-panel figure:
  Left:  Calibration scatter (predicted σ vs actual prediction error)
  Right: Convergence over episode (Kalman estimate error vs step)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "episode_data.npy"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "model_calibration_plot.svg"


def load_episode_data(data_path):
    """Load per-step episode data saved as a list of sample dicts."""
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if isinstance(data, dict):
        data = [data]
    return list(data)


def calibration_and_convergence_plot(samples, nn_only=False, n_bins=10,
                                     save_path='calibration_plot.svg'):
    """
    Two-panel figure:
      Left:  calibration scatter (predicted σ vs |prediction - truth|)
      Right: convergence (mean |estimate - truth| vs step, with ±1σ band)

    Returns ECE (expected calibration error).
    """
    cmap = ListedColormap(cc.gouldian)
    label_size = 9
    tick_size = 9
    tick_length = 4
    tick_width = 1

    # --- Extract per-element data ---
    pred_uncertainties = []
    pred_errors = []      # |est + delta - truth|  (model prediction error)
    estimate_errors = []  # |est - truth|           (Kalman estimate error)
    step_positions = []

    for sample in samples:
        est = np.array(sample['current_estimate'])
        delta = np.array(sample['model_values'])
        truth = np.array(sample['capacitance'])
        unc = np.exp(0.5 * np.array(sample['model_log_vars']))
        step = sample.get('step', 0)

        if nn_only:
            est, delta, truth, unc = est[:1], delta[:1], truth[:1], unc[:1]

        pred_uncertainties.extend(unc)
        pred_errors.extend(np.abs(est + delta - truth))
        estimate_errors.extend(np.abs(est - truth))
        step_positions.extend([step] * len(est))

    pred_uncertainties = np.array(pred_uncertainties)
    pred_errors = np.array(pred_errors)
    estimate_errors = np.array(estimate_errors)
    step_positions = np.array(step_positions)

    step_range = step_positions.max() - step_positions.min()
    normalized_steps = ((step_positions - step_positions.min()) / step_range
                        if step_range > 0 else np.zeros_like(step_positions, dtype=float))

    # --- Binned calibration curve ---
    sorted_idx = np.argsort(pred_uncertainties)
    sorted_unc = pred_uncertainties[sorted_idx]
    sorted_err = pred_errors[sorted_idx]

    bin_edges = np.linspace(0, len(sorted_unc), n_bins + 1).astype(int)
    binned_unc, binned_err = [], []
    for i in range(n_bins):
        s, e = bin_edges[i], bin_edges[i + 1]
        if s < e:
            binned_unc.append(sorted_unc[s:e].mean())
            binned_err.append(sorted_err[s:e].mean())

    # --- Convergence curves (NN vs NNN, by output index) ---
    # output index 0 = NN, indices 1,2 = NNN
    step_to_nn_errs = defaultdict(list)
    step_to_nnn_errs = defaultdict(list)

    idx = 0
    for sample in samples:
        est = np.array(sample['current_estimate'])
        truth = np.array(sample['capacitance'])
        step = int(sample.get('step', 0))
        errs = np.abs(est - truth)

        if nn_only:
            step_to_nn_errs[step].append(errs[0])
        else:
            step_to_nn_errs[step].append(errs[0])
            for k in range(1, len(errs)):
                # Skip edge elements (truth == 0 and est == 0)
                if abs(truth[k]) > 1e-6 or abs(est[k]) > 1e-6:
                    step_to_nnn_errs[step].append(errs[k])

    steps_sorted = sorted(step_to_nn_errs.keys())
    conv_steps = np.array(steps_sorted)
    nn_mean = np.array([np.mean(step_to_nn_errs[s]) for s in steps_sorted])
    nn_std = np.array([np.std(step_to_nn_errs[s]) for s in steps_sorted])

    has_nnn = len(step_to_nnn_errs) > 0
    if has_nnn:
        nnn_mean = np.array([np.mean(step_to_nnn_errs[s]) for s in steps_sorted])
        nnn_std = np.array([np.std(step_to_nnn_errs[s]) for s in steps_sorted])

    # ===================== PLOT =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.88, wspace=0.35)

    # --- Left panel: calibration scatter ---
    colors = cmap(normalized_steps)
    ax1.scatter(pred_uncertainties, pred_errors, alpha=0.3, s=1, c=colors)
    ax1.scatter(binned_unc, binned_err, marker='s', s=10, color='k', ec='black', zorder=5)

    max_val = max(max(binned_unc), max(binned_err))
    x_line = np.linspace(0, max_val, 100)
    ax1.plot(x_line, x_line * np.sqrt(2 / np.pi), '--', color='k', linewidth=1)

    ax1.set_xlabel(r'Predicted Std Dev $\sigma$', fontsize=label_size)
    ax1.set_ylabel(r'Prediction Error $|\hat{c} - c|$', fontsize=label_size)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, 0.4)
    ax1.set_title('Calibration', fontsize=label_size)

    # Colorbar on left panel
    cax = fig.add_axes((0.09, 0.92, 0.2, 0.02))
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Early', 'Late'])
    cbar.set_label('Step in episode', fontsize=label_size - 1, labelpad=-15)

    # --- Right panel: convergence ---
    ax2.fill_between(conv_steps, nn_mean - nn_std, nn_mean + nn_std,
                     alpha=0.15, color='C0')
    ax2.plot(conv_steps, nn_mean, '-', color='C0', linewidth=1.5, label='NN')

    if has_nnn:
        ax2.fill_between(conv_steps, nnn_mean - nnn_std, nnn_mean + nnn_std,
                         alpha=0.15, color='C1')
        ax2.plot(conv_steps, nnn_mean, '-', color='C1', linewidth=1.5, label='NNN')

    ax2.axhline(0.05, color='gray', linestyle=':', linewidth=1, label='0.05')
    ax2.set_xlabel('Step in episode', fontsize=label_size)
    ax2.set_ylabel(r'$|\mathrm{estimate} - \mathrm{truth}|$', fontsize=label_size)
    ax2.set_xlim(0, conv_steps.max())
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=label_size - 1, loc='upper right')
    ax2.set_title('Convergence', fontsize=label_size)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(tick_width)
        ax.tick_params(which='both', direction='in', labelsize=tick_size,
                       top=True, right=True, length=tick_length, width=tick_width)

    # --- ECE ---
    expected = np.array(binned_unc) * np.sqrt(2 / np.pi)
    calib_error = np.mean(np.abs(expected - np.array(binned_err)))

    save_path = Path(save_path)
    plt.savefig(save_path, transparent=True, format="svg")
    plt.savefig(save_path.with_suffix('.png'), dpi=300, format="png")
    plt.close()
    print(f"Saved '{save_path}' and '{save_path.with_suffix('.png')}'")

    return calib_error


def main():
    parser = argparse.ArgumentParser(description="Generate calibration + convergence plot")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Input .npy path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output .svg path")
    parser.add_argument("--bins", type=int, default=15, help="Number of calibration bins")
    parser.add_argument("--nn-only", action="store_true", help="Only plot NN couplings (output[0] per scan)")
    args = parser.parse_args()

    print(f"Loading episode data from: {args.data}")
    samples = load_episode_data(args.data)
    print(f"Loaded {len(samples)} samples")

    calib_error = calibration_and_convergence_plot(
        samples, nn_only=args.nn_only, n_bins=args.bins, save_path=args.output,
    )
    print(f"Expected Calibration Error: {calib_error:.6f}")


if __name__ == "__main__":
    main()
