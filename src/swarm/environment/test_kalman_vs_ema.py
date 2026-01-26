"""
Compare Kalman vs EMA capacitance updates.

Runs the same environment seed with both methods and visualizes:
1. VGM convergence over time
2. Scans at each step showing virtualization effect
3. Accept/reject stats for Kalman
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import torch
from swarm.environment.env import QuantumDeviceEnv


def run_comparison(num_steps=10, seed=42):
    """Run both methods on the same environment and compare."""

    # Create two environments with different update methods
    # Reset numpy random state before each env creation for reproducibility
    print("Initializing Kalman environment...")
    np.random.seed(seed)
    env_kalman = QuantumDeviceEnv(training=False, config_path="test_config_kalman.yaml")
    obs_k, info_k = env_kalman.reset(seed=seed)

    print("Initializing EMA environment...")
    np.random.seed(seed)  # Reset to same state
    env_ema = QuantumDeviceEnv(training=False, config_path="test_config_ema.yaml")
    obs_e, info_e = env_ema.reset(seed=seed)

    # Both should have same ground truth (same seed)
    true_cgd = env_kalman.array.model.Cgd.copy()
    print(f"\nTrue Cgd matrix:\n{true_cgd}")

    # Verify same ground truth
    if not np.allclose(env_kalman.array.model.Cgd, env_ema.array.model.Cgd):
        print("WARNING: Cgd matrices differ slightly, using Kalman env's Cgd as reference")

    # Get ground truth voltages and barriers
    gt_voltages = env_kalman.device_state["gate_ground_truth"]
    barriers = env_kalman.device_state["current_barrier_voltages"]

    # Extract true nearest-neighbor couplings
    n_dots = env_kalman.num_dots
    true_couplings = []
    coupling_labels = []
    for i in range(n_dots - 1):
        true_couplings.append(true_cgd[i+1, i])  # RL
        true_couplings.append(true_cgd[i, i+1])  # LR
        coupling_labels.append(f"C[{i+1},{i}]")
        coupling_labels.append(f"C[{i},{i+1}]")

    print(f"\nTrue couplings: {[f'{c:.3f}' for c in true_couplings]}")

    # Get references to predictors and ML model
    kalman_predictor = env_kalman.capacitance_model["capacitance_predictor"]
    ema_predictor = env_ema.capacitance_model["capacitance_predictor"]
    ml_model = env_kalman.capacitance_model["ml_model"]
    device = env_kalman.capacitance_model["device"]

    # Storage for tracking
    kalman_history = {label: [] for label in coupling_labels}
    ema_history = {label: [] for label in coupling_labels}
    kalman_var_history = {label: [] for label in coupling_labels}
    ml_predictions = []
    ml_variances = []

    print(f"\n{'='*60}")
    print(f"Running {num_steps} update steps...")
    print(f"{'='*60}")

    np.random.seed(seed + 1000)  # Separate seed for voltage offsets

    for step in range(num_steps):
        # Get observation at current gate voltages (near ground truth with some offset)
        offset = np.random.uniform(-5, 5, size=n_dots)
        gate_voltages = gt_voltages + offset

        # Get observations from each env (different VGMs = different virtualization)
        raw_obs_kalman = env_kalman.array._get_obs(gate_voltages, barriers)
        raw_obs_ema = env_ema.array._get_obs(gate_voltages, barriers)

        obs_norm_kalman = env_kalman._normalise_obs(raw_obs_kalman)
        obs_norm_ema = env_ema._normalise_obs(raw_obs_ema)

        print(f"\nStep {step + 1}:")

        # Get ML predictions for Kalman env
        image_k = obs_norm_kalman["image"]
        batch_k = torch.from_numpy(image_k).float().permute(2, 0, 1).unsqueeze(1).to(device)
        with torch.no_grad():
            values_k, log_vars_k = ml_model(batch_k)
        values_k_np = values_k.cpu().numpy()
        log_vars_k_np = log_vars_k.cpu().numpy()

        # Get ML predictions for EMA env
        image_e = obs_norm_ema["image"]
        batch_e = torch.from_numpy(image_e).float().permute(2, 0, 1).unsqueeze(1).to(device)
        with torch.no_grad():
            values_e, log_vars_e = ml_model(batch_e)
        values_e_np = values_e.cpu().numpy()
        log_vars_e_np = log_vars_e.cpu().numpy()

        print(f"  Kalman ML pred: {values_k_np.flatten()}, vars: {np.exp(log_vars_k_np).flatten()}")
        print(f"  EMA ML pred:    {values_e_np.flatten()}, vars: {np.exp(log_vars_e_np).flatten()}")

        ml_predictions.append(values_k_np.flatten())
        ml_variances.append(np.exp(log_vars_k_np).flatten())

        # Determine output mode from config
        nn_mode = env_kalman.capacitance_model.get("nearest_neighbour", True)
        num_outputs = 2 if nn_mode else 3

        # Update Kalman predictor with Kalman env's predictions
        for i in range(n_dots - 1):
            if num_outputs == 3:
                # NNN mode: [NN, NNN_right, NNN_left]
                ml_outputs = [
                    (float(values_k_np[i, 0]), float(log_vars_k_np[i, 0])),
                    (float(values_k_np[i, 1]), float(log_vars_k_np[i, 1])),
                    (float(values_k_np[i, 2]), float(log_vars_k_np[i, 2])),
                ]
            else:
                # Legacy NN mode: [RL, LR]
                ml_outputs = [
                    (float(values_k_np[i, 0]), float(log_vars_k_np[i, 0])),
                    (float(values_k_np[i, 1]), float(log_vars_k_np[i, 1])),
                ]
            accepted, rejected = kalman_predictor.update_from_scan(left_dot=i, ml_outputs=ml_outputs)
            if step == 0:
                print(f"  Pair {i}: accepted={accepted}, rejected={rejected}")

        # Update EMA predictor with EMA env's predictions
        for i in range(n_dots - 1):
            if num_outputs == 3:
                ml_outputs = [
                    (float(values_e_np[i, 0]), float(log_vars_e_np[i, 0])),
                    (float(values_e_np[i, 1]), float(log_vars_e_np[i, 1])),
                    (float(values_e_np[i, 2]), float(log_vars_e_np[i, 2])),
                ]
            else:
                ml_outputs = [
                    (float(values_e_np[i, 0]), float(log_vars_e_np[i, 0])),
                    (float(values_e_np[i, 1]), float(log_vars_e_np[i, 1])),
                ]
            ema_predictor.update_from_scan(left_dot=i, ml_outputs=ml_outputs)

        # Apply VGM updates to both arrays (this is what happens in the real pipeline)
        kalman_vgm = kalman_predictor.get_full_matrix()
        ema_vgm = ema_predictor.get_full_matrix()
        env_kalman.array._update_virtual_gate_matrix(kalman_vgm)
        env_ema.array._update_virtual_gate_matrix(ema_vgm)

        # Record current estimates
        label_idx = 0
        for i in range(n_dots - 1):
            # RL coupling
            k_mean, k_var = kalman_predictor.get_capacitance_stats(i+1, i)
            e_mean, _ = ema_predictor.get_capacitance_stats(i+1, i)
            kalman_history[coupling_labels[label_idx]].append(k_mean)
            kalman_var_history[coupling_labels[label_idx]].append(k_var)
            ema_history[coupling_labels[label_idx]].append(e_mean)
            label_idx += 1

            # LR coupling
            k_mean, k_var = kalman_predictor.get_capacitance_stats(i, i+1)
            e_mean, _ = ema_predictor.get_capacitance_stats(i, i+1)
            kalman_history[coupling_labels[label_idx]].append(k_mean)
            kalman_var_history[coupling_labels[label_idx]].append(k_var)
            ema_history[coupling_labels[label_idx]].append(e_mean)
            label_idx += 1

    print(f"\nKalman stats: {kalman_predictor.get_stats()}")

    # Convert ML predictions to array for plotting
    ml_predictions = np.array(ml_predictions)  # (num_steps, 6)
    ml_variances = np.array(ml_variances)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = range(1, num_steps + 1)

    # Plot 1: Kalman ML predictions over time (should converge toward 0)
    ax1 = axes[0, 0]
    for idx in range(ml_predictions.shape[1]):
        ax1.plot(steps, ml_predictions[:, idx], 'o-', label=coupling_labels[idx % len(coupling_labels)], markersize=4, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Target (0)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('ML Prediction (Delta)')
    ax1.set_title('Kalman: ML Predictions Should Converge to 0')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Kalman VGM estimate convergence
    ax2 = axes[0, 1]
    for idx, label in enumerate(coupling_labels):
        ax2.plot(steps, kalman_history[label], 'o-', label=label, markersize=4)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Prior (0)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('VGM Estimate')
    ax2.set_title('Kalman VGM Estimates')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Kalman variance evolution
    ax3 = axes[1, 0]
    for label in coupling_labels:
        ax3.plot(steps, kalman_var_history[label], 'o-', label=label, markersize=4)
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Threshold (0.05)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Variance')
    ax3.set_title('Kalman Variance Evolution (log scale)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: EMA VGM estimate
    ax4 = axes[1, 1]
    for idx, label in enumerate(coupling_labels):
        ax4.plot(steps, ema_history[label], 'o-', label=label, markersize=4)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Target (0)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('VGM Estimate')
    ax4.set_title('EMA VGM Estimates')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent / 'kalman_vs_ema_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()

    # Print summary
    kalman_final = [kalman_history[label][-1] for label in coupling_labels]
    ema_final = [ema_history[label][-1] for label in coupling_labels]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nKalman VGM final: {[f'{c:.4f}' for c in kalman_final]}")
    print(f"EMA VGM final:    {[f'{c:.4f}' for c in ema_final]}")

    # Check if ML predictions are converging toward 0 (VGM is correcting)
    ml_pred_first = np.abs(ml_predictions[0]).mean()
    ml_pred_last = np.abs(ml_predictions[-1]).mean()
    print(f"\nML prediction magnitude (should decrease):")
    print(f"  First step: {ml_pred_first:.4f}")
    print(f"  Last step:  {ml_pred_last:.4f}")
    print(f"  Reduction:  {(1 - ml_pred_last/ml_pred_first)*100:.1f}%")

    print(f"\nKalman acceptance rate: {kalman_predictor.get_stats()['acceptance_rate']:.1%}")
    print(f"Kalman total: {kalman_predictor.get_stats()['total_accepted']} accepted, {kalman_predictor.get_stats()['total_rejected']} rejected")


def run_scan_episode(num_steps=10, seed=42):
    """Visualize scans over an episode to show VGM convergence."""

    print("Initializing environment...")
    np.random.seed(seed)
    env = QuantumDeviceEnv(training=False, config_path="test_config_kalman.yaml")
    obs, info = env.reset(seed=seed)

    # Get references
    true_cgd = env.array.model.Cgd.copy()
    gt_voltages = env.device_state["gate_ground_truth"]
    barriers = env.device_state["current_barrier_voltages"]
    n_dots = env.num_dots

    kalman_predictor = env.capacitance_model["capacitance_predictor"]
    ml_model = env.capacitance_model["ml_model"]
    device = env.capacitance_model["device"]

    print(f"True Cgd off-diagonals: {[f'{true_cgd[i+1,i]:.3f}' for i in range(n_dots-1)]}")
    print(f"Ground truth voltages: {gt_voltages}")

    # Storage for scans and predictions
    all_scans = []  # Store scans at each step
    all_predictions = []
    all_variances = []
    vgm_history = []
    kalman_variance_history = []  # Track Kalman variances for uncertainty labels

    np.random.seed(seed + 1000)

    for step in range(num_steps):
        # Use ground truth voltages for scans (ideal conditions)
        gate_voltages = gt_voltages.copy()

        # Get observation
        raw_obs = env.array._get_obs(gate_voltages, barriers)
        obs_normalized = env._normalise_obs(raw_obs)

        # Store all channel scans
        scans = []
        for ch in range(n_dots - 1):
            scan = raw_obs["image"][:, :, ch].copy()
            # Normalize for display
            p_low, p_high = np.percentile(scan, [1, 99])
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scans.append(np.clip(scan_norm, 0, 1))
        all_scans.append(scans)

        # Get ML predictions
        image = obs_normalized["image"]
        batch_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(1).to(device)
        with torch.no_grad():
            values, log_vars = ml_model(batch_tensor)
        values_np = values.cpu().numpy()
        log_vars_np = log_vars.cpu().numpy()

        all_predictions.append(values_np.flatten())
        all_variances.append(np.exp(log_vars_np).flatten())

        # Store current VGM estimate and variances
        vgm_history.append(kalman_predictor.get_full_matrix().copy())
        kalman_variance_history.append(kalman_predictor.variances.copy())

        # Debug: print current VGM
        current_vgm = env.array.model.gate_voltage_composer.virtual_gate_matrix
        print(f"Step {step+1}: pred={values_np.flatten()[:2]}, VGM[1,0]={current_vgm[1,0]:.4f}")

        # Update Kalman predictor (negate predictions due to qarray sign convention)
        nn_mode = env.capacitance_model.get("nearest_neighbour", True)
        num_outputs = 2 if nn_mode else 3
        for i in range(n_dots - 1):
            if num_outputs == 3:
                # NNN mode: [NN, NNN_right, NNN_left]
                ml_outputs = [
                    (-float(values_np[i, 0]), float(log_vars_np[i, 0])),  # NN negated
                    (-float(values_np[i, 1]), float(log_vars_np[i, 1])),  # NNN_right negated
                    (-float(values_np[i, 2]), float(log_vars_np[i, 2])),  # NNN_left negated
                ]
            else:
                # Legacy NN mode: [RL, LR]
                ml_outputs = [
                    (-float(values_np[i, 0]), float(log_vars_np[i, 0])),  # negated
                    (-float(values_np[i, 1]), float(log_vars_np[i, 1])),  # negated
                ]
            kalman_predictor.update_from_scan(left_dot=i, ml_outputs=ml_outputs)

        # Apply VGM update
        kalman_vgm = kalman_predictor.get_full_matrix()
        print(f"        Kalman estimate[1,0]={kalman_vgm[1,0]:.4f}, applying to VGM...")
        env.array._update_virtual_gate_matrix(kalman_vgm)

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_variances = np.array(all_variances)

    # Extract true NN couplings (nearest neighbor)
    true_nn_couplings = [true_cgd[i+1, i] for i in range(n_dots-1)]

    # Extract true NNN couplings (next-nearest neighbor)
    # For channel i: NNN_right = Cgd[i, i+2], NNN_left = Cgd[i+1, i-1]
    true_nnn_right = []
    true_nnn_left = []
    nnn_labels = []
    for i in range(n_dots - 1):
        # NNN_right: Cgd[i, i+2]
        if i + 2 < n_dots:
            true_nnn_right.append(true_cgd[i, i + 2])
        else:
            true_nnn_right.append(0.0)
        # NNN_left: Cgd[i+1, i-1]
        if i - 1 >= 0:
            true_nnn_left.append(true_cgd[i + 1, i - 1])
        else:
            true_nnn_left.append(0.0)
        nnn_labels.append(f'Ch{i}')

    # Extract VGM estimates over time for NN
    vgm_nn_estimates = []
    for vgm in vgm_history:
        vgm_nn_estimates.append([vgm[i+1, i] for i in range(n_dots-1)])
    vgm_nn_estimates = np.array(vgm_nn_estimates)

    # Extract VGM estimates over time for NNN
    vgm_nnn_right_estimates = []
    vgm_nnn_left_estimates = []
    for vgm in vgm_history:
        nnn_r = []
        nnn_l = []
        for i in range(n_dots - 1):
            if i + 2 < n_dots:
                nnn_r.append(vgm[i, i + 2])
            else:
                nnn_r.append(0.0)
            if i - 1 >= 0:
                nnn_l.append(vgm[i + 1, i - 1])
            else:
                nnn_l.append(0.0)
        vgm_nnn_right_estimates.append(nnn_r)
        vgm_nnn_left_estimates.append(nnn_l)
    vgm_nnn_right_estimates = np.array(vgm_nnn_right_estimates)
    vgm_nnn_left_estimates = np.array(vgm_nnn_left_estimates)

    # Extract Kalman variances over time for NN
    kalman_nn_variances = []
    for var_matrix in kalman_variance_history:
        kalman_nn_variances.append([var_matrix[i+1, i] for i in range(n_dots-1)])
    kalman_nn_variances = np.array(kalman_nn_variances)

    # Extract Kalman variances over time for NNN
    kalman_nnn_right_variances = []
    kalman_nnn_left_variances = []
    for var_matrix in kalman_variance_history:
        nnn_r_var = []
        nnn_l_var = []
        for i in range(n_dots - 1):
            if i + 2 < n_dots:
                nnn_r_var.append(var_matrix[i, i + 2])
            else:
                nnn_r_var.append(0.0)
            if i - 1 >= 0:
                nnn_l_var.append(var_matrix[i + 1, i - 1])
            else:
                nnn_l_var.append(0.0)
        kalman_nnn_right_variances.append(nnn_r_var)
        kalman_nnn_left_variances.append(nnn_l_var)
    kalman_nnn_right_variances = np.array(kalman_nnn_right_variances)
    kalman_nnn_left_variances = np.array(kalman_nnn_left_variances)

    # Plot: scans + predictions + NN bar chart + NNN bar chart
    n_channels = min(3, n_dots - 1)
    fig, axes = plt.subplots(n_channels + 3, num_steps, figsize=(2.5 * num_steps, 2.5 * (n_channels + 3)))

    for step in range(num_steps):
        # Row 0 to n_channels-1: Scans
        for ch in range(n_channels):
            ax = axes[ch, step]
            ax.imshow(all_scans[step][ch], cmap='viridis', origin='lower', aspect='equal')
            pred = all_predictions[step, ch*3] if all_predictions.shape[1] >= n_channels*3 else all_predictions[step, ch*2]
            ax.set_title(f'NN pred={pred:.2f}', fontsize=8)
            ax.axis('off')
            if step == 0:
                ax.set_ylabel(f'Ch{ch}', fontsize=10)

        # Row n_channels: Predictions over time (all 3 outputs per channel)
        ax = axes[n_channels, step]
        for ch in range(n_channels):
            base_idx = ch * 3 if all_predictions.shape[1] >= n_channels*3 else ch * 2
            ax.plot(range(1, step+2), all_predictions[:step+1, base_idx], 'o-', markersize=3, label=f'Ch{ch} NN')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(0.5, num_steps + 0.5)
        ax.set_ylim(-0.8, 0.5)
        ax.grid(True, alpha=0.3)
        if step == 0:
            ax.set_ylabel('NN Pred', fontsize=10)
            ax.legend(fontsize=6)

        # Row n_channels+1: NN VGM estimates vs true Cgd (nearest neighbor)
        ax = axes[n_channels + 1, step]
        x = np.arange(n_channels)
        width = 0.35
        ax.bar(x - width/2, true_nn_couplings[:n_channels], width, label='True NN', color='green', alpha=0.7)
        bars = ax.bar(x + width/2, vgm_nn_estimates[step, :n_channels], width, label='VGM Est', color='blue', alpha=0.7)
        # Add uncertainty labels on VGM bars
        for idx, bar in enumerate(bars):
            var = kalman_nn_variances[step, idx]
            std = np.sqrt(var)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'±{std:.2f}', ha='center', va='bottom', fontsize=6, color='blue')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C[{i+1},{i}]' for i in range(n_channels)], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        if step == 0:
            ax.set_ylabel('NN Cgd', fontsize=10)
            ax.legend(fontsize=6, loc='upper right')

        # Row n_channels+2: NNN VGM estimates vs true Cgd (next-nearest neighbor)
        ax = axes[n_channels + 2, step]
        x = np.arange(n_channels * 2)  # 2 NNN values per channel
        width = 0.35
        # Interleave NNN_right and NNN_left for each channel
        true_nnn_all = []
        vgm_nnn_all = []
        nnn_variances_all = []
        nnn_xlabels = []
        for ch in range(n_channels):
            true_nnn_all.append(true_nnn_right[ch])
            true_nnn_all.append(true_nnn_left[ch])
            vgm_nnn_all.append(vgm_nnn_right_estimates[step, ch])
            vgm_nnn_all.append(vgm_nnn_left_estimates[step, ch])
            nnn_variances_all.append(kalman_nnn_right_variances[step, ch])
            nnn_variances_all.append(kalman_nnn_left_variances[step, ch])
            i = ch
            nnn_xlabels.append(f'C[{i},{i+2}]')  # NNN_right
            nnn_xlabels.append(f'C[{i+1},{i-1}]')  # NNN_left
        ax.bar(x - width/2, true_nnn_all, width, label='True NNN', color='orange', alpha=0.7)
        bars = ax.bar(x + width/2, vgm_nnn_all, width, label='VGM Est', color='purple', alpha=0.7)
        # Add uncertainty labels on VGM bars
        for idx, bar in enumerate(bars):
            var = nnn_variances_all[idx]
            std = np.sqrt(var)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'±{std:.2f}', ha='center', va='bottom', fontsize=5, color='purple')
        ax.set_xticks(x)
        ax.set_xticklabels(nnn_xlabels, fontsize=6, rotation=45, ha='right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3, axis='y')
        if step == 0:
            ax.set_ylabel('NNN Cgd', fontsize=10)
            ax.legend(fontsize=6, loc='upper right')

    # Column headers
    for step in range(num_steps):
        axes[0, step].set_title(f'Step {step+1}', fontsize=9)

    plt.suptitle(f'Episode: True NN = {[f"{c:.2f}" for c in true_nn_couplings]}, NNN_R = {[f"{c:.2f}" for c in true_nnn_right]}\nPredictions should → 0, VGM should → True Cgd', fontsize=10)
    plt.tight_layout()

    output_path = Path(__file__).parent / 'scan_episode_convergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scan episode to: {output_path}")
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("EPISODE SUMMARY")
    print(f"{'='*60}")
    pred_first = np.abs(all_predictions[0]).mean()
    pred_last = np.abs(all_predictions[-1]).mean()
    print(f"Mean |prediction|: {pred_first:.4f} → {pred_last:.4f} ({(1-pred_last/pred_first)*100:.1f}% reduction)")
    print(f"Kalman stats: {kalman_predictor.get_stats()}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # run_comparison(num_steps=10, seed=42)
    run_scan_episode(num_steps=10, seed=42)
