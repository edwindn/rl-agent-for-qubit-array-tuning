import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for clean imports
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def test_capacitance_values(num_steps=10, seed=42):
    """
    Test script that uses env.step() to update voltages and let env handle Kalman updates.
    Uses small random actions around ground truth, similar to test_kalman_vs_ema.py.
    """
    print("Initializing environment...")
    np.random.seed(seed)
    env = QuantumDeviceEnv(training=False, num_dots=4, use_barriers=True)
    obs, info = env.reset(seed=seed)

    # Get references
    true_cgd = env.array.model.Cgd.copy()
    n_dots = env.num_dots
    num_channels = obs["image"].shape[2]

    print(f"True Cgd off-diagonals: {[f'{true_cgd[i+1,i]:.3f}' for i in range(n_dots-1)]}")

    # Storage for scans and predictions
    all_scans = {i: [] for i in range(num_channels)}
    vgm_history = []

    # Set random seed for voltage offsets
    np.random.seed(seed + 1000)

    for step in range(num_steps):
        # Get current ground truth
        gt_voltages = env.device_state["gate_ground_truth"]
        gt_barriers = env.device_state["barrier_ground_truth"]

        # Add small random offset to voltages (similar to test_kalman_vs_ema)
        offset = np.random.uniform(-5, 5, size=n_dots)
        target_voltages = gt_voltages + offset

        # Normalize to action space [-1, 1]
        normalized_gates = (target_voltages - env.plunger_min) / (env.plunger_max - env.plunger_min)
        normalized_gates = normalized_gates * 2 - 1
        normalized_gates = np.clip(normalized_gates, -1, 1)

        normalized_barriers = (gt_barriers - env.barrier_min) / (env.barrier_max - env.barrier_min)
        normalized_barriers = normalized_barriers * 2 - 1
        normalized_barriers = np.clip(normalized_barriers, -1, 1)

        # Create action dictionary
        action = {
            "action_gate_voltages": normalized_gates,
            "action_barrier_voltages": normalized_barriers,
        }

        # Store VGM before step
        vgm_before = env.array.model.gate_voltage_composer.virtual_gate_matrix
        kalman_estimate_before = env.capacitance_model["capacitance_predictor"].get_full_matrix().copy()

        print(f"\nStep {step + 1}/{num_steps}:")
        print(f"  VGM[1,0] before step: {vgm_before[1,0]:.4f}")
        print(f"  Kalman estimate[1,0] before: {kalman_estimate_before[1,0]:.4f}")

        # Step the environment - this will call _update_virtual_gate_matrix internally
        obs, reward, terminated, truncated, info = env.step(action)

        # Store VGM after step
        vgm_after = env.array.model.gate_voltage_composer.virtual_gate_matrix
        kalman_estimate_after = env.capacitance_model["capacitance_predictor"].get_full_matrix().copy()
        vgm_history.append(kalman_estimate_after)

        print(f"  VGM[1,0] after step: {vgm_after[1,0]:.4f}")
        print(f"  Kalman estimate[1,0] after: {kalman_estimate_after[1,0]:.4f}")

        # Store each channel separately
        for channel_idx in range(num_channels):
            all_scans[channel_idx].append(obs["image"][:, :, channel_idx])

    # Extract true couplings
    true_couplings = [true_cgd[i+1, i] for i in range(n_dots-1)]

    # Extract VGM estimates over time
    vgm_estimates = []
    for vgm in vgm_history:
        vgm_estimates.append([vgm[i+1, i] for i in range(n_dots-1)])
    vgm_estimates = np.array(vgm_estimates)

    # Create plot with scans + VGM convergence
    n_channels_plot = min(3, num_channels)
    fig, axes = plt.subplots(n_channels_plot + 1, num_steps, figsize=(2.5 * num_steps, 2.5 * (n_channels_plot + 1)))

    for step in range(num_steps):
        # Rows 0 to n_channels_plot-1: Scans
        for ch in range(n_channels_plot):
            ax = axes[ch, step]
            scan = all_scans[ch][step]
            ax.imshow(scan, cmap='viridis', origin='lower', aspect='equal', vmin=0, vmax=1)
            ax.axis('off')
            if step == 0:
                ax.set_ylabel(f'Ch{ch}', fontsize=10)
            if ch == 0:
                ax.set_title(f'Step {step+1}', fontsize=8)

        # Bottom row: VGM estimates vs true Cgd
        ax = axes[n_channels_plot, step]
        x = np.arange(n_channels_plot)
        width = 0.35
        ax.bar(x - width/2, true_couplings[:n_channels_plot], width, label='True Cgd', color='green', alpha=0.7)
        ax.bar(x + width/2, vgm_estimates[step, :n_channels_plot], width, label='VGM Est', color='blue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i+1},{i}' for i in range(n_channels_plot)], fontsize=7)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, axis='y')
        if step == 0:
            ax.set_ylabel('Cgd values', fontsize=10)
            ax.legend(fontsize=6)

    plt.suptitle(f'Env.step() Test: True Cgd = {[f"{c:.2f}" for c in true_couplings]}\nVGM should → True Cgd', fontsize=11)
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'test_capacitance_scans.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved test scans to {output_path}")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"True Cgd: {[f'{c:.4f}' for c in true_couplings]}")
    print(f"Initial VGM: {[f'{vgm_estimates[0, i]:.4f}' for i in range(len(true_couplings))]}")
    print(f"Final VGM: {[f'{vgm_estimates[-1, i]:.4f}' for i in range(len(true_couplings))]}")
    print(f"\nKalman stats: {env.capacitance_model['capacitance_predictor'].get_stats()}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_capacitance_values(num_steps=10, seed=42)
