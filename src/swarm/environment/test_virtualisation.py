"""
Test virtualisation by applying the exact Cgd matrix to update_virtual_gate_matrix.

This script verifies that when we use the exact (ground truth) Cgd matrix to update
the virtual gate matrix, subsequent scans appear perfectly virtualised with no crosstalk.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import QuantumDeviceEnv


def test_virtualisation(step_size=10, output_path="virtualisation_test.png",
                        v0_min=None, v0_max=None, v1_min=None, v1_max=None):
    """
    Test virtualisation by comparing scans before and after applying exact Cgd.

    Creates two grids:
    1. Before virtualisation: Scans with default (imperfect) virtual gate matrix
    2. After virtualisation: Scans after updating with exact Cgd matrix

    Args:
        step_size: Voltage step size for sampling (default: 10V)
        output_path: Path to save the output image
        v0_min, v0_max: Override min/max voltage for gate 0
        v1_min, v1_max: Override min/max voltage for gate 1
    """
    # Initialize environment
    env = QuantumDeviceEnv(training=True)
    obs, info = env.reset()

    # Ensure virtual gate matrix is identity (should already be from reset)
    print("\n=== Initial virtual gate matrix (should be identity) ===")
    print(env.array.model.gate_voltage_composer.virtual_gate_matrix)

    # Get voltage ranges for first two plunger gates
    plunger_min = env.plunger_min[:2].copy()
    plunger_max = env.plunger_max[:2].copy()

    print(f"Environment voltage ranges:")
    print(f"  Gate 0: [{plunger_min[0]:.2f}, {plunger_max[0]:.2f}] V")
    print(f"  Gate 1: [{plunger_min[1]:.2f}, {plunger_max[1]:.2f}] V")

    # Override with user-specified ranges if provided
    if v0_min is not None:
        plunger_min[0] = v0_min
    if v0_max is not None:
        plunger_max[0] = v0_max
    if v1_min is not None:
        plunger_min[1] = v1_min
    if v1_max is not None:
        plunger_max[1] = v1_max

    print(f"\nUsing voltage ranges:")
    print(f"  Gate 0: [{plunger_min[0]:.2f}, {plunger_max[0]:.2f}] V")
    print(f"  Gate 1: [{plunger_min[1]:.2f}, {plunger_max[1]:.2f}] V")
    print(f"  Ground truth: {info['current_device_state']['gate_ground_truth'][:2]}")

    # Create voltage grids
    v0_points = np.arange(plunger_min[0], plunger_max[0] + step_size, step_size)
    v1_points = np.arange(plunger_min[1], plunger_max[1] + step_size, step_size)

    n_cols = len(v0_points)
    n_rows = len(v1_points)

    print(f"\nCreating {n_rows} x {n_cols} grid ({n_rows * n_cols} scans)")
    print(f"Step size: {step_size} V")

    # Get other gate voltages
    other_gates = env.device_state["current_gate_voltages"][2:]
    barriers = env.device_state["current_barrier_voltages"]

    # --- BEFORE VIRTUALISATION ---
    print("\n=== Collecting scans BEFORE perfect virtualisation ===")
    if env.use_barriers:
        print(f"Original virtual gate matrix:\n{env.array.model.gate_voltage_composer.virtual_gate_matrix}")
    else:
        print(f"Original virtual gate matrix:\n{env.array.model.gate_voltage_composer.virtual_gate_matrix}")

    scans_before = []
    for v1 in v1_points:
        row_scans = []
        for v0 in v0_points:
            gate_voltages = np.array([v0, v1] + list(other_gates))
            raw_obs = env.array._get_obs(gate_voltages, barriers)
            scan = raw_obs["image"][:, :, 0]

            # Normalize
            p_low = np.percentile(scan, 0.5)
            p_high = np.percentile(scan, 99.5)
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scan_norm = np.clip(scan_norm, 0, 1)

            row_scans.append(scan_norm)
        scans_before.append(row_scans)

    # --- APPLY PERFECT VIRTUALISATION ---
    print("\n=== Applying EXACT Cgd matrix to virtual gate matrix ===")

    # Get the exact Cgd matrix from the model
    if env.use_barriers:
        # For barrier model, use the full cgd matrix
        exact_cgd = env.array.model.cgd_full[:env.num_dots, :env.num_dots+1]  # (n_dot, n_gate+1)
        print(f"Exact Cgd shape: {exact_cgd.shape}")
        print(f"Exact Cgd matrix:\n{exact_cgd}")
    else:
        # For non-barrier model
        exact_cgd = env.array.model.cgd
        print(f"Exact Cgd shape: {exact_cgd.shape}")
        print(f"Exact Cgd matrix:\n{exact_cgd}")

    # Update virtual gate matrix with exact Cgd
    env.array._update_virtual_gate_matrix(exact_cgd)

    print(f"Updated virtual gate matrix:\n{env.array.model.gate_voltage_composer.virtual_gate_matrix}")

    # --- AFTER VIRTUALISATION ---
    print("\n=== Collecting scans AFTER perfect virtualisation ===")
    scans_after = []
    for v1 in v1_points:
        row_scans = []
        for v0 in v0_points:
            gate_voltages = np.array([v0, v1] + list(other_gates))
            raw_obs = env.array._get_obs(gate_voltages, barriers)
            scan = raw_obs["image"][:, :, 0]

            # Normalize
            p_low = np.percentile(scan, 0.5)
            p_high = np.percentile(scan, 99.5)
            if p_high > p_low:
                scan_norm = (scan - p_low) / (p_high - p_low)
            else:
                scan_norm = np.zeros_like(scan)
            scan_norm = np.clip(scan_norm, 0, 1)

            row_scans.append(scan_norm)
        scans_after.append(row_scans)

    # --- CREATE COMPARISON PLOT ---
    print("\n=== Creating comparison plot ===")
    fig = plt.figure(figsize=(2 * n_cols, 4 * n_rows + 2))

    # Create two sets of subplots: before (top) and after (bottom)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Before virtualisation
    gs_before = gs[0].subgridspec(n_rows, n_cols, hspace=0.1, wspace=0.1)
    # After virtualisation
    gs_after = gs[1].subgridspec(n_rows, n_cols, hspace=0.1, wspace=0.1)

    # Plot BEFORE
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs_before[n_rows - 1 - row_idx, col_idx])
            scan = scans_before[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            v0 = v0_points[col_idx]
            v1 = v1_points[row_idx]
            ax.set_title(f"({v0:.0f}, {v1:.0f})", fontsize=8)

    # Plot AFTER
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs_after[n_rows - 1 - row_idx, col_idx])
            scan = scans_after[row_idx][col_idx]

            ax.imshow(scan, cmap='viridis', origin='lower', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            v0 = v0_points[col_idx]
            v1 = v1_points[row_idx]
            ax.set_title(f"({v0:.0f}, {v1:.0f})", fontsize=8)

    # Add section labels
    fig.text(0.5, 0.97, 'BEFORE Perfect Virtualisation (Default Virtual Gate Matrix)',
             ha='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.48, 'AFTER Perfect Virtualisation (Exact Cgd Applied)',
             ha='center', fontsize=14, weight='bold')

    # Add axis labels
    fig.text(0.5, 0.01, f'Gate 0 Voltage (V) →', ha='center', fontsize=12)
    fig.text(0.01, 0.5, f'Gate 1 Voltage (V) →', va='center', rotation='vertical', fontsize=12)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved virtualisation test to: {output_path}")
    plt.close()

    # Calculate and print crosstalk metrics
    print("\n=== Crosstalk Analysis ===")
    print("In perfectly virtualised scans, charge transitions should appear perfectly")
    print("horizontal and vertical (no diagonal features), indicating zero crosstalk.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test virtualisation with exact Cgd matrix")
    parser.add_argument("--step-size", type=float, default=20.0,
                        help="Voltage step size in volts (default: 20)")
    parser.add_argument("--output", type=str, default="virtualisation_test.png",
                        help="Output image path (default: virtualisation_test.png)")
    parser.add_argument("--v0-min", type=float, default=-40,
                        help="Minimum voltage for gate 0")
    parser.add_argument("--v0-max", type=float, default=40,
                        help="Maximum voltage for gate 0")
    parser.add_argument("--v1-min", type=float, default=-40,
                        help="Minimum voltage for gate 1")
    parser.add_argument("--v1-max", type=float, default=40,
                        help="Maximum voltage for gate 1")

    args = parser.parse_args()

    test_virtualisation(
        step_size=args.step_size,
        output_path=args.output,
        v0_min=args.v0_min,
        v0_max=args.v0_max,
        v1_min=args.v1_min,
        v1_max=args.v1_max
    )
