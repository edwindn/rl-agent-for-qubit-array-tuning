"""
Test script to visualize crosstalk effects on charge stability diagrams.
Varies the first and last plunger voltages and observes changes in the CSDs of the middle two dots.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from env import QuantumDeviceEnv

def test_crosstalk_with_scans():
    """Test crosstalk by varying first and last plungers and scanning the middle two dots."""

    # Create output directory
    output_dir = "crosstalk_scans"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving scans to: {output_dir}/")

    # Initialize environment
    print("\nInitializing environment...")
    env = QuantumDeviceEnv()
    obs, info = env.reset()

    print("\n" + "="*80)
    print("CROSSTALK TEST: Observing CSD changes in middle dots (1-2)")
    print("="*80)

    # Get initial state and ground truth voltages
    initial_voltages = info["current_device_state"]["current_gate_voltages"]
    ground_truth_voltages = info["current_device_state"]["gate_ground_truth"]  # Get ground truth voltages
    print(f"\nInitial plunger voltages: {initial_voltages}")
    print(f"Ground truth voltages: {ground_truth_voltages}")
    print(f"Plunger ranges: min={env.plunger_min}, max={env.plunger_max}")

    # Set middle two dots (plungers 1 and 2) to their ground truth values
    # Convert ground truth voltages to normalized action space [-1, 1]
    p1_gt = ground_truth_voltages[1]
    p2_gt = ground_truth_voltages[2]

    p1_normalized = 2 * (p1_gt - env.plunger_min[1]) / (env.plunger_max[1] - env.plunger_min[1]) - 1
    p2_normalized = 2 * (p2_gt - env.plunger_min[2]) / (env.plunger_max[2] - env.plunger_min[2]) - 1

    print(f"\nSetting middle plungers (1 and 2) to ground truth:")
    print(f"  Plunger 1: {p1_gt:.4f} V (normalized: {p1_normalized:.4f})")
    print(f"  Plunger 2: {p2_gt:.4f} V (normalized: {p2_normalized:.4f})")

    # Save initial scan (before any changes)
    initial_scan = obs["image"][:, :, 1]  # Channel 1 is the scan between dots 1-2
    save_scan_image(initial_scan, 0, initial_voltages, env, output_dir, scan_channel="dots_1-2")
    print(f"\nSaved initial scan: step_000.png")

    # Run test for multiple steps
    num_steps = 15
    print(f"\nRunning {num_steps} steps with random actions on first and last plungers:")
    print("-" * 80)

    for step in range(num_steps):
        # Create action: random for first and last plungers, ground truth for middle two
        action = {
            "action_gate_voltages": np.array([
                np.random.uniform(-0.5, 0.5),  # Random action on plunger 0
                p1_normalized,                  # Ground truth for plunger 1
                p2_normalized,                  # Ground truth for plunger 2
                np.random.uniform(-0.5, 0.5)   # Random action on plunger 3
            ], dtype=np.float32),
            "action_barrier_voltages": np.zeros(3, dtype=np.float32)  # No barrier changes
        }

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)

        # Get current voltages
        current_voltages = info["current_device_state"]["current_gate_voltages"]

        # Get the CSD scan for the middle pair (dots 1-2)
        # This is channel 1 in the observation (channel 0: dots 0-1, channel 1: dots 1-2, channel 2: dots 2-3)
        scan = obs["image"][:, :, 1]

        # Save the scan
        save_scan_image(scan, step + 1, current_voltages, env, output_dir, scan_channel="dots_1-2")

        # Print progress
        print(f"Step {step + 1:2d}: V0={current_voltages[0]:7.3f} (Δ={current_voltages[0]-initial_voltages[0]:+.3f}), "
              f"V1={current_voltages[1]:7.3f} (Δ={current_voltages[1]-p1_gt:+.3f}), "
              f"V2={current_voltages[2]:7.3f} (Δ={current_voltages[2]-p2_gt:+.3f}), "
              f"V3={current_voltages[3]:7.3f} (Δ={current_voltages[3]-initial_voltages[3]:+.3f}) | "
              f"Saved step_{step+1:03d}.png")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

    # Summary
    final_voltages = info["current_device_state"]["current_gate_voltages"]
    total_change = final_voltages - initial_voltages

    print(f"\nAll {num_steps + 1} scans saved to: {output_dir}/")
    print("\nVoltage Summary:")
    print(f"  Initial voltages:     {initial_voltages}")
    print(f"  Ground truth voltages: {ground_truth_voltages}")
    print(f"  Final voltages:       {final_voltages}")
    print(f"  Total changes:        {total_change}")
    print(f"\nExpected behavior:")
    print(f"  - If crosstalk exists: CSDs should change despite V1 and V2 being at ground truth")
    print(f"  - If perfectly virtualized: CSDs should remain identical across all steps")


def save_scan_image(scan, step_num, voltages, env, output_dir, scan_channel="dots_1-2"):
    """Save a CSD scan as a PNG image."""

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the scan
    im = ax.imshow(
        scan,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    # Labels and title
    ax.set_xlabel("V2 (V)", fontsize=12)
    ax.set_ylabel("V1 (V)", fontsize=12)
    ax.set_title(f"Step {step_num}: CSD for Dots 1-2\n"
                 f"V0={voltages[0]:.2f}, V1={voltages[1]:.2f}, V2={voltages[2]:.2f}, V3={voltages[3]:.2f}",
                 fontsize=11)

    # Colorbar
    plt.colorbar(im, ax=ax, label="Charge Sensor Signal")

    # Save
    filename = os.path.join(output_dir, f"step_{step_num:03d}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    test_crosstalk_with_scans()
