import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


def main():
    # Configuration
    NUM_STEPS = 200
    RADIAL_DISTANCE = 1.0  # Threshold for locking voltages

    # Initialize environment with default config
    env = QuantumDeviceEnv(training=True)

    # Set update method to "perfect" if not already
    if env.capacitance_model is not None:
        print(f"WARNING: Removing capacitance update method: {env.capacitance_model}")
        env.capacitance_model = None

    # Disable barriers if using perfect virtualization (not yet implemented for barriers)
    if env.use_barriers and env.capacitance_model == "perfect":
        print("WARNING: Disabling barriers because perfect virtualization is not implemented for barriers")
        env.use_barriers = False

    # Reset environment
    obs, info = env.reset()

    # Get ground truths from device state
    gate_ground_truth = env.device_state["gate_ground_truth"]
    barrier_ground_truth = env.device_state["barrier_ground_truth"]

    # Initialize distance tracking (per gate)
    gate_distances = [[] for _ in range(env.num_plunger_voltages)]
    barrier_distances = [[] for _ in range(env.num_barrier_voltages)]

    # Track which gates are locked
    locked_gates = np.zeros(env.num_plunger_voltages, dtype=bool)
    locked_gate_values = np.zeros(env.num_plunger_voltages)

    print(f"Running random policy for {NUM_STEPS} steps...")
    print(f"Number of gates: {env.num_plunger_voltages}")
    print(f"Number of barriers: {env.num_barrier_voltages}")
    print(f"Radial distance threshold: {RADIAL_DISTANCE}")

    # Run random steps
    for step in range(NUM_STEPS):
        # Sample random actions from action space
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get current voltages
        current_gate_voltages = env.device_state["current_gate_voltages"].copy()
        current_barrier_voltages = env.device_state["current_barrier_voltages"]

        # Check which gates are within radial distance and lock them
        gate_dists_signed = current_gate_voltages - gate_ground_truth
        gate_dists_individual = np.abs(gate_dists_signed)
        newly_locked = (gate_dists_individual <= RADIAL_DISTANCE) & (~locked_gates)

        if np.any(newly_locked):
            locked_gates |= newly_locked
            locked_gate_values[newly_locked] = current_gate_voltages[newly_locked]
            print(f"Step {step + 1}: Locked gate(s) {np.where(newly_locked)[0].tolist()}")

        # Apply locked values
        current_gate_voltages[locked_gates] = locked_gate_values[locked_gates]
        env.device_state["current_gate_voltages"] = current_gate_voltages

        # Calculate signed distances from ground truth for each individual gate and barrier
        gate_dists = current_gate_voltages - gate_ground_truth
        barrier_dists = current_barrier_voltages - barrier_ground_truth

        # Append signed distances for each gate
        for i in range(env.num_plunger_voltages):
            gate_distances[i].append(gate_dists[i])

        # Append signed distances for each barrier
        for i in range(env.num_barrier_voltages):
            barrier_distances[i].append(barrier_dists[i])

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{NUM_STEPS} - Locked gates: {np.sum(locked_gates)}/{env.num_plunger_voltages}")

        # Check if all gates are locked
        if np.all(locked_gates):
            print(f"\nAll gates locked at step {step + 1}! Ending rollout.")
            break

    # Plot gate distances (one line per gate)
    fig, ax = plt.subplots(figsize=(10, 6))
    num_steps_taken = len(gate_distances[0])
    steps = np.arange(1, num_steps_taken + 1)

    for i in range(env.num_plunger_voltages):
        ax.plot(steps, gate_distances[i], alpha=0.7, linewidth=1, label=f"Gate {i}")

    # Add horizontal lines for radial distance thresholds
    ax.axhline(y=RADIAL_DISTANCE, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Radial Distance Threshold')
    ax.axhline(y=-RADIAL_DISTANCE, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Signed Distance from Ground Truth")
    ax.set_title("Random Policy: Gate Voltage Distances")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = current_dir / "gate_distances.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved gate distances plot to: {output_path}")

    # Plot barrier distances (one line per barrier)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(env.num_barrier_voltages):
        ax.plot(steps, barrier_distances[i], alpha=0.7, linewidth=1, label=f"Barrier {i}")

    # Add horizontal lines for radial distance thresholds
    ax.axhline(y=RADIAL_DISTANCE, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Radial Distance Threshold')
    ax.axhline(y=-RADIAL_DISTANCE, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Signed Distance from Ground Truth")
    ax.set_title("Random Policy: Barrier Voltage Distances")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = current_dir / "barrier_distances.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved barrier distances plot to: {output_path}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    for i in range(env.num_plunger_voltages):
        gate_dist_array = np.array(gate_distances[i])
        print(f"Gate {i} - Mean: {np.mean(gate_dist_array):.4f}, Std: {np.std(gate_dist_array):.4f}, Final: {gate_dist_array[-1]:.4f}")

    for i in range(env.num_barrier_voltages):
        barrier_dist_array = np.array(barrier_distances[i])
        print(f"Barrier {i} - Mean: {np.mean(barrier_dist_array):.4f}, Std: {np.std(barrier_dist_array):.4f}, Final: {barrier_dist_array[-1]:.4f}")


if __name__ == "__main__":
    main()
