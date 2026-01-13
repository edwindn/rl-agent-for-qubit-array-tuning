import sys
from pathlib import Path
import numpy as np

from scipy.optimize import minimize

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from src.swarm.environment.env import QuantumDeviceEnv 
from utils import score_functions, normalize_voltages


def main():
    # Initialize the environment
    env = QuantumDeviceEnv(training=False)
    obs, info = env.reset()

    # Extract ground truth and initial voltages from device state
    device_state = info["current_device_state"]
    plunger_gt = device_state["gate_ground_truth"]
    barrier_gt = device_state["barrier_ground_truth"]

    # Initialize simplex with starting voltages from the environment
    current_plunger_voltages = device_state["current_gate_voltages"].copy()
    current_barrier_voltages = device_state["current_barrier_voltages"].copy()

    # Get voltage ranges for constraints
    plunger_min = env.plunger_min
    plunger_max = env.plunger_max
    barrier_min = env.barrier_min
    barrier_max = env.barrier_max

    print(f"Starting optimization...")
    print(f"Plunger ground truth: {plunger_gt}")
    print(f"Barrier ground truth: {barrier_gt}")
    print(f"Initial plunger voltages: {current_plunger_voltages}")
    print(f"Initial barrier voltages: {current_barrier_voltages}")

    # TODO: Initialize Nelder-Mead simplex
    # - Create initial simplex vertices (n+1 vertices for n-dimensional problem)
    # - Each vertex represents a complete set of plunger + barrier voltages
    # - Simplex should be created around the initial voltages
    simplex = None

    # Nelder-Mead parameters
    # TODO: Set appropriate hyperparameters (alpha, gamma, rho, sigma)
    max_iterations = 1000
    tolerance = 1e-6

    # Optimization loop
    for iteration in range(max_iterations):

        # TODO: Evaluate objective function at each simplex vertex
        # - For each vertex (set of voltages), compute the score
        # - Use score_functions from utils.py
        # - Store scores for all vertices

        # TODO: Sort simplex vertices by their scores (best to worst)

        # TODO: Check convergence
        # - If the range of scores is below tolerance, stop
        # - If we're close enough to ground truth, stop

        # Evaluate current best solution
        # TODO: Get the best vertex from simplex
        best_plunger_voltages = current_plunger_voltages  # Placeholder
        best_barrier_voltages = current_barrier_voltages  # Placeholder

        plunger_score, barrier_score = score_functions(
            best_plunger_voltages,
            best_barrier_voltages,
            plunger_gt,
            barrier_gt
        )

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Plunger score = {plunger_score:.6f}, Barrier score = {barrier_score:.6f}")

        # TODO: Compute centroid of all vertices except the worst

        # TODO: Reflection step
        # - Reflect the worst vertex through the centroid
        # - Evaluate the reflected point

        # TODO: Expansion step (if reflection is very good)
        # - Try expanding further in the reflection direction
        # - Keep expanded point if it's better, otherwise keep reflection

        # TODO: Contraction step (if reflection is not good enough)
        # - Try contracting toward the centroid
        # - If contraction fails, proceed to shrink

        # TODO: Shrink step (if contraction fails)
        # - Shrink all vertices toward the best vertex

        # TODO: Apply voltage constraints
        # - Clip all voltages to stay within [min, max] ranges
        # - Ensure no vertex violates physical constraints

        # TODO: Update simplex with new vertices

        # Optional: Step the environment to get visual feedback (CSD images)
        # This is not required for optimization but useful for debugging
        # action = {
        #     "action_gate_voltages": normalize_voltages(best_plunger_voltages, plunger_min, plunger_max),
        #     "action_barrier_voltages": normalize_voltages(best_barrier_voltages, barrier_min, barrier_max)
        # }
        # obs, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    main()