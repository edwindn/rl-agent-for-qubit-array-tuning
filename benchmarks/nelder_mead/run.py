"""
Nelder-Mead optimization benchmark for quantum dot array tuning.

Usage:
    python run.py --num_dots 2 --num_trials 10 --max_iter 500 --seed 42 --mode joint
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent
sys.path.insert(0, str(benchmarks_dir))

from env_init import create_benchmark_env, get_voltage_ranges, random_initial_voltages
from objective import create_objective_fn, get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary
from convergence_tracker import ConvergenceTracker


def run_joint_optimization(
    env,
    x0: np.ndarray,
    max_iter: int,
    tracker: ConvergenceTracker,
    tol: float = 1e-6,
) -> dict:
    """
    Run Nelder-Mead optimization on all voltages jointly.

    Args:
        env: QuantumDeviceEnv instance
        x0: Initial voltage array [plungers, barriers]
        max_iter: Maximum iterations
        tracker: ConvergenceTracker for recording distances
        tol: Convergence tolerance

    Returns:
        dict with optimization results
    """
    objective = create_objective_fn(env)
    num_pairs = env.num_plunger_voltages - 1  # num_dots - 1

    # Track convergence history
    history = []
    iteration_count = [0]  # Use list to allow mutation in closure

    # Record initial distances (scan 0)
    plunger_dists, barrier_dists = get_distances(x0, env)
    tracker.record(plunger_dists, barrier_dists, scan_number=0)

    def callback(xk):
        iteration_count[0] += 1
        history.append(objective(xk))
        # Record distances for convergence tracking
        plunger_dists, barrier_dists = get_distances(xk, env)
        current_scan = num_pairs * iteration_count[0]
        tracker.record(plunger_dists, barrier_dists, current_scan)

    # Get bounds from voltage ranges
    ranges = get_voltage_ranges(env)
    bounds = []
    for i in range(env.num_plunger_voltages):
        bounds.append((float(ranges["plunger_min"][i]), float(ranges["plunger_max"][i])))
    for i in range(env.num_barrier_voltages):
        bounds.append((float(ranges["barrier_min"][i]), float(ranges["barrier_max"][i])))

    # Run optimization with bounds
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        callback=callback,
        options={
            'maxiter': max_iter,
            'xatol': tol,
            'fatol': tol,
            'disp': False,
        }
    )

    return {
        "x": result.x,
        "fun": result.fun,
        "nit": result.nit,
        "nfev": result.nfev,
        "success": result.success,
        "history": history,
    }


def run_sliding_window_optimization(
    env,
    x0: np.ndarray,
    tracker: ConvergenceTracker,
    max_iter_per_set: int = 100,
    max_sweeps: int = 50,
    max_scans: int = None,
    cap_per_plunger: float = 5.0,
    cap_per_barrier: float = 4.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
    simplex_step_plunger: float = 35.0,
    simplex_step_barrier: float = 4.0,
    xatol: float = 0.1,
    fatol: float = 0.1,
) -> dict:
    """
    Sliding window optimization.

    Slides one dot at a time: (0,1), (1,2), (2,3), etc.
    Each window includes 2 adjacent plungers + the barrier between them.
    Sweeps until all windows remain below threshold.

    Args:
        env: QuantumDeviceEnv instance
        x0: Initial voltage array [plungers, barriers]
        tracker: ConvergenceTracker for recording distances at each step
        max_iter_per_set: Maximum Nelder-Mead iterations per window
        max_sweeps: Maximum sweeps through all windows
        cap_per_plunger: Cap in V^2 per plunger (default 5.0)
        cap_per_barrier: Cap in V^2 per barrier (default 4.0)
        threshold_per_plunger: Convergence threshold in V (L1) per plunger (default 0.5)
        threshold_per_barrier: Convergence threshold in V (L1) per barrier (default 1.0)
        simplex_step_plunger: Initial simplex step size for plungers in V (default 20.0)
        simplex_step_barrier: Initial simplex step size for barriers in V (default 4.0)
        xatol: Nelder-Mead convergence tolerance on x (default 0.1)
        fatol: Nelder-Mead convergence tolerance on f (default 0.1)

    Note: Convergence threshold is computed as (sum of L1 thresholds)^2

    Returns:
        dict with optimization results
    """
    num_plungers = env.num_plunger_voltages
    num_barriers = env.num_barrier_voltages

    current_voltages = x0.copy()
    objective = create_objective_fn(env)

    # Get voltage bounds from environment
    ranges = get_voltage_ranges(env)
    plunger_bounds = [(float(ranges["plunger_min"][i]), float(ranges["plunger_max"][i]))
                      for i in range(num_plungers)]
    barrier_bounds = [(float(ranges["barrier_min"][i]), float(ranges["barrier_max"][i]))
                      for i in range(num_barriers)]

    # Build windows: list of (plunger_indices, barrier_indices)
    # Sliding by 1: (0,1)+[0], (1,2)+[1], (2,3)+[2], ...
    # Barrier i is between plungers i and i+1
    windows = []
    for i in range(num_plungers - 1):
        plungers = [i, i + 1]
        barriers = [i] if i < num_barriers else []
        windows.append((plungers, barriers))

    history = []  # Local objective at end of each sweep
    global_history = []  # Global objective at each function evaluation
    voltage_history = []  # Voltages at each function evaluation
    total_nfev = 0
    sweep = 0
    all_below_threshold = False

    # Record initial state (scan 0)
    initial_global_obj = objective(current_voltages)  # Full objective (all gates)
    global_history.append(initial_global_obj)
    voltage_history.append(current_voltages.tolist())
    # Record initial distances for convergence tracking
    plunger_dists, barrier_dists = get_distances(current_voltages, env)
    tracker.record(plunger_dists, barrier_dists, scan_number=0)

    for sweep in range(max_sweeps):
        all_below_threshold = True

        for plungers, barriers in windows:
            # Compute cap and threshold for this window (2 plungers + 1 barrier)
            window_cap = cap_per_plunger * len(plungers) + cap_per_barrier * len(barriers)
            # Threshold: square of sum of L1 thresholds (sum then square)
            window_threshold = (threshold_per_plunger * len(plungers) + threshold_per_barrier * len(barriers)) ** 2

            # Check current objective for this window
            current_obj = objective(
                current_voltages,
                plungers=plungers,
                barriers=barriers,
                cap=window_cap,
            )

            if current_obj < window_threshold:
                continue  # Already converged

            all_below_threshold = False

            # Closure captures current values - need to copy to avoid late binding
            p_idx = list(plungers)
            b_idx = list(barriers)
            w_cap = window_cap

            def sub_objective(subset_v):
                # Check max_scans limit BEFORE doing work
                if max_scans is not None and len(global_history) >= max_scans:
                    # Return high value to stop optimization
                    return 1e10

                # Build full voltage vector with updated subset
                full_v = current_voltages.copy()
                for idx, p in enumerate(p_idx):
                    full_v[p] = subset_v[idx]
                for idx, b in enumerate(b_idx):
                    full_v[num_plungers + b] = subset_v[len(p_idx) + idx]

                # Compute local objective for optimization
                local_obj = objective(full_v, plungers=p_idx, barriers=b_idx, cap=w_cap)

                # Track global objective (all gates, no cap) for visualization
                global_obj = objective(full_v)
                global_history.append(global_obj)
                voltage_history.append(full_v.tolist())

                # Record distances for convergence tracking
                # In pairwise mode, each function eval = 1 scan
                plunger_dists, barrier_dists = get_distances(full_v, env)
                tracker.record(plunger_dists, barrier_dists, scan_number=len(global_history))

                return local_obj

            # Current values for subset
            subset_x0 = np.array(
                [current_voltages[p] for p in plungers] +
                [current_voltages[num_plungers + b] for b in barriers]
            )

            # Build bounds for subset
            subset_bounds = [plunger_bounds[p] for p in plungers] + \
                            [barrier_bounds[b] for b in barriers]

            # Build initial simplex: N+1 vertices for N variables
            # Each vertex is x0 with one dimension stepped by the appropriate amount
            n_vars = len(subset_x0)
            step_sizes = [simplex_step_plunger] * len(p_idx) + [simplex_step_barrier] * len(b_idx)
            initial_simplex = np.zeros((n_vars + 1, n_vars))
            initial_simplex[0] = subset_x0
            for i in range(n_vars):
                initial_simplex[i + 1] = subset_x0.copy()
                initial_simplex[i + 1, i] += step_sizes[i]
                # Clip simplex vertices to bounds
                initial_simplex[i + 1, i] = np.clip(
                    initial_simplex[i + 1, i],
                    subset_bounds[i][0],
                    subset_bounds[i][1]
                )

            # Run Nelder-Mead with bounds
            result = minimize(
                sub_objective,
                subset_x0,
                method='Nelder-Mead',
                bounds=subset_bounds,
                options={
                    'maxiter': max_iter_per_set,
                    'xatol': xatol,
                    'fatol': fatol,
                    'initial_simplex': initial_simplex,
                },
            )

            # Update voltages
            for idx, p in enumerate(plungers):
                current_voltages[p] = result.x[idx]
            for idx, b in enumerate(barriers):
                current_voltages[num_plungers + b] = result.x[len(plungers) + idx]

            total_nfev += result.nfev

            # Check if we've exceeded max_scans
            if max_scans is not None and len(global_history) >= max_scans:
                break

        # Check if we've exceeded max_scans (break out of sweep loop too)
        if max_scans is not None and len(global_history) >= max_scans:
            break

        # Record full objective after each sweep
        full_obj = objective(current_voltages)
        history.append(full_obj)

        if all_below_threshold:
            break

    return {
        "x": current_voltages,
        "fun": objective(current_voltages),
        "nit": sweep + 1,
        "nfev": total_nfev,
        "success": all_below_threshold,
        "history": history,
        "global_history": global_history,
        "voltage_history": voltage_history,
    }


def run_single_trial(
    env,
    trial_idx: int,
    seed: int,
    max_iter: int,
    mode: str,
    success_threshold: float,
    tracker: ConvergenceTracker,
    max_sweeps: int = 50,
    max_scans: int = None,
    cap_per_plunger: float = 5.0,
    cap_per_barrier: float = 4.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
    simplex_step_plunger: float = 35.0,
    simplex_step_barrier: float = 4.0,
    xatol: float = 0.1,
    fatol: float = 0.1,
) -> TrialResult:
    """Run a single optimization trial."""
    rng = np.random.default_rng(seed)

    # Generate random initial voltages
    x0 = random_initial_voltages(env, rng)

    # Reset tracker for this trial
    tracker.reset()

    # Run optimization
    if mode == "joint":
        opt_result = run_joint_optimization(env, x0, max_iter, tracker)
        # For joint mode: scans = ceil(num_plungers/2) * iterations
        # (multiple pairs scanned simultaneously per iteration)
        num_pairs = math.ceil(env.num_plunger_voltages / 2)
        num_scans = num_pairs * opt_result["nit"]
    else:
        opt_result = run_sliding_window_optimization(
            env, x0, tracker,
            max_iter_per_set=max_iter,
            max_sweeps=max_sweeps,
            max_scans=max_scans,
            cap_per_plunger=cap_per_plunger,
            cap_per_barrier=cap_per_barrier,
            threshold_per_plunger=threshold_per_plunger,
            threshold_per_barrier=threshold_per_barrier,
            simplex_step_plunger=simplex_step_plunger,
            simplex_step_barrier=simplex_step_barrier,
            xatol=xatol,
            fatol=fatol,
        )
        # For pairwise mode: scans = actual evaluations recorded in global_history
        num_scans = len(opt_result.get("global_history", []))

    # Get final distances
    plunger_dists, barrier_dists = get_distances(opt_result["x"], env)

    # Check success
    success = check_success(opt_result["x"], env, success_threshold)

    return TrialResult(
        trial_idx=trial_idx,
        seed=seed,
        success=success,
        num_iterations=opt_result["nit"],
        num_function_evals=opt_result["nfev"],
        num_scans=num_scans,
        final_objective=opt_result["fun"],
        final_plunger_distances=plunger_dists.tolist(),
        final_barrier_distances=barrier_dists.tolist(),
        convergence_history=opt_result["history"],
        global_objective_history=opt_result.get("global_history", opt_result["history"]),
        voltage_history=opt_result.get("voltage_history", []),
        # New distance tracking fields
        scan_numbers=tracker.scan_numbers.copy(),
        plunger_distance_history=tracker.plunger_distance_history.copy(),
        barrier_distance_history=tracker.barrier_distance_history.copy(),
        plunger_range=tracker.plunger_range,
        barrier_range=tracker.barrier_range,
    )


def main():
    parser = argparse.ArgumentParser(description="Nelder-Mead benchmark for quantum dot tuning")
    parser.add_argument("--mode", choices=["joint", "pairwise"], default="pairwise", help="Optimization mode")
    parser.add_argument("--num_dots", type=int, default=2, help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--max_iter", type=int, default=500, help="Maximum iterations per trial")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--use_barriers", action="store_true", default=True, help="Include barrier optimization")
    parser.add_argument("--no_barriers", action="store_true", help="Exclude barrier optimization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold in volts")
    parser.add_argument("--max_sweeps", type=int, default=50, help="Maximum sweeps through all pairs (pairwise mode)")
    parser.add_argument("--max_scans", type=int, default=None, help="Maximum total scans/function evaluations per trial")
    parser.add_argument("--cap_per_plunger", type=float, default=20.0, help="Cap in V^2 per plunger (pairwise mode)")
    parser.add_argument("--cap_per_barrier", type=float, default=10.0, help="Cap in V^2 per barrier (pairwise mode)")
    parser.add_argument("--threshold_per_plunger", type=float, default=0.5, help="Convergence threshold in V (L1) per plunger (pairwise mode)")
    parser.add_argument("--threshold_per_barrier", type=float, default=1.0, help="Convergence threshold in V (L1) per barrier (pairwise mode)")
    parser.add_argument("--simplex_step_plunger", type=float, default=35.0, help="Initial simplex step size for plungers in V (pairwise mode)")
    parser.add_argument("--simplex_step_barrier", type=float, default=4.0, help="Initial simplex step size for barriers in V (pairwise mode)")
    parser.add_argument("--xatol", type=float, default=0.1, help="Nelder-Mead convergence tolerance on x (pairwise mode)")
    parser.add_argument("--fatol", type=float, default=0.1, help="Nelder-Mead convergence tolerance on f (pairwise mode)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    use_barriers = args.use_barriers and not args.no_barriers

    print(f"Running Nelder-Mead benchmark")
    print(f"  Mode: {args.mode}")
    print(f"  Dots: {args.num_dots}, Barriers: {use_barriers}")
    print(f"  Trials: {args.num_trials}, Max iter: {args.max_iter}")
    print(f"  Seed: {args.seed}, Success threshold: {args.threshold}V")
    if args.max_scans:
        print(f"  Max scans: {args.max_scans}")
    if args.mode == "pairwise":
        print(f"  Max sweeps: {args.max_sweeps}")
        print(f"  Cap: {args.cap_per_plunger} V^2/plunger, {args.cap_per_barrier} V^2/barrier")
        print(f"  Convergence: {args.threshold_per_plunger} V/plunger, {args.threshold_per_barrier} V/barrier")
        print(f"  Simplex step: {args.simplex_step_plunger}V plunger, {args.simplex_step_barrier}V barrier")
        print(f"  Tolerances: xatol={args.xatol}, fatol={args.fatol}")
    print()

    # Initialize benchmark result
    result = BenchmarkResult(
        method="nelder_mead",
        mode=args.mode,
        num_dots=args.num_dots,
        use_barriers=use_barriers,
        num_trials=args.num_trials,
        max_iterations=args.max_iter,
        success_threshold=args.threshold,
        # Pairwise mode parameters
        max_sweeps=args.max_sweeps if args.mode == "pairwise" else None,
        cap_per_plunger=args.cap_per_plunger if args.mode == "pairwise" else None,
        cap_per_barrier=args.cap_per_barrier if args.mode == "pairwise" else None,
        threshold_per_plunger=args.threshold_per_plunger if args.mode == "pairwise" else None,
        threshold_per_barrier=args.threshold_per_barrier if args.mode == "pairwise" else None,
        simplex_step_plunger=args.simplex_step_plunger if args.mode == "pairwise" else None,
        simplex_step_barrier=args.simplex_step_barrier if args.mode == "pairwise" else None,
        xatol=args.xatol if args.mode == "pairwise" else None,
        fatol=args.fatol if args.mode == "pairwise" else None,
    )

    base_rng = np.random.default_rng(args.seed)

    # Create tracker once (will be reused across trials)
    # Need a temporary env to initialize tracker with correct dimensions
    temp_env = create_benchmark_env(
        num_dots=args.num_dots,
        use_barriers=use_barriers,
        seed=0,
    )
    tracker = ConvergenceTracker.from_env(temp_env)

    start_time = time.time()
    for trial_idx in range(args.num_trials):
        trial_seed = base_rng.integers(0, 2**31)

        # Create fresh environment for each trial
        env = create_benchmark_env(
            num_dots=args.num_dots,
            use_barriers=use_barriers,
            seed=trial_seed,
        )

        trial_result = run_single_trial(
            env=env,
            trial_idx=trial_idx,
            seed=trial_seed,
            max_iter=args.max_iter,
            mode=args.mode,
            success_threshold=args.threshold,
            tracker=tracker,
            max_sweeps=args.max_sweeps,
            max_scans=args.max_scans,
            cap_per_plunger=args.cap_per_plunger,
            cap_per_barrier=args.cap_per_barrier,
            threshold_per_plunger=args.threshold_per_plunger,
            threshold_per_barrier=args.threshold_per_barrier,
            simplex_step_plunger=args.simplex_step_plunger,
            simplex_step_barrier=args.simplex_step_barrier,
            xatol=args.xatol,
            fatol=args.fatol,
        )

        result.trials.append(trial_result)

        status = "SUCCESS" if trial_result.success else "FAIL"
        print(f"Trial {trial_idx + 1}/{args.num_trials}: {status} "
              f"(obj={trial_result.final_objective:.4f}, scans={trial_result.num_scans})")

    # Compute and display stats
    # Record total time
    result.total_time_seconds = time.time() - start_time

    result.compute_stats()
    print_summary(result)
    print(f"Total time: {result.total_time_seconds:.1f}s ({result.total_time_seconds/args.num_trials:.1f}s/trial)")

    # Save results
    output_path = save_results(result, args.output)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
