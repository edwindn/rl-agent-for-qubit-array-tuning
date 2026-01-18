"""
Bayesian optimization benchmark for quantum dot array tuning.

Uses scikit-optimize (skopt) with Gaussian Process surrogate model.

Usage:
    python run.py --num_dots 2 --num_trials 10 --max_iter 100 --seed 42 --mode joint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent
sys.path.insert(0, str(benchmarks_dir))

from env_init import create_benchmark_env, get_voltage_ranges, random_initial_voltages
from objective import create_objective_fn, get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary


def run_joint_optimization(
    env,
    x0: np.ndarray,
    max_iter: int,
    n_initial_points: int = 20,
    seed: int = None,
) -> dict:
    """
    Run Bayesian optimization on all voltages jointly.

    Args:
        env: QuantumDeviceEnv instance
        x0: Initial voltage array [plungers, barriers] (for reference)
        max_iter: Maximum iterations (function evaluations)
        n_initial_points: Number of LHS initial points before GP modeling
        seed: Random seed for reproducibility

    Returns:
        dict with optimization results
    """
    objective = create_objective_fn(env)
    ranges = get_voltage_ranges(env)

    # Build search space
    dimensions = []
    for i in range(env.num_plunger_voltages):
        dimensions.append(Real(float(ranges["plunger_min"][i]), float(ranges["plunger_max"][i]), name=f"p{i}"))
    for i in range(env.num_barrier_voltages):
        dimensions.append(Real(float(ranges["barrier_min"][i]), float(ranges["barrier_max"][i]), name=f"b{i}"))

    # Create GP with Matern 5/2 kernel
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    # Wrap objective to convert list to numpy array
    def wrapped_objective(x):
        return objective(np.array(x))

    # Run Bayesian optimization with LHS initial sampling
    result = gp_minimize(
        func=wrapped_objective,
        dimensions=dimensions,
        n_calls=max_iter,
        n_initial_points=n_initial_points,
        initial_point_generator="lhs",
        base_estimator=gp,
        acq_func="EI",
        random_state=seed,
        verbose=False,
    )

    return {
        "x": np.array(result.x),
        "fun": result.fun,
        "nit": max_iter,
        "nfev": len(result.func_vals),
        "success": True,
        "history": result.func_vals.tolist(),
    }


def run_sliding_window_optimization(
    env,
    x0: np.ndarray,
    n_calls_per_window: int = 30,
    n_initial_points: int = 20,
    max_sweeps: int = 50,
    max_scans: int = None,
    cap_per_plunger: float = 20.0,
    cap_per_barrier: float = 10.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
    seed: int = None,
) -> dict:
    """
    Sliding window Bayesian optimization matching Nelder-Mead structure.

    Slides one dot at a time: (0,1), (1,2), (2,3), etc.
    Each window includes 2 adjacent plungers + the barrier between them.
    Sweeps until all windows remain below threshold.

    Args:
        env: QuantumDeviceEnv instance
        x0: Initial voltage array [plungers, barriers]
        n_calls_per_window: Function evaluations per window optimization
        n_initial_points: Number of LHS initial points per window
        max_sweeps: Maximum sweeps through all windows
        cap_per_plunger: Cap in V^2 per plunger
        cap_per_barrier: Cap in V^2 per barrier
        threshold_per_plunger: Convergence threshold in V (L1) per plunger
        threshold_per_barrier: Convergence threshold in V (L1) per barrier
        seed: Random seed for reproducibility

    Returns:
        dict with optimization results
    """
    num_plungers = env.num_plunger_voltages
    num_barriers = env.num_barrier_voltages

    current_voltages = x0.copy()
    objective = create_objective_fn(env)
    ranges = get_voltage_ranges(env)

    # Build plunger and barrier bounds
    plunger_bounds = [(float(ranges["plunger_min"][i]), float(ranges["plunger_max"][i]))
                      for i in range(num_plungers)]
    barrier_bounds = [(float(ranges["barrier_min"][i]), float(ranges["barrier_max"][i]))
                      for i in range(num_barriers)]

    # Build windows: list of (plunger_indices, barrier_indices)
    # Sliding by 1: (0,1)+[0], (1,2)+[1], (2,3)+[2], ...
    windows = []
    for i in range(num_plungers - 1):
        plungers = [i, i + 1]
        barriers = [i] if i < num_barriers else []
        windows.append((plungers, barriers))

    history = []
    global_history = []
    voltage_history = []
    total_nfev = 0
    sweep = 0
    all_below_threshold = False

    # Record initial global objective
    initial_global_obj = objective(current_voltages)
    global_history.append(initial_global_obj)
    voltage_history.append(current_voltages.tolist())

    rng = np.random.default_rng(seed)

    for sweep in range(max_sweeps):
        all_below_threshold = True

        for plungers, barriers in windows:
            # Compute cap and threshold for this window
            window_cap = cap_per_plunger * len(plungers) + cap_per_barrier * len(barriers)
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

            # Capture indices for closure
            p_idx = list(plungers)
            b_idx = list(barriers)
            w_cap = window_cap

            def sub_objective(subset_v):
                subset_v = np.array(subset_v)  # Convert from list
                full_v = current_voltages.copy()
                for idx, p in enumerate(p_idx):
                    full_v[p] = subset_v[idx]
                for idx, b in enumerate(b_idx):
                    full_v[num_plungers + b] = subset_v[len(p_idx) + idx]

                local_obj = objective(full_v, plungers=p_idx, barriers=b_idx, cap=w_cap)

                global_obj = objective(full_v)
                global_history.append(global_obj)
                voltage_history.append(full_v.tolist())

                return local_obj

            # Build dimensions for this window
            dimensions = [Real(plunger_bounds[p][0], plunger_bounds[p][1]) for p in plungers]
            dimensions += [Real(barrier_bounds[b][0], barrier_bounds[b][1]) for b in barriers]

            # Create GP with Matern 5/2 kernel
            window_seed = int(rng.integers(0, 2**31))
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=window_seed)

            # Run Bayesian optimization for this window
            result = gp_minimize(
                func=sub_objective,
                dimensions=dimensions,
                n_calls=n_calls_per_window,
                n_initial_points=min(n_initial_points, n_calls_per_window - 1),
                initial_point_generator="lhs",
                base_estimator=gp,
                acq_func="EI",
                random_state=window_seed,
                verbose=False,
            )

            # Update voltages
            for idx, p in enumerate(plungers):
                current_voltages[p] = result.x[idx]
            for idx, b in enumerate(barriers):
                current_voltages[num_plungers + b] = result.x[len(plungers) + idx]

            total_nfev += len(result.func_vals)

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
    n_initial_points: int = 20,
    max_sweeps: int = 50,
    max_scans: int = None,
    n_calls_per_window: int = 30,
    cap_per_plunger: float = 20.0,
    cap_per_barrier: float = 10.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
) -> TrialResult:
    """Run a single optimization trial."""
    rng = np.random.default_rng(seed)

    # Generate random initial voltages
    x0 = random_initial_voltages(env, rng)

    # Run optimization
    if mode == "joint":
        opt_result = run_joint_optimization(
            env, x0, max_iter,
            n_initial_points=n_initial_points,
            seed=seed,
        )
        num_scans = opt_result["nfev"]
    else:
        opt_result = run_sliding_window_optimization(
            env, x0,
            n_calls_per_window=n_calls_per_window,
            n_initial_points=n_initial_points,
            max_sweeps=max_sweeps,
            max_scans=max_scans,
            cap_per_plunger=cap_per_plunger,
            cap_per_barrier=cap_per_barrier,
            threshold_per_plunger=threshold_per_plunger,
            threshold_per_barrier=threshold_per_barrier,
            seed=seed,
        )
        num_scans = opt_result["nfev"]

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
        global_objective_history=opt_result.get("global_history", []),
        voltage_history=opt_result.get("voltage_history", []),
    )


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization benchmark for quantum dot tuning")
    parser.add_argument("--mode", choices=["joint", "pairwise"], default="pairwise", help="Optimization mode")
    parser.add_argument("--num_dots", type=int, default=2, help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum function evaluations per trial (joint mode)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--use_barriers", action="store_true", default=True, help="Include barrier optimization")
    parser.add_argument("--no_barriers", action="store_true", help="Exclude barrier optimization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold in volts")
    parser.add_argument("--n_initial", type=int, default=20, help="Number of LHS initial points")
    parser.add_argument("--max_sweeps", type=int, default=50, help="Maximum sweeps through all windows (pairwise mode)")
    parser.add_argument("--max_scans", type=int, default=None, help="Maximum total scans/function evaluations per trial")
    parser.add_argument("--n_calls_per_window", type=int, default=30, help="Function evaluations per window (pairwise mode)")
    parser.add_argument("--cap_per_plunger", type=float, default=20.0, help="Cap in V^2 per plunger (pairwise mode)")
    parser.add_argument("--cap_per_barrier", type=float, default=10.0, help="Cap in V^2 per barrier (pairwise mode)")
    parser.add_argument("--threshold_per_plunger", type=float, default=0.5, help="Convergence threshold in V per plunger (pairwise mode)")
    parser.add_argument("--threshold_per_barrier", type=float, default=1.0, help="Convergence threshold in V per barrier (pairwise mode)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    use_barriers = args.use_barriers and not args.no_barriers

    print(f"Running Bayesian optimization benchmark")
    print(f"  Mode: {args.mode}")
    print(f"  Dots: {args.num_dots}, Barriers: {use_barriers}")
    print(f"  Trials: {args.num_trials}")
    print(f"  Seed: {args.seed}, Success threshold: {args.threshold}V")
    print(f"  LHS initial points: {args.n_initial}, Kernel: Matern 5/2")
    if args.max_scans:
        print(f"  Max scans: {args.max_scans}")
    if args.mode == "pairwise":
        print(f"  Max sweeps: {args.max_sweeps}, Calls per window: {args.n_calls_per_window}")
        print(f"  Cap: {args.cap_per_plunger} V^2/plunger, {args.cap_per_barrier} V^2/barrier")
        print(f"  Convergence: {args.threshold_per_plunger} V/plunger, {args.threshold_per_barrier} V/barrier")
    else:
        print(f"  Max iterations: {args.max_iter}")
    print()

    # Initialize benchmark result
    result = BenchmarkResult(
        method="bayesian",
        mode=args.mode,
        num_dots=args.num_dots,
        use_barriers=use_barriers,
        num_trials=args.num_trials,
        max_iterations=args.max_iter,
        success_threshold=args.threshold,
        max_sweeps=args.max_sweeps if args.mode == "pairwise" else None,
        cap_per_plunger=args.cap_per_plunger if args.mode == "pairwise" else None,
        cap_per_barrier=args.cap_per_barrier if args.mode == "pairwise" else None,
        threshold_per_plunger=args.threshold_per_plunger if args.mode == "pairwise" else None,
        threshold_per_barrier=args.threshold_per_barrier if args.mode == "pairwise" else None,
    )

    base_rng = np.random.default_rng(args.seed)

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
            n_initial_points=args.n_initial,
            max_sweeps=args.max_sweeps,
            max_scans=args.max_scans,
            n_calls_per_window=args.n_calls_per_window,
            cap_per_plunger=args.cap_per_plunger,
            cap_per_barrier=args.cap_per_barrier,
            threshold_per_plunger=args.threshold_per_plunger,
            threshold_per_barrier=args.threshold_per_barrier,
        )

        result.trials.append(trial_result)

        status = "SUCCESS" if trial_result.success else "FAIL"
        print(f"Trial {trial_idx + 1}/{args.num_trials}: {status} "
              f"(obj={trial_result.final_objective:.4f}, scans={trial_result.num_scans})")

    # Compute and display stats
    result.compute_stats()
    print_summary(result)

    # Save results
    output_path = save_results(result, args.output)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
