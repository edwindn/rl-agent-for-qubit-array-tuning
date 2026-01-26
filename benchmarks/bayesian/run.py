"""
Bayesian optimization benchmark for quantum dot array tuning.

Uses BoTorch with GPU acceleration and Gaussian Process surrogate model.

Usage:
    python run.py --num_dots 2 --num_trials 10 --max_iter 100 --seed 42 --mode joint
    python run.py --num_dots 4 --num_trials 10 --max_iter 200 --mode pairwise --device cuda
    python run.py --num_dots 2 --num_trials 5 --max_iter 50 --batch_size 4  # batch q-EI
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent
sys.path.insert(0, str(benchmarks_dir))

from env_init import create_benchmark_env, get_voltage_ranges, random_initial_voltages
from objective import create_objective_fn, get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary
from convergence_tracker import ConvergenceTracker


# -----------------------------------------------------------------------------
# Device Handling
# -----------------------------------------------------------------------------

def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device with auto-detection.

    Args:
        device: "cuda", "cpu", or "auto" (default)

    Returns:
        torch.device instance
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# -----------------------------------------------------------------------------
# BO Helper Functions
# -----------------------------------------------------------------------------

def generate_initial_points(
    bounds: Tensor,
    n_points: int,
    seed: int = None,
) -> Tensor:
    """
    Generate initial points using Latin Hypercube Sampling.

    Args:
        bounds: Tensor of shape (2, d) with [lower_bounds, upper_bounds]
        n_points: Number of initial points
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (n_points, d) with initial points in original space
    """
    d = bounds.shape[1]
    sampler = qmc.LatinHypercube(d=d, seed=int(seed) if seed is not None else None)
    # Generate points in [0, 1]^d
    unit_points = sampler.random(n=n_points)
    unit_points = torch.tensor(unit_points, dtype=bounds.dtype, device=bounds.device)
    # Scale to bounds
    return bounds[0] + (bounds[1] - bounds[0]) * unit_points


def create_gp_model(
    train_X: Tensor,
    train_Y: Tensor,
    bounds: Tensor,
) -> SingleTaskGP:
    """
    Create a SingleTaskGP with Matern 5/2 kernel.

    Uses input normalization and output standardization for numerical stability.

    Args:
        train_X: Training inputs, shape (n, d)
        train_Y: Training outputs, shape (n, 1)
        bounds: Parameter bounds, shape (2, d)

    Returns:
        SingleTaskGP model (not yet fitted)
    """
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    return model


def fit_gp_model(model: SingleTaskGP) -> None:
    """Fit GP hyperparameters via maximum likelihood estimation."""
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def optimize_acquisition(
    model: SingleTaskGP,
    bounds: Tensor,
    best_f: float,
    batch_size: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> Tensor:
    """
    Optimize Expected Improvement acquisition function.

    Args:
        model: Fitted GP model
        bounds: Parameter bounds, shape (2, d)
        best_f: Best observed objective value (for minimization)
        batch_size: Number of candidates to return (1 for sequential EI, >1 for q-EI)
        num_restarts: Number of L-BFGS restarts for acquisition optimization
        raw_samples: Number of initial random samples for multi-start optimization

    Returns:
        Tensor of shape (batch_size, d) with next evaluation point(s)
    """
    if batch_size == 1:
        # Sequential Log Expected Improvement (numerically stable)
        acq_func = LogExpectedImprovement(model=model, best_f=best_f)
    else:
        # Batch q-Log Expected Improvement
        acq_func = qLogExpectedImprovement(model=model, best_f=best_f)

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return candidates


# -----------------------------------------------------------------------------
# Joint Optimization
# -----------------------------------------------------------------------------

def run_joint_optimization(
    env,
    x0: np.ndarray,
    max_iter: int,
    tracker: ConvergenceTracker,
    n_initial_points: int = 20,
    batch_size: int = 1,
    seed: int = None,
    device: torch.device = None,
) -> dict:
    """
    Run Bayesian optimization on all voltages jointly.

    Args:
        env: QuantumDeviceEnv instance
        x0: Initial voltage array [plungers, barriers] (for reference)
        max_iter: Maximum iterations (function evaluations)
        tracker: ConvergenceTracker for recording distances
        n_initial_points: Number of Sobol initial points before GP modeling
        batch_size: Number of candidates per acquisition (1=sequential, >1=batch q-EI)
        seed: Random seed for reproducibility
        device: Torch device (cuda/cpu)

    Returns:
        dict with optimization results
    """
    if device is None:
        device = get_device()

    objective = create_objective_fn(env)
    ranges = get_voltage_ranges(env)
    num_pairs = env.num_plunger_voltages - 1  # num_dots - 1

    # Build bounds tensor
    lower = []
    upper = []
    for i in range(env.num_plunger_voltages):
        lower.append(float(ranges["plunger_min"][i]))
        upper.append(float(ranges["plunger_max"][i]))
    for i in range(env.num_barrier_voltages):
        lower.append(float(ranges["barrier_min"][i]))
        upper.append(float(ranges["barrier_max"][i]))

    bounds = torch.tensor([lower, upper], dtype=torch.float64, device=device)

    # Generate initial points via Sobol
    train_X = generate_initial_points(bounds, n_initial_points, seed=seed)

    # Evaluate initial points and record distances
    train_Y_list = []
    for i, x in enumerate(train_X):
        x_np = x.cpu().numpy()
        y = objective(x_np)
        train_Y_list.append([y])
        # Record distances for convergence tracking
        plunger_dists, barrier_dists = get_distances(x_np, env)
        current_scan = num_pairs * (i + 1)
        tracker.record(plunger_dists, barrier_dists, current_scan)

    train_Y = torch.tensor(train_Y_list, dtype=torch.float64, device=device)

    history = train_Y.squeeze(-1).tolist()

    # BO loop
    n_remaining = max_iter - n_initial_points
    n_bo_iterations = (n_remaining + batch_size - 1) // batch_size if n_remaining > 0 else 0

    for _ in range(n_bo_iterations):
        # Fit GP
        model = create_gp_model(train_X, train_Y, bounds)
        model = model.to(device)
        fit_gp_model(model)

        # Get best observed value (for minimization, EI expects best_f to be the minimum)
        best_f = train_Y.min().item()

        # Optimize acquisition
        candidates = optimize_acquisition(
            model=model,
            bounds=bounds,
            best_f=best_f,
            batch_size=min(batch_size, max_iter - len(history)),
            num_restarts=10,
            raw_samples=512,
        )

        # Evaluate candidates and record distances
        new_Y_list = []
        for c in candidates:
            c_np = c.cpu().numpy()
            y = objective(c_np)
            new_Y_list.append([y])
            # Record distances for convergence tracking
            plunger_dists, barrier_dists = get_distances(c_np, env)
            current_scan = num_pairs * (len(history) + len(new_Y_list))
            tracker.record(plunger_dists, barrier_dists, current_scan)

        new_Y = torch.tensor(new_Y_list, dtype=torch.float64, device=device)

        # Update training data
        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        history.extend(new_Y.squeeze(-1).tolist())

        # Check if we've reached max_iter
        if len(history) >= max_iter:
            break

    # Get best result
    best_idx = train_Y.argmin()
    best_x = train_X[best_idx].cpu().numpy()
    best_y = train_Y[best_idx].item()

    return {
        "x": best_x,
        "fun": best_y,
        "nit": max_iter,
        "nfev": len(history),
        "success": True,
        "history": history,
    }


# -----------------------------------------------------------------------------
# Sliding Window (Pairwise) Optimization
# -----------------------------------------------------------------------------

def run_sliding_window_optimization(
    env,
    x0: np.ndarray,
    tracker: ConvergenceTracker,
    n_calls_per_window: int = 30,
    n_initial_points: int = 20,
    max_sweeps: int = 50,
    max_scans: int = None,
    cap_per_plunger: float = 20.0,
    cap_per_barrier: float = 10.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
    batch_size: int = 1,
    seed: int = None,
    device: torch.device = None,
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
        n_initial_points: Number of Sobol initial points per window
        max_sweeps: Maximum sweeps through all windows
        max_scans: Maximum total scans/function evaluations per trial
        cap_per_plunger: Cap in V^2 per plunger
        cap_per_barrier: Cap in V^2 per barrier
        threshold_per_plunger: Convergence threshold in V (L1) per plunger
        threshold_per_barrier: Convergence threshold in V (L1) per barrier
        batch_size: Number of candidates per acquisition
        seed: Random seed for reproducibility
        device: Torch device

    Returns:
        dict with optimization results
    """
    if device is None:
        device = get_device()

    num_plungers = env.num_plunger_voltages
    num_barriers = env.num_barrier_voltages

    current_voltages = x0.copy()
    objective = create_objective_fn(env)
    ranges = get_voltage_ranges(env)

    # Build plunger and barrier bounds
    plunger_bounds = [
        (float(ranges["plunger_min"][i]), float(ranges["plunger_max"][i]))
        for i in range(num_plungers)
    ]
    barrier_bounds = [
        (float(ranges["barrier_min"][i]), float(ranges["barrier_max"][i]))
        for i in range(num_barriers)
    ]

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
    all_below_threshold = False

    # Record initial state (scan 0)
    initial_global_obj = objective(current_voltages)
    global_history.append(initial_global_obj)
    voltage_history.append(current_voltages.tolist())
    # Record initial distances for convergence tracking
    plunger_dists, barrier_dists = get_distances(current_voltages, env)
    tracker.record(plunger_dists, barrier_dists, scan_number=0)

    rng = np.random.default_rng(seed)

    for sweep in range(max_sweeps):
        all_below_threshold = True

        for plungers, barriers in windows:
            # Compute cap and threshold for this window
            window_cap = cap_per_plunger * len(plungers) + cap_per_barrier * len(barriers)
            window_threshold = (
                threshold_per_plunger * len(plungers) + threshold_per_barrier * len(barriers)
            ) ** 2

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

            # Build window-specific bounds
            window_lower = [plunger_bounds[p][0] for p in plungers] + [
                barrier_bounds[b][0] for b in barriers
            ]
            window_upper = [plunger_bounds[p][1] for p in plungers] + [
                barrier_bounds[b][1] for b in barriers
            ]
            window_bounds = torch.tensor(
                [window_lower, window_upper], dtype=torch.float64, device=device
            )

            # Capture indices for closure
            p_idx = list(plungers)
            b_idx = list(barriers)
            w_cap = window_cap

            def make_sub_objective(p_idx, b_idx, w_cap):
                """Create sub-objective with captured indices."""

                def sub_objective(subset_v: np.ndarray) -> float:
                    full_v = current_voltages.copy()
                    for idx, p in enumerate(p_idx):
                        full_v[p] = subset_v[idx]
                    for idx, b in enumerate(b_idx):
                        full_v[num_plungers + b] = subset_v[len(p_idx) + idx]

                    local_obj = objective(full_v, plungers=p_idx, barriers=b_idx, cap=w_cap)

                    global_obj = objective(full_v)
                    global_history.append(global_obj)
                    voltage_history.append(full_v.tolist())

                    # Record distances for convergence tracking
                    # In pairwise mode, each function eval = 1 scan
                    plunger_dists, barrier_dists = get_distances(full_v, env)
                    tracker.record(plunger_dists, barrier_dists, scan_number=len(global_history))

                    return local_obj

                return sub_objective

            sub_objective = make_sub_objective(p_idx, b_idx, w_cap)

            # Run BO for this window
            window_seed = int(rng.integers(0, 2**31))
            window_result = _run_window_bo(
                sub_objective=sub_objective,
                bounds=window_bounds,
                n_calls=n_calls_per_window,
                n_initial=min(n_initial_points, n_calls_per_window - 1),
                batch_size=batch_size,
                seed=window_seed,
                device=device,
            )

            # Update voltages from window result
            best_x = window_result["x"]
            for idx, p in enumerate(plungers):
                current_voltages[p] = best_x[idx]
            for idx, b in enumerate(barriers):
                current_voltages[num_plungers + b] = best_x[len(plungers) + idx]

            total_nfev += window_result["nfev"]

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


def _run_window_bo(
    sub_objective,
    bounds: Tensor,
    n_calls: int,
    n_initial: int,
    batch_size: int = 1,
    seed: int = None,
    device: torch.device = None,
) -> dict:
    """
    Run BO for a single window (subset of variables).

    Args:
        sub_objective: Callable that takes numpy array and returns float
        bounds: Tensor of shape (2, d)
        n_calls: Total function evaluations
        n_initial: Number of initial Sobol points
        batch_size: Candidates per acquisition
        seed: Random seed
        device: Torch device

    Returns:
        dict with "x" (best point as numpy), "fun" (best value), "nfev" (evaluations)
    """
    # Generate initial points
    train_X = generate_initial_points(bounds, n_initial, seed=seed)

    # Evaluate initial points
    train_Y = torch.tensor(
        [[sub_objective(x.cpu().numpy())] for x in train_X],
        dtype=torch.float64,
        device=device,
    )

    nfev = n_initial

    # BO loop
    n_remaining = n_calls - n_initial
    n_bo_iterations = (n_remaining + batch_size - 1) // batch_size if n_remaining > 0 else 0

    for _ in range(n_bo_iterations):
        if nfev >= n_calls:
            break

        # Fit GP
        model = create_gp_model(train_X, train_Y, bounds)
        model = model.to(device)
        fit_gp_model(model)

        # Optimize acquisition
        best_f = train_Y.min().item()
        actual_batch = min(batch_size, n_calls - nfev)

        candidates = optimize_acquisition(
            model=model,
            bounds=bounds,
            best_f=best_f,
            batch_size=actual_batch,
            num_restarts=5,  # Fewer restarts for window optimization
            raw_samples=256,
        )

        # Evaluate candidates
        new_Y = torch.tensor(
            [[sub_objective(c.cpu().numpy())] for c in candidates],
            dtype=torch.float64,
            device=device,
        )

        train_X = torch.cat([train_X, candidates], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        nfev += len(candidates)

    # Get best result
    best_idx = train_Y.argmin()
    best_x = train_X[best_idx].cpu().numpy()
    best_y = train_Y[best_idx].item()

    return {"x": best_x, "fun": best_y, "nfev": nfev}


# -----------------------------------------------------------------------------
# Trial Runner
# -----------------------------------------------------------------------------

def run_single_trial(
    env,
    trial_idx: int,
    seed: int,
    max_iter: int,
    mode: str,
    success_threshold: float,
    tracker: ConvergenceTracker,
    n_initial_points: int = 20,
    max_sweeps: int = 50,
    max_scans: int = None,
    n_calls_per_window: int = 30,
    cap_per_plunger: float = 20.0,
    cap_per_barrier: float = 10.0,
    threshold_per_plunger: float = 0.5,
    threshold_per_barrier: float = 1.0,
    batch_size: int = 1,
    device: torch.device = None,
) -> TrialResult:
    """Run a single optimization trial."""
    rng = np.random.default_rng(seed)

    # Generate random initial voltages
    x0 = random_initial_voltages(env, rng)

    # Reset tracker for this trial
    tracker.reset()

    # Run optimization
    if mode == "joint":
        opt_result = run_joint_optimization(
            env,
            x0,
            max_iter,
            tracker=tracker,
            n_initial_points=n_initial_points,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        # For joint mode: scans = (num_dots - 1) * num_evaluations
        num_pairs = env.num_plunger_voltages - 1
        num_scans = num_pairs * opt_result["nfev"]
    else:
        opt_result = run_sliding_window_optimization(
            env,
            x0,
            tracker=tracker,
            n_calls_per_window=n_calls_per_window,
            n_initial_points=n_initial_points,
            max_sweeps=max_sweeps,
            max_scans=max_scans,
            cap_per_plunger=cap_per_plunger,
            cap_per_barrier=cap_per_barrier,
            threshold_per_plunger=threshold_per_plunger,
            threshold_per_barrier=threshold_per_barrier,
            batch_size=batch_size,
            seed=seed,
            device=device,
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization benchmark for quantum dot tuning (BoTorch/GPU)"
    )
    parser.add_argument(
        "--mode", choices=["joint", "pairwise"], default="pairwise", help="Optimization mode"
    )
    parser.add_argument("--num_dots", type=int, default=2, help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum function evaluations per trial (joint mode)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--use_barriers", action="store_true", default=True, help="Include barrier optimization"
    )
    parser.add_argument("--no_barriers", action="store_true", help="Exclude barrier optimization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold in volts")
    parser.add_argument("--n_initial", type=int, default=20, help="Number of Sobol initial points")
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=50,
        help="Maximum sweeps through all windows (pairwise mode)",
    )
    parser.add_argument(
        "--max_scans",
        type=int,
        default=None,
        help="Maximum total scans/function evaluations per trial",
    )
    parser.add_argument(
        "--n_calls_per_window",
        type=int,
        default=20,
        help="Function evaluations per window (pairwise mode)",
    )
    parser.add_argument(
        "--cap_per_plunger",
        type=float,
        default=20.0,
        help="Cap in V^2 per plunger (pairwise mode)",
    )
    parser.add_argument(
        "--cap_per_barrier",
        type=float,
        default=10.0,
        help="Cap in V^2 per barrier (pairwise mode)",
    )
    parser.add_argument(
        "--threshold_per_plunger",
        type=float,
        default=0.5,
        help="Convergence threshold in V per plunger (pairwise mode)",
    )
    parser.add_argument(
        "--threshold_per_barrier",
        type=float,
        default=1.0,
        help="Convergence threshold in V per barrier (pairwise mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for q-EI acquisition (1=sequential EI, >1=batch q-EI)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for BO computation (auto detects CUDA)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    use_barriers = args.use_barriers and not args.no_barriers
    device = get_device(args.device)

    print("Running Bayesian optimization benchmark (BoTorch)")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"  Dots: {args.num_dots}, Barriers: {use_barriers}")
    print(f"  Trials: {args.num_trials}")
    print(f"  Seed: {args.seed}, Success threshold: {args.threshold}V")
    print(f"  Latin Hypercube initial points: {args.n_initial}, Kernel: Matern 5/2")
    print(f"  Batch size: {args.batch_size} ({'sequential EI' if args.batch_size == 1 else 'q-EI'})")
    if args.max_scans:
        print(f"  Max scans: {args.max_scans}")
    if args.mode == "pairwise":
        print(f"  Max sweeps: {args.max_sweeps}, Calls per window: {args.n_calls_per_window}")
        print(f"  Cap: {args.cap_per_plunger} V^2/plunger, {args.cap_per_barrier} V^2/barrier")
        print(
            f"  Convergence: {args.threshold_per_plunger} V/plunger, "
            f"{args.threshold_per_barrier} V/barrier"
        )
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

    # Create tracker once (will be reused across trials)
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
            n_initial_points=args.n_initial,
            max_sweeps=args.max_sweeps,
            max_scans=args.max_scans,
            n_calls_per_window=args.n_calls_per_window,
            cap_per_plunger=args.cap_per_plunger,
            cap_per_barrier=args.cap_per_barrier,
            threshold_per_plunger=args.threshold_per_plunger,
            threshold_per_barrier=args.threshold_per_barrier,
            batch_size=args.batch_size,
            device=device,
        )

        result.trials.append(trial_result)

        status = "SUCCESS" if trial_result.success else "FAIL"
        print(
            f"Trial {trial_idx + 1}/{args.num_trials}: {status} "
            f"(obj={trial_result.final_objective:.4f}, scans={trial_result.num_scans})"
        )

    # Record total time
    result.total_time_seconds = time.time() - start_time

    # Compute and display stats
    result.compute_stats()
    print_summary(result)
    print(f"Total time: {result.total_time_seconds:.1f}s ({result.total_time_seconds/args.num_trials:.1f}s/trial)")

    # Save results
    output_path = save_results(result, args.output)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
