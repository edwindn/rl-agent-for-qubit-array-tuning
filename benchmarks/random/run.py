"""Random sampling benchmark for quantum dot tuning."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add benchmarks directory to path
benchmarks_dir = Path(__file__).parent.parent
sys.path.insert(0, str(benchmarks_dir))

from env_init import create_benchmark_env, get_voltage_ranges
from objective import create_objective_fn, get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary


def run_single_trial(
    env,
    trial_idx: int,
    seed: int,
    max_samples: int,
    success_threshold: float,
    batch_size: int = 1000,
) -> TrialResult:
    """Run a single random sampling trial with batched evaluation."""
    rng = np.random.default_rng(seed)
    objective = create_objective_fn(env)
    ranges = get_voltage_ranges(env)

    num_plungers = env.num_plunger_voltages
    num_barriers = env.num_barrier_voltages

    # Get optimal voltages for vectorized success check
    from objective import PhysicalObjective
    phys_obj = PhysicalObjective(env)
    ref = np.concatenate([env.device_state['gate_ground_truth'],
                          env.device_state['barrier_ground_truth']])
    opt_plungers, opt_barriers = phys_obj.get_optimal_voltages(ref)
    optimal = np.concatenate([opt_plungers, opt_barriers])

    global_history = []
    success = False
    final_sample = max_samples
    voltages = None

    samples_done = 0
    while samples_done < max_samples:
        current_batch = min(batch_size, max_samples - samples_done)

        # Batch sample random voltages
        plunger_v = rng.uniform(
            low=ranges["plunger_min"],
            high=ranges["plunger_max"],
            size=(current_batch, num_plungers)
        )
        barrier_v = rng.uniform(
            low=ranges["barrier_min"],
            high=ranges["barrier_max"],
            size=(current_batch, num_barriers)
        )
        batch_voltages = np.concatenate([plunger_v, barrier_v], axis=1)

        # Evaluate objectives for batch
        for i in range(current_batch):
            voltages = batch_voltages[i]
            obj_val = objective(voltages)
            global_history.append(obj_val)

            # Vectorized success check: all voltages within threshold
            if np.all(np.abs(voltages - optimal) <= success_threshold):
                success = True
                final_sample = samples_done + i + 1
                break

        if success:
            break
        samples_done += current_batch

    # Get final distances
    plunger_dists, barrier_dists = get_distances(voltages, env)

    return TrialResult(
        trial_idx=trial_idx,
        seed=seed,
        success=success,
        num_iterations=final_sample,
        num_function_evals=final_sample,
        num_scans=final_sample,
        final_objective=global_history[-1],
        final_plunger_distances=plunger_dists.tolist(),
        final_barrier_distances=barrier_dists.tolist(),
        convergence_history=[],
        global_objective_history=global_history,
    )


def main():
    parser = argparse.ArgumentParser(description="Random sampling benchmark for quantum dot tuning")
    parser.add_argument("--num_dots", type=int, default=2, help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--max_samples", type=int, default=100000, help="Maximum samples per trial")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold in volts")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    print(f"Running random sampling benchmark")
    print(f"  Dots: {args.num_dots}")
    print(f"  Trials: {args.num_trials}, Max samples: {args.max_samples}")
    print(f"  Seed: {args.seed}, Threshold: {args.threshold}V")
    print()

    result = BenchmarkResult(
        method="random",
        mode="sampling",
        num_dots=args.num_dots,
        use_barriers=True,
        num_trials=args.num_trials,
        max_iterations=args.max_samples,
        success_threshold=args.threshold,
    )

    base_rng = np.random.default_rng(args.seed)

    for trial_idx in range(args.num_trials):
        trial_seed = base_rng.integers(0, 2**31)

        env = create_benchmark_env(
            num_dots=args.num_dots,
            use_barriers=True,
            seed=trial_seed,
        )

        trial_result = run_single_trial(
            env=env,
            trial_idx=trial_idx,
            seed=trial_seed,
            max_samples=args.max_samples,
            success_threshold=args.threshold,
        )

        result.trials.append(trial_result)

        status = "SUCCESS" if trial_result.success else "FAIL"
        print(f"Trial {trial_idx + 1}/{args.num_trials}: {status} "
              f"(obj={trial_result.final_objective:.4f}, samples={trial_result.num_scans})")

    result.compute_stats()
    print_summary(result)

    output_path = save_results(result, args.output)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
