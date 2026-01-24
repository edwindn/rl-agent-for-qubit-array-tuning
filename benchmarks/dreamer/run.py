"""
DreamerV3 evaluation benchmark for quantum dot array tuning.

Evaluates a trained DreamerV3 agent and produces BenchmarkResult compatible output.

Usage:
    python run.py --checkpoint /path/to/logdir --num_dots 2 --num_trials 10 --gpu 5
"""

import argparse
import os
import sys
from pathlib import Path

# Parse --gpu early before JAX imports
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument('--gpu', type=int, default=5, help='GPU device index')
_gpu_args, _ = _gpu_parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_args.gpu)

# Disable JAX transfer guard (qarray needs host-to-device transfers)
import jax
jax.config.update('jax_transfer_guard', 'allow')

import numpy as np

# Setup paths
folder = Path(__file__).parent
benchmarks_dir = folder.parent
sys.path.insert(0, str(folder))
sys.path.insert(1, str(benchmarks_dir))
sys.path.insert(2, str(benchmarks_dir.parent / 'src'))

import ninjax_patch  # Patch ninjax 3.6.2 debug print (see ninjax_patch.py)

import elements
import ruamel.yaml as yaml

from env_init import create_benchmark_env, get_ground_truth
from objective import get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary
from wrapper import make_dreamer_env
from convergence_tracker import ConvergenceTracker


def load_agent_from_checkpoint(checkpoint_path: str, env):
    """
    Load a trained DreamerV3 agent from checkpoint.

    Args:
        checkpoint_path: Path to the training logdir containing checkpoint
        env: Environment instance for obs/act spaces

    Returns:
        Loaded agent ready for evaluation
    """
    from agent import Agent
    from embodied.envs import from_gym
    import embodied

    # Load config from checkpoint
    config_path = Path(checkpoint_path) / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path) as f:
        config_dict = yaml.YAML(typ='safe').load(f)
    config = elements.Config(config_dict)

    # Wrap env and get spaces
    gym_env = make_dreamer_env(
        num_dots=env.num_dots,
        use_barriers=env.use_barriers,
        max_steps=env.max_steps
    )
    wrapped_env = from_gym.FromGym(gym_env)

    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in wrapped_env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in wrapped_env.act_space.items() if k != 'reset'}
    wrapped_env.close()

    # Create agent with config
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(checkpoint_path),
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    ))

    # Load checkpoint using elements.Checkpoint
    # Use absolute path - elements' custom pathlib doesn't handle relative paths well
    ckpt_dir = (Path(checkpoint_path) / 'ckpt').resolve()
    if ckpt_dir.exists():
        cp = elements.Checkpoint(ckpt_dir)
        cp.agent = agent
        cp.load()
        print(f"Loaded checkpoint from {ckpt_dir}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_dir}, using fresh agent")

    return agent


def run_single_trial(
    agent,
    num_dots: int,
    use_barriers: bool,
    max_steps: int,
    trial_idx: int,
    seed: int,
    success_threshold: float,
    tracker: ConvergenceTracker,
) -> TrialResult:
    """
    Evaluate trained DreamerV3 agent on a single trial.

    Args:
        agent: Trained DreamerV3 agent
        num_dots: Number of quantum dots
        use_barriers: Whether barriers are controlled
        max_steps: Maximum steps per episode
        trial_idx: Trial index
        seed: Random seed for this trial
        success_threshold: Distance threshold for success
        tracker: ConvergenceTracker for recording distances

    Returns:
        TrialResult with trial metrics
    """
    from embodied.envs import from_gym
    import embodied

    # Create fresh evaluation environment
    gym_env = make_dreamer_env(num_dots, use_barriers, max_steps, seed=seed)
    env = from_gym.FromGym(gym_env)

    # Apply wrappers (same as training)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)

    # Also create benchmark env for ground truth calculations
    benchmark_env = create_benchmark_env(
        num_dots=num_dots,
        use_barriers=use_barriers,
        seed=seed
    )

    # Initialize policy state
    state = agent.init_policy(batch_size=1)

    # Reset environment via step with reset=True
    reset_action = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items() if k != 'reset'}
    reset_action['reset'] = np.array(True)
    obs = env.step(reset_action)
    obs = {k: v[None] for k, v in obs.items()}  # Add batch dimension

    reward_history = []
    voltage_history = []
    global_objective_history = []
    step_count = 0

    # Reset tracker for this trial
    tracker.reset()

    for step_idx in range(max_steps):
        # Get action from policy (returns: carry, acts, outs)
        state, act, _ = agent.policy(state, obs, mode='eval')

        # Remove batch dimension and step
        act_single = {k: v[0] for k, v in act.items()}
        act_single['reset'] = np.array(False)
        obs_new = env.step(act_single)

        # Extract reward and done from obs dict
        rew = obs_new['reward']
        done = obs_new['is_last']

        reward_history.append(float(rew))
        step_count += 1

        # Get current voltages from the wrapper's device state
        try:
            device_state = gym_env.device_state
            current_gates = device_state["current_gate_voltages"]
            current_barriers = device_state["current_barrier_voltages"]
            voltages = np.concatenate([current_gates, current_barriers])
            voltage_history.append(voltages.tolist())

            # Calculate global objective (sum of squared distances) for convergence plot
            plunger_dists, barrier_dists = get_distances(voltages, benchmark_env)
            global_obj = np.sum(plunger_dists**2) + np.sum(barrier_dists**2)
            global_objective_history.append(float(global_obj))

            # Record distances for convergence tracking
            # In RL mode, each step = 1 scan
            tracker.record(plunger_dists, barrier_dists, scan_number=step_count)
        except:
            voltage_history.append([])
            global_objective_history.append(float('inf'))

        if done:
            break

        # Add batch dimension for next iteration
        obs = {k: v[None] for k, v in obs_new.items()}

    env.close()

    # Calculate final metrics (before closing benchmark_env)
    if voltage_history and voltage_history[-1]:
        final_voltages = np.array(voltage_history[-1])
        plunger_dists, barrier_dists = get_distances(final_voltages, benchmark_env)
        success = check_success(final_voltages, benchmark_env, success_threshold)
        final_objective = np.mean(plunger_dists) + np.mean(barrier_dists)
    else:
        plunger_dists = np.zeros(num_dots)
        barrier_dists = np.zeros(num_dots - 1)
        success = False
        final_objective = float('inf')

    benchmark_env.close()

    return TrialResult(
        trial_idx=trial_idx,
        seed=seed,
        success=success,
        num_iterations=step_count,
        num_function_evals=step_count,
        num_scans=step_count,  # Each step = 1 CSD scan
        final_objective=final_objective,
        final_plunger_distances=plunger_dists.tolist(),
        final_barrier_distances=barrier_dists.tolist(),
        convergence_history=reward_history,
        global_objective_history=global_objective_history,
        voltage_history=voltage_history,
        # New distance tracking fields
        scan_numbers=tracker.scan_numbers.copy(),
        plunger_distance_history=tracker.plunger_distance_history.copy(),
        barrier_distance_history=tracker.barrier_distance_history.copy(),
        plunger_range=tracker.plunger_range,
        barrier_range=tracker.barrier_range,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 agent on quantum dots")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to training logdir with checkpoint")
    parser.add_argument("--num_dots", type=int, default=2,
                       help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=10,
                       help="Number of evaluation trials")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Success distance threshold")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for results JSON")
    parser.add_argument("--use_barriers", action="store_true", default=True,
                       help="Control barrier voltages")
    parser.add_argument("--gpu", type=int, default=5,
                       help="GPU device (already applied)")
    args = parser.parse_args()

    print(f"=== DreamerV3 Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dots: {args.num_dots}, Barriers: {args.use_barriers}")
    print(f"Trials: {args.num_trials}, Max steps: {args.max_steps}")
    print()

    # Create dummy env for agent loading
    class DummyEnv:
        num_dots = args.num_dots
        use_barriers = args.use_barriers
        max_steps = args.max_steps

    # Load trained agent
    print("Loading agent from checkpoint...")
    agent = load_agent_from_checkpoint(args.checkpoint, DummyEnv())

    # Re-enable transfers after agent loading (internal.setup() disables them)
    jax.config.update('jax_transfer_guard', 'allow')

    # Initialize result container
    result = BenchmarkResult(
        method="dreamerv3",
        mode="rl",
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        num_trials=args.num_trials,
        max_iterations=args.max_steps,
        success_threshold=args.threshold,
    )

    # Run trials
    base_rng = np.random.default_rng(args.seed)

    # Create tracker once (will be reused across trials)
    temp_env = create_benchmark_env(
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        seed=0,
    )
    tracker = ConvergenceTracker.from_env(temp_env)
    temp_env.close()

    for trial_idx in range(args.num_trials):
        trial_seed = int(base_rng.integers(0, 2**31))

        print(f"Trial {trial_idx + 1}/{args.num_trials} (seed={trial_seed})...", end=" ")

        trial_result = run_single_trial(
            agent=agent,
            num_dots=args.num_dots,
            use_barriers=args.use_barriers,
            max_steps=args.max_steps,
            trial_idx=trial_idx,
            seed=trial_seed,
            success_threshold=args.threshold,
            tracker=tracker,
        )

        result.trials.append(trial_result)
        status = "SUCCESS" if trial_result.success else "FAIL"
        print(f"{status} (steps={trial_result.num_scans}, obj={trial_result.final_objective:.4f})")

    # Compute statistics and save
    result.compute_stats()
    print()
    print_summary(result)

    output_path = save_results(result, args.output)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
