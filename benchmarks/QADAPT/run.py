"""
QADAPT benchmark for quantum dot array tuning using trained RL policies.

Uses the same MultiAgentEnvWrapper as training to ensure correct observation/action handling.

Usage:
    python run.py --checkpoint ../../artifacts/rl_checkpoint_best:v3482 --num_dots 4 --num_trials 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add paths
benchmarks_dir = Path(__file__).parent.parent
project_root = benchmarks_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(benchmarks_dir))
sys.path.insert(0, str(src_dir))

from env_init import create_benchmark_env
from objective import get_distances, check_success
from utils import BenchmarkResult, TrialResult, save_results, print_summary
from convergence_tracker import ConvergenceTracker

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
from swarm.environment.env import QuantumDeviceEnv
from policy_loader import load_policies, get_deterministic_action


class BenchmarkMultiAgentWrapper(MultiAgentEnvWrapper):
    """MultiAgentEnvWrapper that accepts num_dots parameter."""

    def __init__(self, num_dots: int, training: bool = True, return_voltage: bool = True):
        # Don't call super().__init__ - we'll set up manually
        from gymnasium import spaces
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        MultiAgentEnv.__init__(self)

        self.return_voltage = return_voltage
        self.distance_data_dir = None
        self.distance_history = None
        self.gif_config = None

        # Create base env with specified num_dots
        self.base_env = QuantumDeviceEnv(training=training, num_dots=num_dots, use_barriers=True)

        self.num_gates = self.base_env.num_dots
        self.use_barriers = self.base_env.use_barriers
        self.num_barriers = self.base_env.num_dots - 1
        self.num_image_channels = self.base_env.num_dots - 1

        # Set up agent IDs and channel maps (copied from parent)
        self._setup_agents()

    def _setup_agents(self):
        """Set up agent IDs, channel maps, and spaces."""
        from gymnasium import spaces
        import numpy as np

        # Create agent IDs
        self.gate_agent_ids = [f"plunger_{i}" for i in range(self.num_gates)]
        self.barrier_agent_ids = [f"barrier_{i}" for i in range(self.num_barriers)]
        self.all_agent_ids = self.gate_agent_ids + self.barrier_agent_ids

        # Create channel map
        self.agent_channel_map = {}
        num_csds = self.num_image_channels

        for i, agent_id in enumerate(self.gate_agent_ids):
            if i == 0:
                self.agent_channel_map[agent_id] = [0, 0]
            elif i == self.num_gates - 1:
                self.agent_channel_map[agent_id] = [num_csds - 1, num_csds - 1]
            else:
                self.agent_channel_map[agent_id] = [i - 1, i]

        for i, agent_id in enumerate(self.barrier_agent_ids):
            self.agent_channel_map[agent_id] = [i]

        # Create observation and action spaces
        resolution = self.base_env.resolution

        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self.gate_agent_ids:
            if self.return_voltage:
                self.observation_spaces[agent_id] = spaces.Dict({
                    'image': spaces.Box(low=0.0, high=1.0, shape=(resolution, resolution, 2), dtype=np.float32),
                    'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                })
            else:
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0.0, high=1.0, shape=(resolution, resolution, 2), dtype=np.float32
                )
            self.action_spaces[agent_id] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        for agent_id in self.barrier_agent_ids:
            if self.return_voltage:
                self.observation_spaces[agent_id] = spaces.Dict({
                    'image': spaces.Box(low=0.0, high=1.0, shape=(resolution, resolution, 1), dtype=np.float32),
                    'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                })
            else:
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0.0, high=1.0, shape=(resolution, resolution, 1), dtype=np.float32
                )
            self.action_spaces[agent_id] = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_spaces = spaces.Dict(**self.observation_spaces)
        self.action_spaces = spaces.Dict(**self.action_spaces)

        self._agent_ids = set(self.all_agent_ids)
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces
        self.agents = self._agent_ids.copy()
        self.possible_agents = self._agent_ids.copy()


def run_single_trial(
    num_dots: int,
    policies: dict,
    trial_idx: int,
    seed: int,
    max_steps: int,
    success_threshold: float,
    tracker: ConvergenceTracker,
    device: str = "cuda",
) -> TrialResult:
    """
    Run a single RL episode using MultiAgentEnvWrapper.
    """
    # Create multi-agent wrapped environment with correct num_dots
    env = BenchmarkMultiAgentWrapper(num_dots=num_dots, training=True, return_voltage=True)

    num_pairs = num_dots - 1

    # Reset environment
    obs_dict, info = env.reset(seed=seed)

    # Reset tracker
    tracker.reset()

    # Get initial voltages and distances
    plunger_v = env.base_env.device_state["current_gate_voltages"]
    barrier_v = env.base_env.device_state["current_barrier_voltages"]
    current_voltages = np.concatenate([plunger_v, barrier_v])

    # Record initial state
    plunger_dists, barrier_dists = get_distances(current_voltages, env.base_env)
    tracker.record(plunger_dists, barrier_dists, scan_number=0)

    # Tracking
    global_history = []
    voltage_history = [current_voltages.tolist()]

    success = False
    final_step = 0

    for step in range(max_steps):
        # Get actions from policies for each agent
        actions = {}

        for agent_id, agent_obs in obs_dict.items():
            # Determine which policy to use
            policy_name = "plunger_policy" if "plunger" in agent_id else "barrier_policy"

            # Convert observation to torch tensor with batch dimension
            torch_obs = {
                "image": torch.tensor(agent_obs["image"], dtype=torch.float32, device=device).unsqueeze(0),
                "voltage": torch.tensor(agent_obs["voltage"], dtype=torch.float32, device=device).unsqueeze(0),
            }

            # Get deterministic action
            action = get_deterministic_action(policies[policy_name], torch_obs)
            actions[agent_id] = action.cpu().numpy().flatten()

        # Step environment with combined actions
        obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)

        # Get current voltages
        plunger_v = env.base_env.device_state["current_gate_voltages"]
        barrier_v = env.base_env.device_state["current_barrier_voltages"]
        current_voltages = np.concatenate([plunger_v, barrier_v])

        # Track distances
        plunger_dists, barrier_dists = get_distances(current_voltages, env.base_env)
        current_scan = num_pairs * (step + 1)
        tracker.record(plunger_dists, barrier_dists, current_scan)

        # Track objective
        obj_val = float(np.sum(plunger_dists**2) + np.sum(barrier_dists**2))
        global_history.append(obj_val)
        voltage_history.append(current_voltages.tolist())

        final_step = step + 1

        # Check termination
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
        if done:
            break

        # Check success
        if check_success(current_voltages, env.base_env, success_threshold):
            success = True
            break

    # Final metrics
    plunger_dists, barrier_dists = get_distances(current_voltages, env.base_env)
    final_obj = float(np.sum(plunger_dists**2) + np.sum(barrier_dists**2))
    num_scans = num_pairs * final_step

    return TrialResult(
        trial_idx=trial_idx,
        seed=seed,
        success=success,
        num_iterations=final_step,
        num_function_evals=final_step,
        num_scans=num_scans,
        final_objective=final_obj,
        final_plunger_distances=plunger_dists.tolist(),
        final_barrier_distances=barrier_dists.tolist(),
        convergence_history=global_history,
        global_objective_history=global_history,
        voltage_history=voltage_history,
        scan_numbers=tracker.scan_numbers.copy(),
        plunger_distance_history=tracker.plunger_distance_history.copy(),
        barrier_distance_history=tracker.barrier_distance_history.copy(),
        plunger_range=tracker.plunger_range,
        barrier_range=tracker.barrier_range,
    )


def main():
    parser = argparse.ArgumentParser(
        description="QADAPT benchmark for quantum dot tuning using trained RL policies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib checkpoint directory",
    )
    parser.add_argument("--num_dots", type=int, default=4, help="Number of quantum dots")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Success threshold in volts")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for policy inference",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Running QADAPT benchmark (trained RL policies)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Dots: {args.num_dots}, Barriers: True")
    print(f"  Trials: {args.num_trials}, Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}, Success threshold: {args.threshold}V")
    print()

    # Load policies
    print("Loading policies...")
    policies = load_policies(args.checkpoint, device=device)
    print(f"  Loaded: {list(policies.keys())}")
    print()

    # Initialize benchmark result
    result = BenchmarkResult(
        method="QADAPT",
        mode="rl",
        num_dots=args.num_dots,
        use_barriers=True,
        num_trials=args.num_trials,
        max_iterations=args.max_steps,
        success_threshold=args.threshold,
    )

    base_rng = np.random.default_rng(args.seed)

    # Create temp env for tracker initialization
    temp_env = BenchmarkMultiAgentWrapper(num_dots=args.num_dots, training=True, return_voltage=True)
    tracker = ConvergenceTracker.from_env(temp_env.base_env)

    start_time = time.time()
    for trial_idx in range(args.num_trials):
        trial_seed = int(base_rng.integers(0, 2**31))

        trial_result = run_single_trial(
            num_dots=args.num_dots,
            policies=policies,
            trial_idx=trial_idx,
            seed=trial_seed,
            max_steps=args.max_steps,
            success_threshold=args.threshold,
            tracker=tracker,
            device=device,
        )

        result.trials.append(trial_result)

        status = "SUCCESS" if trial_result.success else "FAIL"
        print(
            f"Trial {trial_idx + 1}/{args.num_trials}: {status} "
            f"(obj={trial_result.final_objective:.4f}, steps={trial_result.num_iterations}, "
            f"scans={trial_result.num_scans})"
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
