"""
Rollout runner: loads a FACMAC policy via eval_adapter, runs N trials through
MultiAgentEnvWrapper, records per-step distance-to-ground-truth, and writes
JSON matching the schema used by benchmarks/results/final_Ndot/*.json.

Usage:
    uv run --extra facmac python benchmarks/MARL/facmac/run_eval_trials.py \\
        --checkpoint-dir benchmarks/MARL/facmac/results/models/<run>/1500 \\
        --env-config benchmarks/MARL/facmac/configs/env_config_smoke.yaml \\
        --num-trials 100 \\
        --output benchmarks/MARL/facmac/results/facmac_4dots.json

Output JSON schema (matches ppo_4dots.json etc.):
    {
        "method": "facmac",
        "num_dots": N,
        "use_barriers": true,
        "trials": [
            {
                "plunger_distance_history": [...],   # sum over plungers per step
                "barrier_distance_history": [...],   # sum over barriers per step
                "scan_numbers":             [1,2,...],
                "plunger_range":            float,
                "barrier_range":            float
            }, ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_SRC = _THIS_DIR.parent.parent / "src"
for _p in (_THIS_DIR, _PROJECT_SRC):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from eval_adapter import load_policy
from qadapt.environment.multi_agent_wrapper import MultiAgentEnvWrapper


def _random_policy(action_dict_template: dict[str, np.ndarray], rng: np.random.Generator):
    """Used for the baseline sanity comparison."""
    def policy(obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            agent_id: rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            for agent_id in action_dict_template
        }
    return policy


def _compute_distances(base_env) -> tuple[float, float]:
    """Sum of |current - GT| across plungers / barriers, from device_state."""
    ds = base_env.device_state
    plunger_dist = float(np.sum(np.abs(ds["current_gate_voltages"] - ds["gate_ground_truth"])))
    barrier_dist = float(np.sum(np.abs(ds["current_barrier_voltages"] - ds["barrier_ground_truth"])))
    return plunger_dist, barrier_dist


def _per_agent_distances(base_env) -> tuple[np.ndarray, np.ndarray]:
    """Per-plunger and per-barrier |current - GT| arrays (1D over agents)."""
    ds = base_env.device_state
    plunger_per = np.abs(ds["current_gate_voltages"] - ds["gate_ground_truth"]).ravel()
    barrier_per = np.abs(ds["current_barrier_voltages"] - ds["barrier_ground_truth"]).ravel()
    return plunger_per, barrier_per


def _extract_ranges(base_env) -> tuple[float, float]:
    # plunger_max/min are per-plunger arrays sharing the same scalar range,
    # so mean gives that scalar back.
    plunger_range = float(np.mean(base_env.plunger_max - base_env.plunger_min))
    barrier_range = float(np.mean(base_env.barrier_max - base_env.barrier_min))
    return plunger_range, barrier_range


def run_trials(
    policy,
    env: MultiAgentEnvWrapper,
    num_trials: int,
    seed_base: int = 0,
    npy_output_dir: Path | None = None,
) -> list[dict]:
    """Run N trials, return JSON-shaped trial summaries.

    If npy_output_dir is set, also writes per-agent per-trial .npy files in the
    layout that ablation_metrics.py / compute_table.py consumes:
        <npy_output_dir>/plunger_<i>/<trialID>_<random>.npy
        <npy_output_dir>/barrier_<i>/<trialID>_<random>.npy
    Each .npy is a 1D float array of |current - GT| per step for that agent
    in that trial.
    """
    trials = []
    max_steps = env.base_env.max_steps
    rng = np.random.default_rng(seed_base)

    for trial_idx in range(num_trials):
        obs, _ = env.reset(seed=seed_base + trial_idx)
        plunger_range, barrier_range = _extract_ranges(env.base_env)

        plunger_history: list[float] = []
        barrier_history: list[float] = []
        scan_numbers: list[int] = []
        # Per-agent per-step distances: shape (max_steps, num_plungers) etc.
        plunger_per_step: list[np.ndarray] = []
        barrier_per_step: list[np.ndarray] = []

        for step in range(max_steps):
            action = policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            pd, bd = _compute_distances(env.base_env)
            plunger_history.append(pd)
            barrier_history.append(bd)
            scan_numbers.append(step + 1)
            if npy_output_dir is not None:
                pp, bp = _per_agent_distances(env.base_env)
                plunger_per_step.append(pp)
                barrier_per_step.append(bp)
            if terminated["__all__"] or truncated["__all__"]:
                break

        trials.append({
            "plunger_distance_history": plunger_history,
            "barrier_distance_history": barrier_history,
            "scan_numbers":             scan_numbers,
            "plunger_range":            plunger_range,
            "barrier_range":            barrier_range,
        })

        if npy_output_dir is not None and plunger_per_step:
            # plunger_per_step[t] shape (num_plungers,) — stack to (T, num_plungers)
            plunger_arr = np.stack(plunger_per_step, axis=0)
            barrier_arr = np.stack(barrier_per_step, axis=0) if barrier_per_step else None
            trial_id = f"{trial_idx + 1:04d}"
            rand_tag = f"{rng.integers(0, 10**6):06d}"
            for i in range(plunger_arr.shape[1]):
                agent_dir = npy_output_dir / f"plunger_{i}"
                agent_dir.mkdir(parents=True, exist_ok=True)
                np.save(agent_dir / f"{trial_id}_{rand_tag}.npy", plunger_arr[:, i])
            if barrier_arr is not None:
                for i in range(barrier_arr.shape[1]):
                    agent_dir = npy_output_dir / f"barrier_{i}"
                    agent_dir.mkdir(parents=True, exist_ok=True)
                    np.save(agent_dir / f"{trial_id}_{rand_tag}.npy", barrier_arr[:, i])

    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="results/models/<run>/<step>/ containing agent_{plunger,barrier}.th")
    parser.add_argument("--env-config", type=Path, required=True,
                        help="env_config yaml defining num_dots, resolution, max_steps")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output (legacy schema). Optional if --npy-output-dir is set.")
    parser.add_argument("--npy-output-dir", type=Path, default=None,
                        help="If set, also write per-agent .npy distance trajectories in the "
                             "<dir>/plunger_<i>/<NNNN>_<random>.npy layout that "
                             "ablation_metrics.py consumes.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-dots", type=int, default=None,
                        help="override for cross-size eval; defaults to env_config's num_dots")
    parser.add_argument("--random-baseline", action="store_true",
                        help="skip loading FACMAC and run a uniform-random policy instead")
    args = parser.parse_args()

    env = MultiAgentEnvWrapper(
        training=False,
        return_voltage=False,
        env_config_path=str(args.env_config),
        capacitance_model_checkpoint=None,
    )
    num_dots = args.num_dots if args.num_dots is not None else env.base_env.num_dots

    if args.random_baseline:
        method_name = "random"
        rng = np.random.default_rng(args.seed_base)
        template = {agent_id: None for agent_id in env.all_agent_ids}
        policy = _random_policy(template, rng)
    else:
        method_name = "facmac"
        policy = load_policy(
            checkpoint_dir=args.checkpoint_dir,
            env_config_path=args.env_config,
            num_dots=num_dots,
            device=args.device,
        )

    if args.output is None and args.npy_output_dir is None:
        parser.error("Must provide at least one of --output or --npy-output-dir.")
    if args.npy_output_dir is not None:
        args.npy_output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    trials = run_trials(
        policy, env, args.num_trials,
        seed_base=args.seed_base,
        npy_output_dir=args.npy_output_dir,
    )
    elapsed = time.time() - t0

    if args.output is not None:
        output = {
            "method": method_name,
            "num_dots": num_dots,
            "use_barriers": env.use_barriers,
            "trials": trials,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output))

    final_p = [t["plunger_distance_history"][-1] for t in trials]
    final_b = [t["barrier_distance_history"][-1] for t in trials]
    print(
        f"Ran {args.num_trials} trials ({method_name}, num_dots={num_dots}) in {elapsed:.1f}s\n"
        f"  final plunger distance: mean={np.mean(final_p):.2f}  median={np.median(final_p):.2f}\n"
        f"  final barrier distance: mean={np.mean(final_b):.2f}  median={np.median(final_b):.2f}"
    )
    if args.output is not None:
        print(f"  wrote {args.output}")
    if args.npy_output_dir is not None:
        print(f"  wrote .npy layout under {args.npy_output_dir}")


if __name__ == "__main__":
    main()
