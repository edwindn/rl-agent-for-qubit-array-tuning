"""
Task 5 diagnostics — validates the eval adapter and rollout runner.

Runs a small number of trials (10 by default) with (a) the trained FACMAC
policy and (b) a uniform-random policy on the same seeds, then:

    1. summary.txt          — final-step + mid-episode distance stats, per-agent
                              action-distribution ranges, JSON-schema checks
    2. convergence.png      — mean + spread of plunger / barrier distance-to-GT
                              vs step, FACMAC vs random, on the same seed set
    3. action_histograms.png — per-agent action distributions (should be well
                               inside [-1, 1], not saturated at the clamp boundaries)
    4. facmac_sample.json    — sample output matching benchmark schema; used to
                               assert keys + dtypes match an existing ppo_*.json

Usage (after running task-4c training):

    uv run --extra facmac python benchmarks/MARL/facmac/tests/task_5_diagnostics.py \\
        --checkpoint-dir benchmarks/MARL/facmac/results/models/<run>/<step>/ \\
        --env-config benchmarks/MARL/facmac/configs/env_config_smoke.yaml \\
        --num-trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent.parent
_PROJECT_SRC = _BENCH_DIR.parent.parent / "src"
for p in (_BENCH_DIR, _PROJECT_SRC):
    sys.path.insert(0, str(p))

from eval_adapter import load_policy
from run_eval_trials import run_trials, _random_policy
from qadapt.environment.multi_agent_wrapper import MultiAgentEnvWrapper

OUT_DIR = _BENCH_DIR / "diagnostics" / "task_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_JSON = _BENCH_DIR.parent / "results" / "final_4dot" / "ppo_4dots.json"


def _latest_checkpoint(auto_discover: bool) -> Path | None:
    if not auto_discover:
        return None
    models_root = _BENCH_DIR / "results" / "models"
    grouped = [
        d for d in models_root.iterdir()
        if d.is_dir() and d.name.startswith("facmac_quantum_smoke_grouped__")
    ]
    if not grouped:
        return None
    latest_run = max(grouped, key=lambda p: p.stat().st_mtime)
    step_dirs = sorted(
        [d for d in latest_run.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )
    return step_dirs[-1] if step_dirs else None


def _collect_trials(policy_name: str, policy, env, num_trials: int, seed_base: int) -> list[dict]:
    env.reset(seed=seed_base)
    trials = run_trials(policy, env, num_trials, seed_base=seed_base)
    return trials


def _stats_over_trials(trials: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) along step axis across trials."""
    arrs = [np.asarray(t[key]) for t in trials]
    min_len = min(len(a) for a in arrs)
    arrs = np.stack([a[:min_len] for a in arrs])
    return arrs.mean(axis=0), arrs.std(axis=0)


def plot_convergence(facmac_trials, random_trials) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    lines = []
    for ax, key, label in zip(
        axes,
        ("plunger_distance_history", "barrier_distance_history"),
        ("plunger distance-to-GT", "barrier distance-to-GT"),
    ):
        for trials, color, name in (
            (facmac_trials, "C1", "FACMAC"),
            (random_trials, "C0", "random"),
        ):
            mean, std = _stats_over_trials(trials, key)
            steps = np.arange(1, len(mean) + 1)
            ax.plot(steps, mean, label=f"{name} (mean)", color=color, linewidth=1.5)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
            lines.append(
                f"    - {name} {key}: start={mean[0]:.2f}  end={mean[-1]:.2f}  delta={mean[-1]-mean[0]:+.2f}"
            )
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("Σ |current - GT|")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Task 5 convergence — FACMAC vs uniform-random policy (same seeds)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(0, f"  [written] {out}")
    return lines


def plot_action_histograms(env: MultiAgentEnvWrapper, policy, num_steps: int = 50) -> list[str]:
    """Runs one trial, logs every action per agent, plots per-agent histograms."""
    obs, _ = env.reset(seed=99)
    action_log: dict[str, list[float]] = {a: [] for a in env.all_agent_ids}
    for _ in range(num_steps):
        actions = policy(obs)
        for a, v in actions.items():
            action_log[a].append(float(v.ravel()[0]))
        obs, _, term, trunc, _ = env.step(actions)
        if term["__all__"] or trunc["__all__"]:
            break

    n = len(env.all_agent_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 3.5), sharey=True)
    clamped = []
    for ax, agent_id in zip(axes, env.all_agent_ids):
        vals = np.asarray(action_log[agent_id])
        ax.hist(vals, bins=30, color="C1", edgecolor="none")
        ax.axvline(-1.0, color="red", linewidth=0.5)
        ax.axvline(1.0, color="red", linewidth=0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{agent_id}\nmean={vals.mean():+.2f}", fontsize=8)
        ax.set_xlabel("action", fontsize=8)
        pct_clamped = float(np.mean((np.abs(vals) > 0.99))) * 100
        if pct_clamped > 25:
            clamped.append((agent_id, pct_clamped))

    fig.suptitle("Task 5 per-agent action histogram (one rollout, deterministic policy)", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "action_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    lines = [f"  [written] {out}"]
    if clamped:
        lines.append("  WARN — actions saturating (>25% near ±1) for agents:")
        for a, pct in clamped:
            lines.append(f"    {a}: {pct:.0f}% near boundary")
    else:
        lines.append("  OK — no agent saturates at the action clamp boundary")
    return lines


def schema_check(trials: list[dict]) -> list[str]:
    """Verify our JSON output matches the schema used by existing ppo_*.json."""
    lines = ["=== JSON schema check ==="]
    expected_keys = {
        "plunger_distance_history",
        "barrier_distance_history",
        "scan_numbers",
        "plunger_range",
        "barrier_range",
    }
    actual_keys = set(trials[0].keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    lines.append(f"  expected keys: {sorted(expected_keys)}")
    lines.append(f"  actual   keys: {sorted(actual_keys)}")
    lines.append(f"  missing: {sorted(missing) or 'none'}")
    lines.append(f"  extra:   {sorted(extra) or 'none'}")

    if REFERENCE_JSON.exists():
        ref = json.load(REFERENCE_JSON.open())
        ref_keys = set(ref["trials"][0].keys())
        diff = actual_keys.symmetric_difference(ref_keys)
        lines.append(f"  vs ppo_4dots.json: {'EXACT MATCH' if not diff else f'DIFFERS BY {sorted(diff)}'}")
    else:
        lines.append(f"  (ppo_4dots.json not found at {REFERENCE_JSON}; skipping cross-check)")

    # Types
    t0 = trials[0]
    lines.append(f"  plunger_distance_history: list of {type(t0['plunger_distance_history'][0]).__name__}, len={len(t0['plunger_distance_history'])}")
    lines.append(f"  plunger_range: {type(t0['plunger_range']).__name__} value={t0['plunger_range']:.2f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Defaults to latest facmac_quantum_smoke_grouped run's final step")
    parser.add_argument("--env-config", type=Path,
                        default=_BENCH_DIR / "configs" / "env_config_smoke.yaml")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt = args.checkpoint_dir or _latest_checkpoint(auto_discover=True)
    if ckpt is None:
        raise SystemExit("No checkpoint found. Pass --checkpoint-dir explicitly.")
    print(f"checkpoint: {ckpt}")
    print(f"env_config: {args.env_config}")

    env = MultiAgentEnvWrapper(
        training=False, return_voltage=False,
        env_config_path=str(args.env_config),
        capacitance_model_checkpoint=None,
    )
    template = {a: None for a in env.all_agent_ids}

    summary: list[str] = [f"checkpoint: {ckpt}", f"env_config: {args.env_config}", ""]

    facmac_policy = load_policy(
        checkpoint_dir=ckpt, env_config_path=args.env_config, device=args.device,
    )
    print(f"[1/4] running {args.num_trials} FACMAC trials")
    facmac_trials = _collect_trials("facmac", facmac_policy, env, args.num_trials, args.seed_base)

    rng = np.random.default_rng(args.seed_base)
    random_policy = _random_policy(template, rng)
    print(f"[2/4] running {args.num_trials} random-policy trials")
    env_r = MultiAgentEnvWrapper(
        training=False, return_voltage=False,
        env_config_path=str(args.env_config),
        capacitance_model_checkpoint=None,
    )
    random_trials = _collect_trials("random", random_policy, env_r, args.num_trials, args.seed_base)

    print("[3/4] convergence plot + action histograms")
    summary.append("=== convergence (mean over trials, end-of-episode distance) ===")
    summary.extend(plot_convergence(facmac_trials, random_trials))
    summary.append("")
    summary.append("=== action histograms (one rollout, deterministic policy) ===")
    summary.extend(plot_action_histograms(env, facmac_policy))
    summary.append("")

    print("[4/4] schema check + sample JSON")
    summary.extend(schema_check(facmac_trials))
    sample_path = OUT_DIR / "facmac_sample.json"
    sample_path.write_text(json.dumps({
        "method": "facmac", "num_dots": env.num_gates, "use_barriers": env.use_barriers,
        "trials": facmac_trials,
    }))
    summary.append(f"  [written] {sample_path}")

    (OUT_DIR / "summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\nDone. Artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
