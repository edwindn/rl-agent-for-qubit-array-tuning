"""Eval the trained grouped-policy SuperSims checkpoint at variable N_QUBITS.

For appendix scaling figures (All-XY violins + multi-N convergence). Runs
greedy + random rollouts at the N selected via the SUPERSIMS_PARAM_CFG env
var (see SuperSims/parameter_generation.py), saving per-step P(|1>) staircases
and per-step per-qubit rewards as a single .npz per N.

The trained `grouped` policy is N-agnostic — both freq_policy and env_policy
take per-qubit obs and emit per-qubit actions, so a 4-qubit checkpoint runs
zero-shot at any N.

Usage:
  CUDA_VISIBLE_DEVICES=0 SUPERSIMS_PARAM_CFG=parameter_config_N6.json \\
    uv run python scripts/eval_multi_N.py --n-seeds 100 \\
        --checkpoint checkpoints_supersims_grouped/iteration_28 \\
        --out plots_supersims_diagnostic/staircase_scan_N6.npz

For N=4 set SUPERSIMS_PARAM_CFG=parameter_config.json (the default).
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from qadapt_for_supersim.eval import (  # noqa: E402
    load_modules_from_checkpoint,
    greedy_action,
)
from qadapt_for_supersim.env import SuperSimsEnv  # noqa: E402


def _build_env_config_for_param_cfg(param_cfg_name: str) -> str:
    """Write a temporary env config matching SUPERSIMS_PARAM_CFG and return its path.

    The canonical env config has parameter_config_filename: parameter_config.json
    hardcoded, which fails the assertion that env-sampling N matches the module-level
    N when we override SUPERSIMS_PARAM_CFG. Mirror the canonical config but swap
    parameter_config_filename to match.
    """
    canonical = (_REPO / "src" / "qadapt" / "environment"
                 / "supersims_env_config.yaml")
    cfg = yaml.safe_load(canonical.read_text())
    cfg["parameter_config_filename"] = param_cfg_name
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="supersims_env_cfg_",
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _staircase_to_p1(staircase_norm: np.ndarray) -> np.ndarray:
    """Env stores P1 normalised to [-1, 1] (= 2·P1 - 1). Reverse for plotting."""
    return ((staircase_norm + 1.0) / 2.0).astype(np.float32)


def run_episode(env, policy_split, modules, seed, mode, rng):
    """Returns (rewards, staircase_p1).

    Shapes:
      rewards:       (n_steps+1, n_qubits)
      staircase_p1:  (n_steps+1, n_qubits, n_allxy) — P(|1>) in [0, 1]
    """
    obs, info = env.reset(seed=seed)
    rewards = [info["per_qubit_rewards"].copy()]
    staircases = [_staircase_to_p1(obs["staircase"])]
    for _ in range(env.max_steps):
        if mode == "greedy":
            action = greedy_action(policy_split, modules,
                                   obs["staircase"], obs["params"])
        else:
            n_q = obs["params"].shape[0]
            action = rng.uniform(-1.0, 1.0, size=(n_q, 5)).astype(np.float32)
        obs, _, terminated, _, info = env.step(action)
        rewards.append(info["per_qubit_rewards"].copy())
        staircases.append(_staircase_to_p1(obs["staircase"]))
        if terminated:
            break
    return np.asarray(rewards), np.asarray(staircases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str,
                    help="Path to checkpoints_supersims_grouped/iteration_N")
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--out", type=str, required=True,
                    help="Output .npz path")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Start seeds from this offset (for split-across-GPU runs)")
    ap.add_argument("--time-only", action="store_true",
                    help="Run a single 1-step probe and report timing; no full rollout, no save")
    args = ap.parse_args()

    cfg_name = os.environ.get("SUPERSIMS_PARAM_CFG", "parameter_config.json")
    print(f"SUPERSIMS_PARAM_CFG = {cfg_name}")

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {ckpt}")
    print(f"Seeds:      {args.n_seeds}  (offset {args.seed_offset})")
    print(f"Output:     {out}")

    print("\nLoading policy...")
    policy_split, modules = load_modules_from_checkpoint(ckpt)
    print(f"  policy_split={policy_split}")

    t_env_start = time.perf_counter()
    print("\nBuilding env (JIT warmup, can be 30-60s at large N)...")
    env_cfg_path = _build_env_config_for_param_cfg(cfg_name)
    env = SuperSimsEnv(config_path=env_cfg_path)
    print(f"  built in {time.perf_counter() - t_env_start:.1f}s  "
          f"n_qubits={env.n_qubits}  n_allxy={env.n_allxy}  max_steps={env.max_steps}")

    if args.time_only:
        print("\n[time-only] resetting + stepping a few times to measure...")
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        obs, info = env.reset(seed=0)
        print(f"  reset: {time.perf_counter() - t0:.2f}s")
        # First step pays JIT compile.
        t0 = time.perf_counter()
        action = rng.uniform(-1, 1, (env.n_qubits, 5)).astype(np.float32)
        obs, _, _, _, _ = env.step(action)
        t_step1 = time.perf_counter() - t0
        print(f"  step 1 (compile): {t_step1:.2f}s")
        # Subsequent steps are the steady-state cost.
        steady = []
        for k in range(3):
            t0 = time.perf_counter()
            action = rng.uniform(-1, 1, (env.n_qubits, 5)).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            steady.append(time.perf_counter() - t0)
        print(f"  steady steps: {[f'{s:.2f}s' for s in steady]}  mean {np.mean(steady):.2f}s")
        print(f"\n[time-only] cost per full episode (20 steps): "
              f"~{t_step1 + 20 * np.mean(steady):.1f}s "
              f"(first ep) / {21 * np.mean(steady):.1f}s (subsequent)")
        return

    n_steps_plus_1 = env.max_steps + 1
    n_q = env.n_qubits
    n_allxy = env.n_allxy

    rewards_g = np.empty((args.n_seeds, n_steps_plus_1, n_q), dtype=np.float32)
    rewards_r = np.empty((args.n_seeds, n_steps_plus_1, n_q), dtype=np.float32)
    sc_g = np.empty((args.n_seeds, n_steps_plus_1, n_q, n_allxy), dtype=np.float32)
    sc_r = np.empty((args.n_seeds, n_steps_plus_1, n_q, n_allxy), dtype=np.float32)

    t_total_start = time.perf_counter()
    for i in range(args.n_seeds):
        s = args.seed_offset + i
        t0 = time.perf_counter()
        rewards, staircase = run_episode(env, policy_split, modules,
                                         seed=s, mode="greedy", rng=None)
        t_g = time.perf_counter() - t0
        # Pad if early termination (shouldn't happen for max_steps=20 setting,
        # but defensive).
        if rewards.shape[0] < n_steps_plus_1:
            pad = n_steps_plus_1 - rewards.shape[0]
            rewards = np.concatenate([rewards,
                                      np.tile(rewards[-1:], (pad, 1))], axis=0)
            staircase = np.concatenate([staircase,
                                        np.tile(staircase[-1:], (pad, 1, 1))], axis=0)
        rewards_g[i] = rewards
        sc_g[i] = staircase

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed=s)
        rewards, staircase = run_episode(env, policy_split, modules,
                                         seed=s, mode="random", rng=rng)
        t_r = time.perf_counter() - t0
        if rewards.shape[0] < n_steps_plus_1:
            pad = n_steps_plus_1 - rewards.shape[0]
            rewards = np.concatenate([rewards,
                                      np.tile(rewards[-1:], (pad, 1))], axis=0)
            staircase = np.concatenate([staircase,
                                        np.tile(staircase[-1:], (pad, 1, 1))], axis=0)
        rewards_r[i] = rewards
        sc_r[i] = staircase

        # Progress every 5 seeds (or at end).
        if (i + 1) % 5 == 0 or i == args.n_seeds - 1:
            elapsed = time.perf_counter() - t_total_start
            mean_per_seed = elapsed / (i + 1)
            remaining = mean_per_seed * (args.n_seeds - i - 1)
            print(f"  seed {i + 1}/{args.n_seeds}  greedy={t_g:.1f}s "
                  f"random={t_r:.1f}s  elapsed {elapsed/60:.1f}m  "
                  f"eta {remaining/60:.1f}m  "
                  f"g_final={rewards_g[i, -1].mean():.3f} "
                  f"r_final={rewards_r[i, -1].mean():.3f}")

    np.savez(
        out,
        reward_greedy=rewards_g,
        reward_random=rewards_r,
        staircase_greedy=sc_g,
        staircase_random=sc_r,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
        n_qubits=n_q,
        n_allxy=n_allxy,
        n_steps=env.max_steps,
        checkpoint=str(ckpt),
        param_config=cfg_name,
    )
    print(f"\nWrote {out}")
    print(f"  reward_greedy {rewards_g.shape}  reward_random {rewards_r.shape}")
    print(f"  staircase_greedy {sc_g.shape}  staircase_random {sc_r.shape}")
    print(f"  greedy final mean: {rewards_g[:, -1].mean():.3f}  "
          f"random final mean: {rewards_r[:, -1].mean():.3f}")


if __name__ == "__main__":
    main()
