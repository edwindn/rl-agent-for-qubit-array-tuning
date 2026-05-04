#!/usr/bin/env python3
"""
Pull a wandb checkpoint + run the ablation eval for one algo.

Reads ablation_config.yaml, looks up `--algo`, downloads the latest
'rl_checkpoint_best' artifact for that run, applies any env_overrides, and
invokes src/eval_runs/main.py for `--num-episodes` rollouts.

Usage:
  uv run python run_ablation.py --algo qadapt --gpu 7
  CUDA_VISIBLE_DEVICES=7 uv run python run_ablation.py --algo w_o_kalman

Outputs land under <output_root>/<algo_name>/, matching the run dir layout
that ablation_metrics.py consumes (one .npy per agent per episode).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import wandb
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]   # qaduub-mappo
EVAL_MAIN = REPO_ROOT / "src" / "eval_runs" / "main.py"
CONFIG_PATH = Path(__file__).resolve().parent / "ablation_config.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _resolve_run(project: str, run_number: int):
    """Find the wandb Run whose display_name ends with -<run_number>."""
    api = wandb.Api()
    runs = list(api.runs(project, per_page=500))
    suffix = f"-{run_number}"
    hits = [r for r in runs if (r.display_name or "").endswith(suffix)]
    if not hits:
        raise RuntimeError(
            f"No run in {project} with display_name ending '{suffix}'. "
            f"Searched {len(runs)} runs."
        )
    if len(hits) > 1:
        names = [r.display_name for r in hits]
        raise RuntimeError(f"Multiple runs match suffix '{suffix}': {names}")
    return hits[0]


def _best_artifact_by_reward(run, reward_key: str = "episode_return_mean"):
    """Pick the 'rl_checkpoint_best' artifact whose logging iteration had the
    highest `reward_key` in the run's history.

    The 'best' artifacts share a global versioned namespace
    (rl_checkpoint_best:v{NNN} interleaves checkpoints from many runs), so
    'highest version' is not a reliable proxy for 'best for this run'. Instead
    we match each artifact's created_at timestamp to the iteration whose
    history `_timestamp` is closest at-or-before, then pick the artifact whose
    iteration has the highest reward.
    """
    arts = [a for a in run.logged_artifacts() if a.name.startswith("rl_checkpoint_best")]
    if not arts:
        raise RuntimeError(f"No rl_checkpoint_best artifact for run {run.display_name}")

    # Pull the full per-iter history once.
    rows = []
    for r in run.scan_history(keys=["iteration", reward_key, "_timestamp"]):
        ts = r.get("_timestamp")
        rew = r.get(reward_key)
        it = r.get("iteration")
        if ts is None or rew is None:
            continue
        rows.append((float(ts), float(rew), it))
    if not rows:
        raise RuntimeError(
            f"Run {run.display_name} has no history rows with both "
            f"'_timestamp' and '{reward_key}'; can't pick best artifact."
        )
    rows.sort()  # by timestamp

    def _iso_to_ts(s: str) -> float:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    best_art = None
    best_reward = float("-inf")
    best_iter = None
    for a in arts:
        a_ts = _iso_to_ts(a.created_at)
        # Last history row at-or-before artifact creation
        candidates = [(t, rew, it) for (t, rew, it) in rows if t <= a_ts]
        if not candidates:
            continue
        _, rew, it = candidates[-1]
        if rew > best_reward:
            best_reward = rew
            best_art = a
            best_iter = it

    if best_art is None:
        raise RuntimeError(
            f"Could not match any artifact's creation time to a history row "
            f"for run {run.display_name}."
        )
    print(f"  picked {best_art.name} at iter {best_iter} with {reward_key}={best_reward:.4f}")
    return best_art


def _apply_overrides(env_config: dict, overrides: dict) -> dict:
    """Apply nested overrides via dotted keys: {'a.b.c': v} -> env_config['a']['b']['c']=v."""
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        node = env_config
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return env_config


def _prepare_checkpoint(
    algo_name: str,
    algo_cfg: dict,
    weights_root: Path,
    num_episodes: int,
) -> Path:
    """Pull weights + rewrite training_config + env_config. Return the checkpoint dir."""
    project = algo_cfg["wandb_project"]
    run_number = algo_cfg["wandb_run_number"]

    print(f"[{algo_name}] resolving wandb run -{run_number} in {project}...")
    run = _resolve_run(project, run_number)
    print(f"[{algo_name}]   -> {run.display_name} ({run.id}) state={run.state}")

    art = _best_artifact_by_reward(run)
    print(f"[{algo_name}]   selected best artifact: {art.name}")

    ckpt_dir = weights_root / algo_name
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True)

    print(f"[{algo_name}] downloading to {ckpt_dir}...")
    art.download(root=str(ckpt_dir))

    # Override training_config.yaml's defaults.num_iterations — main.py uses it
    # as the episode count for eval rollouts.
    train_cfg_path = ckpt_dir / "training_config.yaml"
    if not train_cfg_path.exists():
        raise RuntimeError(f"{train_cfg_path} not in artifact")
    with train_cfg_path.open() as fh:
        train_cfg = yaml.safe_load(fh)
    train_cfg.setdefault("defaults", {})["num_iterations"] = int(num_episodes)
    with train_cfg_path.open("w") as fh:
        yaml.safe_dump(train_cfg, fh, default_flow_style=False)
    print(f"[{algo_name}] set defaults.num_iterations = {num_episodes}")

    # Apply env overrides if any
    overrides = algo_cfg.get("env_overrides") or {}
    if overrides:
        env_config_path = ckpt_dir / "env_config.yaml"
        if not env_config_path.exists():
            raise RuntimeError(f"{env_config_path} not in artifact; cannot apply overrides")
        with env_config_path.open() as fh:
            env_config = yaml.safe_load(fh)
        _apply_overrides(env_config, overrides)
        with env_config_path.open("w") as fh:
            yaml.safe_dump(env_config, fh, default_flow_style=False)
        print(f"[{algo_name}] applied env overrides: {overrides}")

    return ckpt_dir


def _run_eval(algo_name: str, ckpt_dir: Path, num_episodes: int, gpu: int | None):
    """Invoke main.py for inference. Output dir is created automatically by main.py."""
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # main.py reads num_episodes from training_config (or its own default); pass
    # via env var as a generic override hook — our main.py respects N_EVAL_EPISODES
    # if set (added 2026-05-04 for ablation reproducibility).
    env["ABLATION_NUM_EPISODES"] = str(num_episodes)
    env["ABLATION_ALGO_NAME"] = algo_name

    cmd = [
        "uv", "run", "python", str(EVAL_MAIN),
        "--load-checkpoint", str(ckpt_dir),
        "--disable-wandb",
    ]
    print(f"[{algo_name}] running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env, cwd=str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, help="Key into ablation_config.yaml")
    ap.add_argument("--gpu", type=int, default=None, help="Set CUDA_VISIBLE_DEVICES")
    ap.add_argument("--num-episodes", type=int, default=None,
                    help="Override num_episodes (defaults to ablation_config.yaml's defaults)")
    ap.add_argument("--weights-root", default="/tmp/ablation_weights",
                    help="Where to download checkpoints to")
    ap.add_argument("--skip-download", action="store_true",
                    help="Reuse existing checkpoint dir instead of re-downloading")
    args = ap.parse_args()

    cfg = _load_config()
    if args.algo not in cfg["algos"]:
        print(f"Unknown algo '{args.algo}'. Available: {list(cfg['algos'])}", file=sys.stderr)
        sys.exit(1)

    algo_cfg = cfg["algos"][args.algo]
    if algo_cfg.get("pipeline", "rlmodel") != "rlmodel":
        print(f"[{args.algo}] pipeline={algo_cfg['pipeline']} — only 'rlmodel' is wired up. "
              f"Single-agent / facmac TODO.", file=sys.stderr)
        sys.exit(2)

    num_episodes = args.num_episodes or cfg["defaults"]["num_episodes"]
    weights_root = Path(args.weights_root).resolve()
    weights_root.mkdir(parents=True, exist_ok=True)

    if args.skip_download and (weights_root / args.algo).exists():
        ckpt_dir = weights_root / args.algo
        print(f"[{args.algo}] reusing existing {ckpt_dir}")
    else:
        ckpt_dir = _prepare_checkpoint(args.algo, algo_cfg, weights_root, num_episodes)

    _run_eval(args.algo, ckpt_dir, num_episodes, args.gpu)
    print(f"[{args.algo}] done.")


if __name__ == "__main__":
    main()
