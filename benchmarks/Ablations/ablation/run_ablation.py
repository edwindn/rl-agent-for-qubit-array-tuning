#!/usr/bin/env python3
"""
Pull a wandb checkpoint + run the ablation eval for one algo.

Reads ablation_config.yaml, looks up `--algo`, downloads the latest
'rl_checkpoint_best' artifact for that run, applies any env_overrides, and
invokes benchmarks/Ablations/main.py for `--num-episodes` rollouts.

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


REPO_ROOT = Path(__file__).resolve().parents[3]   # <repo>
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


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay onto base; overlay values win on conflicts.

    Used to fill in any env_config keys that the env code now expects but the
    wandb-logged config (from an older training run) didn't include — e.g.
    reward.sparse_reward was added to env.py after run_473 was trained.
    """
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


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

    # Reconstruct training_config.yaml + env_config.yaml from the run's
    # wandb-logged config. Wandb artifacts only ship the RLlib checkpoint
    # itself; the original yamls aren't in the artifact, but run.config holds
    # the merged dict that train.py logged at startup. Splitting out the
    # env_config sub-dict gives us a faithful pair of yamls that main.py can
    # consume.
    full_cfg = dict(run.config)
    env_cfg = full_cfg.pop("env_config", None)
    if env_cfg is None:
        raise RuntimeError(f"run.config has no 'env_config' key for {run.display_name}")

    # Override num_iterations to ablation episode count.
    full_cfg.setdefault("defaults", {})["num_iterations"] = int(num_episodes)

    # Inference-friendly Ray topology: pure local eval — no remote env runners
    # at all (sampling or evaluation). Training-time multi-runner setup
    # (num_env_runners: 12 @ 0.25 GPU each) trips a Ray 2.51 race in
    # EnvRunnerGroup.get_spaces on this server, and even
    # evaluation_num_env_runners=1 stalls between reset and step 1 for some
    # checkpoints (PPO+IMPALA, run_473) — local eval avoids both.
    rl_cfg = full_cfg.setdefault("rl_config", {})
    rl_cfg.setdefault("env_runners", {})["num_env_runners"] = 0
    rl_cfg["env_runners"]["num_gpus_per_env_runner"] = 0
    rl_cfg.setdefault("evaluation", {})["evaluation_num_env_runners"] = 0
    rl_cfg.setdefault("learners", {})["num_gpus_per_learner"] = 1.0

    # Disable GIF + scan-image capture for ablation — both write a PNG per
    # step per agent, costing ~1 sec/step in I/O alone. We only need the
    # per-episode .npy distance trajectories that the ablation_metrics
    # pipeline consumes.
    full_cfg.setdefault("gif_config", {})["enabled"] = False
    full_cfg["defaults"]["save_distance_data"] = True   # confirm .npy stays on

    # Skip per-iteration checkpoint saves. The eval loop runs algo.evaluate()
    # which doesn't change weights, so saving each iteration is wasteful. More
    # importantly, parallel ablation runs share cwd=<repo> and would
    # all race on the same ./checkpoints/iteration_N/ path — observed crash:
    # FileNotFoundError on .../iteration_5/learner_group/state.pkl when two
    # processes save concurrently.
    full_cfg.setdefault("checkpoints", {})["save_per_iter"] = False
    # The scan-saving wrapper writes if env var SCAN_SAVE_ENABLED is truthy;
    # main.py sets the dir unconditionally. Safest is to flag via the wrapper
    # path: setting scan_save_dir to None in env_config disables it (handled
    # below in env_overrides).

    # Patch ray.runtime_env.excludes — eval_runs/ accumulates >500MB of past
    # collected_data on this machine and Ray's working_dir upload caps at 512MB.
    # Older runs were logged before eval_runs ballooned, so their stored
    # excludes don't cover it.
    runtime_env = full_cfg.setdefault("ray", {}).setdefault("runtime_env", {})
    excludes = list(runtime_env.get("excludes") or [])
    for extra in ("eval_runs", "**/eval_runs/**", "**/scan_captures/**",
                  "**/rollout_scans/**", "**/__pycache__/**", "**/collected_data/**"):
        if extra not in excludes:
            excludes.append(extra)
    runtime_env["excludes"] = excludes

    # Deep-merge canonical env_config underneath wandb-logged env_config:
    # the wandb config wins on existing keys (preserves the run's training
    # settings), but new keys added to env.py since the training run get sane
    # defaults from the current canonical (e.g. reward.sparse_reward).
    canonical_env_path = REPO_ROOT / "src" / "eval_runs" / "env_config.yaml"
    if canonical_env_path.exists():
        with canonical_env_path.open() as fh:
            canonical_env_cfg = yaml.safe_load(fh)
        env_cfg = _deep_merge(canonical_env_cfg, env_cfg)

    # Apply user env_overrides on top of the merged env_config.
    overrides = algo_cfg.get("env_overrides") or {}
    if overrides:
        _apply_overrides(env_cfg, overrides)

    train_cfg_path = ckpt_dir / "training_config.yaml"
    env_cfg_path = ckpt_dir / "env_config.yaml"
    with train_cfg_path.open("w") as fh:
        yaml.safe_dump(full_cfg, fh, default_flow_style=False)
    with env_cfg_path.open("w") as fh:
        yaml.safe_dump(env_cfg, fh, default_flow_style=False)
    print(f"[{algo_name}] wrote configs (num_iterations={num_episodes}, "
          f"env_overrides={overrides or 'none'})")

    return ckpt_dir


SINGLE_AGENT_TRAIN = REPO_ROOT / "benchmarks" / "MARL" / "single_agent_ppo" / "train.py"
SINGLE_AGENT_SAC_TRAIN = REPO_ROOT / "benchmarks" / "MARL" / "single_agent_sac" / "train.py"
FACMAC_EVAL = REPO_ROOT / "benchmarks" / "MARL" / "facmac" / "run_eval_trials.py"


def _run_eval(
    algo_name: str,
    ckpt_dir: Path,
    num_episodes: int,
    gpu: int | None,
    pipeline: str = "rlmodel",
):
    """Invoke the appropriate eval entry point based on pipeline kind.

    - rlmodel: benchmarks/Ablations/main.py (PPO/MAPPO/SAC swarm)
    - single_agent: benchmarks/MARL/single_agent_ppo/train.py --eval-only
    - facmac: benchmarks/MARL/facmac/run_eval_trials.py --npy-output-dir
    """
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ABLATION_NUM_EPISODES"] = str(num_episodes)
    env["ABLATION_ALGO_NAME"] = algo_name
    # Unbuffered stdout/stderr so progress prints land in the log immediately
    # (default block-buffering when redirected to a file makes runs look hung).
    env["PYTHONUNBUFFERED"] = "1"

    if pipeline == "rlmodel":
        cmd = [
            "uv", "run", "python", "-u", str(EVAL_MAIN),
            "--load-checkpoint", str(ckpt_dir),
            "--disable-wandb",
        ]
    elif pipeline == "single_agent":
        cmd = [
            "uv", "run", "python", "-u", str(SINGLE_AGENT_TRAIN),
            "--load-checkpoint", str(ckpt_dir),
            "--load-configs",
            "--eval-only",
            "--disable-wandb",
        ]
    elif pipeline == "single_agent_sac":
        cmd = [
            "uv", "run", "python", "-u", str(SINGLE_AGENT_SAC_TRAIN),
            "--load-checkpoint", str(ckpt_dir),
            "--load-configs",
            "--eval-only",
            "--disable-wandb",
        ]
    elif pipeline == "facmac":
        # FACMAC eval writes per-trial .npy files into the same
        # collected_data/{ts}_{algo} layout the rlmodel pipeline uses.
        # FACMAC also requires an env_config yaml (num_dots, max_steps, etc.).
        # For wandb-sourced ckpts _prepare_checkpoint writes one; for local
        # ckpts we fall back to the canonical FACMAC env_quantum_full.yaml.
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "src" / "eval_runs" / "collected_data" / f"{ts}_{algo_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        env_cfg_path = ckpt_dir / "env_config.yaml"
        if not env_cfg_path.exists():
            # env_quantum_full.yaml is the *sacred* wrapper (env_args.env_config_path);
            # the underlying env_config.yaml that QArray actually consumes is this:
            env_cfg_path = REPO_ROOT / "benchmarks" / "facmac" / "configs" / "env_config_full.yaml"
        cmd = [
            "uv", "run", "--extra", "facmac", "python", "-u", str(FACMAC_EVAL),
            "--checkpoint-dir", str(ckpt_dir),
            "--env-config", str(env_cfg_path),
            "--num-trials", str(num_episodes),
            "--npy-output-dir", str(out_dir),
        ]
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    print(f"[{algo_name}] running: {' '.join(cmd)}", flush=True)
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
    pipeline = algo_cfg.get("pipeline", "rlmodel")
    if pipeline not in ("rlmodel", "single_agent", "single_agent_sac", "facmac"):
        print(f"[{args.algo}] pipeline={pipeline} — supported: rlmodel, single_agent, "
              f"single_agent_sac, facmac.", file=sys.stderr)
        sys.exit(2)

    num_episodes = args.num_episodes or cfg["defaults"]["num_episodes"]
    weights_root = Path(args.weights_root).resolve()
    weights_root.mkdir(parents=True, exist_ok=True)

    if "local_checkpoint" in algo_cfg:
        ckpt_dir = Path(algo_cfg["local_checkpoint"]).resolve()
        if not ckpt_dir.exists():
            print(f"[{args.algo}] local_checkpoint not found: {ckpt_dir}", file=sys.stderr)
            sys.exit(3)
        print(f"[{args.algo}] using local_checkpoint {ckpt_dir}")
    elif args.skip_download and (weights_root / args.algo).exists():
        ckpt_dir = weights_root / args.algo
        print(f"[{args.algo}] reusing existing {ckpt_dir}")
    else:
        ckpt_dir = _prepare_checkpoint(args.algo, algo_cfg, weights_root, num_episodes)

    _run_eval(args.algo, ckpt_dir, num_episodes, args.gpu, pipeline=pipeline)
    print(f"[{args.algo}] done.")


if __name__ == "__main__":
    main()
