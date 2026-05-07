#!/usr/bin/env python3
"""
Train a fresh QADAPT (run-473 hyperparams) with capacitance_model.update_method
set to "direct" — i.e. *trained* in direct virtualisation mode, not just
evaluated in it. The existing `w_o_kalman` ablation entry uses run-473's
weights at eval time only with the direct override, which is unfair: the
policy was trained against Kalman-smoothed virtualisation. This script gives
us a clean apples-to-apples direct-mode-trained QADAPT for the ablation.

Steps:
  1. Pull run-473 from wandb project rl_agents_for_tuning/RLModel
  2. Reconstruct training_config.yaml + env_config.yaml from run.config (the
     ablation pipeline already does this so we re-use the same logic)
  3. Apply override env.capacitance_model.update_method = "direct"
  4. Launch train.py --config ... --env-config ... on the requested GPU

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_qadapt_direct_473.py
  # or, equivalently:
  uv run python scripts/train_qadapt_direct_473.py --gpu 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "src" / "qadapt" / "training" / "train.py"

# Reuse the ablation pipeline's wandb-config reconstruction helpers.
sys.path.insert(0, str(REPO_ROOT / "src" / "eval_runs" / "ablation"))
from run_ablation import _resolve_run, _apply_overrides, _deep_merge  # noqa: E402


def prepare_configs(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[qadapt_direct_473] resolving wandb run -473 ...")
    run = _resolve_run("rl_agents_for_tuning/RLModel", 473)
    print(f"  -> {run.display_name} ({run.id})")

    full_cfg = dict(run.config)
    env_cfg = full_cfg.pop("env_config", None)
    if env_cfg is None:
        raise RuntimeError("run.config has no 'env_config' for run-473")

    canonical_env_path = REPO_ROOT / "src" / "eval_runs" / "env_config.yaml"
    if canonical_env_path.exists():
        env_cfg = _deep_merge(yaml.safe_load(canonical_env_path.read_text()), env_cfg)

    _apply_overrides(env_cfg, {"capacitance_model.update_method": "direct"})
    print(f"  applied env override: capacitance_model.update_method = direct")

    # Lab-server fit: original run-473 used 12 runners × 0.25 GPU + 1 learner × 0.75 GPU.
    # On 20GB A4000s each env runner consumes ~9.4 GB (capacitance model + JAX + torch),
    # so 12-on-4-GPUs OOMs. Spread to 6 runners × 1 GPU + 1 learner × 1 GPU = 7 GPUs.
    # This trades ~2× slower sample collection for stable execution.
    rl_cfg = full_cfg.setdefault("rl_config", {})
    rl_cfg.setdefault("env_runners", {})["num_env_runners"] = 6
    rl_cfg["env_runners"]["num_gpus_per_env_runner"] = 1.0
    rl_cfg.setdefault("learners", {})["num_gpus_per_learner"] = 1.0
    rl_cfg["learners"]["num_learners"] = 1
    # Original learner used minibatch_size=2048 — peak activations OOM on 20 GB A4000
    # during compute_values. Shrink 4× to fit; trade-off is more passes per epoch but
    # same effective gradient (full train_batch_size_per_learner=16384 still consumed).
    rl_cfg.setdefault("training", {})["minibatch_size"] = 512
    print("  applied training override: env_runners=6 × 1 GPU + learner × 1 GPU + minibatch_size=512 (A4000 fit)")

    train_cfg_path = out_dir / "training_config.yaml"
    env_cfg_path = out_dir / "env_config.yaml"
    train_cfg_path.write_text(yaml.safe_dump(full_cfg, default_flow_style=False))
    env_cfg_path.write_text(yaml.safe_dump(env_cfg, default_flow_style=False))
    print(f"  wrote configs to {out_dir}/")
    return train_cfg_path, env_cfg_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES (comma-separated, e.g. '0,1,2,3'). "
                         "Run-473 needs 4 GPUs: 12 env_runners × 0.25 + 1 learner = 4.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/qadapt_direct_473"))
    ap.add_argument("--no-launch", action="store_true",
                    help="Prepare configs but don't launch train.py")
    args = ap.parse_args()

    train_cfg, env_cfg = prepare_configs(args.out)

    if args.no_launch:
        print("--no-launch set; skipping train.py invocation")
        return

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_RUN_NAME"] = "qadapt_direct_473"

    cmd = [
        "uv", "run", "python", "-u", str(TRAIN_PY),
        "--config", str(train_cfg),
        "--env-config", str(env_cfg),
    ]
    print(f"[qadapt_direct_473] launching: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
