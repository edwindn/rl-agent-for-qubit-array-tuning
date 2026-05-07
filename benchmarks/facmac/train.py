"""
Thin wrapper around vendor/main.py that:
  1. Sets up sys.path to let vendor imports + swarm imports both resolve.
  2. Registers PyMARLEnvWrapper under "pymarl_quantum" in vendor's env REGISTRY.
  3. Copies our task-specific config YAMLs into vendor/config/{algs,envs}/ so
     sacred's config loader can find them.
  4. chdir's into benchmarks/facmac so `local_results_path: results` ends up
     co-located with the sacred observer output.
  5. Delegates to vendor/main.py via runpy.run_path, preserving sacred CLI semantics.

Usage:
    # task-3 smoke run, CPU, ~10 min
    uv run --extra facmac python benchmarks/facmac/train.py \\
        --config=facmac_quantum_smoke --env-config=env_quantum_smoke

    # add extra sacred overrides after `with`:
    uv run --extra facmac python benchmarks/facmac/train.py \\
        --config=facmac_quantum_smoke --env-config=env_quantum_smoke \\
        with seed=7 t_max=800
"""

from __future__ import annotations

import os
import sys

# Route JAX (QArray backend) to match torch's device. Must be set BEFORE any
# import chain that loads jax (which happens inside _register_env()).
# - use_cuda=True on the CLI -> JAX on GPU, with memory-sharing flags so it
#   coexists with torch on the same device (mirrors the main training config).
# - otherwise -> JAX on CPU to avoid BLAS init fighting with torch's lazy CUDA.
_wants_cuda = any("use_cuda=True" in a for a in sys.argv)
if _wants_cuda:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import logging
import runpy
import shutil
from functools import partial
from pathlib import Path


def _silence_noisy_loggers() -> None:
    """
    Vendor's utils/logging.get_logger() sets the root logger to DEBUG, which
    cascades to JAX, git, sacred, etc. We pin specific child loggers to WARNING
    so their DEBUG output is filtered while the vendor's own logger stays chatty.
    """
    for name in ("jax", "jax._src", "absl", "git", "git.cmd", "git.util", "sacred.config"):
        logging.getLogger(name).setLevel(logging.WARNING)

BENCH_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BENCH_DIR / "vendor"
PROJECT_SRC = BENCH_DIR.parent.parent / "src"

CONFIGS_SRC = BENCH_DIR / "configs"
ALG_CONFIGS_DEST = VENDOR_DIR / "config" / "algs"
ENV_CONFIGS_DEST = VENDOR_DIR / "config" / "envs"

ENV_CONFIG_SMOKE = CONFIGS_SRC / "env_config_smoke.yaml"
ENV_CONFIG_FULL = CONFIGS_SRC / "env_config_full.yaml"
CAPACITANCE_CKPT = PROJECT_SRC / "qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"


def _pick_default_env_config(argv: list[str]) -> Path:
    """Select the underlying QuantumDeviceEnv config based on the sacred algo config name."""
    for a in argv:
        if a.startswith("--config=") and "full" in a:
            return ENV_CONFIG_FULL
    return ENV_CONFIG_SMOKE


def _setup_paths() -> None:
    for p in (VENDOR_DIR, PROJECT_SRC, BENCH_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _sync_configs() -> None:
    ALG_CONFIGS_DEST.mkdir(parents=True, exist_ok=True)
    ENV_CONFIGS_DEST.mkdir(parents=True, exist_ok=True)
    for yaml_path in CONFIGS_SRC.glob("*.yaml"):
        name = yaml_path.name
        if name.startswith("facmac_"):
            shutil.copy(yaml_path, ALG_CONFIGS_DEST / name)
        elif name.startswith("env_quantum"):
            shutil.copy(yaml_path, ENV_CONFIGS_DEST / name)


def _register_env() -> None:
    from envs import REGISTRY as env_REGISTRY, env_fn
    from env_wrapper import PyMARLEnvWrapper
    env_REGISTRY["pymarl_quantum"] = partial(env_fn, env=PyMARLEnvWrapper)


def _register_mac_and_agents() -> None:
    from controllers import REGISTRY as mac_REGISTRY
    from modules.agents import REGISTRY as vendor_agent_REGISTRY
    from grouped_mac import GroupedMAC
    from agents.plunger_cnn import PlungerCNNAgent
    from agents.barrier_cnn import BarrierCNNAgent

    mac_REGISTRY["grouped_mac"] = GroupedMAC
    vendor_agent_REGISTRY["plunger_cnn"] = PlungerCNNAgent
    vendor_agent_REGISTRY["barrier_cnn"] = BarrierCNNAgent


def _register_runners() -> None:
    from runners import REGISTRY as runner_REGISTRY
    from multi_gpu_parallel_runner import MultiGPUParallelRunner

    runner_REGISTRY["multi_gpu_parallel"] = MultiGPUParallelRunner


def _inject_default_env_args(argv: list[str]) -> list[str]:
    """
    Ensures absolute env_args paths are present. Only injects if the user hasn't
    already supplied them via `with env_args.xxx=...`.
    """
    user_with_args = []
    if "with" in argv:
        user_with_args = argv[argv.index("with") + 1:]

    needs_env_config = not any(a.startswith("env_args.env_config_path=") for a in user_with_args)
    needs_ckpt = not any(a.startswith("env_args.capacitance_model_checkpoint=") for a in user_with_args)

    additions = []
    if needs_env_config:
        additions.append(f"env_args.env_config_path={_pick_default_env_config(argv)}")
    if needs_ckpt:
        additions.append(f"env_args.capacitance_model_checkpoint={CAPACITANCE_CKPT}")

    if not additions:
        return argv
    if "with" in argv:
        idx = argv.index("with") + 1
        return argv[:idx] + additions + argv[idx:]
    return argv + ["with"] + additions


def main() -> None:
    _silence_noisy_loggers()
    _setup_paths()
    _sync_configs()
    _register_env()
    _register_mac_and_agents()
    _register_runners()

    os.chdir(BENCH_DIR)
    sys.argv = ["main.py"] + _inject_default_env_args(sys.argv[1:])
    runpy.run_path(str(VENDOR_DIR / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
