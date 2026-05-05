"""
Modal wrapper for running FACMAC / MADDPG training in the cloud.

Both algorithms share the same entrypoint (benchmarks/facmac/train.py); only
the --config and --env-config flags differ. Pass `algo` to pick the alg config
under benchmarks/facmac/vendor/config/algs/.

Usage (from project root):
    # MADDPG training, default env_quantum
    uv run --extra facmac modal run modal_scripts/modal_facmac.py

    # explicit
    uv run --extra facmac modal run modal_scripts/modal_facmac.py \
        --algo maddpg_quantum --env env_quantum

    # FACMAC retrain (if ever needed)
    uv run --extra facmac modal run modal_scripts/modal_facmac.py \
        --algo facmac_quantum --env env_quantum
"""
import modal
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent

modalignore_path = project_root / ".modalignore"
ignore_patterns = []
if modalignore_path.exists():
    with open(modalignore_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.append(line)

# FACMAC needs the `[facmac]` extras (sacred, jsonpickle, tensorboard-logger,
# gym==0.10.8). uv sync --extra facmac installs them. sacred's GitPython
# dependency further requires the `git` binary at import time, so we add it
# to the apt layer.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv", "wandb")
    .add_local_dir(
        str(project_root),
        remote_path="/root/quantum-rl-project",
        ignore=ignore_patterns,
        copy=True,
    )
    .env({"MODAL_MOUNT_TIMEOUT": "600"})
    .run_commands(
        "cd /root/quantum-rl-project && uv sync --frozen --extra facmac"
    )
)

app = modal.App("quantum-rl-facmac")

# Persistent volume so checkpoints survive between runs and can be pulled back
# locally for eval. PyMARL writes to results/models/<run_name>__<timestamp>/...
volume = modal.Volume.from_name("facmac-results", create_if_missing=True)


@app.function(
    gpu="H100",
    image=image,
    timeout=86400,  # 24h
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/results": volume},
)
def train(algo: str = "maddpg_quantum", env: str = "env_quantum", t_max: int | None = None):
    """Run benchmarks/facmac/train.py inside Modal.

    Args:
        algo: name of vendor/config/algs/<algo>.yaml (without extension)
        env:  name of vendor/config/envs/<env>.yaml (without extension)
        t_max: optional override for the sacred t_max param
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    os.chdir("/root/quantum-rl-project")

    cmd = [
        "uv", "run", "--extra", "facmac", "python",
        "benchmarks/facmac/train.py",
        f"--config={algo}",
        f"--env-config={env}",
    ]
    if t_max is not None:
        cmd += ["with", f"t_max={t_max}"]

    print(f"[modal_facmac] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # Copy results out to the persistent volume so they survive container exit.
    src = Path("benchmarks/facmac/results")
    dst = Path("/results") / algo
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for sub in ("models", "sacred"):
            s = src / sub
            if s.exists():
                shutil.copytree(s, dst / sub, dirs_exist_ok=True)
                print(f"[modal_facmac] copied {s} -> {dst / sub}")
    volume.commit()
    print(f"[modal_facmac] done. Checkpoints in volume facmac-results under /{algo}/")


@app.local_entrypoint()
def main(
    algo: str = "maddpg_quantum",
    env: str = "env_quantum",
    t_max: int | None = None,
):
    print(f"[modal_facmac] launching cloud run: algo={algo}, env={env}, t_max={t_max}")
    train.remote(algo=algo, env=env, t_max=t_max)
    print("[modal_facmac] cloud run finished.")
