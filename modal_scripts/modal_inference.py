"""
Modal inference script for running RL agent rollouts on cloud GPUs.

This script runs inference in a SINGLE PROCESS (no distributed workers) to avoid
the hanging issues caused by Ray worker coordination in the eval pipeline.

Key difference from modal_eval.py:
- Uses num_env_runners=0 and num_learners=0 for local execution
- No subprocess call - runs inference directly

Usage:
    modal run modal_scripts/modal_inference.py
    modal run modal_scripts/modal_inference.py --num-dots 8 --num-rollouts 100
    modal run modal_scripts/modal_inference.py --checkpoint "rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3482"
"""
import modal
from pathlib import Path

# Get absolute path to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Read ignore patterns from .modalignore file
modalignore_path = project_root / ".modalignore"
ignore_patterns = []
if modalignore_path.exists():
    with open(modalignore_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.append(line)

# Create image with project dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv", "wandb")
    .add_local_dir(
        str(project_root),
        remote_path="/root/quantum-rl-project",
        ignore=ignore_patterns,
        copy=True,
    )
    .env({"MODAL_MOUNT_TIMEOUT": "600"})
    .run_commands(
        "cd /root/quantum-rl-project && uv sync --frozen"
    )
    .add_local_file(
        str(project_root / "src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    )
)

app = modal.App("quantum-rl-inference")


@app.function(
    gpu="H100",  # Single H100 for inference
    image=image,
    timeout=7200,  # 2 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_inference(
    checkpoint_artifact: str,
    num_dots: int = 8,
    num_rollouts: int = 10,
):
    """Run inference via subprocess (uses uv run to activate venv).

    Args:
        checkpoint_artifact: Wandb artifact path for the checkpoint
        num_dots: Number of quantum dots (default: 8)
        num_rollouts: Number of rollouts to run (default: 10)
    """
    import subprocess
    import os
    import yaml

    os.chdir("/root/quantum-rl-project")

    # Download checkpoint from wandb
    import wandb
    import shutil
    print(f"Downloading checkpoint artifact: {checkpoint_artifact}")
    run = wandb.init(project="RLModel", entity="rl_agents_for_tuning", job_type="inference")
    artifact = run.use_artifact(checkpoint_artifact, type='model_checkpoint')
    checkpoint_dir = artifact.download()
    wandb.finish()
    print(f"Checkpoint downloaded to: {checkpoint_dir}")

    # Copy training config if not present in checkpoint
    training_config_path = Path(checkpoint_dir) / "training_config.yaml"
    if not training_config_path.exists():
        default_config = Path("src/swarm/training/configs/ppo_impala.yaml")
        print(f"No training_config.yaml in checkpoint, copying from {default_config}")
        shutil.copy(default_config, training_config_path)

    # Update env_config with Modal-specific paths
    env_config_path = Path("/root/quantum-rl-project/src/eval_runs/env_config.yaml")
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    # Override capacitance model weights path for Modal
    modal_weights_path = "/root/quantum-rl-project/src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    env_config["capacitance_model"]["weights_path"] = modal_weights_path

    # Write updated config back
    with open(env_config_path, 'w') as f:
        yaml.dump(env_config, f, default_flow_style=False)

    # Build command
    cmd = [
        "uv", "run", "python", "src/eval_runs/inference.py",
        "--checkpoint", checkpoint_dir,
        "--num-dots", str(num_dots),
        "--num-rollouts", str(num_rollouts),
        "--upload-to-wandb",
    ]

    print(f"\nRunning inference command:")
    print(f"  {' '.join(cmd)}\n")

    # Run the inference script
    subprocess.run(cmd, check=True)

    print("\nInference completed!")


@app.local_entrypoint()
def main(
    checkpoint: str = "rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3482",
    num_dots: int = 8,
    num_rollouts: int = 10,
):
    """Entry point when running 'modal run modal_inference.py'

    Args:
        checkpoint: Wandb artifact path for the checkpoint
        num_dots: Number of quantum dots (default: 8)
        num_rollouts: Number of rollouts to run (default: 10)
    """
    print("=" * 60)
    print("Modal Inference Pipeline")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Num dots: {num_dots}")
    print(f"  Num rollouts: {num_rollouts}")
    print(f"  GPU: H100 (single process execution)")
    print("=" * 60)

    run_inference.remote(
        checkpoint_artifact=checkpoint,
        num_dots=num_dots,
        num_rollouts=num_rollouts,
    )

    print("Inference completed!")
