"""
Modal wrapper for running quantum device RL training in the cloud.

This script sets up a Modal container with all required dependencies and runs
the existing train.py script on cloud GPUs.

Usage (from project root):
    modal run modal_scripts/modal_train.py

Or to customize GPU type:
    # Edit the gpu parameter in the @app.function decorator below
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
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                ignore_patterns.append(line)

# Create image with project dependencies
# Modal automatically caches this based on requirements.txt hash
# Only rebuilds if requirements.txt changes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv", "wandb")
    .add_local_dir(
        str(project_root),
        remote_path="/root/quantum-rl-project",
        ignore=ignore_patterns,
        copy=True,
    )
    .env({"MODAL_MOUNT_TIMEOUT": "600"})  # 10 min mount timeout
    .run_commands(
        "cd /root/quantum-rl-project && uv sync --frozen"
    )
    .add_local_file(
        str(project_root / "src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    )
)

app = modal.App("quantum-rl-training")


@app.function(
    gpu="A100:5",  # Single GPU - Change to "A100:2", "A100:4", or "A100:8" for multiple GPUs
    # Options: "A100", "H100", "L40S", "L4", "T4", etc.
    # Multi-GPU: "A100:2" (2 GPUs), "H100:8" (8 GPUs), etc.
    # Note: H100, A100, L40S, L4, T4 support up to 8 GPUs; A10 supports up to 4
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(checkpoint_artifact: str = None, config_path: str = "src/qadapt/training/configs/ppo_impala.yaml"):
    """Run the training script inside Modal container.

    Args:
        checkpoint_artifact: Optional wandb artifact path to resume from
                            e.g. 'rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3510'
        config_path: Path to config file relative to project root
    """
    import subprocess
    import os

    # Change to project directory
    os.chdir("/root/quantum-rl-project")

    # Build command
    cmd = ["uv", "run", "python", "src/qadapt/training/train.py", "--config", config_path]

    # If checkpoint artifact specified, download it first
    if checkpoint_artifact:
        import wandb
        print(f"Downloading checkpoint artifact: {checkpoint_artifact}")
        run = wandb.init(project="RLModel", entity="rl_agents_for_tuning")
        artifact = run.use_artifact(checkpoint_artifact, type='model_checkpoint')
        artifact_dir = artifact.download()
        wandb.finish()
        print(f"Checkpoint downloaded to: {artifact_dir}")

        # Add checkpoint loading argument
        cmd.extend(["--load-checkpoint", artifact_dir])

    # Run the training script using uv to use the virtual environment
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(
    checkpoint: str = None,
    config: str = "src/qadapt/training/configs/ppo_impala.yaml",
):
    """Entry point when running 'modal run modal_train.py'

    Args:
        checkpoint: Optional wandb artifact path to resume from
                   e.g. 'rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3510'
        config: Path to config file relative to project root
    """
    print("Starting quantum device RL training on Modal...")
    print("This will run train.py on cloud GPUs")
    print(f"Using config: {config}")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    train.remote(checkpoint_artifact=checkpoint, config_path=config)
    print("Training completed!")
