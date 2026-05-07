"""
Modal wrapper for running TD3 (Twin Delayed DDPG) training in the cloud.

This script sets up a Modal container with all required dependencies and runs
the TD3 training script from algo_ablations on cloud GPUs.

Usage (from project root):
    modal run modal_scripts/modal_td3.py

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
    .pip_install("uv")
    .add_local_dir(
        str(project_root),
        remote_path="/root/quantum-rl-project",
        ignore=ignore_patterns,
        copy=True
    )
    .run_commands(
        "cd /root/quantum-rl-project && uv sync --frozen"
    )
    .add_local_file(
        str(project_root / "src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    )
)

app = modal.App("quantum-rl-td3-training")


@app.function(
    gpu="A100:5",  # 5 GPUs as requested
    # Options: "A100", "H100", "L40S", "L4", "T4", etc.
    # Multi-GPU: "A100:2" (2 GPUs), "H100:8" (8 GPUs), etc.
    # Note: H100, A100, L40S, L4, T4 support up to 8 GPUs; A10 supports up to 4
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(checkpoint_artifact: str = None):
    """Run the TD3 training script inside Modal container.

    Args:
        checkpoint_artifact: Optional wandb artifact path to resume from
                           e.g. 'anon-entity/AlgorithmAblations/rl_checkpoint_best:v6'
    """
    import subprocess
    import os

    # Change to project directory
    os.chdir("/root/quantum-rl-project")

    # Build the command
    cmd = ["uv", "run", "python", "src/qadapt/algo_ablations/td3_train.py", "--config", "configs/td3_training_config.yaml"]

    # If checkpoint artifact is provided, download it using uv run and add to command
    if checkpoint_artifact:
        print(f"Downloading checkpoint artifact: {checkpoint_artifact}")

        # Create a small script to download the artifact
        download_script = f'''
import wandb
run = wandb.init(project="AlgorithmAblations", entity="anon-entity", job_type="checkpoint_download")
artifact = run.use_artifact("{checkpoint_artifact}", type="model_checkpoint")
artifact_dir = artifact.download()
wandb.finish()
print(artifact_dir)
'''
        # Run the download script using uv to access wandb
        result = subprocess.run(
            ["uv", "run", "python", "-c", download_script],
            capture_output=True,
            text=True,
            check=True
        )

        # Get the artifact directory from the output (last non-empty line)
        artifact_dir = result.stdout.strip().split('\n')[-1]
        print(f"Checkpoint downloaded to: {artifact_dir}")
        cmd.extend(["--load-checkpoint", artifact_dir])

    # Run the TD3 training script using uv to use the virtual environment
    # Uses the TD3 config from algo_ablations/configs
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(checkpoint: str = None):
    """Entry point when running 'modal run modal_td3.py'

    Args:
        checkpoint: Optional wandb artifact to resume from
                   e.g. --checkpoint 'anon-entity/AlgorithmAblations/rl_checkpoint_best:v6'
    """
    print("Starting TD3 (Twin Delayed DDPG) training on Modal...")
    print("This will run td3_train.py on cloud GPUs")
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    train.remote(checkpoint_artifact=checkpoint)
    print("Training completed!")
