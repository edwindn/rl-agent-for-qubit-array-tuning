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

# Create image with project dependencies
# Modal automatically caches this based on requirements.txt hash
# Only rebuilds if requirements.txt changes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip_install_private_repos
    .pip_install_private_repos(
        # Install qarray-latched from private repo at specific commit
        "github.com/b-vanstraaten/qarray-latched@fcc472276f27e7633bb3aafc6f0d6c92966875d7",
        git_user="rahul-marchand",  # Your GitHub username
        secrets=[modal.Secret.from_name("github-private")],
    )
    .pip_install_from_requirements(str(project_root / "requirements.txt"))
    .add_local_dir(str(project_root), remote_path="/root/quantum-rl-project")
)

app = modal.App("quantum-rl-training")


@app.function(
    gpu="A100:8",  # Single GPU - Change to "A100:2", "A100:4", or "A100:8" for multiple GPUs
    # Options: "A100", "H100", "L40S", "L4", "T4", etc.
    # Multi-GPU: "A100:2" (2 GPUs), "H100:8" (8 GPUs), etc.
    # Note: H100, A100, L40S, L4, T4 support up to 8 GPUs; A10 supports up to 4
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[
        modal.Secret.from_name("wandb-secret"),  # W&B logging
    ],
)
def train():
    """Run the training script inside Modal container."""
    import subprocess
    import os

    # Change to project directory
    os.chdir("/root/quantum-rl-project")

    # Run the training script
    # You can add any command-line arguments here
    subprocess.run(
        ["python", "src/swarm/training/train.py"],
        check=True
    )

@app.local_entrypoint()
def main():
    """Entry point when running 'modal run modal_train.py'"""
    print("Starting quantum device RL training on Modal...")
    print("This will run train.py on cloud GPUs")
    train.remote()
    print("Training completed!")
