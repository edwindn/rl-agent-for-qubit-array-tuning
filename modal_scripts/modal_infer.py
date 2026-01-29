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

# Checkpoint path (relative to project root)
CHECKPOINT_PATH = "src/swarm/inference/checkpoints/run_418"

# Derived paths
LOCAL_CHECKPOINT_PATH = project_root / CHECKPOINT_PATH
REMOTE_CHECKPOINT_PATH = f"/root/quantum-rl-project/{CHECKPOINT_PATH}"

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
    .add_local_file(
        str(project_root / "src/swarm/capacitance_model/symmetric_weights/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/swarm/capacitance_model/symmetric_weights/mobilenet_barrier_weights.pth",
        copy=True
    )
    .add_local_dir(
        str(LOCAL_CHECKPOINT_PATH),
        remote_path=REMOTE_CHECKPOINT_PATH,
        copy=True
    )
    .run_commands(
        "cd /root/quantum-rl-project && uv sync --frozen"
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
def train():
    """Run the training script inside Modal container."""
    import subprocess
    import os

    # Change to project directory
    os.chdir("/root/quantum-rl-project")

    # Run the training script using uv to use the virtual environment
    # You can add any command-line arguments here
    subprocess.run(
        ["uv", "run", "python", "src/swarm/training/train_inference.py", "--load-checkpoint", CHECKPOINT_PATH, "--load-configs"],
        check=True
    )

@app.local_entrypoint()
def main():
    """Entry point when running 'modal run modal_train.py'"""
    print("Starting quantum device RL training on Modal...")
    print("This will run train.py on cloud GPUs")
    train.remote()
    print("Training completed!")
