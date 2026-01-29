"""
Modal wrapper for running SAC (Soft Actor-Critic) training in the cloud.

This script sets up a Modal container with all required dependencies and runs
the SAC training script from algo_ablations on cloud GPUs.

Usage (from project root):
    modal run modal_scripts/modal_sac.py

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
        str(project_root / "src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    )
)

app = modal.App("quantum-rl-sac-training")


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
    """Run the SAC training script inside Modal container."""
    import subprocess
    import os

    # Change to project directory
    os.chdir("/root/quantum-rl-project")

    # Run the SAC training script using uv to use the virtual environment
    # Uses the SAC config from algo_ablations/configs
    subprocess.run(
        ["uv", "run", "python", "src/swarm/algo_ablations/sac_train.py", "--config", "configs/sac_training_config.yaml"],
        check=True
    )

@app.local_entrypoint()
def main():
    """Entry point when running 'modal run modal_sac.py'"""
    print("Starting SAC (Soft Actor-Critic) training on Modal...")
    print("This will run sac_train.py on cloud GPUs")
    train.remote()
    print("Training completed!")
