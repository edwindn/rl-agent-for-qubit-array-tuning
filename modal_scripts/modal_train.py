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

# Create image with project dependencies
# Modal automatically caches this based on requirements.txt hash
# Only rebuilds if requirements.txt changes
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("../requirements.txt")
    .add_local_dir("..", remote_path="/root/quantum-rl-project")
)

app = modal.App("quantum-rl-training")


@app.function(
    gpu="A100",  # Options: "A100", "H100", "L40S", "L4", "T4", etc.
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],  # Optional: for W&B logging
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
