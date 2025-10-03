#!/usr/bin/env python
"""
Modal initialization script - builds and deploys the training environment.
Run this first with: modal deploy modal_init.py
"""

import modal

# Create Modal app
app = modal.App("qarray-rl-training")

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "openssh-client")
    .run_commands(
        # Setup SSH for git
        "mkdir -p /root/.ssh",
        "ssh-keyscan github.com >> /root/.ssh/known_hosts",
        # Write SSH key from environment variable (provided by secret)
        'echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_ed25519',
        "chmod 600 /root/.ssh/id_ed25519",
        secrets=[modal.Secret.from_name("github-ssh")],
    )
    .pip_install(
        # Core ML/RL dependencies
        "torch",
        "torchvision[models]",
        "ray[rllib]",
        "wandb[media]",
        "gymnasium",
        "jax[cuda12]",
        # Data and visualization
        "numpy",
        "matplotlib",
        "pyyaml",
        "scipy",
        "scikit-learn",
        "scienceplots",
        # QArray
        "qarray",
        "git+ssh://git@github.com/b-vanstraaten/qarray-latched.git@c076d4cef57a071dd6e52458ad5937589747c18f",
    )
    # Set environment for unbuffered output
    .env({"PYTHONUNBUFFERED": "1"})
    # Mount the source code
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu="A100:8",
    cpu=24,
    memory=160 * 1024,  # 160GB
    timeout=86400,  # 24h
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train():
    """Run the training script with GPU verification."""
    import subprocess
    import sys
    import os
    from datetime import datetime

    # Set unbuffered output
    os.environ["PYTHONUNBUFFERED"] = "1"

    print("========= MODAL ENVIRONMENT =========")
    print(f"Date: {datetime.now()}")
    print("=====================================")

    # GPU Check
    print("========= GPU CHECK =========")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception as e:
        print(f"nvidia-smi failed: {e}")
        sys.exit(1)

    # PyTorch GPU check
    try:
        import torch
        print(f"Torch CUDA Available: {torch.cuda.is_available()}")
        print(f"Torch Device Count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Torch Current Device: {torch.cuda.current_device()}")
    except ModuleNotFoundError:
        print("Torch not installed in this environment")
        sys.exit(1)
    print("=============================")

    # Run Training Script with unbuffered output
    print("🚀 Launching training...")
    print("=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()

    result = subprocess.run(
        ["python", "-u", "/root/src/swarm/training/train.py"],
        check=False,
        stdout=sys.stdout,  # Stream directly to Modal logs
        stderr=sys.stderr,  # Stream errors directly to Modal logs
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    print("=" * 60)
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print("Training completed successfully")
