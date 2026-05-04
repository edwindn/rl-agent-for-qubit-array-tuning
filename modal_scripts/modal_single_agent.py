"""
Modal wrapper for running single-agent PPO training on cloud GPUs.

Uses the training code in src/swarm/single_agent_ablations/

Usage (from project root):
    modal run modal_scripts/modal_single_agent.py

    # Or with custom num_dots:
    modal run modal_scripts/modal_single_agent.py --num-dots 4
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
    .pip_install("uv")
    .add_local_dir(
        str(project_root),
        remote_path="/root/quantum-rl-project",
        ignore=ignore_patterns,
        copy=True
    )
    .run_commands(
        "cd /root/quantum-rl-project && UV_HTTP_TIMEOUT=300 uv sync --frozen"
    )
    .add_local_file(
        str(project_root / "src/swarm/capacitance_model/weights/best_model_barriers.pth"),
        remote_path="/root/quantum-rl-project/src/swarm/capacitance_model/weights/best_model_barriers.pth"
    )
)

app = modal.App("quantum-rl-single-agent")


@app.function(
    gpu="A100:5",  # 5 A100 GPUs
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(num_dots: int = 2, num_iterations: int = 150):
    """Run the single-agent training script inside Modal container."""
    import subprocess
    import os

    os.chdir("/root/quantum-rl-project")

    cmd = [
        "uv", "run", "python", "src/swarm/single_agent_ablations/train.py",
        "--num-dots", str(num_dots),
        "--defaults.num_iterations", str(num_iterations),
        "--rl_config.env_runners.sample_timeout_s", "1800",
    ]

    print(f"Running single-agent PPO with {num_dots} dots on A100")
    print(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(num_dots: int = 2, num_iterations: int = 150):
    """Entry point when running 'modal run modal_single_agent.py'"""
    print(f"Starting single-agent PPO training on Modal (5 A100s)...")
    print(f"  num_dots: {num_dots}")
    print(f"  num_iterations: {num_iterations}")
    print(f"  env_runners: 12 (0.3 GPU each)")
    print(f"  learner: 1 (1.0 GPU)")
    train.remote(num_dots=num_dots, num_iterations=num_iterations)
    print("Training completed!")
