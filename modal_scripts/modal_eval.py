"""
Modal wrapper for running eval data collection on cloud GPUs.

This script downloads a checkpoint from wandb, runs eval with specified num_dots,
and uploads the distance data back to wandb as an artifact.

Usage (from project root):
    # Test with 2 rollouts first
    modal run modal_scripts/modal_eval.py --num-rollouts 2

    # Full run with 100 rollouts
    modal run modal_scripts/modal_eval.py --num-rollouts 100

    # Custom checkpoint
    modal run modal_scripts/modal_eval.py --checkpoint "anon-entity/RLModel/rl_checkpoint_best:vN"
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
        str(project_root / "src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"),
        remote_path="/root/quantum-rl-project/src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    )
)

app = modal.App("quantum-rl-eval")


@app.function(
    gpu="A100:5",  # Same as training - config expects 12 env_runners + 1 learner
    image=image,
    timeout=86400,  # 24 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_eval(
    checkpoint_artifact: str,
    num_dots: int = 8,
    num_rollouts: int = 100,
):
    """Run eval data collection inside Modal container.

    Args:
        checkpoint_artifact: Wandb artifact path for the checkpoint
                            e.g. 'anon-entity/RLModel/rl_checkpoint_best:vN'
        num_dots: Number of dots for the environment (default: 8)
        num_rollouts: Number of rollouts to collect (default: 100)
    """
    import subprocess
    import os
    import yaml

    os.chdir("/root/quantum-rl-project")

    # Download checkpoint from wandb
    import wandb
    import shutil
    print(f"Downloading checkpoint artifact: {checkpoint_artifact}")
    run = wandb.init(project="RLModel", entity="anon-entity")
    artifact = run.use_artifact(checkpoint_artifact, type='model_checkpoint')
    artifact_dir = artifact.download()
    wandb.finish()
    print(f"Checkpoint downloaded to: {artifact_dir}")

    # Copy training config if not present in checkpoint
    training_config_path = Path(artifact_dir) / "training_config.yaml"
    if not training_config_path.exists():
        default_config = Path("src/qadapt/training/configs/ppo_impala.yaml")
        print(f"No training_config.yaml in checkpoint, copying from {default_config}")
        shutil.copy(default_config, training_config_path)

    # Create modified env_config with num_dots override
    env_config_path = Path("benchmarks/Ablations/env_config.yaml")
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    # Override num_dots
    original_num_dots = env_config["simulator"]["num_dots"]
    env_config["simulator"]["num_dots"] = num_dots
    print(f"Overriding num_dots: {original_num_dots} -> {num_dots}")

    # Override capacitance model weights path for Modal
    modal_weights_path = "/root/quantum-rl-project/src/qadapt/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth"
    env_config["capacitance_model"]["weights_path"] = modal_weights_path
    print(f"Overriding capacitance weights path to: {modal_weights_path}")

    # Write modified config back
    with open(env_config_path, "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)

    # Build command - uses config values for env_runners/learners (same as training)
    cmd = [
        "uv", "run", "python", "benchmarks/Ablations/main.py",
        "--load-checkpoint", artifact_dir,
        "--collect-data",
        "--upload-to-wandb",
        "--num-rollouts", str(num_rollouts),
    ]

    print(f"\nRunning eval with {num_rollouts} rollouts, {num_dots} dots (using config for env_runners/learners)...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run the eval script
    subprocess.run(cmd, check=True)

    print("\nEval completed! Distance data uploaded to wandb.")


@app.local_entrypoint()
def main(
    checkpoint: str = "anon-entity/RLModel/rl_checkpoint_best:vN",
    num_dots: int = 8,
    num_rollouts: int = 100,
):
    """Entry point when running 'modal run modal_eval.py'

    Args:
        checkpoint: Wandb artifact path for the checkpoint (default: run 473)
        num_dots: Number of dots for environment (default: 8)
        num_rollouts: Number of rollouts to collect (default: 100)
    """
    print("Starting quantum device RL eval on Modal...")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Num dots: {num_dots}")
    print(f"  Num rollouts: {num_rollouts}")
    print("  Using config values for env_runners/learners (same as training)")

    run_eval.remote(
        checkpoint_artifact=checkpoint,
        num_dots=num_dots,
        num_rollouts=num_rollouts,
    )
    print("Eval completed!")
