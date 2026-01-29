#!/usr/bin/env python3
"""Download wandb artifact and run inference evaluation."""
import subprocess
import sys

import wandb

def main():
    # Download the model checkpoint artifact
    print("Downloading wandb artifact...")
    run = wandb.init()
    artifact = run.use_artifact(
        'rl_agents_for_tuning/RLModel/rl_checkpoint_best:v3783',
        type='model_checkpoint'
    )
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")
    wandb.finish()

    # Run inference with the downloaded checkpoint
    # 100 steps (configured in env_config.yaml), 2 dots, 100 trials
    cmd = [
        sys.executable,
        "main.py",
        "--load-checkpoint", artifact_dir,
        "--collect-data",
        "--num-rollouts", "100",
        "--disable-wandb",
    ]

    print(f"\nRunning inference: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
