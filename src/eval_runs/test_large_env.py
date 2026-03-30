#!/usr/bin/env python3
"""
Test script for instantiating the quantum environment without Ray or training pipeline.
Simply loads env config, creates environment, and calls reset() to verify setup.
"""
import os
import sys
from pathlib import Path
import yaml
import tempfile
import time

# Set JAX environment variables before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["JAX_PLATFORM_NAME"] = "cuda"

# Add src directory to path for clean imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.scan_saving_wrapper import ScanSavingWrapper


def load_env_config():
    """Load environment configuration from eval_runs/env_config.yaml."""
    config_file = Path(__file__).parent / "env_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"env_config.yaml not found in {config_file.parent}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_test_env(env_config_path=None):
    """Create environment for testing (no Ray workers)."""
    import jax

    try:
        jax.clear_backends()
    except:
        pass

    gif_config = {
        'enabled': False,
        'save_dir': str(Path(__file__).parent / "test_gif_captures"),
        'target_agent_type': 'plunger',
        'target_agent_indices': [1],
        'fps': 0.5,
    }

    env_config = load_env_config()
    capacitance_weights_path = env_config.get("capacitance_model", {}).get("weights_path")

    env = ScanSavingWrapper(
        return_voltage=True,
        gif_config=gif_config,
        distance_data_dir=None,
        env_config_path=env_config_path,
        capacitance_model_checkpoint=capacitance_weights_path,
        scan_save_dir=None,
        scan_save_enabled=False,
        is_collecting_data=False,
    )

    return env


def main():
    """Test environment instantiation and reset."""
    env_config = load_env_config()

    temp_env_config_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, dir='/tmp'
    )
    yaml.dump(env_config, temp_env_config_file, default_flow_style=False)
    temp_env_config_path = temp_env_config_file.name
    temp_env_config_file.close()

    env = create_test_env(env_config_path=temp_env_config_path)
    start = time.time()
    obs, info = env.reset()
    print(time.time() - start)

    os.unlink(temp_env_config_path)


if __name__ == "__main__":
    main()
