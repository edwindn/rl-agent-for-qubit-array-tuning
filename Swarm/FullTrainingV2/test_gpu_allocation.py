#!/usr/bin/env python3
"""
Test script to verify Ray automatically distributes environments across GPUs.
"""
import os
import sys
from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# Add Swarm directory to path
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent
sys.path.append(str(swarm_dir))
os.environ['SWARM_PROJECT_ROOT'] = str(swarm_dir)

import ray
from ray.tune.registry import register_env

# Import the create_env function we just fixed
from train import create_env, setup_gpu_environment

def test_gpu_distribution():
    """Test that environments are properly distributed across GPUs."""
    
    # Setup with multiple GPUs (modify as needed for your system)
    train_gpus = "1, 2, 3, 4"  # Adjust based on available GPUs
    num_gpus = setup_gpu_environment(train_gpus)
    
    # Initialize Ray
    ray.init(
        include_dashboard=False,
        runtime_env={
            "working_dir": str(swarm_dir),
            "py_modules": [
                str(swarm_dir / "Environment"),
                str(swarm_dir / "VoltageAgent")
            ],
            "env_vars": {
                "SWARM_PROJECT_ROOT": str(swarm_dir),
            }
        }
    )
    
    try:
        # Register environment
        register_env("test_env", create_env)
        
        # Create multiple environment instances to test distribution
        env_config = {"num_dots": 4}
        
        print("\n=== Testing Environment GPU Allocation ===")
        
        # Create environments on different workers
        @ray.remote(num_gpus=0.1)
        def create_test_env():
            env = create_env(env_config)
            return f"Worker created env on device: {env.base_env.gpu if hasattr(env.base_env, 'gpu') else 'unknown'}"
        
        # Launch multiple remote environment creations
        futures = [create_test_env.remote() for _ in range(10)]
        results = ray.get(futures)
        
        print("\nEnvironment creation results:")
        for i, result in enumerate(results):
            print(f"  Worker {i}: {result}")
            
        print(f"\nExpected: Environments should be distributed across {num_gpus} GPU(s)")
        
    finally:
        ray.shutdown()

if __name__ == "__main__":
    test_gpu_distribution()