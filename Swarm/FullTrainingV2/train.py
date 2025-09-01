#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.
"""
import os
import sys
from typing import Any, Optional, List

# Configure JAX to use CPU-only before any other imports
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env

# Memory monitoring imports
import psutil
import gc
import time
import logging
from datetime import datetime
import wandb

# Add current directory to path for imports
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent  # Get Swarm directory
sys.path.append(str(swarm_dir))

# Set environment variable for Swarm directory so Ray workers can find project files
os.environ['SWARM_PROJECT_ROOT'] = str(swarm_dir)

from utils.policy_mapping import create_rl_module_spec
from utils.logging_utils import (
    log_memory_usage_wandb, memory_checkpoint_wandb, log_training_metrics_wandb,
    setup_memory_logging, log_memory_usage, memory_checkpoint
)
from metrics_utils import extract_training_metrics


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from Environment.multi_agent_wrapper import MultiAgentQuantumWrapper
    import torch

    num_dots = config["num_dots"]    
    
    # Get Ray's GPU assignment for this worker
    try:
        assigned_gpu_ids = ray.get_gpu_ids()
    except Exception as e:
        print("[ENV DEBUG] ray.get_gpu_ids() failed:", e)
        assigned_gpu_ids = []

    print(f'[ENV DEBUG] Ray assigned GPU IDs to this worker: {assigned_gpu_ids}')
    
    # Determine device based on Ray's GPU allocation
    if assigned_gpu_ids:
        # Ray assigns fractional GPUs, so multiple workers may share the same GPU
        # Use the first assigned GPU ID as the local CUDA device
        # With CUDA_VISIBLE_DEVICES set, Ray's GPU IDs map to local CUDA ordinals
        local_gpu_ordinal = 0  # Always use ordinal 0 since Ray manages visibility
        device = f"cuda:{local_gpu_ordinal}"
        
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_gpu_ordinal)
                print(f"[ENV DEBUG] Set torch device to cuda:{local_gpu_ordinal}")
            else:
                print("[ENV DEBUG] CUDA not available")
                raise RuntimeError("CUDA not available")
        except Exception as e:
            print(f"[ENV DEBUG] torch.cuda.set_device({local_gpu_ordinal}) failed: {e}")
            raise RuntimeError("CUDA not available")
    else:
        print("[ENV DEBUG] No GPUs assigned by Ray")
        raise RuntimeError("No CUDA device assigned")

    print(f"[ENV DEBUG] Final device for environment: {device}")
    return MultiAgentQuantumWrapper(num_dots=num_dots, training=True, gpu=device)


def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs. Ray 2.49.0 passes agent_id and episode."""
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type. "
            f"Expected format: 'plunger_X' or 'barrier_X' where X is the agent number."
        )


def setup_gpu_environment(train_gpus: str):
    """
    Setup GPU environment variables for Ray and PyTorch.
    
    Args:
        train_gpus: Comma-separated string of GPU indices for training
        
    Returns:
        int: Number of GPUs available for training
    """
    # Parse training GPUs
    gpu_indices = [int(gpu.strip()) for gpu in train_gpus.split(',') if gpu.strip()]
    
    # Set CUDA_VISIBLE_DEVICES for Ray to only see the specified GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
    
    print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Training will use {len(gpu_indices)} GPU(s): {gpu_indices}")
    print("Ray will automatically distribute environments across available GPUs")
    
    return len(gpu_indices)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    parser.add_argument("--num-dots", type=int, default=8, help="Number of quantum dots")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--train-gpus", type=str, default="0", help="Comma-separated GPU indices to use (e.g., '0,1,2')")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def main():
    """Main training function using Ray RLlib 2.49.0 modern API with wandb logging."""
    args = parse_arguments()
    
    # Setup GPU environment and get actual number of available GPUs
    actual_num_gpus = setup_gpu_environment(args.train_gpus)
    
    # Update num_gpus to match actual available GPUs
    args.num_gpus = actual_num_gpus
    
    # Initialize Weights & Biases if not disabled
    if not args.disable_wandb:
        run_name = f"qarray-{args.num_dots}dots-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            entity="rl_agents_for_tuning",
            project="RLModel",
            name=run_name,
            config={
                "num_dots": args.num_dots,
                "num_iterations": args.num_iterations,
                "num_gpus": args.num_gpus,
            }
        )
        memory_checkpoint_wandb("STARTUP", "Training script started")
    else:
        print("Wandb logging disabled")
        # Still setup basic memory logging for console output
        memory_logger = setup_memory_logging()
        memory_checkpoint(memory_logger, "STARTUP", "Training script started")
    
    # Set environment variables for Ray
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    # Add VoltageAgent to path for custom RLModule
    sys.path.append(str(swarm_dir / "VoltageAgent"))
    
    # Initialize Ray with runtime environment
    ray_config = {
        "include_dashboard": False,
        "runtime_env": {
            "working_dir": str(swarm_dir),
            "py_modules": [
                str(swarm_dir / "Environment"),
                str(swarm_dir / "VoltageAgent")
            ],
            "env_vars": {
                #"JAX_PLATFORM_NAME": "cpu",
                #"JAX_PLATFORMS": "cpu",
                #"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
                #"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "SWARM_PROJECT_ROOT": str(swarm_dir),
                #'NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR': "1"
            }
        }
    }

    policy_config = {
        "batch_mode": "complete_episodes",
        #"batch_mode": "truncate_episodes",
        "lstm_cell_size": 256
    }

    env_config = {
        "num_dots": args.num_dots
    }
    
    start_time = time.time()
    ray.init(**ray_config)
    ray_init_time = time.time() - start_time
    
    try:
        register_env("qarray_multiagent_env", create_env)
        env_instance = create_env(**env_config)

        rl_module_spec = create_rl_module_spec(env_instance, policy_config)

        config = (
            PPOConfig()
            .environment(
                env="qarray_multiagent_env", 
                env_config=env_config
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=["plunger_policy", "barrier_policy"],
                policies_to_train=["plunger_policy", "barrier_policy"]
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=40,
                rollout_fragment_length='auto',
                sample_timeout_s=180.0,
                num_gpus_per_env_runner=0.1, # 1/this is how many envs a single gpu runs
            )
            .training(
                train_batch_size=250,
                minibatch_size=50,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
                num_sgd_iter=4  # Fewer SGD iterations to speed up training
            )
            .resources(
                num_gpus=args.num_gpus # how many gpus the trainer uses
            )
        )

        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")
        
        start_time = time.time()
        algo = config.build_algo() # creates a PPO object
        build_time = time.time() - start_time
        
        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance
        
        print(f"\nStarting training for {args.num_iterations} iterations...\n")
        
        for i in range(args.num_iterations):
            iteration_start_time = time.time()
            
            try:
                result = algo.train()
                iteration_time = time.time() - iteration_start_time
                print(result)
                
                # Extract focused training metrics
                metrics = extract_training_metrics(result, iteration_time)
                
                # Log metrics to wandb or console
                if not args.disable_wandb:
                    log_training_metrics_wandb(metrics, i)
                    # Log memory usage every iteration for the first 5, then every 10
                    if i < 5 or i % 10 == 0:
                        memory_checkpoint_wandb(f"ITERATION_{i}_COMPLETE", 
                                              f"Iteration {i} completed in {iteration_time:.2f} seconds", step=i)
                else:
                    # Fallback logging to console and file
                    if i < 5 or i % 10 == 0:
                        memory_checkpoint(memory_logger, f"ITERATION_{i}_COMPLETE", 
                                        f"Iteration {i} completed in {iteration_time:.2f} seconds")
                
                # Console output for all modes
                print(f"Iteration {i:3d}: {metrics['summary']}")
                
            except Exception as e:
                error_msg = f"Error in iteration {i}: {str(e)}"
                raise
        
        # Save final checkpoint
        checkpoint_path = algo.save()

        print(f"\nTraining completed. Checkpoint saved to: {checkpoint_path}\n")
        
    finally:
        if ray.is_initialized():
            ray.shutdown()
            
        if not args.disable_wandb:
            wandb.finish()
            print("Wandb session finished")
        else:
            memory_logger.info("=" * 100)
            memory_logger.info("TRAINING SESSION COMPLETED")
            memory_logger.info("=" * 100)


if __name__ == "__main__":
    main()