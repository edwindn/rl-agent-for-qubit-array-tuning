#!/usr/bin/env python3
"""
Main training script for multi-agent quantum device RL.
Orchestrates the entire training process with Ray RLlib and W&B logging.
"""

import os
import sys
import argparse
from pathlib import Path
import ray
from typing import Optional
import psutil
import gc
import torch
import logging
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.config_loader import ConfigLoader, load_config_from_file
from utils.wandb_logger import setup_wandb_logging
from utils.policy_mapping import get_policy_mapping_fn, create_rl_module_spec

# Add VoltageAgent to path for imports
sys.path.append(str(current_dir.parent))
from VoltageAgent import get_trainer_class

# Setup memory logging
def setup_memory_logger():
    """Setup memory usage logger to write to file."""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"memory_debug_{timestamp}.log"
    
    logger = logging.getLogger("memory_debug")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    print(f"Memory debug logging to: {log_file}")
    return logger

# Global logger instance
memory_logger = None

def log_memory_usage(stage: str, prefix: str = "MEMORY"):
    """Log current memory usage for debugging out of memory issues."""
    global memory_logger
    if memory_logger is None:
        memory_logger = setup_memory_logger()
    
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()
    
    # System memory info
    sys_mem = psutil.virtual_memory()
    
    # GPU memory if available
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_info = f" | GPU: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved, {gpu_mem_total:.2f}GB total"
    
    log_message = (f"{prefix} [{stage}] Process: {mem_info.rss / 1024**3:.2f}GB ({mem_percent:.1f}%) | "
                   f"System: {sys_mem.used / 1024**3:.2f}GB/{sys_mem.total / 1024**3:.2f}GB ({sys_mem.percent:.1f}%){gpu_info}")
    
    memory_logger.info(log_message)
    # Also print to console for immediate feedback
    print(log_message)


def force_garbage_collect(stage: str):
    """Force garbage collection and log memory freed."""
    global memory_logger
    if memory_logger is None:
        memory_logger = setup_memory_logger()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    before_mem = psutil.Process().memory_info().rss / 1024**3
    collected = gc.collect()
    after_mem = psutil.Process().memory_info().rss / 1024**3
    freed = before_mem - after_mem
    
    log_message = f"GC [{stage}] Collected {collected} objects, freed {freed:.3f}GB memory"
    memory_logger.info(log_message)
    print(log_message)


def setup_environment():
    """Setup environment variables and paths."""
    # Set matplotlib backend to avoid GUI issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Force JAX to use CPU only to avoid CUDA initialization issues
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'True'
    
    # Disable Ray dashboard completely to avoid pydantic compatibility issues
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    os.environ['RAY_DISABLE_RUNTIME_ENV_LOG_TO_DRIVER'] = '1'
    os.environ['RAY_DEDUP_LOGS'] = '0'
    
    # Add Swarm directory to Python path to import environment
    swarm_path = current_dir.parent
    if swarm_path.exists():
        sys.path.append(str(swarm_path))
    else:
        print(f"Warning: Swarm directory not found at {swarm_path}")


def import_environment():
    """Import the multi-agent quantum device environment wrapper."""
    try:
        # Add Swarm directory to path for package imports
        swarm_dir = current_dir.parent
        if str(swarm_dir) not in sys.path:
            sys.path.insert(0, str(swarm_dir))
        
        # Import multi-agent wrapper and base environment
        from Environment.multi_agent_wrapper import MultiAgentQuantumWrapper
        from Environment.env import QuantumDeviceEnv
        
        # Return factory function that creates wrapped environment
        def create_wrapped_env(num_quantum_dots=8, **kwargs):
            # Pass num_quantum_dots to base environment through config override
            base_env = QuantumDeviceEnv(training=True, **kwargs)
            # Override the num_dots configuration
            base_env.config['simulator']['num_dots'] = num_quantum_dots
            # Reinitialize environment with new configuration
            base_env._initialize_environment()
            return MultiAgentQuantumWrapper(base_env, num_quantum_dots=num_quantum_dots)
        
        return create_wrapped_env
    except Exception as e:
        print(f"Failed to import MultiAgentQuantumWrapper: {e}")
        print("Please ensure the environment is properly set up in Swarm/Environment/")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(current_dir / "configs" / "config.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num-quantum-dots", 
        type=int, 
        default=8,
        help="Number of quantum dots (N)"
    )
    
    parser.add_argument(
        "--num-iterations", 
        type=int, 
        default=None,
        help="Number of training iterations (overrides config)"
    )
    
    parser.add_argument(
        "--resume-from", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None,
        help="Override experiment name from config"
    )
    
    parser.add_argument(
        "--disable-wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--ray-address", 
        type=str, 
        default=None,
        help="Ray cluster address (for distributed training)"
    )
    
    parser.add_argument(
        "--test-env", 
        action="store_true",
        help="Test environment setup and exit"
    )
    
    return parser.parse_args()


def test_environment_setup(env_instance, num_quantum_dots: int = 8):
    """Test multi-agent environment setup and print information."""
    print("Testing multi-agent environment setup...")
    log_memory_usage("env_test_start")
    
    try:
        print(f"✓ Environment created successfully")
        print(f"  Agent IDs: {env_instance.get_agent_ids()}")
        print(f"  Number of agents: {len(env_instance.get_agent_ids())}")
        
        # Test reset
        log_memory_usage("before_env_reset")
        obs, info = env_instance.reset()
        log_memory_usage("after_env_reset")
        print(f"✓ Environment reset successful")
        print(f"  Got observations for {len(obs)} agents")
        
        # Check observation structure for a few agents
        sample_agents = env_instance.get_agent_ids()[:3]  # Check first 3 agents
        for agent_id in sample_agents:
            agent_obs = obs[agent_id]
            print(f"  {agent_id}:")
            print(f"    Image shape: {agent_obs['image'].shape}")
            print(f"    Voltage shape: {agent_obs['voltage'].shape}")
        
        # Test step with sample actions
        log_memory_usage("before_env_step")
        actions = {}
        for agent_id in env_instance.get_agent_ids():
            actions[agent_id] = env_instance.action_space[agent_id].sample()
        
        obs, rewards, terminated, truncated, info = env_instance.step(actions)
        log_memory_usage("after_env_step")
        print(f"✓ Environment step successful")
        print(f"  Got rewards for {len(rewards)} agents")
        print(f"  Sample rewards: {dict(list(rewards.items())[:3])}")
        print(f"  Termination status: {terminated[env_instance.get_agent_ids()[0]]}")
        
        env_instance.close()
        log_memory_usage("after_env_close")
        print("✓ Multi-agent environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training function."""
    log_memory_usage("startup")
    
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    log_memory_usage("environment_setup")
    
    # Import environment factory function
    env_factory = import_environment()
    log_memory_usage("environment_import")
    
    # Test environment if requested
    if args.test_env:
        log_memory_usage("before_test_env_creation")
        # Create test environment instance
        test_env = env_factory(num_quantum_dots=args.num_quantum_dots)
        log_memory_usage("after_test_env_creation")
        success = test_environment_setup(test_env, args.num_quantum_dots)
        force_garbage_collect("after_env_test")
        sys.exit(0 if success else 1)
    
    # Load configuration
    try:
        config = load_config_from_file(args.config)
        log_memory_usage("config_loaded")
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    
    if args.disable_wandb:
        config["logging"]["wandb"]["enabled"] = False
    
    # Initialize Ray
    log_memory_usage("before_ray_init")
    # Setup Ray runtime environment
    swarm_dir = current_dir.parent
    ray_config = {
        "num_gpus": config["ray"]["num_gpus"],
        "object_store_memory": config["ray"]["object_store_memory"],
        "include_dashboard": False,  # Disable dashboard to avoid pydantic compatibility issues
        "_node_ip_address": "127.0.0.1",  # Force local IP to avoid network issues
        "dashboard_host": "127.0.0.1",
        "dashboard_port": None,  # Disable dashboard port
        "_temp_dir": "/tmp/ray_temp",  # Set explicit temp dir
        "runtime_env": {
            "working_dir": str(swarm_dir),
            "py_modules": [
                str(swarm_dir / "Environment"),
                str(swarm_dir / "CapacitanceModel")
            ]
        }
    }
    
    if args.ray_address:
        ray.init(address=args.ray_address)
        print(f"Connected to Ray cluster at: {args.ray_address}")
    else:
        ray.init(**ray_config)
        print("Initialized local Ray cluster")
    
    log_memory_usage("after_ray_init")
    force_garbage_collect("after_ray_init")
    
    try:
        # Setup W&B logging
        wandb_logger, callback = setup_wandb_logging(config)
        log_memory_usage("wandb_initialized")
        print("W&B logging initialized")
        
        # Get trainer class and create instance
        trainer_type = config.get("trainer_type", "recurrent_ppo") # use recurrent by default
        trainer_class = get_trainer_class(trainer_type)
        log_memory_usage("before_trainer_creation")
        trainer = trainer_class(config, env_factory)
        log_memory_usage("after_trainer_creation")
        print(f"{trainer_type.upper()} trainer created")
        
        # Create environment instance for policy setup
        log_memory_usage("before_policy_env_creation")
        env_instance = env_factory(num_quantum_dots=args.num_quantum_dots)
        log_memory_usage("after_policy_env_creation")

        # Setup RLModule specifications and mapping functions
        log_memory_usage("before_rlmodule_setup")
        rl_module_spec = create_rl_module_spec(env_instance)
        policy_mapping_fn = get_policy_mapping_fn(args.num_quantum_dots)
        log_memory_usage("after_rlmodule_creation")
        
        # Setup training configuration
        ppo_config = trainer.setup_training(
            rl_module_spec=rl_module_spec,
            policy_mapping_fn=policy_mapping_fn,
            callback_class=callback,
            num_quantum_dots=args.num_quantum_dots
        )
        log_memory_usage("after_training_setup")
        force_garbage_collect("after_training_setup")
        print(f"Training configuration setup for {args.num_quantum_dots} quantum dots")
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            # Note: Checkpoint resuming would be implemented based on specific needs
        
        # Start training
        log_memory_usage("before_training_start")
        print("Starting training...")
        print(f"Configuration: {config['experiment']['name']}")
        print(f"Total workers: {config['ray']['num_workers']}")
        print(f"GPUs: {config['ray']['num_gpus']}")
        print(f"Stopping criteria: {config['stopping_criteria']}")
        
        algorithm = trainer.train(args.num_iterations)
        log_memory_usage("after_training_complete")
        
        # Save final checkpoint
        final_checkpoint = algorithm.save()
        log_memory_usage("after_checkpoint_save")
        print(f"Training completed. Final checkpoint saved to: {final_checkpoint}")
        
        # Cleanup
        wandb_logger.finish()
        trainer.cleanup()
        log_memory_usage("after_cleanup")
        force_garbage_collect("final_cleanup")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'wandb_logger' in locals():
            wandb_logger.finish()
        if 'trainer' in locals():
            trainer.cleanup()
        sys.exit(1)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if 'wandb_logger' in locals():
            wandb_logger.finish()
        if 'trainer' in locals():
            trainer.cleanup()
        sys.exit(1)
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main() 