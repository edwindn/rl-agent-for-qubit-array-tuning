#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.
"""
import os
import sys

# Suppress Ray warnings and verbose output
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

import argparse
import glob
import logging
import re
import yaml
from functools import partial

# Memory monitoring imports
import time
from pathlib import Path

import ray
import torch
import wandb
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.tune.registry import register_env

# Set logging level to reduce verbosity
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.WARNING)

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

from swarm.training.utils import (  # noqa: E402
    log_to_wandb,
    print_training_progress,
    setup_wandb_metrics,
    upload_checkpoint_artifact,
    policy_mapping_fn,
    cleanup_gif_files,
    process_and_log_gifs,
)

from swarm.voltage_model import create_rl_module_spec
from swarm.training.utils.custom_ppo_learner import PPOLearnerWithValueStats # for logging

def parse_config_overrides(unknown_args):
    """Parse config override arguments in the format --key.subkey value or --key=value (allows dynamically overriding settings when calling train.py)"""
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            # Handle both --key=value and --key value formats
            if '=' in arg:
                # Format: --key=value
                key_value = arg[2:]  # Remove '--' prefix
                key, value = key_value.split('=', 1)
                i += 1
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Format: --key value
                key = arg[2:]  # Remove '--' prefix
                value = unknown_args[i + 1]
                i += 2
            else:
                # Standalone flag or no value
                i += 1
                continue
            
            # Type conversion
            try:
                # Handle None as string
                if value.lower() == 'none':
                    value = None
                elif value.lower() in ('true', 'false'):
                    # Handle boolean
                    value = value.lower() == 'true'
                else:
                    # Try to convert to number (handles both int, float, and scientific notation)
                    try:
                        # First try float (handles scientific notation like 1e-05)
                        float_val = float(value)
                        # If it's a whole number, convert to int
                        if float_val.is_integer() and 'e' not in value.lower() and '.' not in value:
                            value = int(float_val)
                        else:
                            value = float_val
                    except ValueError:
                        # Keep as string if not a number
                        pass
            except (ValueError, AttributeError):
                pass  # Keep as string
                
            overrides[key] = value
        else:
            i += 1
    return overrides


def map_sweep_parameters(overrides):
    """Map sweep parameter names to config paths for wandb sweep compatibility."""
    # Mapping from sweep parameter names to config paths
    sweep_param_mapping = {
        # Core training parameters
        'minibatch_size': 'rl_config.training.minibatch_size',
        'num_epochs': 'rl_config.training.num_epochs',
        'lr': 'rl_config.training.lr',
        'gamma': 'rl_config.training.gamma',
        'lambda_': 'rl_config.training.lambda_',
        'clip_param': 'rl_config.training.clip_param',
        'entropy_coeff': 'rl_config.training.entropy_coeff',
        'vf_loss_coeff': 'rl_config.training.vf_loss_coeff',
        'kl_target': 'rl_config.training.kl_target',
        'grad_clip': 'rl_config.training.grad_clip',
        'grad_clip_by': 'rl_config.training.grad_clip_by',
        'train_batch_size': 'rl_config.training.train_batch_size',
        
        # Algorithm choice
        'algorithm': 'rl_config.algorithm',
        
        # Training control
        'num_iterations': 'defaults.num_iterations',
    }
    
    mapped_overrides = {}
    
    for key, value in overrides.items():
        if key in sweep_param_mapping:
            # Map sweep parameter to config path
            config_path = sweep_param_mapping[key]
            mapped_overrides[config_path] = value
            print(f"Mapped sweep parameter: {key} -> {config_path} = {value}")
        else:
            # Keep original key (might be a nested config path already)
            mapped_overrides[key] = value
            print(f"Direct config override: {key} = {value}")
    
    return mapped_overrides


def apply_config_overrides(config, overrides):
    """Apply config overrides using dot notation to nested dictionary."""
    # First map sweep parameters to config paths
    mapped_overrides = map_sweep_parameters(overrides)
    
    for key, value in mapped_overrides.items():
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
        print(f"Config override applied: {key} = {value}")
    
    return config


def fix_optimizer_betas_after_checkpoint_load(algo):
    """Fix optimizer beta parameters that may have been saved as tensors.

    When checkpoints are saved with LR schedules, optimizer parameters like betas
    can be stored as tensors. This causes errors when loading with certain optimizer
    configurations. This function converts any tensor betas back to scalar values.

    Args:
        algo: The RLlib Algorithm instance after restore_from_path() has been called
    """
    def fix_betas(learner):
        for name, optimizer in learner._named_optimizers.items():
            for param_group in optimizer.param_groups:
                if 'betas' in param_group:
                    betas = param_group['betas']
                    param_group['betas'] = (
                        float(betas[0]) if torch.is_tensor(betas[0]) else betas[0],
                        float(betas[1]) if torch.is_tensor(betas[1]) else betas[1]
                    )

    algo.learner_group.foreach_learner(fix_betas)


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the given directory.
    
    Args:
        checkpoint_dir (str or Path): Directory containing checkpoint folders
        
    Returns:
        tuple: (checkpoint_path, iteration_number) or (None, None) if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None, None
    
    # Find all iteration directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))
    
    if not iteration_dirs:
        return None, None
    
    # Extract iteration numbers and find the maximum
    max_iteration = 0
    latest_checkpoint = None
    
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        match = re.search(r'iteration_(\d+)', iteration_dir)
        if match:
            iteration_num = int(match.group(1))
            if iteration_num > max_iteration:
                max_iteration = iteration_num
                latest_checkpoint = iteration_dir
    
    return latest_checkpoint, max_iteration


def clean_checkpoint_folder(checkpoint_dir):
    """Clean checkpoint folder at start of training, keeping yaml files."""
    import shutil
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Remove all iteration_* directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))
    
    for iteration_dir in iteration_dirs:
        try:
            shutil.rmtree(iteration_dir)
        except Exception as e:
            print(f"Warning: Could not delete {iteration_dir}: {e}")


def delete_old_checkpoint_if_needed(checkpoint_dir):
    """Keep only the latest checkpoint, delete all older ones."""
    import shutil
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Find all iteration directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))
    
    if len(iteration_dirs) <= 1:
        return  # Keep at least 1 checkpoint
    
    # Extract iteration numbers and sort
    iteration_info = []
    for iteration_dir in iteration_dirs:
        match = re.search(r'iteration_(\d+)', iteration_dir)
        if match:
            iteration_num = int(match.group(1))
            iteration_info.append((iteration_num, iteration_dir))
    
    # Sort by iteration number (oldest first)
    iteration_info.sort(key=lambda x: x[0])
    
    # Delete all but the latest checkpoint
    for iteration_num, iteration_dir in iteration_info[:-1]:
        try:
            shutil.rmtree(iteration_dir)
        except Exception as e:
            print(f"Warning: Could not delete old checkpoint {iteration_dir}: {e}")




def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(description='Multi-agent RL training for quantum device tuning')
    
    parser.add_argument(
        '--load-checkpoint', 
        type=str, 
        help='Path to specific checkpoint directory to load'
    )
    
    parser.add_argument(
        '--resume-latest', 
        action='store_true',
        help='Resume training from the most recent checkpoint in the default checkpoints directory'
    )

    parser.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    # Parse known args to allow for dynamic config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides from remaining arguments
    config_overrides = parse_config_overrides(unknown)
    args.config_overrides = config_overrides
    
    return args



def create_env(config=None, gif_config=None, distance_data_dir=None):
    """Create multi-agent quantum environment with JAX safety."""
    import os
    import jax

    # Ensure JAX settings are applied in worker processes
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    assert gif_config is not None, "Gif config dict required to set up rollout visualisation"

    # Try to clear any existing JAX state
    try:
        # Force JAX to use a fresh backend in each worker
        jax.clear_backends()
    except:
        pass

    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper

    # Wrap in multi-agent wrapper (config unused but required by RLlib)
    # need return_voltage=True if we are using deltas + LSTM/Transformer
    # RLlib handles temporal sequences via ConnectorV2, not via environment
    return MultiAgentEnvWrapper(return_voltage=True, gif_config=gif_config, distance_data_dir=distance_data_dir)


def create_env_to_module_connector(env, spaces, device, use):
    """
    Creates module connector for action to memory handling.
    Note: do not modify the signature, ray expects arguments 0-2
    
    Args:
        env: The (vectorized) gym environment
        spaces: Dict with space info like {'__env__': ([obs_space, act_space]), '__env_single__': ([obs_space, act_space])}
        device: Torch device (can be None)
        use: Whether to use the custom connector or not
    """
    if use:
        from swarm.voltage_model.prev_action_handling import CustomPrevActionHandling
        return [CustomPrevActionHandling()]
    else:
        # Return empty list - let Ray handle everything with defaults
        return []


def load_config():
    """Load training configuration from YAML file."""
    config_path = Path(__file__).parent / "training_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function using Ray RLlib 2.49.0 API with wandb logging."""

    # Parse command line arguments
    args = parse_arguments()

    use_wandb = not args.disable_wandb
    
    config = load_config()
    
    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Create timestamped data folder if save_distance_data is enabled
    distance_data_dir = None
    if config['defaults']['save_distance_data']:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        distance_data_dir = Path(__file__).parent / "data" / timestamp
        distance_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDistance data will be saved to: {distance_data_dir}\n")

    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project']
        )
        # Note: We'll update wandb config with merged config later after env creation
        setup_wandb_metrics(config['wandb']['ema_period'])

    
    # Initialize Ray with runtime environment from config
    ray_config = {
        "include_dashboard": config['ray']['include_dashboard'],
        "log_to_driver": config['ray']['log_to_driver'],
        "logging_level": config['ray']['logging_level'],
        "runtime_env": {
            "working_dir": str(src_dir),
            "excludes": config['ray']['runtime_env']['excludes'],
            "env_vars": {
                **config['ray']['runtime_env']['env_vars'],
                "SWARM_PROJECT_ROOT": str(project_root),
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        gif_config = config["gif_config"]
        # Clean up any previous GIF capture lock files
        cleanup_gif_files(gif_config['save_dir'])

        create_env_fn = partial(create_env, gif_config=gif_config, distance_data_dir=distance_data_dir)
        register_env("qarray_multiagent_env", create_env_fn)
        env_instance = create_env_fn()

        # Extract environment config and merge with training config for wandb
        if use_wandb:
            env_config = env_instance.base_env.config
            
            # Create a copy of the training config and merge with env config
            import copy
            merged_config = copy.deepcopy(config)
            merged_config['env_config'] = env_config
            
            # Update wandb config with the merged config
            wandb.config.update(merged_config)
            
            # Save the merged config as a file artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                yaml.dump(merged_config, tmp_file, default_flow_style=False, sort_keys=False)
                merged_config_path = tmp_file.name
            
            # Log the complete merged config as an artifact
            merged_config_artifact = wandb.Artifact("full_training_config", type="config", metadata=merged_config)
            merged_config_artifact.add_file(merged_config_path, "full_training_config.yaml")
            wandb.log_artifact(merged_config_artifact)
            
            # Clean up temporary file
            os.unlink(merged_config_path)

        # Optionally update the rl module config to allow log_std clamping, shared log_std vector etc.
        rl_module_config = {
            "plunger_policy": {
                **config['neural_networks']['plunger_policy'],
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            },
            "barrier_policy": {
                **config['neural_networks']['barrier_policy'],
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            }
        }
        
        algo = config['rl_config']['algorithm'].lower()

        rl_module_spec = create_rl_module_spec(env_instance, algo=algo, config=rl_module_config)

        # Configure custom callbacks for logging to Wandb
        # log_images = config['wandb']['log_images']
        # custom_callbacks = partial(CustomCallbacks, log_images=log_images)

        # Specify algorithm-specific training parameters
        ppo_train_config = {
            "lr": config['rl_config']['training']['lr'],
            "gamma": config['rl_config']['training']['gamma'],
            "lambda_": config['rl_config']['training']['lambda_'],
            "clip_param": config['rl_config']['training']['clip_param'],
            "entropy_coeff": config['rl_config']['training']['entropy_coeff'],
            "vf_loss_coeff": config['rl_config']['training']['vf_loss_coeff'],
            "kl_target": config['rl_config']['training']['kl_target'],
        }

        sac_train_config = {
            "actor_lr": config['rl_config']['training']['actor_lr'],
            "critic_lr": config['rl_config']['training']['critic_lr'],
            "alpha_lr": config['rl_config']['training']['alpha_lr'],
            "twin_q": config['rl_config']['training']['twin_q'],
            "tau": config['rl_config']['training']['tau'],
            "initial_alpha": config['rl_config']['training']['initial_alpha'],
            "target_entropy": config['rl_config']['training']['target_entropy'],
            "n_step": config['rl_config']['training']['n_step'],
            "clip_actions": config['rl_config']['training']['clip_actions'],
            "target_network_update_freq": config['rl_config']['training']['target_network_update_freq'],
            "num_steps_sampled_before_learning_starts": config['rl_config']['training']['num_steps_sampled_before_learning_starts'],
            "replay_buffer_config": config['rl_config']['training']['replay_buffer_config'],
        }

        if algo == "ppo":
            algo_config_builder = PPOConfig
            train_config = ppo_train_config
        elif algo == "sac":
            algo_config_builder = SACConfig
            train_config = sac_train_config
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        # Handle voltage parsing to memory manually
        use_deltas = env_instance.base_env.use_deltas
        memory_layer = config['neural_networks']['plunger_policy']['backbone'].get('memory_layer')
        has_lstm = memory_layer == 'lstm'
        has_transformer = memory_layer == 'transformer'

        # Determine if we need frame stacking for temporal models (transformer)
        use_frame_stacking = has_transformer
        num_frames = 1
        if use_frame_stacking:
            num_frames = config['neural_networks']['plunger_policy']['backbone']['transformer']['max_seq_len']
            print(f"\n[Frame Stacking] Enabled with {num_frames} frames for transformer\n")

        # Build env-to-module connector
        # Note: We prioritize frame stacking over custom LSTM connector when both are applicable
        if use_frame_stacking:
            env_to_module_connector = lambda env: FrameStackingEnvToModule(
                num_frames=num_frames,
                multi_agent=True
            )
        elif use_deltas and has_lstm:
            env_to_module_connector = partial(create_env_to_module_connector, use=True)
        else:
            env_to_module_connector = None

        # Build learner connector for frame stacking
        learner_connector = None
        if use_frame_stacking:
            learner_connector = lambda obs_space, act_space: FrameStackingLearner(
                num_frames=num_frames,
                multi_agent=True
            )

        algo_config = (
            algo_config_builder()
            .environment(
                env="qarray_multiagent_env",
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=config['rl_config']['multi_agent']['policies'],
                policies_to_train=config['rl_config']['multi_agent']['policies_to_train'],
                count_steps_by=config['rl_config']['multi_agent']['count_steps_by'],
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=config['rl_config']['env_runners']['num_env_runners'],
                rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=config['rl_config']['env_runners']['num_gpus_per_env_runner'],
                env_to_module_connector=env_to_module_connector,
                add_default_connectors_to_env_to_module_pipeline=True,  # Let Ray handle defaults
            )
            .learners(
                num_learners=config['rl_config']['learners']['num_learners'],
                num_gpus_per_learner=config['rl_config']['learners']['num_gpus_per_learner'],
            )
            .training(
                train_batch_size=config['rl_config']['training']['train_batch_size'],
                minibatch_size=config['rl_config']['training']['minibatch_size'],
                num_epochs=config['rl_config']['training']['num_epochs'],
                grad_clip=config['rl_config']['training']['grad_clip'],
                grad_clip_by=config['rl_config']['training']['grad_clip_by'],
                learner_class=PPOLearnerWithValueStats if algo == "ppo" else None,
                learner_connector=learner_connector,
                **train_config,
            )
            .resources(num_gpus=config['resources']['num_gpus'])
            # .callbacks([custom_callbacks] if use_wandb else [])
        )

        # Build the algorithm
        print(f"\nBuilding {algo} algorithm...\n")

        algo = algo_config.build()


        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance

        # Handle checkpoint loading if requested
        start_iteration = 0
        checkpoint_loaded = False
        
        if args.load_checkpoint:
            # Load specific checkpoint
            checkpoint_path = Path(args.load_checkpoint)
            if checkpoint_path.exists():
                print(f"\nLoading checkpoint from: {checkpoint_path}")
                try:
                    algo.restore_from_path(str(checkpoint_path))
                    fix_optimizer_betas_after_checkpoint_load(algo)

                    # Extract iteration number from path
                    match = re.search(r'iteration_(\d+)', str(checkpoint_path))
                    if match:
                        start_iteration = int(match.group(1))
                        checkpoint_loaded = True
                        print(f"Checkpoint loaded successfully. Resuming from iteration {start_iteration + 1}")
                    else:
                        print("Warning: Could not determine iteration number from checkpoint path")
                        
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Continuing with fresh training...")
            else:
                print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
                print("Continuing with fresh training...")
                
        elif args.resume_latest:
            # Find and load most recent checkpoint
            checkpoint_dir = Path(config['checkpoints']['save_dir'])
            latest_checkpoint, latest_iteration = find_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                print(f"\nFound latest checkpoint: {latest_checkpoint} (iteration {latest_iteration})")
                try:
                    algo.restore_from_path(str(Path(latest_checkpoint).absolute()))
                    fix_optimizer_betas_after_checkpoint_load(algo)

                    start_iteration = latest_iteration
                    checkpoint_loaded = True
                    print(f"Latest checkpoint loaded successfully. Resuming from iteration {start_iteration + 1}")
                except Exception as e:
                    print(f"Error loading latest checkpoint: {e}")
                    print("Continuing with fresh training...")
            else:
                print("\nNo checkpoints found in checkpoint directory.")
                print("Starting fresh training...")
        
        if not checkpoint_loaded:
            print("\nStarting fresh training from iteration 1...")
        else:
            print(f"Training will continue from iteration {start_iteration + 1} to {config['defaults']['num_iterations']}")
            # Validate that we haven't already completed training
            if start_iteration >= config['defaults']['num_iterations']:
                print(f"Training already completed! Loaded checkpoint is at iteration {start_iteration}, "
                      f"but max iterations is {config['defaults']['num_iterations']}.")
                return

        # Save training config to checkpoint directory for easy reference
        checkpoint_base_dir = Path(config['checkpoints']['save_dir'])
        checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old checkpoints if delete_old_checkpoints is enabled and starting fresh training
        if config['defaults']['delete_old_checkpoints'] and not checkpoint_loaded:
            clean_checkpoint_folder(checkpoint_base_dir)
        
        config_save_path = checkpoint_base_dir / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Also save environment config to checkpoint directory for inference
        env_config_src = Path(__file__).parent.parent / "environment" / "env_config.yaml"
        if env_config_src.exists():
            env_config_dst = checkpoint_base_dir / "env_config.yaml"
            import shutil
            shutil.copy2(env_config_src, env_config_dst)
            print(f"Environment config saved to: {env_config_dst}")

        remaining_iterations = config['defaults']['num_iterations'] - start_iteration
        print(f"\nStarting training for {remaining_iterations} iterations (from iteration {start_iteration + 1} to {config['defaults']['num_iterations']})...\n")
        print(f"Training config saved to: {config_save_path}\n")


        training_start_time = time.time()
        best_reward = float("-inf")  # Track best performance for artifact upload

        for i in range(start_iteration, config['defaults']['num_iterations']):
            result = algo.train()

            # Clean console output and wandb logging
            print_training_progress(result, i, training_start_time)

            # Log metrics to wandb (EMA is calculated automatically in metrics_logger)
            log_to_wandb(result, i, distance_data_dir)

            # Process and log GIFs if enabled
            if config['gif_config']['enabled'] and use_wandb:
                process_and_log_gifs(i + 1, config, use_wandb)

            # Save checkpoint using modern RLlib API
            local_checkpoint_dir = Path(config['checkpoints']['save_dir']) / f"iteration_{i+1}"
            local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = algo.save_to_path(str(local_checkpoint_dir.absolute()))

            # Delete old checkpoints if enabled (keep only latest)
            if config['defaults']['delete_old_checkpoints']:
                delete_old_checkpoint_if_needed(Path(config['checkpoints']['save_dir']))

            # Upload checkpoint as wandb artifact if performance improved
            if config['checkpoints']['upload_best_only']:
                current_reward = result.get("env_runners", {}).get(
                    "episode_return_mean", float("-inf")
                )
                if current_reward is not None and current_reward > best_reward:
                    best_reward = current_reward
                    upload_checkpoint_artifact(checkpoint_path, i + 1, current_reward)
            else:
                upload_checkpoint_artifact(checkpoint_path, i + 1, 0.0)

            print(f"\nIteration {i+1} completed. Checkpoint saved to: {checkpoint_path}\n")

    finally:
        if ray.is_initialized():
            ray.shutdown()

        if use_wandb:
            wandb.finish()
            print("Wandb session finished")


if __name__ == "__main__":
    main()
