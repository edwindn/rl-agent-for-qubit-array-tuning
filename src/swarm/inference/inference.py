#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.

example: python inference.py --load-checkpoint ../training/checkpoints/run_361 --load-configs
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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
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
    print_training_progress,
    policy_mapping_fn,
    cleanup_gif_files,
    CustomFrameStackingEnvToModule,
    CustomFrameStackingLearner,
)

from swarm.training.train_utils import (  # noqa: E402
    parse_config_overrides,
    map_sweep_parameters,
    apply_config_overrides,
    fix_optimizer_betas_after_checkpoint_load,
    find_latest_checkpoint,
    clean_checkpoint_folder,
    delete_old_checkpoint_if_needed,
    create_env_to_module_connector,
)

from swarm.voltage_model import create_rl_module_spec
from swarm.training.utils.custom_ppo_learner import PPOLearnerWithValueStats # for logging

# Import local inference utilities - need to add inference dir to path
inference_dir = Path(__file__).parent
sys.path.insert(0, str(inference_dir))
from utils import save_scans_to_iteration_folder, save_distance_plots  # noqa: E402


def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(description='Multi-agent RL training for quantum device tuning')
    
    parser.add_argument(
        '--load-checkpoint', 
        type=str, 
        help='Path to specific checkpoint directory to load'
    )

    parser.add_argument(
        '--num-iterations',
        type=int,
        default=4,
        help='Number of inference iterations to run'
    )

    parser.add_argument(
        '--disable-cleanup',
        action='store_true',
        help='Disable cleanup of temporary folders (distance_data and gif_captures) after inference'
    )

    parser.add_argument(
        '--load-configs',
        action='store_true',
        help='Load training_config.yaml and env_config.yaml from checkpoint directory instead of default config files'
    )

    parser.add_argument(
        '--sample',
        action='store_true',
        help='Sample from action distribution instead of using deterministic (mean) actions'
    )

    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect distance data by preserving the distance_data folder after inference and running 100 rollouts by default'
    )

    # Parse known args to allow for dynamic config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides from remaining arguments
    config_overrides = parse_config_overrides(unknown)
    args.config_overrides = config_overrides
    
    return args


def create_env(config=None, gif_config=None, distance_data_dir=None, env_config_path=None, scan_save_dir="inference_scans"):
    """Create multi-agent quantum environment with JAX safety."""
    import os
    import jax

    # Ensure JAX settings are applied in worker processes
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
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

    from swarm.environment.scan_saving_wrapper import ScanSavingWrapper

    # Wrap in scan-saving wrapper (which inherits from MultiAgentEnvWrapper)
    # need return_voltage=True if we are using deltas + LSTM/Transformer
    # store_history=True when using transformer to maintain observation history
    # env_config_path is passed to MultiAgentEnvWrapper which passes it to QuantumDeviceEnv
    return ScanSavingWrapper(
        return_voltage=True,
        gif_config=gif_config,
        distance_data_dir=distance_data_dir,
        env_config_path=env_config_path,
        scan_save_dir=scan_save_dir,
        scan_save_enabled=scan_save_dir is not None,
    )


def load_config(checkpoint_path=None):
    """Load training configuration from YAML file."""
    if checkpoint_path:
        # Load from checkpoint directory
        config_path = Path(checkpoint_path) / "training_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Must provide config files within checkpoint: training_config.yaml not found in {checkpoint_path}")
    else:
        # Load from training directory since we're in inference folder
        config_path = Path(__file__).parent.parent / "training" / "training_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_env_config(checkpoint_path=None):
    """Load environment configuration from YAML file."""
    if checkpoint_path:
        # Load from checkpoint directory
        config_path = Path(checkpoint_path) / "env_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Must provide config files within checkpoint: env_config.yaml not found in {checkpoint_path}")
    else:
        config_path = Path(__file__).parent.parent / "environment" / "env_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_dict_keys_recursively(override_dict, reference_dict, path=""):
    """
    Recursively validate that all keys in override_dict exist in reference_dict.

    Args:
        override_dict: Dictionary with override values
        reference_dict: Dictionary with reference structure
        path: Current path in nested structure (for error messages)

    Raises:
        ValueError: If any key in override_dict doesn't exist in reference_dict
    """
    for key, value in override_dict.items():
        current_path = f"{path}.{key}" if path else key

        if key not in reference_dict:
            raise ValueError(f"Invalid override key: '{current_path}' not found in reference config")

        # If both are dicts, recurse
        if isinstance(value, dict) and isinstance(reference_dict[key], dict):
            validate_dict_keys_recursively(value, reference_dict[key], current_path)


def validate_config_overrides(overrides, env_config, train_config):
    """
    Validate that config_overrides.yaml has valid structure matching original configs.

    Args:
        overrides: Dictionary loaded from config_overrides.yaml
        env_config: Dictionary loaded from env_config.yaml
        train_config: Dictionary loaded from training_config.yaml

    Raises:
        ValueError: If structure doesn't match
    """
    if not isinstance(overrides, dict):
        raise ValueError("config_overrides.yaml must contain a dictionary at root level")

    # Validate env overrides
    if 'env' in overrides and overrides['env'] is not None:
        if not isinstance(overrides['env'], dict):
            raise ValueError("'env' section in config_overrides.yaml must be a dictionary")
        validate_dict_keys_recursively(overrides['env'], env_config, path="env")

    # Validate train overrides
    if 'train' in overrides and overrides['train'] is not None:
        if not isinstance(overrides['train'], dict):
            raise ValueError("'train' section in config_overrides.yaml must be a dictionary")
        validate_dict_keys_recursively(overrides['train'], train_config, path="train")


def apply_overrides_recursively(base_dict, override_dict):
    """
    Recursively apply override values to base dictionary (modifies in-place).

    Args:
        base_dict: Dictionary to modify
        override_dict: Dictionary with override values
    """
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            # Recurse into nested dicts
            apply_overrides_recursively(base_dict[key], value)
        else:
            # Override the value
            base_dict[key] = value


def main():
    """Main inference function - single iteration with local saving."""

    # Parse command line arguments
    args = parse_arguments()

    # Determine checkpoint path for config loading if --load-configs is set
    checkpoint_path = args.load_checkpoint if args.load_configs else None

    config = load_config(checkpoint_path)

    # Override num_iterations based on command line arg
    # If --collect-data is set, use 100 rollouts unless explicitly overridden
    if args.collect_data and args.num_iterations == 4:  # 4 is the default
        config['defaults']['num_iterations'] = 100
    else:
        config['defaults']['num_iterations'] = args.num_iterations

    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Load config_overrides.yaml if it exists
    config_overrides_path = Path(__file__).parent / "config_overrides.yaml"
    temp_env_config_path = None

    if config_overrides_path.exists():
        print(f"\nLoading config overrides from: {config_overrides_path}")
        with open(config_overrides_path, 'r') as f:
            config_overrides = yaml.safe_load(f)

        # Load env_config for validation (will be loaded again later)
        env_config = load_env_config(checkpoint_path)

        # Validate overrides
        try:
            validate_config_overrides(config_overrides, env_config, config)
            print("Config overrides validated successfully")
        except ValueError as e:
            raise ValueError(f"Invalid config_overrides.yaml: {e}") from e

        # Apply env overrides to env_config
        if 'env' in config_overrides and config_overrides['env'] is not None:
            print("\nApplying environment config overrides...")
            apply_overrides_recursively(env_config, config_overrides['env'])

            # Write the overridden env_config to a temporary file
            import tempfile
            temp_env_config_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False, dir='/tmp'
            )
            yaml.dump(env_config, temp_env_config_file, default_flow_style=False)
            temp_env_config_path = temp_env_config_file.name
            temp_env_config_file.close()
            print(f"Written overridden env config to: {temp_env_config_path}")

        # Apply train overrides
        if 'train' in config_overrides and config_overrides['train'] is not None:
            print("\nApplying training config overrides...")
            apply_overrides_recursively(config, config_overrides['train'])
            print("Training config overrides applied")
    else:
        config_overrides = None

    # Save distance data to current directory (use absolute path for Ray workers)
    distance_data_dir = Path(__file__).parent.resolve() / "distance_data"
    distance_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDistance data will be saved to: {distance_data_dir}\n")

    # Create scan save directory (use absolute path for Ray workers)
    scan_save_dir = Path(__file__).parent.resolve() / "scan_captures"
    scan_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scan images will be saved to: {scan_save_dir}\n")

    
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
                "SWARM_PROJECT_ROOT": str(swarm_package_dir),
                "JAX_PLATFORMS": "cuda",
                "JAX_PLATFORM_NAME": "cuda",
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        gif_config = config["gif_config"]
        # Override gif save dir to local inference directory
        gif_config['save_dir'] = str(Path(__file__).parent / "gif_captures")
        # Override fps to 2 frames per second for faster playback
        gif_config['fps'] = 2

        # Check if plunger policy uses transformer memory layer
        use_transformer = config['neural_networks']['plunger_policy']['backbone']['memory_layer'] == 'transformer'

        create_env_fn = partial(
            create_env,
            gif_config=gif_config,
            distance_data_dir=distance_data_dir,
            env_config_path=temp_env_config_path,
            scan_save_dir=str(scan_save_dir)
        )
        register_env("qarray_multiagent_env", create_env_fn)

        # Load env config for driver process (workers will load from temp file if it exists)
        if temp_env_config_path:
            # Load the overridden config that was written to temp file
            with open(temp_env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
        else:
            # No overrides, load default
            env_config = load_env_config(checkpoint_path)

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

        rl_module_spec = create_rl_module_spec(env_config, algo=algo, config=rl_module_config)

        # Filter training parameters based on algorithm
        # PPO-specific parameters that should NOT be passed to SAC
        ppo_only_params = {'lr', 'lambda_', 'clip_param', 'entropy_coeff', 'vf_loss_coeff', 'kl_target', 'num_epochs', 'minibatch_size', 'train_batch_size'}
        # SAC-specific parameters that should NOT be passed to PPO
        sac_only_params = {'actor_lr', 'critic_lr', 'alpha_lr', 'twin_q', 'tau', 'initial_alpha', 'target_entropy', 'n_step',
                          'clip_actions', 'target_network_update_freq', 'num_steps_sampled_before_learning_starts', 'replay_buffer_config',
                          'reward_scale'}

        training_params = config['rl_config']['training'].copy()

        if algo == 'sac':
            # Remove PPO-only parameters when using SAC
            for param in ppo_only_params:
                training_params.pop(param, None)
            algo_config_builder = SACConfig
        elif algo == 'ppo':
            # Remove SAC-only parameters when using PPO
            for param in sac_only_params:
                training_params.pop(param, None)
            algo_config_builder = PPOConfig
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: ppo, sac")

        # Handle voltage parsing to memory manually
        use_deltas = env_config['simulator']['use_deltas']
        memory_layer = config['neural_networks']['plunger_policy']['backbone'].get('memory_layer')
        has_lstm = memory_layer == 'lstm'
        env_to_module_connector = partial(create_env_to_module_connector, use=use_deltas and has_lstm)

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
                num_env_runners=1,  # Use single worker for inference
                rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=0.95,  # Use 0.95 GPU to force separate physical device from learner
                env_to_module_connector=env_to_module_connector,
                add_default_connectors_to_env_to_module_pipeline=True,  # Let Ray handle defaults
            )
            .learners(
                num_learners=1,
                num_gpus_per_learner=0.95  # Use 0.95 GPU to force separate physical device from env runner
            )
            .training(
                # Pass filtered training params based on algorithm
                **training_params,
                learner_class=PPOLearnerWithValueStats if algo == "ppo" else None,
            )
            .evaluation(
                evaluation_num_env_runners=1,
                evaluation_duration=1,  # Run 1 episode per evaluation
                evaluation_duration_unit="episodes",
                evaluation_sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],  # Use same timeout as env_runners
                evaluation_config={
                    "explore": args.sample,  # Use --sample flag to control exploration
                },
            )
            # .callbacks([custom_callbacks] if use_wandb else [])
        )

        # Build the algorithm
        print(f"\nBuilding {algo} algorithm...\n")

        algo = algo_config.build()


        # Handle checkpoint loading if requested
        start_iteration = 0
        checkpoint_loaded = False
        
        if args.load_checkpoint:
            # Load specific checkpoint
            checkpoint_path = Path(args.load_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

            print(f"\nLoading checkpoint from: {checkpoint_path}")
            try:
                algo.restore_from_path(str(checkpoint_path.absolute()))
                fix_optimizer_betas_after_checkpoint_load(algo)

                # Extract iteration number from path
                match = re.search(r'iteration_(\d+)', str(checkpoint_path))
                if match:
                    start_iteration = int(match.group(1))
                    checkpoint_loaded = True
                    print(f"Checkpoint loaded successfully. Resuming from iteration {start_iteration + 1}")
                else:
                    print("Warning: Could not determine iteration number from checkpoint path")
                    start_iteration = 0
                    checkpoint_loaded = True
                    print(f"Checkpoint loaded successfully. Starting from iteration 1")

            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint: {e}") from e
        
        if not checkpoint_loaded:
            print("\nNo checkpoint loaded - starting fresh...\n")
        else:
            print(f"\nCheckpoint loaded successfully. Running inference...\n")


        inference_start_time = time.time()

        for i in range(args.num_iterations):
            print(f"\nRunning inference rollout {i+1}/{args.num_iterations}...")

            # Use Ray's evaluate method which runs 1 episode without training
            result = algo.evaluate()

            # Extract metrics from evaluation result
            eval_metrics = result.get('evaluation', {})
            avg_reward = eval_metrics.get('env_runners', {}).get('episode_reward_mean', 0)

            print(f"Rollout {i+1} reward: {avg_reward:.2f}")

            # Wait briefly for async worker processes to finish writing data to disk
            time.sleep(0.5)

            # Save scans to iteration folder if enabled (skip when collecting data)
            if config['gif_config']['enabled'] and not args.collect_data:
                save_scans_to_iteration_folder(i + 1, config)

            # Save distance plots (skip when collecting data)
            if not args.collect_data:
                save_distance_plots(distance_data_dir, iteration=i + 1)

            print(f"\nRollout {i+1}/{args.num_iterations} completed.\n")

        print(f"\nInference completed. All outputs saved to {Path(__file__).parent}\n")

        # Clean up temporary distance data directory
        # Skip cleanup if --collect-data or --disable-cleanup is set
        if not args.disable_cleanup and not args.collect_data:
            import shutil
            if distance_data_dir.exists():
                print(f"Cleaning up temporary distance data directory: {distance_data_dir}")
                shutil.rmtree(distance_data_dir, ignore_errors=True)
        else:
            print(f"Cleanup disabled - distance data preserved at: {distance_data_dir}")

    finally:
        # Clean up algorithm before shutting down Ray
        try:
            if 'algo' in locals():
                print("Stopping algorithm...")
                algo.stop()
        except Exception as e:
            print(f"Warning: Error stopping algorithm: {e}")

        if ray.is_initialized():
            print("Shutting down Ray...")
            ray.shutdown()

        # Clean up temporary env config file AFTER Ray shutdown
        if 'temp_env_config_path' in locals() and temp_env_config_path:
            try:
                import os
                os.remove(temp_env_config_path)
                print(f"Cleaned up temporary config file: {temp_env_config_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary config file: {e}")

        print("Inference completed")


if __name__ == "__main__":
    main()