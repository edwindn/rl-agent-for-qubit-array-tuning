#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.
"""
import os
import sys
import warnings

# Suppress deprecation/future warnings on driver (before Ray/RLlib imports)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0" #not neccessary but suppresses warning, opts into future behaviour of Ray accel

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
from ray.tune.registry import register_env

# Set logging level to reduce verbosity
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

# Apply patch for Ray 2.49.0 replay buffer bug with complex observations (DQN/SAC)
# See: https://github.com/ray-project/ray/pull/57017
# Remove after upgrading to Ray >= 2.51.0
from qadapt.training.patches.ray_episode_patch import apply_patch  # noqa: E402
apply_patch()

from utils.metrics_logger import (  # noqa: E402
    log_to_wandb,
    print_training_progress,
    setup_wandb_metrics,
    upload_checkpoint_artifact,
)

from qadapt.training.train_utils import (  # noqa: E402
    parse_config_overrides,
    map_sweep_parameters,
    apply_config_overrides,
    fix_optimizer_betas_after_checkpoint_load,
    find_latest_checkpoint,
    clean_checkpoint_folder,
    delete_old_checkpoint_if_needed,
    create_env_to_module_connector,
)

from utils.factory import create_rl_module_spec
from utils.custom_ppo_learner import PPOLearnerWithValueStats  # for logging
from qadapt.training.utils.custom_sac_learner import SACLearnerWithRewardScaling  # for reward scaling


def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(description='Single-agent RL training for quantum device tuning')
    
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

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training config YAML file (default: training_config.yaml)'
    )

    parser.add_argument(
        '--load-configs',
        action='store_true',
        help='Load training_config.yaml and env_config.yaml from checkpoint directory instead of default config files'
    )

    parser.add_argument(
        '--num-dots',
        type=int,
        default=None,
        help='Number of quantum dots (overrides env_config.yaml)'
    )

    parser.add_argument(
        '--sac',
        action='store_true',
        help='Use SAC instead of PPO (overrides config rl_config.algorithm)'
    )

    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Eval mode: each iteration calls algo.evaluate() (no gradient updates) '
             'and per-iter checkpoint saves are skipped. Use with --load-checkpoint '
             'to evaluate a trained policy and dump per-episode .npy distance '
             'trajectories under collected_data/.'
    )

    # Parse known args to allow for dynamic config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides from remaining arguments
    config_overrides = parse_config_overrides(unknown)
    args.config_overrides = config_overrides
    
    return args



def create_env(config=None, env_config_path=None, distance_data_dir=None, num_dots_override=None):
    """Create multi-agent wrapped single-agent environment with JAX safety.

    Note: We wrap the single-agent env as multi-agent because RLlib's
    single-agent PPO code path has issues with continuous action spaces.
    Using multi-agent config with a single agent uses the correct code path.
    """
    import os
    import jax

    # Ensure JAX settings are applied in worker processes
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    # Try to clear any existing JAX state
    try:
        # Force JAX to use a fresh backend in each worker
        jax.clear_backends()
    except:
        pass

    from qadapt.single_agent_sac.utils.multiagent_wrapper import SingleAsMultiAgentWrapper

    # Default config path
    if env_config_path is None:
        env_config_path = str(Path(__file__).parent / "single_agent_env_config.yaml")

    # Path to capacitance model weights (required for kalman update method)
    capacitance_model_checkpoint = str(project_root / "src" / "eval_runs" / "mobilenet_barrier_weights.pth")

    # Create multi-agent wrapped environment
    wrapper_config = {
        "config_path": env_config_path,
        "training": True,
        "distance_data_dir": distance_data_dir,
        "num_dots_override": num_dots_override,
        "capacitance_model_checkpoint": capacitance_model_checkpoint,
    }

    return SingleAsMultiAgentWrapper(config=wrapper_config)


def load_config(config_path=None, checkpoint_path=None):
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default training_config.yaml
        checkpoint_path: Path to checkpoint directory. If provided, loads from checkpoint instead of default location
    """
    if checkpoint_path:
        # Load from checkpoint directory
        config_file = Path(checkpoint_path) / "training_config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Must provide config files within checkpoint: training_config.yaml not found in {checkpoint_path}")
    elif config_path is None:
        config_file = Path(__file__).parent / "training_config.yaml"
    else:
        config_file = Path(config_path)
        # Handle relative paths from training directory
        if not config_file.is_absolute() and not config_file.exists():
            config_file = Path(__file__).parent / config_path

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_env_config(checkpoint_path=None):
    """Load environment configuration from YAML file.

    Args:
        checkpoint_path: Path to checkpoint directory. If provided, loads from checkpoint instead of default location
    """
    if checkpoint_path:
        # Load from checkpoint directory
        config_file = Path(checkpoint_path) / "env_config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Must provide config files within checkpoint: env_config.yaml not found in {checkpoint_path}"
            )
    else:
        config_file = Path(__file__).parent / "single_agent_env_config.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Main training function using Ray RLlib 2.49.0 API with wandb logging."""

    # Parse command line arguments
    args = parse_arguments()

    use_wandb = not args.disable_wandb

    # Determine checkpoint path for config loading if --load-configs is set
    checkpoint_path = args.load_checkpoint if args.load_configs else None

    # Validate that --load-configs requires --load-checkpoint
    if args.load_configs and not args.load_checkpoint:
        raise ValueError("--load-configs requires --load-checkpoint to be specified")

    config = load_config(args.config, checkpoint_path)

    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Override algorithm if --sac flag is provided
    if args.sac:
        config['rl_config']['algorithm'] = "SAC"

    # Create timestamped data folder if save_distance_data is enabled.
    # In --eval-only mode, write to src/eval_runs/collected_data/{ts}_{algo}
    # so compute_table.py finds the .npy distance trajectories alongside the
    # multi-agent ablation runs. Otherwise (training), keep the historical
    # location next to single_agent_sac/.
    distance_data_dir = None
    if config['defaults']['save_distance_data']:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.eval_only and args.load_checkpoint:
            checkpoint_name = Path(args.load_checkpoint).name
            data_parent_dir = (
                Path(__file__).resolve().parents[3]
                / "src" / "eval_runs" / "collected_data"
            )
            distance_data_dir = data_parent_dir / f"{timestamp}_{checkpoint_name}"
        else:
            data_parent_dir = Path(__file__).parent / "data"
            distance_data_dir = data_parent_dir / timestamp

        # Create parent data directory first with proper permissions
        data_parent_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        try:
            os.chmod(data_parent_dir, 0o777)
        except:
            pass

        # Create timestamped subdirectory with proper permissions
        distance_data_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        # Ensure permissions are set correctly even if directory already existed
        try:
            os.chmod(distance_data_dir, 0o777)
        except:
            pass  # Directory may not exist or permissions may not be settable
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
                "QADAPT_PROJECT_ROOT": str(swarm_package_dir),
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        gif_config = config.get("gif_config")
        if gif_config:
            # Clean up any previous GIF capture lock files
            from utils.gif_logger import cleanup_gif_files
            cleanup_gif_files(gif_config['save_dir'])

        # Load env config directly from YAML (no GPU initialization needed on driver)
        env_config = load_env_config(checkpoint_path)

        # Override num_dots if specified via command line
        if args.num_dots is not None:
            env_config['simulator']['num_dots'] = args.num_dots

        # Write env_config to a temporary file so workers can load it
        temp_env_config_path = None
        if checkpoint_path:
            import tempfile
            temp_env_config_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False, dir='/tmp'
            )
            yaml.dump(env_config, temp_env_config_file, default_flow_style=False)
            temp_env_config_path = temp_env_config_file.name
            temp_env_config_file.close()
            print(f"Written env config to temporary file: {temp_env_config_path}")

        # Log num_dots override if specified
        if args.num_dots is not None:
            print(f"Overriding num_dots to: {args.num_dots}")

        create_env_fn = partial(
            create_env,
            env_config_path=temp_env_config_path,
            distance_data_dir=distance_data_dir,
            num_dots_override=args.num_dots,
        )
        register_env("qarray_singleagent_env", create_env_fn)

        # Merge with training config for wandb logging
        if use_wandb:
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
            **config['neural_networks']['single_agent_policy'],
            "free_log_std": config['rl_config']['single_agent']['free_log_std'],
            "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
        }
        
        algo = config['rl_config']['algorithm'].lower()

        rl_module_spec = create_rl_module_spec(env_config, algo=algo, config=rl_module_config)

        # Filter training parameters based on algorithm
        # PPO-specific parameters that should NOT be passed to SAC
        ppo_only_params = {
            'lr', 'lambda_', 'clip_param', 'entropy_coeff', 'vf_loss_coeff',
            'kl_target', 'num_epochs', 'minibatch_size',
        }
        # SAC-specific parameters that should NOT be passed to PPO
        sac_only_params = {
            'actor_lr', 'critic_lr', 'alpha_lr', 'twin_q', 'tau', 'initial_alpha',
            'target_entropy', 'n_step', 'clip_actions', 'target_network_update_freq',
            'num_steps_sampled_before_learning_starts', 'replay_buffer_config', 'reward_scale',
        }

        training_params = config['rl_config']['training'].copy()
        # Extract reward_scale for SAC (not a native SACConfig param, handled by custom learner)
        reward_scale = training_params.pop('reward_scale', 1.0)

        if algo == "sac":
            for param in ppo_only_params:
                training_params.pop(param, None)
        elif algo == "ppo":
            for param in sac_only_params:
                training_params.pop(param, None)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: ppo, sac")

        if algo == "sac":
            replay_buffer_config = training_params.get("replay_buffer_config")
            if replay_buffer_config is None:
                training_params["replay_buffer_config"] = {
                    "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                    "capacity": 200000,
                    "alpha": 0.6,
                    "beta": 0.4,
                }
            else:
                replay_buffer_config = dict(replay_buffer_config)
                replay_buffer_config["type"] = "MultiAgentPrioritizedEpisodeReplayBuffer"
                training_params["replay_buffer_config"] = replay_buffer_config

        # Configure custom callbacks for logging to Wandb
        # log_images = config['wandb']['log_images']
        # custom_callbacks = partial(CustomCallbacks, log_images=log_images)

        # Select algorithm config builder
        if algo == "ppo":
            algo_config_builder = PPOConfig
        elif algo == "sac":
            algo_config_builder = SACConfig
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: ppo, sac")

        # Handle voltage parsing to memory manually (no frame stacking in single-agent)
        use_deltas = env_config['simulator']['use_deltas']
        memory_layer = config['neural_networks']['single_agent_policy']['backbone'].get('memory_layer')
        has_lstm = memory_layer == 'lstm'

        if use_deltas and has_lstm:
            env_to_module_connector = partial(create_env_to_module_connector, use=True)
        else:
            env_to_module_connector = None

        learner_connector = None

        # Policy mapping function for multi-agent setup
        def policy_mapping_fn(agent_id, episode, **kwargs):
            return "agent_0"

        algo_config = (
            algo_config_builder()
            .environment(
                env="qarray_singleagent_env",
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies={"agent_0"},
                policies_to_train=["agent_0"],
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
                # Pass filtered training params based on algorithm
                **training_params,
                # Code-level settings
                learner_connector=learner_connector,
                # Algorithm-specific learner classes
                **({"learner_class": PPOLearnerWithValueStats} if algo == "ppo"
                   else {"learner_class": SACLearnerWithRewardScaling} if algo == "sac"
                   else {}),
            )
            # .callbacks([custom_callbacks] if use_wandb else [])
        )

        # Set reward_scale on config for SAC learner to access
        if algo == "sac":
            algo_config.reward_scale = reward_scale

        # Evaluation block — only added when --eval-only so the training path
        # is unchanged. Local eval (no remote env runner) avoids the Ray 2.51
        # race + the PPO+IMPALA stall we hit in the multi-agent pipeline.
        if args.eval_only:
            algo_config = algo_config.evaluation(
                evaluation_num_env_runners=0,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_sample_timeout_s=1800,
                evaluation_config={"explore": False},
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
        env_config_src = Path(__file__).parent / "single_agent_env_config.yaml"
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
            # Eval mode runs algo.evaluate() — no gradient updates, no per-iter
            # checkpoint save, no wandb upload. Used by run_ablation.py for the
            # single_agent pipeline so the loaded weights stay frozen and the
            # env wrapper still dumps per-episode .npy distance trajectories.
            if args.eval_only:
                result = algo.evaluate()
                print_training_progress(result, i, training_start_time)
                if use_wandb:
                    log_to_wandb(result, i, distance_data_dir)
                print(f"\nIteration {i+1} completed (eval-only).\n")
                continue

            result = algo.train()

            # Clean console output and wandb logging
            print_training_progress(result, i, training_start_time)

            # Log metrics to wandb (EMA is calculated automatically in metrics_logger)
            log_to_wandb(result, i, distance_data_dir)

            # Process and log GIFs if enabled
            if gif_config and gif_config.get('enabled') and use_wandb:
                from utils.gif_logger import process_and_log_gifs
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
                # Upload only at every 25 iterations (25, 50, 75, etc.)
                if (i + 1) % 25 == 0:
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
