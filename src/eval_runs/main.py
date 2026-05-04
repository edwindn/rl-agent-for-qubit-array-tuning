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
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent  # src directory
swarm_package_dir = src_dir / "swarm"  # swarm package directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

# Apply patch for Ray 2.49.0 replay buffer bug with complex observations (DQN/SAC)
# See: https://github.com/ray-project/ray/pull/57017
# Remove after upgrading to Ray >= 2.51.0
from swarm.training.patches.ray_episode_patch import apply_patch  # noqa: E402
apply_patch()

from swarm.training.utils import (  # noqa: E402
    log_to_wandb,
    print_training_progress,
    setup_wandb_metrics,
    upload_checkpoint_artifact,
    policy_mapping_fn,
    cleanup_gif_files,
    process_and_log_gifs,
    CustomFrameStackingEnvToModule,
    CustomFrameStackingLearner,
)

# Import inference utilities for scan saving (we'll define locally to work with training directory)
def save_scans_to_iteration_folder(iteration_num, config, save_dir=None):
    """Save scan images to a dedicated folder for this iteration."""
    import shutil

    gif_save_dir = Path(config['gif_config']['save_dir'])
    training_dir = Path(__file__).parent if save_dir is None else Path(save_dir)

    if not gif_save_dir.exists():
        print("No image dir found for scans")
        return

    try:
        print(f"Saving scans for iteration {iteration_num}...")

        # Create rollout_scans parent folder and iteration-specific subfolder
        rollout_scans_dir = training_dir / "rollout_scans"
        iteration_folder = rollout_scans_dir / f"iteration_{iteration_num}_scans"
        iteration_folder.mkdir(parents=True, exist_ok=True)

        # Find all agent subdirectories
        agent_dirs = [d for d in gif_save_dir.iterdir() if d.is_dir()]

        if not agent_dirs:
            print("No agent directories found")
            return

        scans_saved = 0
        # Copy each agent's images to the iteration folder
        for agent_dir in agent_dirs:
            agent_id = agent_dir.name
            image_files = sorted(agent_dir.glob("step_*.png"))

            if not image_files:
                print(f"No images found for {agent_id}")
                continue

            # Create agent subfolder within iteration folder
            agent_iteration_folder = iteration_folder / agent_id
            agent_iteration_folder.mkdir(parents=True, exist_ok=True)

            # Copy all images to the iteration folder
            for img_file in image_files:
                dest_path = agent_iteration_folder / img_file.name
                shutil.copy2(img_file, dest_path)
                scans_saved += 1

        print(f"Saved {scans_saved} scan images to {iteration_folder}")

        # Clean up temporary gif_captures directory
        shutil.rmtree(gif_save_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error saving scans: {e}")
        import traceback
        traceback.print_exc()

from swarm.training.train_utils import (  # noqa: E402
    parse_config_overrides,
    map_sweep_parameters,
    apply_config_overrides,
    fix_optimizer_betas_after_checkpoint_load,
    clean_checkpoint_folder,
    delete_old_checkpoint_if_needed,
    create_env_to_module_connector,
)

from swarm.voltage_model import create_rl_module_spec
from swarm.training.utils.custom_ppo_learner import PPOLearnerWithValueStats  # for logging
from swarm.training.utils.custom_sac_learner import SACLearnerWithRewardScaling  # for reward scaling


def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(description='Multi-agent RL training for quantum device tuning')
    
    parser.add_argument(
        '--load-checkpoint', 
        type=str, 
        required=True,
        help='Path to checkpoint directory to load (must contain training_config.yaml)'
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



def create_env(
    config=None,
    gif_config=None,
    distance_data_dir=None,
    env_config_path=None,
    scan_save_dir=None,
    is_collecting_data=False,
    capacitance_model_checkpoint=None,
    return_global_state=False,
):
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

    from swarm.environment.scan_saving_wrapper import ScanSavingWrapper

    # Wrap in scan-saving wrapper (which inherits from MultiAgentEnvWrapper)
    # need return_voltage=True if we are using deltas + LSTM/Transformer
    # RLlib handles temporal sequences via ConnectorV2, not via environment
    return ScanSavingWrapper(
        return_voltage=True,
        return_global_state=return_global_state,
        gif_config=gif_config,
        distance_data_dir=distance_data_dir,
        env_config_path=env_config_path,
        capacitance_model_checkpoint=capacitance_model_checkpoint,
        scan_save_dir=scan_save_dir,
        scan_save_enabled=scan_save_dir is not None,
        is_collecting_data=is_collecting_data,
    )


def load_config(checkpoint_path=None):
    """Load training configuration from checkpoint directory."""
    if not checkpoint_path:
        raise ValueError("Must provide --load-checkpoint to load training_config.yaml")

    # Load from checkpoint directory
    config_file = Path(checkpoint_path) / "training_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Must provide config files within checkpoint: training_config.yaml not found in {checkpoint_path}"
        )

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_env_config(checkpoint_path: str | None = None):
    """Load environment configuration.

    Prefers <checkpoint_path>/env_config.yaml when supplied — this lets each
    eval run use the env_config that matches its checkpoint (e.g. ablation
    rows that share weights but differ in env settings like update_method).
    Falls back to eval_runs/env_config.yaml for callers that don't pass a
    checkpoint path.
    """
    if checkpoint_path is not None:
        ckpt_env_config = Path(checkpoint_path) / "env_config.yaml"
        if ckpt_env_config.exists():
            with open(ckpt_env_config, "r") as f:
                return yaml.safe_load(f)

    config_file = Path(__file__).parent / "env_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"env_config.yaml not found in {config_file.parent}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Main training function using Ray RLlib 2.49.0 API with wandb logging."""

    # Parse command line arguments
    args = parse_arguments()

    use_wandb = not args.disable_wandb

    # Resolve checkpoint path to absolute path
    args.load_checkpoint = str(Path(args.load_checkpoint).resolve())
    checkpoint_path = args.load_checkpoint

    config = load_config(checkpoint_path)

    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Create timestamped data folder if save_distance_data is enabled
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(checkpoint_path).name
    data_parent_dir = Path("src/eval_runs/collected_data")
    distance_data_dir = data_parent_dir / f"{timestamp}_{checkpoint_name}"

    data_parent_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    try:
        os.chmod(data_parent_dir, 0o777)
    except:
        pass

    distance_data_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    try:
        os.chmod(distance_data_dir, 0o777)
    except:
        pass
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
                "SWARM_PROJECT_ROOT": str(swarm_package_dir),
                "EVAL_RUNS_DIR": str(current_dir),
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        gif_config = config["gif_config"]
        # Override gif save dir to local training directory
        gif_config['save_dir'] = str(Path(__file__).parent / "gif_captures")
        # Clean up any previous GIF capture lock files
        cleanup_gif_files(gif_config['save_dir'])

        # Create scan save directory (use absolute path for Ray workers).
        # Tie scan saving to gif_config.enabled — when both are off (e.g. ablation
        # eval), skip the per-step PNG writes that cost ~2-3s/step in I/O.
        if config["gif_config"].get("enabled", True):
            scan_save_dir = Path(__file__).parent.resolve() / "scan_captures"
            scan_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Scan images will be saved to: {scan_save_dir}\n")
        else:
            scan_save_dir = None
            print("Scan saving disabled (gif_config.enabled=False)\n")

        # Load env config directly from YAML (no GPU initialization needed on driver).
        # Pass checkpoint_path so ablation runs that bundle a per-run env_config.yaml
        # alongside the checkpoint get respected (otherwise we'd silently use the
        # global eval_runs/env_config.yaml for every algo).
        env_config = load_env_config(checkpoint_path)

        capacitance_weights_path = env_config.get("capacitance_model", {}).get("weights_path")

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

        create_env_fn = partial(
            create_env,
            gif_config=gif_config,
            distance_data_dir=distance_data_dir,
            env_config_path=temp_env_config_path,
            capacitance_model_checkpoint=capacitance_weights_path,
            scan_save_dir=str(scan_save_dir) if scan_save_dir is not None else None,
            is_collecting_data=True,
            return_global_state=(config['rl_config']['algorithm'].lower() == "mappo"),
        )
        register_env("qarray_multiagent_env", create_env_fn)

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
        ppo_only_params = {'lr', 'lambda_', 'clip_param', 'entropy_coeff', 'vf_loss_coeff', 'kl_target', 'num_epochs', 'minibatch_size'}
        # SAC-specific parameters that should NOT be passed to PPO
        sac_only_params = {'actor_lr', 'critic_lr', 'alpha_lr', 'twin_q', 'tau', 'initial_alpha', 'target_entropy', 'n_step',
                          'clip_actions', 'target_network_update_freq', 'num_steps_sampled_before_learning_starts', 'replay_buffer_config',
                          'reward_scale'}  # reward_scale used by custom SAC learner

        training_params = config['rl_config']['training'].copy()

        # Extract reward_scale for SAC (not a native SACConfig param, handled by custom learner)
        reward_scale = training_params.pop('reward_scale', 1.0)

        if algo == 'sac':
            # Remove PPO-only parameters when using SAC
            for param in ppo_only_params:
                training_params.pop(param, None)
        elif algo in ('ppo', 'mappo'):
            # Remove SAC-only parameters when using PPO/MAPPO
            for param in sac_only_params:
                training_params.pop(param, None)

        # Configure custom callbacks for logging to Wandb
        # log_images = config['wandb']['log_images']
        # custom_callbacks = partial(CustomCallbacks, log_images=log_images)

        # Select algorithm config builder. MAPPO uses PPOConfig with a
        # centralized critic wired up via the custom RLModule spec; the algo
        # builder is the same as plain PPO.
        if algo in ("ppo", "mappo"):
            algo_config_builder = PPOConfig
        elif algo == "sac":
            algo_config_builder = SACConfig
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: ppo, mappo, sac")

        # Handle voltage parsing to memory manually
        use_deltas = env_config['simulator']['use_deltas']
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
            # Note: RLlib expects signature (env, spaces, device) - spaces and device are optional
            env_to_module_connector = lambda env, spaces=None, device=None: CustomFrameStackingEnvToModule(
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
            learner_connector = lambda obs_space, act_space: CustomFrameStackingLearner(
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
                num_env_runners=config['rl_config']['env_runners'].get('num_env_runners', 1),
                rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=config['rl_config']['env_runners'].get('num_gpus_per_env_runner', 0.9),
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
                **({"learner_class": PPOLearnerWithValueStats} if algo in ("ppo", "mappo")
                   else {"learner_class": SACLearnerWithRewardScaling} if algo == "sac"
                   else {}),
            )
            .evaluation(
                evaluation_num_env_runners=config['rl_config'].get('evaluation', {}).get('evaluation_num_env_runners', 1),
                evaluation_duration=1,  # Run 1 episode per evaluation
                evaluation_duration_unit="episodes",
                evaluation_sample_timeout_s=1800,
                evaluation_config={
                    "explore": True,  # Use stochastic actions
                },
            )
            # .callbacks([custom_callbacks] if use_wandb else [])
        )

        # Set reward_scale on config for SAC learner to access
        if algo == "sac":
            algo_config.reward_scale = reward_scale

        # Build the algorithm
        print(f"\nBuilding {algo} algorithm...\n")

        algo = algo_config.build()

        # Ray 2.51 race: build() returns before learner actor is registered as
        # healthy in worker_manager. restore_from_path -> sync_weights ->
        # learner_group.get_state() then crashes with IndexError because
        # healthy_actor_ids() is empty. Force a probe with healthy_only=False +
        # mark_healthy=True to flip the bit before restore.
        wm = algo.learner_group._worker_manager
        print(f"Pre-probe: actors={len(wm._actors)}, healthy={len(wm.healthy_actor_ids())}")
        wm.foreach_actor(func=lambda w: True, healthy_only=False, mark_healthy=True)
        print(f"Post-probe: healthy={len(wm.healthy_actor_ids())}")

        # Handle checkpoint loading if requested
        start_iteration = 0
        checkpoint_loaded = False
        
        if args.load_checkpoint:
            # Load specific checkpoint
            checkpoint_path = Path(args.load_checkpoint).resolve()
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
        env_config_src = swarm_package_dir / "environment" / "env_config.yaml"
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
            # Use evaluate() instead of train() to skip gradient updates
            result = algo.evaluate()

            # Clean console output and wandb logging
            print_training_progress(result, i, training_start_time)

            # Log metrics to wandb (EMA is calculated automatically in metrics_logger)
            if use_wandb:
                log_to_wandb(result, i, distance_data_dir)

            # Save scans to iteration folder if enabled
            if config['gif_config']['enabled']:
                # Wait briefly for async worker processes to finish writing data to disk
                time.sleep(0.5)
                save_scans_to_iteration_folder(i + 1, config)

            # Process and log GIFs if enabled (uses gif_captures, not scan_captures)
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
