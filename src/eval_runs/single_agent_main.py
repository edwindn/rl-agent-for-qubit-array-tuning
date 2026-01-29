#!/usr/bin/env python3
"""
Single-agent RL evaluation for quantum device tuning using Ray RLlib.

This script mirrors main.py but uses the single-agent training infrastructure
from src/swarm/single_agent_ablations.
"""
import os
import sys
import warnings

# Suppress deprecation/future warnings on driver (before Ray/RLlib imports)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import argparse
import logging
import re
import time
import yaml
from functools import partial
from pathlib import Path

import ray
import wandb
from ray.rllib.algorithms.ppo import PPOConfig
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

from swarm.training.train_utils import (  # noqa: E402
    parse_config_overrides,
    apply_config_overrides,
    fix_optimizer_betas_after_checkpoint_load,
)

from swarm.training.utils import (  # noqa: E402
    print_training_progress,
    CustomFrameStackingEnvToModule,
    CustomFrameStackingLearner,
)

from swarm.single_agent_ablations.utils.factory import create_rl_module_spec  # noqa: E402
from swarm.training.utils.custom_ppo_learner import PPOLearnerWithValueStats  # noqa: E402


def _get_all_distance_files(distance_data_dir):
    """Get set of all .npy files currently in distance_data_dir."""
    distance_dir = Path(distance_data_dir)
    if not distance_dir.exists():
        return set()

    files = set()
    for agent_dir in distance_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        for npy_file in agent_dir.glob("*.npy"):
            files.add(str(npy_file))
    return files


def _upload_new_distances(distance_data_dir, previously_seen, artifact):
    """Upload any new .npy files that weren't in previously_seen set.

    Returns updated set of seen files.
    """
    distance_dir = Path(distance_data_dir)
    if not distance_dir.exists():
        print(f"Warning: distance_data_dir does not exist: {distance_dir}")
        return previously_seen

    current_files = _get_all_distance_files(distance_data_dir)
    new_files = current_files - previously_seen

    files_added = 0
    for filepath in new_files:
        npy_file = Path(filepath)
        agent_dir_name = npy_file.parent.name
        artifact_path = f"{agent_dir_name}/{npy_file.name}"
        artifact.add_file(str(npy_file), name=artifact_path)
        files_added += 1

    if files_added > 0:
        print(f"  Uploaded {files_added} new distance files to artifact")

    return current_files


def download_wandb_artifact(artifact_path: str) -> str:
    """Download a wandb artifact and return the local path.

    Args:
        artifact_path: Full artifact path like 'entity/project/artifact_name:version'

    Returns:
        Local path to the downloaded artifact directory
    """
    print(f"[DEBUG] Downloading wandb artifact: {artifact_path}")

    # Initialize wandb in offline mode just for artifact download
    run = wandb.init(mode="online", job_type="artifact-download")
    try:
        artifact = run.use_artifact(artifact_path, type='model_checkpoint')
        artifact_dir = artifact.download()
        print(f"[DEBUG] Artifact downloaded to: {artifact_dir}")
        return artifact_dir
    finally:
        wandb.finish()


def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(
        description='Single-agent RL evaluation for quantum device tuning'
    )

    # Checkpoint source (one of these is required)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        '--load-checkpoint',
        type=str,
        help='Path to local checkpoint directory (must contain training_config.yaml)'
    )
    checkpoint_group.add_argument(
        '--wandb-artifact',
        type=str,
        help='Wandb artifact path (e.g., "entity/project/artifact_name:version")'
    )

    parser.add_argument(
        '--training-config',
        type=str,
        help='Path to training_config.yaml (if not in checkpoint directory)'
    )

    parser.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect distance data only (default 100 rollouts)'
    )

    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=100,
        help='Number of rollouts for data collection (default: 100)'
    )

    parser.add_argument(
        '--upload-to-wandb',
        action='store_true',
        help='Upload distance data to wandb after each episode (use with --collect-data)'
    )

    parser.add_argument(
        '--num-env-runners',
        type=int,
        default=1,
        help='Number of parallel env runners (default: 1)'
    )

    parser.add_argument(
        '--gpu-fraction',
        type=float,
        default=0.9,
        help='GPU fraction per env runner (default: 0.9)'
    )

    # Parse known args to allow for dynamic config overrides
    args, unknown = parser.parse_known_args()

    # Parse config overrides from remaining arguments
    config_overrides = parse_config_overrides(unknown)
    args.config_overrides = config_overrides

    return args


def create_env(config=None, distance_data_dir=None, env_config_path=None):
    """Create single-agent environment wrapped as multi-agent with JAX safety."""
    import os
    import jax

    # Ensure JAX settings are applied in worker processes
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    # Try to clear any existing JAX state
    try:
        jax.clear_backends()
    except:
        pass

    from swarm.single_agent_ablations.utils.multiagent_wrapper import SingleAsMultiAgentWrapper

    wrapper_config = {
        "config_path": env_config_path,
        "training": False,
        "distance_data_dir": distance_data_dir,
    }
    return SingleAsMultiAgentWrapper(config=wrapper_config)


def load_config(checkpoint_path=None, config_path=None):
    """Load training configuration from config path, checkpoint directory, or default location.

    Args:
        checkpoint_path: Path to checkpoint directory (may contain training_config.yaml)
        config_path: Explicit path to training_config.yaml (takes precedence)

    Returns:
        Loaded config dictionary
    """
    # Priority 1: Explicit config path
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"[DEBUG] Loading config from explicit path: {config_file}")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    # Priority 2: Config in checkpoint directory
    if checkpoint_path:
        config_file = Path(checkpoint_path) / "training_config.yaml"
        if config_file.exists():
            print(f"[DEBUG] Loading config from checkpoint: {config_file}")
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)

    # Priority 3: Default single-agent training config
    default_config = swarm_package_dir / "single_agent_ablations" / "training_config.yaml"
    if default_config.exists():
        print(f"[DEBUG] Loading default single-agent config: {default_config}")
        with open(default_config, 'r') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"No training_config.yaml found. Searched:\n"
        f"  - Checkpoint: {checkpoint_path}/training_config.yaml\n"
        f"  - Default: {default_config}\n"
        f"Use --training-config to specify a config file."
    )


def load_env_config():
    """Load environment configuration from eval_runs/single_agent_env_config.yaml."""
    config_file = Path(__file__).parent / "single_agent_env_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"single_agent_env_config.yaml not found in {config_file.parent}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def policy_mapping_fn(agent_id, episode=None, **kwargs):
    """Map all agents to the single policy."""
    return "agent_0"


def main():
    """Main evaluation function using Ray RLlib with single-agent setup."""

    # Parse command line arguments
    args = parse_arguments()

    use_wandb = not args.disable_wandb and not args.collect_data
    upload_to_wandb = args.upload_to_wandb and args.collect_data

    # Resolve checkpoint path - either from local path or wandb artifact
    if args.wandb_artifact:
        checkpoint_path = download_wandb_artifact(args.wandb_artifact)
    else:
        checkpoint_path = str(Path(args.load_checkpoint).resolve())

    print(f"[DEBUG] Loading config...")
    config = load_config(checkpoint_path, config_path=args.training_config)
    print(f"[DEBUG] Config loaded successfully")

    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Create timestamped data folder if collect_data is enabled
    distance_data_dir = None
    if args.collect_data:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(checkpoint_path).name.replace(":", "_")
        data_parent_dir = Path(__file__).parent / "collected_data"
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
    elif upload_to_wandb:
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            job_type="eval-data-collection"
        )
        print(f"Wandb initialized for artifact upload")

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

    print("[DEBUG] Initialising ray...")
    ray.init(**ray_config)
    print("[DEBUG] Ray initialized successfully")

    try:
        # Load env config from eval_runs directory
        env_config = load_env_config()

        # Write env_config to a temporary file so workers can load it
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
            distance_data_dir=distance_data_dir,
            env_config_path=temp_env_config_path,
        )
        register_env("qarray_singleagent_env", create_env_fn)

        # Build RLModule config for single-agent
        rl_module_config = {
            **config['neural_networks']['single_agent_policy'],
            "free_log_std": config['rl_config']['single_agent']['free_log_std'],
            "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
        }

        # Handle memory layer max_seq_len
        backbone = rl_module_config.get("backbone", {})
        memory_layer = backbone.get("memory_layer")
        if memory_layer == "lstm":
            rl_module_config["max_seq_len"] = backbone["lstm"]["max_seq_len"]
        elif memory_layer == "transformer":
            rl_module_config["max_seq_len"] = backbone["transformer"]["max_seq_len"]

        rl_module_spec = create_rl_module_spec(env_config, algo="ppo", config=rl_module_config)

        # Determine if we need frame stacking for temporal models
        has_transformer = memory_layer == "transformer"
        use_frame_stacking = has_transformer
        num_frames = 1
        if use_frame_stacking:
            num_frames = backbone["transformer"]["max_seq_len"]
            print(f"\n[Frame Stacking] Enabled with {num_frames} frames for transformer\n")

        # Build env-to-module connector
        if use_frame_stacking:
            env_to_module_connector = lambda env, spaces=None, device=None: CustomFrameStackingEnvToModule(
                num_frames=num_frames,
                multi_agent=True
            )
        else:
            env_to_module_connector = None

        # Build learner connector for frame stacking
        learner_connector = None
        if use_frame_stacking:
            learner_connector = lambda obs_space, act_space: CustomFrameStackingLearner(
                num_frames=num_frames,
                multi_agent=True
            )

        # Filter training parameters for PPO
        ppo_only_params = {'lr', 'lambda_', 'clip_param', 'entropy_coeff', 'vf_loss_coeff', 'kl_target', 'num_epochs', 'minibatch_size'}
        training_params = config['rl_config']['training'].copy()

        # Remove non-PPO parameters
        sac_only_params = {'actor_lr', 'critic_lr', 'alpha_lr', 'twin_q', 'tau', 'initial_alpha', 'target_entropy', 'n_step',
                          'clip_actions', 'target_network_update_freq', 'num_steps_sampled_before_learning_starts', 'replay_buffer_config',
                          'reward_scale'}
        for param in sac_only_params:
            training_params.pop(param, None)

        algo_config = (
            PPOConfig()
            .environment(
                env="qarray_singleagent_env",
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies={"agent_0"},
                policies_to_train=["agent_0"],
                count_steps_by="env_steps",
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=args.num_env_runners,
                rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=args.gpu_fraction,
                env_to_module_connector=env_to_module_connector,
                add_default_connectors_to_env_to_module_pipeline=True,
            )
            .learners(
                # For eval/data collection, we don't need learners
                num_learners=0,
                num_gpus_per_learner=0,
            )
            .training(
                **training_params,
                learner_connector=learner_connector,
                learner_class=PPOLearnerWithValueStats,
            )
            .evaluation(
                evaluation_num_env_runners=1,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_sample_timeout_s=1800,
                evaluation_config={
                    "explore": True,
                },
            )
        )

        # Build the algorithm
        print(f"[DEBUG] Building PPO algorithm...")
        sys.stdout.flush()

        algo = algo_config.build()
        print(f"[DEBUG] Algorithm built successfully")
        sys.stdout.flush()

        # Load checkpoint
        checkpoint_path_obj = Path(checkpoint_path).resolve()
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path_obj}")

        print(f"[DEBUG] Loading checkpoint from: {checkpoint_path_obj}")
        sys.stdout.flush()
        try:
            algo.restore_from_path(str(checkpoint_path_obj.absolute()))
            print(f"[DEBUG] Checkpoint restored, fixing optimizer betas...")
            sys.stdout.flush()
            fix_optimizer_betas_after_checkpoint_load(algo)
            print(f"[DEBUG] Optimizer betas fixed")

            # Extract iteration number from path
            match = re.search(r'iteration_(\d+)', str(checkpoint_path_obj))
            if match:
                start_iteration = int(match.group(1))
                print(f"Checkpoint loaded successfully from iteration {start_iteration}")
            else:
                print("Checkpoint loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}") from e

        print("\nCheckpoint loaded. Starting data collection...\n")

        # Run evaluation/data collection
        training_start_time = time.time()
        num_rollouts = args.num_rollouts
        print(f"[DEBUG] Starting data collection for {num_rollouts} rollouts...")
        sys.stdout.flush()

        # Create wandb artifact for incremental uploads
        if upload_to_wandb:
            artifact_name = f"eval_distances_{Path(checkpoint_path).name}"
            distance_artifact = wandb.Artifact(
                artifact_name,
                type="distance_data",
                metadata={
                    "checkpoint": checkpoint_path,
                    "num_rollouts": num_rollouts,
                    "num_env_runners": args.num_env_runners,
                }
            )
            print(f"[DEBUG] Will upload distance data to wandb artifact: {artifact_name}")
            seen_files = set()

        for i in range(num_rollouts):
            print(f"[DEBUG] Starting rollout {i+1}/{num_rollouts}...")
            sys.stdout.flush()
            result = algo.evaluate()
            print(f"[DEBUG] Rollout {i+1} completed")
            sys.stdout.flush()

            print_training_progress(result, i, training_start_time)

            # Upload any new distance .npy files to wandb
            if upload_to_wandb and distance_data_dir:
                time.sleep(0.5)
                seen_files = _upload_new_distances(distance_data_dir, seen_files, distance_artifact)

        # Finalize and log the artifact
        if upload_to_wandb:
            wandb.log_artifact(distance_artifact)
            print(f"\nUploaded distance artifact to wandb: {artifact_name}")

        print(f"\nData collection complete. Distance data saved to: {distance_data_dir}\n")

    finally:
        if ray.is_initialized():
            ray.shutdown()

        if use_wandb or upload_to_wandb:
            wandb.finish()
            print("Wandb session finished")


if __name__ == "__main__":
    main()
