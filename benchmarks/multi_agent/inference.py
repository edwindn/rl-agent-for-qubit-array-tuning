#!/usr/bin/env python3
"""
Minimal inference script for multi-agent quantum device tuning.
Runs inference using downloaded model weights with Ray coordination.
"""
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

import ray
import torch
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

# Suppress Ray warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.WARNING)

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Default weights directory
WEIGHTS_DIR = current_dir / "weights" / "artifacts"
INFERENCE_CONFIG_PATH = current_dir / "inference_config.yaml"


# ============================================================================
# Configuration Loading
# ============================================================================

def load_inference_config(config_path=INFERENCE_CONFIG_PATH):
    """
    Load inference configuration from YAML file.

    Args:
        config_path: Path to inference_config.yaml

    Returns:
        Dictionary with inference configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Inference config not found: {config_path}\n"
            f"Please create inference_config.yaml in the benchmarks directory."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Checkpoint Discovery and Validation
# ============================================================================

def find_available_checkpoints(weights_dir=WEIGHTS_DIR):
    """
    Find all available checkpoints in the weights directory.

    Returns:
        List of (checkpoint_name, checkpoint_path) tuples
    """
    if not weights_dir.exists():
        return []

    checkpoints = []
    for subdir in weights_dir.iterdir():
        if subdir.is_dir():
            # Check if it's a valid checkpoint (has rllib_checkpoint.json)
            if (subdir / "rllib_checkpoint.json").exists():
                checkpoints.append((subdir.name, subdir))

    return sorted(checkpoints)


def find_checkpoint(checkpoint_name=None):
    """
    Find a checkpoint by name or return the latest one.

    Args:
        checkpoint_name: Name of checkpoint folder (e.g., 'run_240'), or None for latest

    Returns:
        Path to checkpoint directory

    Raises:
        ValueError if no checkpoints found or specified checkpoint doesn't exist
    """
    available = find_available_checkpoints()

    if not available:
        raise ValueError(
            f"No checkpoints found in {WEIGHTS_DIR}\n"
            f"Please download model weights to this directory."
        )

    if checkpoint_name is None:
        # Return the latest checkpoint (last alphabetically)
        checkpoint_name, checkpoint_path = available[-1]
        print(f"Auto-selected latest checkpoint: {checkpoint_name}")
        return checkpoint_path

    # Find specific checkpoint
    for name, path in available:
        if name == checkpoint_name:
            return path

    # Checkpoint not found
    available_names = [name for name, _ in available]
    raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found.\n"
        f"Available checkpoints: {', '.join(available_names)}"
    )


def load_checkpoint_config(checkpoint_path):
    """
    Load the full_training_config.yaml from the checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary with training configuration
    """
    config_path = checkpoint_path / "full_training_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected full_training_config.yaml in checkpoint directory."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")
    return config


# ============================================================================
# Policy Mapping (from src/swarm/training/utils/policy_mapping.py)
# ============================================================================

def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs."""
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type."
        )


# ============================================================================
# Environment Creation (from src/swarm/training/train.py)
# ============================================================================

def create_env(config=None, gif_config=None, distance_data_dir=None):
    """Create multi-agent quantum environment."""
    import jax

    # JAX settings for workers
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    os.environ.setdefault("JAX_ENABLE_X64", "true")

    try:
        jax.clear_backends()
    except:
        pass

    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper

    # For inference, disable training mode and GIF capture
    return MultiAgentEnvWrapper(
        training=False,
        return_voltage=True,
        gif_config=gif_config,
        distance_data_dir=distance_data_dir
    )


# ============================================================================
# Model Loading and Inference (from src/swarm/inference/model_loader.py)
# ============================================================================

def load_model(checkpoint_path, config):
    """
    Load trained RL model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        config: Training configuration dict (from full_training_config.yaml)

    Returns:
        tuple: (RLlib Algorithm instance, config dict)
    """
    if not ray.is_initialized():
        # Use Ray config from checkpoint's training config
        ray_config_from_checkpoint = config.get('ray', {})

        # Initialize Ray with config from checkpoint
        ray_config = {
            "include_dashboard": ray_config_from_checkpoint.get("include_dashboard", False),
            "log_to_driver": ray_config_from_checkpoint.get("log_to_driver", False),
            "logging_level": ray_config_from_checkpoint.get("logging_level", logging.WARNING),
            "runtime_env": {
                "working_dir": str(src_dir),
                "excludes": ray_config_from_checkpoint.get("runtime_env", {}).get("excludes", [
                    "dataset",
                    "dataset_v1",
                    "wandb",
                    "outputs",
                    "test_outputs",
                    "checkpoints",
                    "weights*",
                    "*dataset*"
                ]),
                "env_vars": {
                    **ray_config_from_checkpoint.get("runtime_env", {}).get("env_vars", {}),
                    "SWARM_PROJECT_ROOT": str(project_root),
                }
            }
        }

        print("Initializing Ray with config from checkpoint...")
        ray.init(**ray_config)

    # Register environment before loading checkpoint
    register_env("qarray_multiagent_env", create_env)

    # Load algorithm from checkpoint
    # For inference, we need to manually remove optimizer state from the checkpoint
    # because RLlib's set_state doesn't properly honor not_components for optimizer
    print(f"Loading checkpoint from: {checkpoint_path}")

    import pickle
    from ray.rllib.core import COMPONENT_OPTIMIZER

    # Temporarily patch the learner state file to remove optimizer
    checkpoint_abs_path = Path(checkpoint_path).absolute()
    learner_state_path = checkpoint_abs_path / "learner_group" / "learner" / "state.pkl"

    if learner_state_path.exists():
        print("Removing optimizer state from checkpoint for inference...")
        # Load the learner state
        with open(learner_state_path, 'rb') as f:
            learner_state = pickle.load(f)

        # Remove optimizer component if present
        if COMPONENT_OPTIMIZER in learner_state:
            del learner_state[COMPONENT_OPTIMIZER]
            print(f"  Removed {COMPONENT_OPTIMIZER} from learner state")

        # Save back the modified state
        with open(learner_state_path, 'wb') as f:
            pickle.dump(learner_state, f)

    algo = Algorithm.from_checkpoint(str(checkpoint_abs_path))
    print("Model loaded successfully")

    # Print config summary
    rl_config = config.get('rl_config', {})
    print(f"\nCheckpoint configuration:")
    print(f"  Algorithm: {rl_config.get('algorithm', 'Unknown')}")
    print(f"  Policies: {rl_config.get('multi_agent', {}).get('policies', [])}")

    neural_nets = config.get('neural_networks', {})
    if 'plunger_policy' in neural_nets:
        backbone = neural_nets['plunger_policy'].get('backbone', {})
        print(f"  Plunger backbone: {backbone.get('type', 'Unknown')}")
    if 'barrier_policy' in neural_nets:
        backbone = neural_nets['barrier_policy'].get('backbone', {})
        print(f"  Barrier backbone: {backbone.get('type', 'Unknown')}")
    print()

    return algo, config


# ============================================================================
# Vision Image Saving (adapted from multi_agent_wrapper.py)
# ============================================================================

def _save_vision_image(agent_obs, agent_id, agent_info, step_count, save_dir):
    """
    Save agent vision image to disk for GIF creation.

    Args:
        agent_obs: Agent observation (dict with 'image' key or raw image array)
        agent_id: Agent ID string
        agent_info: Agent info dict containing voltage and ground truth
        step_count: Current step number
        save_dir: Directory to save images
    """
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib as mpl

    # Extract image from observation
    if isinstance(agent_obs, dict) and 'image' in agent_obs:
        agent_image = agent_obs['image']
    else:
        agent_image = agent_obs

    # Extract voltage info if available
    voltage_text = ""
    ground_truth_text = ""
    if "current_voltage" in agent_info and "ground_truth" in agent_info:
        voltage = agent_info["current_voltage"]
        ground_truth = agent_info["ground_truth"]
        voltage_text = f"V: {voltage:.3f}"
        ground_truth_text = f"GT: {ground_truth:.3f}"

    # Process image based on number of channels
    if agent_image.shape[2] == 2:
        # Plunger agent: 2 channels - merge side-by-side with spacing
        channel_images = []
        for channel in range(2):
            channel_data = agent_image[:, :, channel]
            # Normalize to 0-1 for colormap
            channel_data_norm = ((channel_data - channel_data.min()) /
                                (channel_data.max() - channel_data.min() + 1e-8))

            # Apply plasma colormap and convert to RGB
            plasma_cmap = mpl.colormaps['plasma']
            plasma_cm = plasma_cmap(channel_data_norm)
            plasma_rgb = (plasma_cm[:, :, :3] * 255).astype(np.uint8)
            channel_images.append(plasma_rgb)

        # Create white spacer between images (10 pixels wide)
        spacer = np.ones((channel_images[0].shape[0], 10, 3), dtype=np.uint8) * 255

        # Concatenate horizontally: channel_0 + spacer + channel_1
        merged_image = np.concatenate([channel_images[0], spacer, channel_images[1]], axis=1)
        img = Image.fromarray(merged_image, mode='RGB')
    else:
        # Barrier agent: 1 channel
        channel_data = agent_image[:, :, 0]
        # Normalize to 0-1 for colormap
        channel_data_norm = ((channel_data - channel_data.min()) /
                            (channel_data.max() - channel_data.min() + 1e-8))

        # Apply plasma colormap and convert to RGB
        plasma_cmap = mpl.colormaps['plasma']
        plasma_cm = plasma_cmap(channel_data_norm)
        plasma_rgb = (plasma_cm[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(plasma_rgb, mode='RGB')

    # Add text overlay if info is available
    if voltage_text and ground_truth_text:
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a larger font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Draw text with black background for readability
        text = f"{voltage_text}  {ground_truth_text}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position at top-left corner
        text_x = 5
        text_y = 5

        # Draw black background rectangle
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], fill='black')
        # Draw white text
        draw.text((text_x, text_y), text, fill='white', font=font)

    # Save image
    filename = save_dir / f"step_{step_count:06d}.png"
    img.save(filename)


def _save_distance_history(distance_history, save_dir):
    """
    Save distance history to numpy files.

    Args:
        distance_history: Dict mapping agent_id to list of distances
        save_dir: Directory to save distance data
    """
    import random
    import glob

    save_path = Path(save_dir)

    for agent_id, distances in distance_history.items():
        if not distances:
            continue

        # Create agent folder
        agent_folder = save_path / agent_id
        agent_folder.mkdir(parents=True, exist_ok=True)

        # Find existing files to determine next count
        existing_files = glob.glob(str(agent_folder / "*.npy"))

        if len(existing_files) == 0:
            next_count = 1
        else:
            # Extract counts from filenames (format: XXXX_YYYYYY.npy)
            counts = []
            for filepath in existing_files:
                filename = Path(filepath).stem
                count_str = filename.split('_')[0]
                counts.append(int(count_str))
            next_count = max(counts) + 1

        # Generate random 6-digit number
        random_suffix = random.randint(0, 999999)

        # Create filename with 4-digit zero-padded count and 6-digit zero-padded random suffix
        filename = f"{next_count:04d}_{random_suffix:06d}.npy"
        filepath = agent_folder / filename

        # Save the array
        distances_array = np.array(distances)
        np.save(filepath, distances_array)


# ============================================================================
# GIF Creation (adapted from src/swarm/training/utils/gif_logger.py)
# ============================================================================

def _create_episode_gif(episode_num, save_dir):
    """
    Create GIF from saved episode images.

    Args:
        episode_num: Episode number for filename
        save_dir: Directory to save the GIF
    """
    from PIL import Image
    import shutil

    temp_dir = Path("/tmp/inference_gif_captures")
    agent_dir = temp_dir / "plunger_1"

    if not agent_dir.exists():
        print(f"Warning: No images found for episode {episode_num}")
        return

    try:
        # Get all images for this agent
        image_files = sorted(agent_dir.glob("step_*.png"))

        if len(image_files) < 2:
            print(f"Warning: Not enough images for episode {episode_num} (need at least 2)")
            return

        # Load images
        images = []
        for img_file in image_files:
            img = Image.open(img_file)
            images.append(img)

        # Add white frames at start for easy loop detection
        if images:
            white_frame = Image.new('RGB', images[0].size, (255, 255, 255))
            images = [white_frame] * 3 + images + [white_frame] * 2

        # Save as GIF
        gif_filename = save_dir / f"episode_{episode_num:03d}_plunger_1.gif"
        images[0].save(
            gif_filename,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / 0.5),  # 0.5 fps -> 2000ms per frame
            loop=0
        )

        print(f"Saved GIF: {gif_filename}")

        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error creating GIF for episode {episode_num}: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Distance Plotting (adapted from src/swarm/training/utils/metrics_logger.py)
# ============================================================================

def _create_distance_plots(episode_num, save_dir):
    """
    Create distance plots from saved episode data.

    Args:
        episode_num: Episode number for filename
        save_dir: Directory to save the plots
    """
    import glob
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    temp_dir = Path("/tmp/inference_distance_data")

    if not temp_dir.exists():
        print(f"Warning: No distance data found for episode {episode_num}")
        return

    try:
        # Get all agent folders
        plunger_folders = sorted([f for f in temp_dir.iterdir() if f.is_dir() and f.name.startswith("plunger_")])
        barrier_folders = sorted([f for f in temp_dir.iterdir() if f.is_dir() and f.name.startswith("barrier_")])

        # Plot plunger distances
        if plunger_folders:
            fig, ax = plt.subplots(figsize=(10, 6))

            for agent_folder in plunger_folders:
                # Get all .npy files and find the one with highest count
                npy_files = glob.glob(str(agent_folder / "*.npy"))
                if npy_files:
                    # Extract counts and find max
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        # Load and plot the data
                        distances = np.load(latest_file)
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title("Plunger Agent Distances")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plunger_filename = save_dir / f"episode_{episode_num:03d}_plunger_distances.png"
            plt.savefig(plunger_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plunger distance plot: {plunger_filename}")

        # Plot barrier distances
        if barrier_folders:
            fig, ax = plt.subplots(figsize=(10, 6))

            for agent_folder in barrier_folders:
                # Get all .npy files and find the one with highest count
                npy_files = glob.glob(str(agent_folder / "*.npy"))
                if npy_files:
                    # Extract counts and find max
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        # Load and plot the data
                        distances = np.load(latest_file)
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title("Barrier Agent Distances")
            ax.legend()
            ax.grid(True, alpha=0.3)

            barrier_filename = save_dir / f"episode_{episode_num:03d}_barrier_distances.png"
            plt.savefig(barrier_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved barrier distance plot: {barrier_filename}")

        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error creating distance plots for episode {episode_num}: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Multi-Agent Inference (adapted from src/swarm/inference/inference_testing.py)
# ============================================================================

def run_inference_episode(algo, max_steps=100, deterministic=True, verbose=True,
                         save_vision_images=False, vision_save_dir=None,
                         collect_distances=False, distance_save_dir=None):
    """
    Run a single inference episode using multi-agent coordination.

    This uses Ray to coordinate the two policies:
    - plunger_policy: controls all plunger gates
    - barrier_policy: controls all barrier gates

    Args:
        algo: Loaded RLlib algorithm with both policies
        max_steps: Maximum number of steps per episode
        deterministic: If True, use mean action; if False, sample from distribution
        verbose: Print step-by-step information
        save_vision_images: Save vision images for GIF creation
        vision_save_dir: Directory to save vision images
        collect_distances: Collect distance data for plotting
        distance_save_dir: Directory to save distance data

    Returns:
        Episode statistics dict
    """
    # Create environment without special capture configs since we're not in a Ray worker
    env = create_env()

    episode_return = 0.0
    episode_returns_by_agent = {agent_id: 0.0 for agent_id in env.all_agent_ids}

    # Initialize data collection structures
    distance_history = None
    if collect_distances:
        distance_history = {agent_id: [] for agent_id in env.all_agent_ids}

    vision_step_count = 0
    if save_vision_images and vision_save_dir:
        vision_dir = Path(vision_save_dir) / "plunger_1"
        vision_dir.mkdir(parents=True, exist_ok=True)

    try:
        obs, info = env.reset()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting inference episode")
            print(f"Number of plunger agents: {len(env.gate_agent_ids)}")
            print(f"Number of barrier agents: {len(env.barrier_agent_ids)}")
            print(f"{'='*60}\n")

        for step in range(max_steps):
            # Compute actions for all agents using RLModule API
            actions = {}

            for agent_id, agent_obs in obs.items():
                if agent_id in env.all_agent_ids:
                    # Determine which policy to use
                    policy_id = policy_mapping_fn(agent_id)

                    # Get RLModule for this policy (shared weights across agent type)
                    rl_module = algo.get_module(policy_id)

                    # Prepare observation tensor (handle dict observations)
                    if isinstance(agent_obs, dict):
                        # Convert dict observation to tensors
                        obs_tensor = {
                            key: torch.from_numpy(val).unsqueeze(0).float()
                            for key, val in agent_obs.items()
                        }
                    else:
                        # Handle simple numpy array observations
                        obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0).float()

                    # Forward pass through the policy network
                    result = rl_module.forward_inference({"obs": obs_tensor})

                    # Extract action from distribution
                    action_dist_inputs = result["action_dist_inputs"][0]
                    action_dim = action_dist_inputs.shape[0] // 2
                    mean = action_dist_inputs[:action_dim]
                    log_std = action_dist_inputs[action_dim:]

                    # Sample or take mean
                    if deterministic:
                        action = mean
                    else:
                        std = torch.exp(log_std)
                        action = torch.normal(mean, std)

                    # Clip to valid action range
                    action = torch.clamp(action, -1.0, 1.0)
                    actions[agent_id] = action.item()

            # Take step in environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate rewards
            for agent_id, reward in rewards.items():
                if agent_id in env.all_agent_ids:
                    episode_returns_by_agent[agent_id] += reward
                    episode_return += reward

            # Collect distance data if requested
            if collect_distances and distance_history is not None:
                for agent_id in env.all_agent_ids:
                    if agent_id in info and "ground_truth" in info[agent_id] and "current_voltage" in info[agent_id]:
                        distance = info[agent_id]["current_voltage"] - info[agent_id]["ground_truth"]
                        distance_history[agent_id].append(distance)

            # Save vision images if requested (for plunger_1 only)
            if save_vision_images and vision_save_dir:
                agent_id = "plunger_1"
                if agent_id in obs:
                    _save_vision_image(obs[agent_id], agent_id, info.get(agent_id, {}),
                                      vision_step_count, vision_dir)
                    vision_step_count += 1

            if verbose and (step % 10 == 0 or step < 5):
                avg_reward = np.mean(list(rewards.values()))
                print(f"Step {step:3d}: avg_reward={avg_reward:+.4f}, "
                      f"cumulative_return={episode_return:+.4f}")

            # Check if episode is done
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            if done:
                if verbose:
                    print(f"\nEpisode terminated at step {step + 1}")
                break

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode completed")
            print(f"Total steps: {step + 1}")
            print(f"Total return: {episode_return:.4f}")
            print(f"Average return per agent: {episode_return / len(env.all_agent_ids):.4f}")
            print(f"{'='*60}\n")

        # Save distance history if collected
        if collect_distances and distance_history is not None and distance_save_dir:
            _save_distance_history(distance_history, distance_save_dir)

        return {
            "total_return": episode_return,
            "steps": step + 1,
            "returns_by_agent": episode_returns_by_agent,
            "avg_return_per_agent": episode_return / len(env.all_agent_ids),
        }

    finally:
        env.close()


def run_inference_benchmark(checkpoint_path, config, num_episodes=10, max_steps=100,
                           deterministic=True, verbose=True, save_results=False,
                           results_dir=None, save_gifs=False, gif_save_dir=None,
                           save_distance_plots=False, distance_plot_dir=None):
    """
    Run multiple inference episodes and collect statistics.

    Args:
        checkpoint_path: Path to model checkpoint directory
        config: Training configuration dict (from full_training_config.yaml)
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        deterministic: Use deterministic actions
        verbose: Print detailed information
        save_results: Save results to JSON file
        results_dir: Directory to save results
        save_gifs: Save GIFs of agent vision for each episode
        gif_save_dir: Directory to save GIFs (defaults to benchmarks directory)
        save_distance_plots: Save distance plots for each episode
        distance_plot_dir: Directory to save distance plots (defaults to benchmarks directory)

    Returns:
        Benchmark statistics dict
    """
    # Load model (initializes Ray if needed)
    algo, config = load_model(checkpoint_path, config)

    episode_stats = []

    # Set up GIF directory if needed
    if save_gifs:
        if gif_save_dir is None:
            gif_save_dir = current_dir
        gif_save_path = Path(gif_save_dir)
        gif_save_path.mkdir(parents=True, exist_ok=True)

    # Set up distance plot directory if needed
    if save_distance_plots:
        if distance_plot_dir is None:
            distance_plot_dir = current_dir
        distance_plot_path = Path(distance_plot_dir)
        distance_plot_path.mkdir(parents=True, exist_ok=True)

    try:
        for episode in range(num_episodes):
            if verbose:
                print(f"\n{'#'*60}")
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"{'#'*60}")

            # Configure vision image saving for this episode
            vision_save_dir = None
            if save_gifs:
                vision_save_dir = "/tmp/inference_gif_captures"

            # Configure distance data collection for this episode
            distance_data_dir = None
            if save_distance_plots:
                distance_data_dir = "/tmp/inference_distance_data"

            stats = run_inference_episode(
                algo,
                max_steps=max_steps,
                deterministic=deterministic,
                verbose=verbose,
                save_vision_images=save_gifs,
                vision_save_dir=vision_save_dir,
                collect_distances=save_distance_plots,
                distance_save_dir=distance_data_dir
            )
            episode_stats.append(stats)

            # Create and save GIF for this episode
            if save_gifs:
                _create_episode_gif(episode, gif_save_path)

            # Create and save distance plots for this episode
            if save_distance_plots:
                _create_distance_plots(episode, distance_plot_path)

        # Compute aggregate statistics
        returns = [s["total_return"] for s in episode_stats]
        steps = [s["steps"] for s in episode_stats]

        results = {
            "checkpoint": str(checkpoint_path),
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "deterministic": deterministic,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "episode_stats": episode_stats,
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS ({num_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Mean return:   {results['mean_return']:+.4f} ± {results['std_return']:.4f}")
        print(f"Min return:    {results['min_return']:+.4f}")
        print(f"Max return:    {results['max_return']:+.4f}")
        print(f"Mean steps:    {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
        print(f"{'='*60}\n")

        # Save results if requested
        if save_results and results_dir:
            from datetime import datetime
            import json

            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = Path(checkpoint_path).name
            filename = f"inference_{checkpoint_name}_{timestamp}.json"
            filepath = results_path / filename

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to: {filepath}\n")

        return results

    finally:
        # Keep Ray running if user wants to run more benchmarks
        pass


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent inference with downloaded model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All inference settings are configured in inference_config.yaml.

Examples:
  # Run with default config (inference_config.yaml)
  python inference.py

  # Use custom config file
  python inference.py --config my_inference_config.yaml

  # Override checkpoint from config
  python inference.py --checkpoint run_240

  # Use latest checkpoint (auto)
  python inference.py --checkpoint auto
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(INFERENCE_CONFIG_PATH),
        help=f"Path to inference config YAML file (default: {INFERENCE_CONFIG_PATH.name})"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Override checkpoint from config (auto, checkpoint name, or full path)"
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    try:
        # Load inference config
        print(f"Loading inference config from: {args.config}")
        inference_config = load_inference_config(Path(args.config))

        # Get checkpoint (command line overrides config)
        checkpoint_name = args.checkpoint if args.checkpoint else inference_config['checkpoint']['name']

        # Find checkpoint
        if checkpoint_name == "auto":
            checkpoint_path = find_checkpoint(None)
        elif Path(checkpoint_name).exists():
            # Full path provided
            checkpoint_path = Path(checkpoint_name)
        else:
            # Assume it's a checkpoint name
            checkpoint_path = find_checkpoint(checkpoint_name)

        print(f"Using checkpoint: {checkpoint_path}")

        # Load training config from checkpoint
        training_config = load_checkpoint_config(checkpoint_path)

        # Get inference parameters from config
        num_episodes = inference_config['episodes']['num_episodes']
        max_steps = inference_config['episodes']['max_steps']
        deterministic = inference_config['episodes']['deterministic']
        verbose = inference_config['output']['verbose']
        save_results = inference_config['output']['save_results']
        results_dir = inference_config['output']['results_dir']
        save_gifs = inference_config['output'].get('save_gifs', False)
        gif_save_dir = inference_config['output'].get('gif_save_dir', str(current_dir))
        save_distance_plots = inference_config['output'].get('save_distance_plots', False)
        distance_plot_dir = inference_config['output'].get('distance_plot_dir', str(current_dir))

        print(f"\nInference parameters:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Max steps: {max_steps}")
        print(f"  Deterministic: {deterministic}")
        print(f"  Save results: {save_results}")
        if save_results:
            print(f"  Results directory: {results_dir}")
        print(f"  Save GIFs: {save_gifs}")
        if save_gifs:
            print(f"  GIF directory: {gif_save_dir}")
        print(f"  Save distance plots: {save_distance_plots}")
        if save_distance_plots:
            print(f"  Distance plot directory: {distance_plot_dir}")
        print()

        # Run benchmark
        results = run_inference_benchmark(
            checkpoint_path=checkpoint_path,
            config=training_config,
            num_episodes=num_episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            verbose=verbose,
            save_results=save_results,
            results_dir=results_dir,
            save_gifs=save_gifs,
            gif_save_dir=gif_save_dir,
            save_distance_plots=save_distance_plots,
            distance_plot_dir=distance_plot_dir
        )

        return 0

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown complete")


if __name__ == "__main__":
    sys.exit(main())
