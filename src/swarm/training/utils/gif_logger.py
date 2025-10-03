"""
GIF logging utilities for episode visualization during training.

This module handles the conversion of episode scan data and agent vision
to wandb-compatible GIF format for qualitative assessment of agent performance.
"""

import numpy as np
import wandb
from pathlib import Path
import shutil


def process_episode_gif(result, iteration):
    """
    Process episode GIF data from training results and log to wandb.

    Args:
        result: Training result dictionary from Ray RLlib
        iteration: Current training iteration number
    """
    try:
        env_runner_results = result.get("env_runners", {})
        custom_metrics = env_runner_results.get("custom_metrics", {})

        # Look for gif_frames in the custom metrics
        if "gif_frames" in custom_metrics:
            frames_data = custom_metrics["gif_frames"]
            channel = custom_metrics.get("gif_channel", 0)
            num_frames = custom_metrics.get("gif_num_frames", 0)

            # Convert to wandb format: (T, C, H, W) with dtype uint8
            if len(frames_data.shape) == 3:  # (T, H, W)
                # Environment already normalized to [0,1], just convert to uint8
                # Add channel dimension: (T, H, W) -> (T, 1, H, W)
                frames_wandb = frames_data[:, np.newaxis, :, :]

                # Convert from [0,1] to [0,255] uint8
                frames_uint8 = (frames_wandb * 255).astype(np.uint8)

                # Log to wandb as video/GIF
                wandb.log({
                    "episode_scan_evolution": wandb.Video(
                        frames_uint8,
                        format="gif",
                        fps=4,
                        caption=f"Scan evolution - Channel {channel} - Iteration {iteration+1}"
                    ),
                    "gif_num_frames": num_frames,
                    "gif_channel": channel,
                }, step=iteration+1)

    except Exception as e:
        print(f"Error processing episode GIF: {e}")
        # Don't raise to avoid disrupting training


def cleanup_gif_files(gif_save_dir=None):
    """Remove gif capture lock file and images from previous training runs."""
    import os

    # Clean up lock file with stale process detection
    lock_file = "/tmp/gif_capture_worker.lock"
    try:
        # Check if lock file exists and if the process is still alive
        if os.path.exists(lock_file):
            try:
                with open(lock_file, 'r') as f:
                    old_pid = int(f.read().strip())

                # Check if process is still running
                try:
                    os.kill(old_pid, 0)  # Signal 0 checks if process exists
                    print(f"Warning: Lock file exists with active process {old_pid}. Another training may be running.")
                except OSError:
                    # Process doesn't exist - safe to remove stale lock
                    os.remove(lock_file)
                    print(f"Cleaned up stale GIF capture lock file (process {old_pid} no longer exists)")
            except (ValueError, IOError):
                # Couldn't read PID or file is corrupted - try to remove anyway
                try:
                    os.remove(lock_file)
                    print("Cleaned up corrupted GIF capture lock file")
                except PermissionError:
                    print(f"Warning: Lock file exists but cannot be removed due to permissions. Try: sudo rm {lock_file}")
        else:
            pass  # No lock file, nothing to clean
    except FileNotFoundError:
        pass  # Already gone between check and removal
    except PermissionError:
        print(f"Warning: Could not remove GIF lock file due to permissions. Try: sudo rm {lock_file}")
    except Exception as e:
        print(f"Warning: Could not remove GIF lock file: {e}")

    # Clean up previous GIF images if directory is specified
    if gif_save_dir is not None:
        try:
            gif_dir = Path(gif_save_dir)
            if gif_dir.exists():
                shutil.rmtree(gif_dir, ignore_errors=True)
                print(f"Cleaned up previous GIF images from {gif_dir}")
        except Exception as e:
            print(f"Warning: Could not remove previous GIF images: {e}")


def process_and_log_gifs(iteration_num, config, use_wandb=True):
    """Process saved images into GIFs and log to Wandb."""
    gif_save_dir = Path(config['gif_config']['save_dir'])

    if not gif_save_dir.exists():
        print("No image dir found for gif creation")
        return

    try:
        print(f"Processing GIFs for iteration {iteration_num}...")

        # Find all agent subdirectories
        agent_dirs = [d for d in gif_save_dir.iterdir() if d.is_dir()]

        if not agent_dirs:
            print("No agent directories found for GIF creation")
            return

        videos_logged = 0
        # Process each agent's images separately
        for agent_dir in agent_dirs:
            agent_id = agent_dir.name
            image_files = sorted(agent_dir.glob("step_*.png"))

            if len(image_files) < 2:
                print(f"Not enough images for {agent_id} (need at least 2)")
                continue

            # Log to Wandb
            if use_wandb:
                _log_images_as_video_to_wandb(image_files, agent_id, iteration_num, config)
                videos_logged += 1

        # Clean up temporary files
        shutil.rmtree(gif_save_dir, ignore_errors=True)
        print(f"Processed and logged {videos_logged} agent videos for iteration {iteration_num}")

    except Exception as e:
        print(f"Error processing GIFs: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        shutil.rmtree(gif_save_dir, ignore_errors=True)


def _log_images_as_video_to_wandb(image_files, agent_id, iteration_num, config):
    """Convert images to numpy arrays and log as videos to Wandb."""
    try:
        from PIL import Image

        fps = config['gif_config'].get('fps', 0.5)  # Default to 0.5 if not specified

        if len(image_files) < 2:
            print(f"Not enough images for video (need at least 2)")
            return

        # Load images into numpy array
        images = []
        for img_file in image_files:
            img = Image.open(img_file)
            img_array = np.array(img)

            # Convert grayscale to RGB if needed (wandb.Video expects 3 channels)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

            images.append(img_array)

        # Add white frames at start for easy loop detection
        if images:
            # Create white frames with same shape as first image
            white_frame = np.ones_like(images[0]) * 255

            # Add 3 white frames at start and 2 at end
            images = [white_frame] * 3 + images + [white_frame] * 2

        # Convert to numpy array with shape (frames, height, width, channels)
        video_array = np.stack(images, axis=0)

        # Reorder to (frames, channels, height, width) as expected by wandb.Video
        video_array = np.transpose(video_array, (0, 3, 1, 2))

        # Parse agent type and index from agent_id (e.g., "plunger_1" -> "plunger", "1")
        agent_parts = agent_id.split("_")
        agent_type = agent_parts[0]
        agent_index = agent_parts[1]

        # Use agent_id as the wandb key for unique logging
        wandb.log({
            f"agent_vision/{agent_id}": wandb.Video(
                video_array,
                fps=fps,
                format="gif",
                caption=f"{agent_type} {agent_index}, iteration {iteration_num}"
            )
        })
        print(f"Logged {agent_id} vision video to Wandb for iteration {iteration_num}")

    except Exception as e:
        print(f"Error logging videos to Wandb: {e}")
        import traceback
        traceback.print_exc()