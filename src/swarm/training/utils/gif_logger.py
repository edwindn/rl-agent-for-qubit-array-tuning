"""
GIF logging utilities for episode visualization during training.

This module handles the conversion of episode scan data to wandb-compatible
GIF format for qualitative assessment of agent performance.
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
        # Debug: Print what we're getting in the result
        env_runner_results = result.get("env_runners", {})
        custom_metrics = env_runner_results.get("custom_metrics", {})


        # Look for gif_frames in the custom metrics
        if "gif_frames" in custom_metrics:
            frames_data = custom_metrics["gif_frames"]
            channel = custom_metrics.get("gif_channel", 0)
            num_frames = custom_metrics.get("gif_num_frames", 0)

            print(f"[GIF Processing] Found GIF data: {frames_data.shape}, "
                  f"channel {channel}, {num_frames} frames")

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

            else:
                print(f"Unexpected frame data shape: {frames_data.shape}")

    except Exception as e:
        print(f"Error processing episode GIF: {e}")
        # Don't raise to avoid disrupting training


def cleanup_gif_lock_file(gif_save_dir=None):
    """Remove gif capture lock file and images from previous training runs."""
    import os

    # Clean up lock file
    lock_file = "/tmp/gif_capture_worker.lock"
    try:
        os.remove(lock_file)
        print("Cleaned up previous GIF capture lock file")
    except FileNotFoundError:
        pass  # Already gone, that's fine
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
    gif_save_dir = Path(config['gif_capture']['save_dir'])

    if not gif_save_dir.exists() or not any(gif_save_dir.glob("step_*.png")):
        return

    try:
        print(f"Processing GIFs for iteration {iteration_num}...")

        # Get all saved images
        image_files = sorted(gif_save_dir.glob("step_*.png"))

        if not image_files:
            print("No images found for GIF creation")
            return

        # Group images by channel
        channel_files = {}
        for img_file in image_files:
            # Parse filename: step_XXXXXX_channel_Y.png
            parts = img_file.stem.split('_')
            if len(parts) >= 4:
                channel = int(parts[3])  # channel number
                if channel not in channel_files:
                    channel_files[channel] = []
                channel_files[channel].append(img_file)

        # Create numpy arrays and log to Wandb
        if use_wandb and channel_files:
            _log_images_as_video_to_wandb(channel_files, iteration_num, config)

        # Clean up temporary files
        shutil.rmtree(gif_save_dir, ignore_errors=True)
        print(f"Processed {len(channel_files)} channels and cleaned up images")

    except Exception as e:
        print(f"Error processing GIFs: {e}")
        # Don't raise to avoid disrupting training


def _log_images_as_video_to_wandb(channel_files, iteration_num, config):
    """Convert images to numpy arrays and log as videos to Wandb."""
    try:
        import numpy as np
        from PIL import Image

        log_dict = {}
        fps = config['gif_capture'].get('fps', 0.5)  # Default to 0.5 if not specified

        for channel, files in channel_files.items():
            if not files:
                continue

            # Sort files by step number
            sorted_files = sorted(files, key=lambda x: int(x.stem.split('_')[1]))

            if len(sorted_files) < 2:
                print(f"Not enough images for channel {channel} video (need at least 2)")
                continue

            # Load images into numpy array
            images = []
            for img_file in sorted_files:
                img = Image.open(img_file)
                img_array = np.array(img)

                # Convert grayscale to RGB if needed (wandb.Video expects 3 channels)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)

                images.append(img_array)

            # Add black frames at start for easy loop detection
            if images:
                # Create black frames with same shape as first image
                black_frame = np.zeros_like(images[0])

                # Add 3 black frames at start and 2 at end
                images = [black_frame] * 3 + images + [black_frame] * 2

            # Convert to numpy array with shape (frames, height, width, channels)
            video_array = np.stack(images, axis=0)

            # Reorder to (frames, channels, height, width) as expected by wandb.Video
            video_array = np.transpose(video_array, (0, 3, 1, 2))

            # Create wandb.Video with configurable framerate
            log_key = f"agent_vision_channel_{channel}"
            log_dict[log_key] = wandb.Video(
                video_array,
                fps=fps,
                format="gif",
                caption=f"Agent vision channel {channel}, iteration {iteration_num}"
            )

        # Add iteration info
        if log_dict:
            log_dict["gif_iteration"] = iteration_num
            wandb.log(log_dict)
            print(f"Logged {len(log_dict)-1} video channels to Wandb for iteration {iteration_num}")

    except Exception as e:
        print(f"Error logging videos to Wandb: {e}")