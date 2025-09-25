"""
GIF logging utilities for episode visualization during training.

This module handles the conversion of episode scan data to wandb-compatible
GIF format for qualitative assessment of agent performance.
"""

import numpy as np
import wandb


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

        print(f"[GIF Debug] Iteration {iteration+1}")
        print(f"[GIF Debug] env_runners keys: {list(env_runner_results.keys())}")
        print(f"[GIF Debug] custom_metrics keys: {list(custom_metrics.keys())}")

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

                print(f"[GIF Processing] Successfully logged GIF to wandb: "
                      f"{frames_uint8.shape}, iteration {iteration+1}")
            else:
                print(f"[GIF Processing] Unexpected frame data shape: {frames_data.shape}")

    except Exception as e:
        print(f"[GIF Processing] Error processing episode GIF: {e}")
        # Don't raise to avoid disrupting training