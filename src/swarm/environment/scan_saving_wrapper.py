"""
Wrapper around MultiAgentEnvWrapper to save scan images at each step during inference.

This wrapper captures the num_dots-1 charge stability diagram scans at each timestep
and saves them as a single PNG image with all scans arranged in a row.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import matplotlib as mpl

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


class ScanSavingWrapper(MultiAgentEnvWrapper):
    """
    Wrapper that saves all charge stability diagram scans at each step.

    At each step, saves num_dots-1 scans in a single horizontal PNG image.
    Inherits from MultiAgentEnvWrapper and has identical signature.
    """

    def __init__(
        self,
        training: bool = True,
        return_voltage: bool = False,
        gif_config: dict = None,
        distance_data_dir: str = None,
        env_config_path: str = None,
        scan_save_dir: str = None,
        scan_save_enabled: bool = True,
    ):
        """
        Initialize scan saving wrapper with same signature as MultiAgentEnvWrapper.

        Args:
            training: Whether in training mode
            return_voltage: If True, returns dict observation with image and voltage
            gif_config: Configuration for GIF capture
            distance_data_dir: Path to directory for saving distance data
            env_config_path: Optional path to custom env config file
            scan_save_dir: Directory to save scan images (new parameter)
            scan_save_enabled: Whether to enable scan saving (new parameter)
        """
        # Initialize parent MultiAgentEnvWrapper
        super().__init__(
            training=training,
            return_voltage=return_voltage,
            gif_config=gif_config,
            distance_data_dir=distance_data_dir,
            env_config_path=env_config_path,
        )

        # Scan saving specific attributes
        self.scan_save_dir = Path(scan_save_dir) if scan_save_dir else None
        self.scan_save_enabled = scan_save_enabled and (scan_save_dir is not None)
        self.step_count = 0
        self.episode_count = 0

        if self.scan_save_enabled:
            self.scan_save_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
            print(f"[ScanSavingWrapper] Scan images will be saved to: {self.scan_save_dir.absolute()}")

    def reset(self, *, seed=None, options=None):
        """Reset environment and reset step counter."""
        self.step_count = 0
        self.episode_count += 1

        # Create episode-specific directory
        if self.scan_save_enabled:
            self.episode_dir = self.scan_save_dir / f"episode_{self.episode_count:04d}"
            self.episode_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        return super().reset(seed=seed, options=options)

    def step(self, agent_actions: Dict[str, np.ndarray]):
        """
        Step environment and save scan images.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Standard gym step return: (observations, rewards, terminated, truncated, infos)
        """
        # Call parent step
        obs, rewards, terminated, truncated, infos = super().step(agent_actions)

        # Save scans if enabled
        if self.scan_save_enabled:
            self._save_scans(obs)

        self.step_count += 1
        return obs, rewards, terminated, truncated, infos

    def _save_scans(self, agent_observations: Dict[str, np.ndarray]):
        """
        Save all scans from current observation as a single horizontal image.

        Args:
            agent_observations: Multi-agent observations dict
        """
        # Extract scans from barrier agents (each has 1 channel)
        # We have num_gates-1 barrier agents, each corresponding to one scan
        num_scans = self.num_gates - 1
        scans = []

        for i in range(num_scans):
            barrier_agent_id = f"barrier_{i}"
            agent_obs = agent_observations[barrier_agent_id]

            # Handle both dict and array observation formats
            if isinstance(agent_obs, dict):
                scan = agent_obs['image'][:, :, 0]  # Extract single channel
            else:
                scan = agent_obs[:, :, 0]  # Extract single channel

            scans.append(scan)

        # Convert scans to RGB images using plasma colormap
        scan_images = []
        for scan in scans:
            # Normalize to 0-1 for colormap
            scan_norm = (scan - scan.min()) / (scan.max() - scan.min() + 1e-8)

            # Apply plasma colormap and convert to RGB
            plasma_cmap = mpl.colormaps['plasma']
            plasma_cm = plasma_cmap(scan_norm)
            plasma_rgb = (plasma_cm[:, :, :3] * 255).astype(np.uint8)

            scan_images.append(plasma_rgb)

        # Create white spacer between scans (10 pixels wide)
        spacer_width = 10
        h, w = scan_images[0].shape[:2]
        spacer = np.ones((h, spacer_width, 3), dtype=np.uint8) * 255

        # Concatenate horizontally with spacers
        parts = []
        for i, img in enumerate(scan_images):
            parts.append(img)
            if i < len(scan_images) - 1:  # Don't add spacer after last image
                parts.append(spacer)

        merged_image = np.concatenate(parts, axis=1)

        # Save as PNG
        pil_image = Image.fromarray(merged_image, mode='RGB')
        filename = self.episode_dir / f"step_{self.step_count:06d}.png"
        pil_image.save(filename)

        # Debug logging for first few saves
        if self.step_count <= 3 or self.step_count % 20 == 0:
            print(f"[ScanSavingWrapper] Saved {num_scans} scans for step {self.step_count} to {filename}")
