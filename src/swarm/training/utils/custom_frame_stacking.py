"""
custom frame stacking class to handle input observations to the Transformer
stacks the last max_seq_len observations and returns them alongside an attention mask
"""

from functools import partial
from typing import Any, Dict, List, Optional
import numpy as np
import gymnasium as gym
import tree
from ray.rllib.connectors.common.frame_stacking import FrameStacking
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


# NOTE attention_mask is True where there is padding

class CustomFrameStacking(FrameStacking):
    """Custom frame stacking connector that supports Dict observation spaces.

    Extends RLlib's FrameStacking to handle Dict spaces with 'image' and 'voltage' keys.
    """

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        num_frames=1,
        multi_agent=False,
        as_learner_connector=False,
        **kwargs,
    ):
        """Initialize custom frame stacking connector.

        Args:
            num_frames: Number of observation frames to stack.
            multi_agent: Whether this operates on multi-agent observation space.
            as_learner_connector: Whether this is a learner connector pipeline.
        """
        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            num_frames=num_frames,
            multi_agent=multi_agent,
            as_learner_connector=as_learner_connector,
            **kwargs,
        )

    @override(FrameStacking)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        """Perform frame stacking on Dict observations with 'image' and 'voltage' keys.

        Stacks historical observations where:
        - images: (H, W, C) -> (num_frames, H, W, C)
        - voltages: (1,) -> (num_frames,)

        Only applies to plunger agents; barrier agents are passed through unchanged.
        """
        if self._as_learner_connector:
            for sa_episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False
            ):
                # For barrier agents, pass observations through unchanged (no frame stacking)
                if "barrier" in sa_episode.agent_id.lower():
                    # Get observations for training (excluding last observation)
                    # This matches what we'd get without frame stacking connector
                    obs = sa_episode.get_observations(
                        indices=slice(0, len(sa_episode)),
                    )
                    # Add unchanged observations to batch
                    self.add_n_batch_items(
                        batch=batch,
                        column=Columns.OBS,
                        items_to_add=obs,
                        num_items=len(sa_episode),
                        single_agent_episode=sa_episode,
                    )
                    continue

                # Get all observations from episode (except last one not needed for learning)
                # In learner connector mode, this returns a dict of numpy arrays:
                # {"image": np.array(shape=(T, H, W, C)), "voltage": np.array(shape=(T, 1))}
                obs_history = sa_episode.get_observations(
                    indices=slice(-self.num_frames + 1, len(sa_episode)),
                    neg_index_as_lookback=True,
                    fill=None, # disable default padding
                )

                images = obs_history["image"]  # Shape: (actual_frames, H, W, C)
                voltages = obs_history["voltage"]  # Shape: (actual_frames, 1)

                actual_num_frames = images.shape[0]
                batch_size = len(sa_episode)
                H, W, C = images.shape[1:]

                # For sliding windows, we need batch_size + num_frames - 1 total frames
                # to create batch_size windows of num_frames each
                required_frames = batch_size + self.num_frames - 1

                if actual_num_frames >= required_frames:
                    # No padding needed, use stride tricks for efficient stacking
                    stacked_images = np.lib.stride_tricks.as_strided(
                        images,
                        shape=(batch_size, self.num_frames, H, W, C),
                        strides=(images.strides[0],) + images.strides,
                    ).copy()
                    stacked_voltages = np.lib.stride_tricks.as_strided(
                        voltages.squeeze(-1),
                        shape=(batch_size, self.num_frames),
                        strides=(voltages.strides[0], voltages.strides[0]),
                    ).copy()
                    attention_mask = np.zeros((batch_size, self.num_frames), dtype=np.int8)
                else:
                    # Need padding - pad at the beginning to reach required frames
                    num_padding = required_frames - actual_num_frames

                    # Pad images: prepend zeros
                    padding_images = np.zeros((num_padding, H, W, C), dtype=images.dtype)
                    padded_images = np.concatenate([padding_images, images], axis=0)

                    # Pad voltages: prepend zeros
                    voltages_squeezed = voltages.squeeze(-1)
                    padding_voltages = np.zeros((num_padding,), dtype=voltages_squeezed.dtype)
                    padded_voltages = np.concatenate([padding_voltages, voltages_squeezed], axis=0)

                    # Use stride tricks on padded data
                    stacked_images = np.lib.stride_tricks.as_strided(
                        padded_images,
                        shape=(batch_size, self.num_frames, H, W, C),
                        strides=(padded_images.strides[0],) + padded_images.strides,
                    ).copy()
                    stacked_voltages = np.lib.stride_tricks.as_strided(
                        padded_voltages,
                        shape=(batch_size, self.num_frames),
                        strides=(padded_voltages.strides[0], padded_voltages.strides[0]),
                    ).copy()

                    # Create attention mask: varies per timestep based on available history
                    # For each timestep t, we look back num_frames steps
                    # If we padded num_padding frames at the start, early timesteps will have padding
                    attention_mask = np.zeros((batch_size, self.num_frames), dtype=np.int8)
                    for t in range(batch_size):
                        # Frames available for timestep t (including padding) is: t + num_frames
                        # But we only have num_padding + actual_num_frames total
                        # The window for timestep t starts at index t
                        # So frames in the window are [t, t+1, ..., t+num_frames-1]
                        # Padding exists in indices [0, num_padding-1]
                        # So for this window, count how many indices are in the padding range
                        for f in range(self.num_frames):
                            frame_idx = t + f
                            if frame_idx < num_padding:
                                attention_mask[t, f] = True  # This frame is padding

                stacked_obs = {
                    "image": stacked_images,
                    "voltage": stacked_voltages,
                    "attention_mask": attention_mask,
                }

                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.OBS,
                    items_to_add=stacked_obs,
                    num_items=batch_size,
                    single_agent_episode=sa_episode,
                )

        # Env-to-module pipeline. Episodes still operate on lists.
        else:
            for sa_episode in self.single_agent_episode_iterator(episodes):
                assert not sa_episode.is_numpy

                # For barrier agents, pass observations through unchanged (no frame stacking)
                if "barrier" in sa_episode.agent_id.lower():
                    # Get latest observation without frame stacking
                    obs = sa_episode.get_observations(indices=-1)
                    # Add unchanged observation to batch
                    self.add_batch_item(
                        batch=batch,
                        column=Columns.OBS,
                        item_to_add=obs,
                        single_agent_episode=sa_episode,
                    )
                    continue

                # Get the list of last num_frames observations to stack
                obs_stack = sa_episode.get_observations(
                    indices=slice(-self.num_frames, None),
                    fill=None,
                )

                # obs_stack is a list of observation dicts: [obs1_dict, obs2_dict, ...]
                # Each dict has keys 'image' and 'voltage'
                images = [obs["image"] for obs in obs_stack]  # List of (H, W, C) arrays
                voltages = [obs["voltage"] for obs in obs_stack]  # List of (1,) arrays

                actual_num_frames = len(images)

                # Create attention mask: False = real, True = padding
                if actual_num_frames >= self.num_frames:
                    # Truncate to num_frames (take most recent)
                    stacked_images = np.stack(images[-self.num_frames:], axis=0)
                    stacked_voltages = np.array([v[0] for v in voltages[-self.num_frames:]], dtype=np.float32)
                    attention_mask = np.zeros(self.num_frames, dtype=np.int8)
                else:
                    # Need padding - pad at the beginning with zeros
                    num_padding = self.num_frames - actual_num_frames

                    # Get shape from first real observation
                    H, W, C = images[0].shape

                    # Pad images: prepend zeros
                    padding_images = [np.zeros((H, W, C), dtype=images[0].dtype) for _ in range(num_padding)]
                    padded_images = padding_images + images

                    # Pad voltages: prepend zeros
                    padding_voltages = [0.0] * num_padding
                    padded_voltages = padding_voltages + [v[0] for v in voltages]

                    # Stack
                    stacked_images = np.stack(padded_images, axis=0)
                    stacked_voltages = np.array(padded_voltages, dtype=np.float32)

                    # Attention mask: True for padding, False for real
                    attention_mask = np.array([True] * num_padding + [False] * actual_num_frames, dtype=np.int8)

                stacked_obs = {
                    "image": stacked_images,
                    "voltage": stacked_voltages,
                    "attention_mask": attention_mask,
                }

                self.add_batch_item(
                    batch=batch,
                    column=Columns.OBS,
                    item_to_add=stacked_obs,
                    single_agent_episode=sa_episode,
                )

        return batch

    @override(FrameStacking)
    def _convert_individual_space(self, obs_space):
        """Convert observation space to support frame stacking with attention mask.

        Transforms Dict observation space with 'image' and 'voltage' into stacked format
        and adds an 'attention_mask' to indicate which frames are real vs padded.

        For barrier agents (identified by 1-channel images), returns the original space unchanged.
        """
        assert isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces and "voltage" in obs_space.spaces, obs_space

        image_space = obs_space["image"]
        voltage_space = obs_space["voltage"]

        num_channels = image_space.shape[-1]
        assert num_channels in [1, 2]

        # Barrier agents (1 channel) - return original space unchanged
        if num_channels == 1:
            return obs_space

        # Plunger agents (2 channels) - apply frame stacking
        return gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=image_space.low.flat[0],
                high=image_space.high.flat[0],
                shape=(self.num_frames,) + image_space.shape,
                dtype=image_space.dtype,
            ),
            "voltage": gym.spaces.Box(
                low=voltage_space.low[0],
                high=voltage_space.high[0],
                shape=(self.num_frames,),
                dtype=voltage_space.dtype
            ),
            "attention_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_frames,),
                dtype=np.int8,
            ),
        })


CustomFrameStackingEnvToModule = partial(CustomFrameStacking, as_learner_connector=False)
CustomFrameStackingLearner = partial(CustomFrameStacking, as_learner_connector=True)