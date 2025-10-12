"""
Multi-agent wrapper for QuantumDeviceEnv.

This wrapper converts the global observation/action spaces into individual agent spaces
and handles the conversion between single-agent actions and global environment actions.
"""

import sys
import glob
import random
from typing import Dict

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Add src directory to path for clean imports
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


class MultiAgentEnvWrapper(MultiAgentEnv):
    """
    Multi-agent wrapper that converts global env to individual agent interactions.

    Each agent sees:
    - Gate agents: 2-channel image (corresponding to their dot pairs) + single voltage
    - Barrier agents: 1-channel image (corresponding to adjacent dots) + single voltage

    Each agent outputs:
    - A single voltage value (gate or barrier)

    The wrapper combines individual agent actions into global environment actions.

    Internally converts between voltage delta outputs and voltages to be applied to the device (or simulator)
    """

    def __init__(
        self,
        training: bool = True,
        return_voltage: bool = False,
        gif_config: dict = None,
        distance_data_dir: str = None,
    ):
        """
        Initialize multi-agent wrapper.

        Automatically infers the array size from the underlying base env

        Args:
            training: Whether in training mode
            return_voltage: If True, returns dict observation with image, voltage, and is_plunger.
                          If False, returns only the image array.
        """
        super().__init__()

        # if store_history and not return_voltage:
        #     print("WARNING: 'store_history' in MultiAgentEnvWrapper is intended to work with 'return_voltage' only. Setting return_voltage=True.")
        #     return_voltage = True

        self.return_voltage = return_voltage

        self.store_history = False

        self.distance_data_dir = distance_data_dir
        self.distance_history = None

        self.gif_config = gif_config
        if self.gif_config is not None:
            self._init_gif_capture()

        self.base_env = QuantumDeviceEnv(training=training)

        self.num_gates = self.base_env.num_dots
        self.use_barriers = self.base_env.use_barriers
        self.num_barriers = self.base_env.num_dots - 1
        self.num_image_channels = self.base_env.num_dots - 1  # N-1 charge stability diagrams

        # Create agent IDs (0-indexed to match expected format)
        self.gate_agent_ids = [f"plunger_{i}" for i in range(self.num_gates)]
        self.barrier_agent_ids = [f"barrier_{i}" for i in range(self.num_barriers)]
        self.all_agent_ids = self.gate_agent_ids + self.barrier_agent_ids

        if self.store_history:
            # Only store history for plunger agents (gate agents)
            self.plunger_agent_history = {agent_id: [] for agent_id in self.gate_agent_ids}

        if self.distance_data_dir is not None:
            distance_data_path = Path(self.distance_data_dir)
            for agent_id in self.all_agent_ids:
                agent_folder = distance_data_path / agent_id
                agent_folder.mkdir(parents=False, exist_ok=True)

        # Setup channel assignments for agents
        self._setup_channel_assignments()

        # Preserve original spaces for policy mapping
        self.base_observation_space = self.base_env.observation_space
        self.base_action_space = self.base_env.action_space

        # Create individual agent spaces
        self._create_agent_spaces(self.base_observation_space, self.base_action_space)

    def _setup_channel_assignments(self):
        """
        Assign image channels to individual agents.

        For N quantum dots, we have N-1 image channels (charge stability diagrams).
        Gate channel assignment strategy:
        - First gate (plunger_0): Gets [0, 0] (first channel twice)
        - Middle gates: Get adjacent pairs [i-1, i]
        - Last gate (plunger_N-1): Gets [N-2, N-2] (last channel twice)
        - Barrier agents get 1 channel: the channel for dots they separate
        """
        self.agent_channel_map = {}

        # Gate agents: special assignment for ends, pairs for middle
        for agent_id in self.gate_agent_ids:
            i = int(agent_id.split("_")[1])
            if i == 0:
                # First gate agent: first channel twice
                self.agent_channel_map[agent_id] = [0, 0]
            elif i == self.num_gates - 1:
                # Last gate agent: last channel twice
                last_channel = self.num_gates - 2  # N-1 channels, so index N-2
                self.agent_channel_map[agent_id] = [last_channel, last_channel]
            else:
                # Middle gate agents: adjacent channel pairs [i-1, i]
                # Gate 1 gets [0, 1], Gate 2 gets [1, 2], etc.
                self.agent_channel_map[agent_id] = [i - 1, i]

        # Barrier agents: each gets 1 channel for the dots they separate
        for agent_id in self.barrier_agent_ids:
            i = int(agent_id.split("_")[1])
            self.agent_channel_map[agent_id] = [i]  # Barrier i separates dots i and i+1

    def _create_agent_spaces(self, base_obs, base_action):
        """Create observation and action spaces for individual agents."""
        image_shape = base_obs["image"].shape  # (H, W, N-1)

        # Voltage ranges (should all be -1 to 1)
        gate_low = base_action["action_gate_voltages"].low[0]
        gate_high = base_action["action_gate_voltages"].high[0]
        barrier_low = base_action["action_barrier_voltages"].low[0]
        barrier_high = base_action["action_barrier_voltages"].high[0]

        # Create spaces for each agent
        self.observation_spaces = {}
        self.action_spaces = {}

        if self.return_voltage:
            # Gate agents: Dict observation with image, voltage, and is_plunger
            for agent_id in self.gate_agent_ids:
                self.observation_spaces[agent_id] = spaces.Dict({
                    'image': spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(image_shape[0], image_shape[1], 2),  # 2 channels
                        dtype=np.float32,
                    ),
                    'voltage': spaces.Box(
                        low=gate_low,
                        high=gate_high,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                })

                self.action_spaces[agent_id] = spaces.Box(
                    low=gate_low,
                    high=gate_high,
                    shape=(1,),  # Single voltage output
                    dtype=np.float32,
                )

            # Barrier agents: Dict observation with image, voltage, and is_plunger
            for agent_id in self.barrier_agent_ids:
                self.observation_spaces[agent_id] = spaces.Dict({
                    'image': spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(image_shape[0], image_shape[1], 1),  # 1 channel
                        dtype=np.float32,
                    ),
                    'voltage': spaces.Box(
                        low=barrier_low,
                        high=barrier_high,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                })

                self.action_spaces[agent_id] = spaces.Box(
                    low=barrier_low,
                    high=barrier_high,
                    shape=(1,),  # Single voltage output
                    dtype=np.float32,
                )
        else:
            # Gate agents: 2-channel images only
            for agent_id in self.gate_agent_ids:
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(image_shape[0], image_shape[1], 2),  # 2 channels
                    dtype=np.float32,
                )

                self.action_spaces[agent_id] = spaces.Box(
                    low=gate_low,
                    high=gate_high,
                    shape=(1,),  # Single voltage output
                    dtype=np.float32,
                )

            # Barrier agents: 1-channel images only
            for agent_id in self.barrier_agent_ids:
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(image_shape[0], image_shape[1], 1),  # 1 channel
                    dtype=np.float32,
                )

                self.action_spaces[agent_id] = spaces.Box(
                    low=barrier_low,
                    high=barrier_high,
                    shape=(1,),  # Single voltage output
                    dtype=np.float32,
                )

        self.observation_spaces = spaces.Dict(**self.observation_spaces)
        self.action_spaces = spaces.Dict(**self.action_spaces)

        # Set required MultiAgentEnv properties
        self._agent_ids = set(self.all_agent_ids)
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces

        self.agents = self._agent_ids.copy()
        self.possible_agents = self._agent_ids.copy()

    def _extract_agent_observation(
        self, global_obs: Dict[str, np.ndarray], agent_id: str, device_state_info: dict = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract individual agent observation from global observation.

        Args:
            global_obs: Global environment observation
            agent_id: ID of the agent
            device_state_info: Device state info containing voltage and ground truth values

        Returns:
            Individual agent observation
        """
        channels = self.agent_channel_map[agent_id]

        # Extract appropriate channels for this agent
        global_image = global_obs["image"]  # Shape: (H, W, N-1)

        if len(channels) == 2:
            # Gate agent: 2 channels with conditional y-axis flipping
            agent_idx = int(agent_id.split("_")[1])
            img1 = global_image[:, :, channels[0]]
            img2 = global_image[:, :, channels[1]]

            if agent_idx == 0:
                # First agent: no flipping
                agent_image = np.stack([img1, img2], axis=2)
            elif agent_idx == self.num_gates - 1:
                # Final agent: flip both images
                img1 = np.transpose(img1, (1, 0))
                img2 = np.transpose(img2, (1, 0))
                agent_image = np.stack([img1, img2], axis=2)
            else:
                # Middle agents: flip only second image
                img2 = np.transpose(img2, (1, 0))
                agent_image = np.stack([img1, img2], axis=2)
        else:
            # Barrier agent: 1 channel
            agent_image = global_image[:, :, channels[0] : channels[0] + 1]


        if (self.gif_config is not None and hasattr(self, 'should_capture_gifs') and self.should_capture_gifs and self._is_target_agent(agent_id)):
            self._save_agent_image(agent_image, agent_id, device_state_info)
            
        if self.return_voltage:
            # Get agent's current voltage value
            if "plunger" in agent_id:
                agent_idx = int(agent_id.split("_")[1])
                voltage = global_obs["obs_gate_voltages"][agent_idx : agent_idx + 1]
                # is_plunger = np.array([1.0], dtype=np.float32)
            else:  # barrier agent
                agent_idx = int(agent_id.split("_")[1])
                voltage = global_obs["obs_barrier_voltages"][agent_idx : agent_idx + 1]
                # is_plunger = np.array([0.0], dtype=np.float32)

            return {
                'image': agent_image.astype(np.float32),
                'voltage': voltage.astype(np.float32),
            }
        else:
            # Return image only
            return agent_image.astype(np.float32)
          

    def _combine_agent_actions(self, agent_actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Combine individual agent actions into global environment action.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Global environment action
        """
        # Initialize action arrays
        gate_actions = np.zeros(self.num_gates, dtype=np.float32)
        barrier_actions = np.zeros(self.num_barriers, dtype=np.float32)

        # Collect gate actions
        for agent_id in self.gate_agent_ids:
            if agent_id in agent_actions:
                i = int(agent_id.split("_")[1])
                action_value = agent_actions[agent_id]
                # Handle both scalar and array inputs
                if hasattr(action_value, "__len__"):
                    gate_actions[i] = float(action_value[0])
                else:
                    gate_actions[i] = float(action_value)

        # Collect barrier actions
        for agent_id in self.barrier_agent_ids:
            if agent_id in agent_actions:
                i = int(agent_id.split("_")[1])
                action_value = agent_actions[agent_id]
                # Handle both scalar and array inputs
                if hasattr(action_value, "__len__"):
                    barrier_actions[i] = float(action_value[0])
                else:
                    barrier_actions[i] = float(action_value)

        return {
            "action_gate_voltages": gate_actions,
            "action_barrier_voltages": barrier_actions,
        }

    def _distribute_rewards(self, global_rewards: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Distribute global rewards to individual agents.

        Args:
            global_rewards: Global reward dictionary with 'gates' and 'barriers' arrays

        Returns:
            Dictionary mapping agent IDs to individual rewards
        """
        agent_rewards = {}

        # Distribute gate rewards
        if "gates" in global_rewards:
            gate_rewards = global_rewards["gates"]
            for agent_id in self.gate_agent_ids:
                i = int(agent_id.split("_")[1])
                agent_rewards[agent_id] = float(gate_rewards[i])
        else:
            raise ValueError("Missing gate rewards in global_rewards")

        # Distribute barrier rewards
        if "barriers" in global_rewards:
            barrier_rewards = global_rewards["barriers"]
            for agent_id in self.barrier_agent_ids:
                i = int(agent_id.split("_")[1])
                agent_rewards[agent_id] = float(barrier_rewards[i])
        else:
            raise ValueError("Missing barrier rewards in global_rewards")

        return agent_rewards

    def reset(self, *, seed=None, options=None):
        """
        Reset environment and return individual agent observations.

        Returns:
            Tuple of (observations, infos) where infos is a per-agent dict
        """
        # Reset observation history for plunger agents only
        if self.store_history:
            self.plunger_agent_history = {agent_id: [] for agent_id in self.gate_agent_ids}

        global_obs, global_info = self.base_env.reset(seed=seed, options=options)

        if self.distance_history is not None:
            self._save_agent_histories(self.distance_history)

        if self.distance_data_dir is not None:
            self.distance_history = {_id: [] for _id in self.all_agent_ids}

        # Convert to multi-agent observations
        agent_observations = {}
        for agent_id in self.all_agent_ids:
            agent_obs = self._extract_agent_observation(global_obs, agent_id, device_state_info=None)

            # Only apply history to plunger agents
            if self.store_history and agent_id in self.gate_agent_ids:
                self.plunger_agent_history[agent_id].append(agent_obs)
                agent_observations[agent_id] = [agent_obs]
            else:
                agent_observations[agent_id] = agent_obs

        # Create per-agent info dict (MultiAgentEnv requirement)
        agent_infos = {agent_id: global_info for agent_id in self.all_agent_ids}

        return agent_observations, agent_infos

    def step(self, agent_actions: Dict[str, np.ndarray]):
        """
        Step environment with individual agent actions.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        assert len(agent_actions) == len(
            self.all_agent_ids
        ), "Agent actions must match the number of agents"
        assert all(
            agent_id in self.all_agent_ids for agent_id in agent_actions.keys()
        ), "Unknown agent IDs in actions"

        # print("[ENV DEBUG] env.step called")
        # Combine agent actions into global action
        global_action = self._combine_agent_actions(agent_actions)

        # Step the base environment
        global_obs, global_rewards, terminated, truncated, info = self.base_env.step(global_action)

        # Get device state info first (needed for image capture)
        device_state_info = info.get("current_device_state", None)

        # Convert to multi-agent format
        agent_observations = {}
        for agent_id in self.all_agent_ids:
            agent_obs = self._extract_agent_observation(global_obs, agent_id, device_state_info)

            # Only apply history to plunger agents
            if self.store_history and agent_id in self.gate_agent_ids:
                self.plunger_agent_history[agent_id].append(agent_obs)
                agent_observations[agent_id] = self.plunger_agent_history[agent_id]
            else:
                agent_observations[agent_id] = agent_obs

        agent_rewards = self._distribute_rewards(global_rewards)

        # Multi-agent termination/truncation (all agents have same status)
        agent_terminated = dict.fromkeys(self.all_agent_ids, terminated)
        agent_terminated["__all__"] = terminated  # Required by MultiAgentEnv

        agent_truncated = dict.fromkeys(self.all_agent_ids, truncated)
        agent_truncated["__all__"] = truncated  # Required by MultiAgentEnv

        # Create per-agent info dict (MultiAgentEnv requirement)

        if not device_state_info:
            agent_infos = {agent_id: {} for agent_id in self.all_agent_ids}
        else:
            try:
                agent_infos = {}
                plunger_ids = [f"plunger_{i}" for i in range(self.num_gates)]
                barrier_ids = [f"barrier_{i}" for i in range(self.num_barriers)]

                for idx, agent_id in enumerate(plunger_ids):
                    ground_truth = device_state_info["gate_ground_truth"][idx]
                    current_voltage = device_state_info["current_gate_voltages"][idx]
                    
                    agent_infos[agent_id] = {
                        "ground_truth": ground_truth,
                        "current_voltage": current_voltage,
                    }

                    if self.distance_data_dir is not None:
                        self.distance_history[agent_id].append(current_voltage - ground_truth)
                
                for idx, agent_id in enumerate(barrier_ids):
                    ground_truth = device_state_info["barrier_ground_truth"][idx]
                    current_voltage = device_state_info["current_barrier_voltages"][idx]

                    agent_infos[agent_id] = {
                        "ground_truth": ground_truth,
                        "current_voltage": current_voltage,
                    }

                    if self.distance_data_dir is not None:
                        self.distance_history[agent_id].append(current_voltage - ground_truth)

            except Exception as e:
                agent_infos = dict.fromkeys(self.all_agent_ids, {
                    "error": f"Error creating multi-agent info: {e}"
                })
                
        return (
            agent_observations,
            agent_rewards,
            agent_terminated,
            agent_truncated,
            agent_infos,
        )

    
    def _save_agent_histories(self, history: dict):
        assert set(history.keys()) == set(self.all_agent_ids), "Mismatch in agent ids in saved history"

        distance_data_path = Path(self.distance_data_dir)

        for agent_id in self.all_agent_ids:
            dists = history[agent_id]
            dists = np.array(dists)

            # Get agent folder
            agent_folder = distance_data_path / agent_id

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
            np.save(filepath, dists)

    # def _get_obs_images(self, obs: Dict[str, Union[np.ndarray, torch.tensor]]):
    #     barrier_keys = [k for k in obs.keys() if k.lower().startswith('barrier')]
    #     assert len(barrier_keys) == len(self.barrier_agent_ids), "Mismatch between barrier agents and provided observation"
    #     channels = []
    #     for agent_id in self.barrier_agent_ids:
    #         agent_obs = obs[agent_id]
    #         if isinstance(agent_obs, torch.Tensor):
    #             agent_obs = agent_obs.numpy()
    #         channels.append(agent_obs)

    #     channels = np.stack(channels, axis=-1)
    #     channels = np.squeeze(channels)
    #     return channels


    def _init_gif_capture(self):
        """Initialize GIF capture system if this worker is selected."""
        import os

        # GIF capture state
        self.should_capture_gifs = False
        self.gif_step_count = 0

        # Check if we're the selected worker for GIF capture
        if self._is_first_env_runner():
            # Load gif capture config from environment variables (set by Ray runtime_env)
            # self.gif_config = {
            #     "enabled": os.getenv("GIF_CAPTURE_ENABLED", "false").lower() == "true",
            #     "target_agent_type": os.getenv("GIF_CAPTURE_AGENT_TYPE", "plunger"),
            #     "target_agent_index": int(os.getenv("GIF_CAPTURE_AGENT_INDEX", "1")),
            #     "save_dir": os.getenv("GIF_CAPTURE_SAVE_DIR", "./gif_captures")
            # }

            if self.gif_config["enabled"]:
                self.should_capture_gifs = True
                self._setup_gif_directories()
                target_agents = ", ".join(self._get_target_agent_ids())
                print(f"[PID {os.getpid()}] Selected as GIF capture worker - targeting: {target_agents}")
                print(f"[PID {os.getpid()}] Will save images to: {os.path.abspath(self.gif_config['save_dir'])}")
                print(f"[PID {os.getpid()}] Current working directory: {os.getcwd()}")
            else:
                print(f"[PID {os.getpid()}] GIF capture disabled in config")
        else:
            pass

    def _is_first_env_runner(self):
        """Check if this is the first env runner using atomic file creation."""
        if not self._is_env_runner_worker():
            return False

        import os

        lock_file = "/tmp/gif_capture_worker.lock"

        try:
            # Check for stale lock first
            if os.path.exists(lock_file):
                try:
                    with open(lock_file, 'r') as f:
                        old_pid = int(f.read().strip())
                    # Check if that process still exists
                    try:
                        os.kill(old_pid, 0)
                        # Process still exists - race lost
                        print(f"Race Lost - active worker {old_pid} exists")
                        return False
                    except OSError:
                        # Stale lock - remove it
                        os.remove(lock_file)
                        print(f"Removed stale lock from PID {old_pid}")
                except (ValueError, IOError, PermissionError):
                    # Can't read or remove - skip and try to create anyway
                    pass

            # Atomic file creation - only succeeds for first worker
            # Use 0o666 for more permissive access (subject to umask)
            lock_fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)

            # Write our PID to the lock file
            with os.fdopen(lock_fd, 'w') as f:
                f.write(str(os.getpid()))
                f.flush()
            print("Race Won")
            return True

        except OSError:
            # File already exists - another worker got there first
            print("Race Lost")
            return False

    def _is_env_runner_worker(self):
        """Check if this process is an env runner worker (not the driver)."""
        try:
            import ray
            ctx = ray.get_runtime_context()
            actor_id = ctx.get_actor_id()
            # Driver has actor_id = None, workers have actual actor IDs
            return actor_id is not None
        except:
            return False

    def _setup_gif_directories(self):
        """Set up directories for GIF capture."""
        from pathlib import Path
        import os

        base_dir = Path(self.gif_config["save_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"GIF capture directory ready: {base_dir}")

    def _get_target_agent_ids(self):
        """Get the agent IDs we're targeting for GIF capture."""
        agent_type = self.gif_config["target_agent_type"]
        # Handle both old (single index) and new (list of indices) config formats
        if "target_agent_indices" in self.gif_config:
            agent_indices = self.gif_config["target_agent_indices"]
            if not isinstance(agent_indices, list):
                agent_indices = [agent_indices]
        else:
            # Fallback for old config format
            agent_indices = [self.gif_config.get("target_agent_index", 0)]

        return [f"{agent_type}_{idx}" for idx in agent_indices]

    def _is_target_agent(self, agent_id):
        """Check if this agent is one of the targets for GIF capture."""
        return agent_id in self._get_target_agent_ids()

    def _save_agent_image(self, agent_image, agent_id, device_state_info=None):
        """Save agent image(s) to disk for GIF creation with text overlay."""
        from pathlib import Path
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib as mpl
        import os

        # Create agent-specific subdirectory
        base_save_dir = Path(self.gif_config["save_dir"])
        save_dir = base_save_dir / agent_id
        save_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        # Ensure permissions are set correctly even if directory already existed
        try:
            os.chmod(save_dir, 0o777)
        except:
            pass  # Directory may not exist or permissions may not be settable

        # Extract agent info if available
        voltage_text = ""
        ground_truth_text = ""
        if device_state_info is not None:
            agent_idx = int(agent_id.split("_")[1])
            if "plunger" in agent_id:
                voltage = device_state_info["current_gate_voltages"][agent_idx]
                ground_truth = device_state_info["gate_ground_truth"][agent_idx]
            else:  # barrier
                voltage = device_state_info["current_barrier_voltages"][agent_idx]
                ground_truth = device_state_info["barrier_ground_truth"][agent_idx]

            voltage_text = f"V: {voltage:.3f}"
            ground_truth_text = f"GT: {ground_truth:.3f}"

        # Save channels - merge side-by-side for plunger agents
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

        filename = save_dir / f"step_{self.gif_step_count:06d}.png"
        img.save(filename)

        self.gif_step_count += 1

        # Debug logging for first few saves and periodically
        if self.gif_step_count <= 3 or self.gif_step_count % 20 == 0:
            print(f"[GIF DEBUG PID {os.getpid()}] Saved step {self.gif_step_count-1} images to {save_dir.absolute()}")
            print(f"[GIF DEBUG PID {os.getpid()}] Directory contents: {[f.name for f in save_dir.glob('*')][:5]}")

    def close(self):
        """Close the base environment."""
        return self.base_env.close()

    def get_agent_ids(self):
        """Get list of all agent IDs."""
        return self.all_agent_ids.copy()


if __name__ == "__main__":
    """Test the multi-agent wrapper."""
    print("=== Testing Multi-Agent Quantum Wrapper ===")

    try:
        # Test image-only mode (return_voltage=False)
        print("\n--- Testing image-only mode ---")
        wrapper = MultiAgentEnvWrapper(training=True, return_voltage=False)
        print("✓ Created multi-agent wrapper (image-only)")

        print(f"Agent IDs: {wrapper.get_agent_ids()}")

        # Test reset
        obs, info = wrapper.reset()
        print(f"✓ Reset successful - got observations for {len(obs)} agents")

        # Check observation shapes
        for agent_id in wrapper.get_agent_ids()[:2]:  # Check first 2 agents
            agent_obs = obs[agent_id]
            print(f"  {agent_id}:")
            print(f"    Observation type: {type(agent_obs)}")
            print(f"    Observation shape: {agent_obs.shape}")

        # Test step with random actions
        actions = {}
        for agent_id in wrapper.get_agent_ids():
            actions[agent_id] = wrapper.action_spaces[agent_id].sample()

        obs, rewards, terminated, truncated, info = wrapper.step(actions)
        print(f"✓ Step successful - got {len(rewards)} agent rewards")
        wrapper.close()

        # Test dict mode (return_voltage=True)
        print("\n--- Testing dict observation mode ---")
        wrapper = MultiAgentEnvWrapper(training=True, return_voltage=True)
        print("✓ Created multi-agent wrapper (dict mode)")

        # Test reset
        obs, info = wrapper.reset()
        print(f"✓ Reset successful - got observations for {len(obs)} agents")

        # Check observation structure
        for agent_id in wrapper.get_agent_ids()[:2]:  # Check first 2 agents
            agent_obs = obs[agent_id]
            print(f"  {agent_id}:")
            print(f"    Observation type: {type(agent_obs)}")
            print(f"    Keys: {list(agent_obs.keys())}")
            print(f"    Image shape: {agent_obs['image'].shape}")
            print(f"    Voltage shape: {agent_obs['voltage'].shape}")

        # Test step with random actions
        actions = {}
        for agent_id in wrapper.get_agent_ids():
            actions[agent_id] = wrapper.action_spaces[agent_id].sample()

        obs, rewards, terminated, truncated, info = wrapper.step(actions)
        print(f"✓ Step successful - got {len(rewards)} agent rewards")

        wrapper.close()
        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
