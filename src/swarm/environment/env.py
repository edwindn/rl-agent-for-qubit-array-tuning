import os
import sys
import logging
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import spaces

# Add src directory to path for clean imports
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.qarray_base_class import QarrayBaseClass
from swarm.environment.utils.fake_capacitance import fake_capacitance_model


# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib

matplotlib.use("Agg")

from swarm.capacitance_model import CapacitancePredictionModel, CapacitancePredictor, InterpolatedCapacitancePredictor


class QuantumDeviceEnv(gym.Env):
    """
    Simulator environment that handles all gym.env related logic
        - loads in QarrayBaseClass to extract observations
        - holds a device state consisting of plunger and barrier voltages and ground truths, and virtual gate matrix
        - allows plotting of sample CSD scans
        - allows Bayesian updates of the model's internal virtual gate matrix as we gather more information about the inter-dot capacitances
    """

    def __init__(
        self,
        training=True,
        config_path="env_config.yaml",
    ):

        super().__init__()

        # environment parameters
        self.config = self._load_config(config_path)
        self.training = training  # if we are training or not
        self.num_dots = self.config['simulator']['num_dots']
        self.use_barriers = self.config['simulator']['use_barriers']
        self.use_deltas = self.config['simulator']['use_deltas']
        self.max_steps = self.config["simulator"]["max_steps"]
        self.num_plunger_voltages = self.num_dots
        self.num_barrier_voltages = self.num_dots - 1
        self.resolution = self.config['simulator']['resolution']

        #voltage params, set by _voltage_init() called in reset()
        self.plunger_max = None
        self.plunger_min = None
        self.barrier_max = None
        self.barrier_min = None
        self.window_delta = None #size of scan region

        delta_max = self.config['simulator']['delta_max']
        self.plunger_delta_max = delta_max
        self.plunger_delta_min = -delta_max

        #reward parameters
        self.gate_ramp_start = self.config["reward"]["gate_ramp_start"]
        self.gate_quadratic_start = self.config["reward"]["gate_quadratic_start"]
        self.gate_curve_type = self.config["reward"]["gate_curve_type"]
        self.gate_curve_exponent = self.config["reward"]["gate_curve_exponent"]
        self.barrier_ramp_start = self.config["reward"]["barrier_ramp_start"]

        self.action_space = spaces.Dict(
            {
                "action_gate_voltages": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_plunger_voltages,),
                    dtype=np.float32,
                ),
                "action_barrier_voltages": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_barrier_voltages,),
                    dtype=np.float32,
                ),
            }
        )

        self.obs_channels = self.num_dots - 1

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.resolution, self.resolution, self.obs_channels),
                    dtype=np.float32,
                ),
                "obs_gate_voltages": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_plunger_voltages,),
                    dtype=np.float32,
                ),
                "obs_barrier_voltages": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_barrier_voltages,),
                    dtype=np.float32,
                ),
            }
        )

        # Initialize capacitance prediction model
        self._init_capacitance_model()

        self.reset()


    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        This method is called at the beginning of each new episode. It should
        reset the state of the environment and return the first observation that
        the agent will see.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.

        Returns:
            observation (np.ndarray): The initial observation of the space.
            info (dict): A dictionary with the current device state.
        """

        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0

        window_delta_range = self.config['simulator']['window_delta_range']
        self.window_delta = np.random.uniform(
            low=window_delta_range['min'],
            high=window_delta_range['max']
        )

        self.array = QarrayBaseClass(
            num_dots=self.num_dots,
            use_barriers=self.use_barriers,
            obs_voltage_min=-self.window_delta,
            obs_voltage_max=self.window_delta,
            obs_image_size=self.resolution,
        )

        plunger_ground_truth, barrier_ground_truth, _ = self.array.calculate_ground_truth()

        if barrier_ground_truth is None:
            assert not self.use_barriers, "Expected array for barrier_ground_truth, got None"
            barrier_ground_truth = np.zeros(self.num_barrier_voltages, dtype=np.float32)
        else:
            barrer_ground_truth = np.array(barrier_ground_truth).flatten().astype(np.float32)

        plunger_ground_truth = np.array(plunger_ground_truth).flatten().astype(np.float32)

        assert len(plunger_ground_truth) == self.num_plunger_voltages, f"Expected plunger ground truth to be of length {self.num_plunger_voltages}, got {len(plunger_ground_truth)}"
        assert len(barrier_ground_truth) == self.num_barrier_voltages, f"Expected plunger ground truth to be of length {self.num_barrier_voltages}, got {len(barrier_ground_truth)}"

        self._init_voltage_ranges(
            plunger_ground_truth,
            barrier_ground_truth
        )

        plungers, barriers = self._starting_voltages()


        self.device_state = {
            "gate_ground_truth": plunger_ground_truth,
            "barrier_ground_truth": barrier_ground_truth,
            "current_gate_voltages": plungers,
            "current_barrier_voltages": barriers,
            "virtual_gate_matrix": self.array.model.gate_voltage_composer.virtual_gate_matrix,
        }

        # --- Return the initial observation ---
        raw_observation = self.array._get_obs(
            self.device_state["current_gate_voltages"],
            self.device_state["current_barrier_voltages"],
        )

        observation = self._normalise_obs(raw_observation)

        self._update_virtual_gate_matrix(observation)

        info = self._get_info()

        return observation, info


    def step(self, action: dict):
        """
        Updates the environment state based on the agent's action.

        This method is the core of the environment. It takes an action from the
        agent and calculates the next state, the reward, and whether the
        episode has ended.

        Args:
            action: An action provided by the agent.

        Returns:
            observation (np.ndarray): The observation of the environment's state.
            reward (dict): The amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., reached a goal).
            truncated (bool): Whether the episode was cut short (e.g., time limit).
            info (dict): A dictionary with auxiliary diagnostic information.
        """
        self.current_step += 1

        gate_voltages = action["action_gate_voltages"]
        barrier_voltages = action["action_barrier_voltages"]

        gate_voltages = np.array(gate_voltages).flatten().astype(np.float32)
        barrier_voltages = np.array(barrier_voltages).flatten().astype(np.float32)

        # Random actions for debugging
        # gate_voltages = np.random.uniform(low=-1.0, high=1.0, size=self.num_plunger_voltages).astype(np.float32)
        # barrier_voltages = np.random.uniform(low=-1.0, high=1.0, size=self.num_barrier_voltages).astype(np.float32)
        # gate_voltages = np.zeros(self.num_plunger_voltages).astype(np.float32)
        # barrier_voltages = np.zeros(self.num_barrier_voltages).astype(np.float32)
        
        # Rescale voltages from [-1, 1] to actual ranges
        gate_voltages = self._rescale_gate_voltages(gate_voltages)
        barrier_voltages = self._rescale_barrier_voltages(barrier_voltages)
            
        self.device_state["current_gate_voltages"] = gate_voltages
        self.device_state["current_barrier_voltages"] = barrier_voltages

        reward = self._get_reward()

        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        raw_observation = self.array._get_obs(gate_voltages, barrier_voltages)
        observation = self._normalise_obs(raw_observation)

        self._update_virtual_gate_matrix(observation)
        self.device_state["virtual_gate_matrix"] = (
            self.array.model.gate_voltage_composer.virtual_gate_matrix
        )

        info = self._get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def _get_reward(self):
        """
        Get the reward for the current state using a piecewise reward function.

        Gate rewards use a configurable piecewise function:
        - Region 1 (distance > ramp_start): reward = 0
        - Region 2 (quadratic_start < distance <= ramp_start): linear from 0 to 0.5
        - Region 3 (0 < distance <= quadratic_start): configurable curve (polynomial/exponential/linear) from 0.5 to 1.0
        - Region 4 (distance = 0): reward = 1.0

        Barrier rewards use a simple linear function from 0 (at barrier_ramp_start) to 1.0 (at ground truth).
        """

        gate_ground_truth = self.device_state["gate_ground_truth"]
        current_gate_voltages = self.device_state["current_gate_voltages"]
        gate_distances = np.abs(gate_ground_truth - current_gate_voltages)

        barrier_ground_truth = self.device_state["barrier_ground_truth"]
        current_barrier_voltages = self.device_state["current_barrier_voltages"]
        barrier_distances = np.abs(barrier_ground_truth - current_barrier_voltages)

        # Calculate gate rewards using piecewise function
        gate_rewards = np.zeros_like(gate_distances)

        for i, dist in enumerate(gate_distances):
            if dist >= self.gate_ramp_start:
                # Region 1: Beyond ramp start
                gate_rewards[i] = 0.0
            elif dist > self.gate_quadratic_start:
                # Region 2: Linear from 0 to 0.5
                normalized = (self.gate_ramp_start - dist) / (self.gate_ramp_start - self.gate_quadratic_start)
                gate_rewards[i] = 0.5 * normalized
            else:
                # Region 3: Curved approach from 0.5 to 1.0
                normalized = (self.gate_quadratic_start - dist) / self.gate_quadratic_start

                if self.gate_curve_type == "polynomial":
                    curve_value = normalized ** self.gate_curve_exponent
                elif self.gate_curve_type == "exponential":
                    curve_value = (np.exp(self.gate_curve_exponent * normalized) - 1) / (np.exp(self.gate_curve_exponent) - 1)
                elif self.gate_curve_type == "linear":
                    curve_value = normalized
                else:
                    raise ValueError(f"Unknown curve type: {self.gate_curve_type}")

                gate_rewards[i] = 0.5 + 0.5 * curve_value

        # Calculate barrier rewards using simple linear function
        barrier_rewards = np.zeros_like(barrier_distances)
        for i, dist in enumerate(barrier_distances):
            if dist >= self.barrier_ramp_start:
                barrier_rewards[i] = 0.0
            else:
                barrier_rewards[i] = (self.barrier_ramp_start - dist) / self.barrier_ramp_start

        # Ensure rewards are in [0, 1]
        gate_rewards = np.clip(gate_rewards, 0, 1)
        barrier_rewards = np.clip(barrier_rewards, 0, 1)

        rewards = {"gates": gate_rewards, "barriers": barrier_rewards}

        return rewards


    def _get_info(self):
        return {
            "current_device_state": self.device_state
        }


    def _normalise_obs(self, obs):
        """
        Images:
            normalize observations from 0 to 1 based on the middle 99% of data.
            clips the outer 0.5% to 0 and 1 on either end.
        
        Voltages:
            normalize observations to range [-1, 1].

        Args:
            obs (dict): Observation dictionary containing image and voltage data

        Returns:
            dict: Normalized observation dictionary
        """
        assert isinstance(obs, dict), f"Incorrect obs type, expected dict, got {type(obs)}"

        normalized_obs = obs.copy()

        # Normalize the image data
        if "image" in obs:
            image_data = obs["image"]

            # Calculate percentiles for the middle 99% of data
            p_low = np.percentile(image_data, 0.5)  # 0.5th percentile
            p_high = np.percentile(image_data, 99.5)  # 99.5th percentile

            # Normalize to [0, 1] based on middle 99% range
            if p_high > p_low:
                normalized_image = (image_data - p_low) / (p_high - p_low)
            else:
                # Handle edge case where all values are the same
                normalized_image = np.zeros_like(image_data)

            # Clip to [0, 1] range (this clips the outer 0.5% on each end)
            normalized_image = np.clip(normalized_image, 0.0, 1.0)

            # Keep as float32 in [0, 1] range
            normalized_obs["image"] = normalized_image.astype(np.float32)

        if "obs_gate_voltages" in obs:
            v = obs["obs_gate_voltages"].astype(np.float32)

            # note low and high are np arrays
            low = self.plunger_min
            high = self.plunger_max
            
            v = (v - low) / (high - low) # rescale to [0, 1]
            v = v * 2 - 1 # rescale to [-1, 1]

            normalized_obs["obs_gate_voltages"] = v.astype(np.float32)

        if "obs_barrier_voltages" in obs:
            b = obs["obs_barrier_voltages"].astype(np.float32)

            low = self.barrier_min
            high = self.barrier_max

            b = (b - low) / (high - low)
            b = b * 2 - 1

            normalized_obs["obs_barrier_voltages"] = b.astype(np.float32)

        return normalized_obs


    def _update_virtual_gate_matrix(self, obs):
        """
        Update the virtual gate matrix using ML-predicted capacitances from batched scans.

        This method processes multiple charge stability diagrams (one per dot pair) through
        the ML model in a single batch, then updates the Bayesian predictor with the
        predictions for each corresponding dot pair.

        Args:
            obs (dict): Observation containing 'image' key with multi-channel charge
                       stability diagrams of shape (resolution, resolution, num_dots-1)
        """
        if self.capacitance_model is None:
            return  # Skip if capacitance model not available

        if self.capacitance_model == "fake":
            cgd_estimate = fake_capacitance_model(
                self.current_step, self.max_steps, self.array.model.cgd
            )
            self.array._update_virtual_gate_matrix(cgd_estimate)
            return

        # Get the multi-channel scan: shape (resolution, resolution, num_dots-1)
        image = obs["image"]  # Each channel is one dot pair's charge stability diagram

        # Create batch: (num_dots-1, 1, resolution, resolution)
        # Convert (height, width, channels) -> (channels, 1, height, width)
        batch_tensor = (
            torch.from_numpy(image)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(1)
            .to(self.capacitance_model["device"])
        )

        # Run ML model on entire batch
        with torch.no_grad():
            values, log_vars = self.capacitance_model["ml_model"](batch_tensor)

        # Update Bayesian predictor for each dot pair
        # Convert tensors to numpy once
        values_np = values.cpu().numpy()  # Shape: (num_dots-1, 3)
        log_vars_np = log_vars.cpu().numpy()  # Shape: (num_dots-1, 3)

        for i in range(self.num_dots - 1):
            # Get current mean estimates for this dot pair and its neighbors
            current_mean_ij, _ = self.capacitance_model["bayesian_predictor"].get_capacitance_stats(
                i, i + 1
            )
            current_mean_ik, _ = self.capacitance_model["bayesian_predictor"].get_capacitance_stats(
                i, max(0, i - 1) if i > 0 else i + 2
            )
            current_mean_jk, _ = self.capacitance_model["bayesian_predictor"].get_capacitance_stats(
                i + 1, min(self.num_dots - 1, i + 2) if i + 2 < self.num_dots else i
            )

            # Add current means to delta predictions to get absolute values
            absolute_values = [
                current_mean_ij + float(values_np[i, 0]),  # C_ij + delta_ij
                current_mean_ik + float(values_np[i, 1]),  # C_ik + delta_ik
                current_mean_jk + float(values_np[i, 2]),  # C_jk + delta_jk
            ]

            # Create ml_outputs format expected by update_from_scan
            ml_outputs = [(absolute_values[j], float(log_vars_np[i, j])) for j in range(3)]

            # Update the Bayesian predictor for this dot pair
            self.capacitance_model["bayesian_predictor"].update_from_scan((i, i + 1), ml_outputs)

        # Get updated capacitance matrix and apply to quantum array
        cgd_estimate = self.capacitance_model["bayesian_predictor"].get_full_matrix()

        self.array._update_virtual_gate_matrix(cgd_estimate)


    def _init_capacitance_model(self):
        """
        Initialize the capacitance prediction model and Bayesian predictor.

        This method loads the pre-trained neural network for capacitance prediction
        and sets up the Bayesian predictor for uncertainty quantification and
        posterior tracking of capacitance matrix elements.
        """
        try:
            update_method = self.config["capacitance_model"]["update_method"]

            if update_method is None:
                self.capacitance_model = None
                return

            elif update_method == "fake":
                self.capacitance_model = "fake"
                return

            # Determine device (GPU if available, otherwise CPU)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Running capacitance model on {device}")
            else:
                device = torch.device("cpu")
                print("Warning: Failed to find available CUDA device, running on CPU")

            # Initialize the neural network model
            ml_model = CapacitancePredictionModel()

            if "SWARM_PROJECT_ROOT" in os.environ:
                # Ray distributed mode: use environment variable set by training script
                swarm_dir = os.environ["SWARM_PROJECT_ROOT"]
            else:
                # Local development mode: find Swarm directory from current file
                swarm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            weights_path = os.path.join(swarm_dir, "CapacitanceModel", "outputs", "best_model.pth")

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")

            # Load the checkpoint (it contains training metadata)
            checkpoint = torch.load(weights_path, map_location=device)

            # Extract model state dict from checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            ml_model.load_state_dict(state_dict)
            ml_model.to(device)
            ml_model.eval()  # Set to evaluation mode

            # Define distance-based prior configuration for Bayesian predictor
            def distance_prior(i: int, j: int) -> tuple:
                """
                Distance-based prior configuration for capacitance matrix elements.

                Args:
                    i, j: Dot indices

                Returns:
                    (prior_mean, prior_variance): Prior distribution parameters
                """
                if i == j:
                    # Self-capacitance (diagonal elements)
                    return (1, 0.01)
                elif abs(i - j) == 1:
                    # Nearest neighbors
                    return (0.40, 0.2)
                elif abs(i - j) == 2:
                    # Distant pairs
                    return (0.2, 0.1)
                else:
                    return (0.0, 0.1)


            if update_method == "bayesian":
                # Initialize Bayesian predictor
                bayesian_predictor = CapacitancePredictor(
                    n_dots=self.num_dots, prior_config=distance_prior
                )
            elif update_method == "kriging":
                # Initialize spatially aware predictor
                bayesian_predictor = InterpolatedCapacitancePredictor(
                    n_dots=self.num_dots, prior_config=distance_prior
                )
            else:
                raise ValueError(f"Unknown update method: {update_method}")

            # Store both components in the capacitance model
            self.capacitance_model = {
                "ml_model": ml_model,
                "bayesian_predictor": bayesian_predictor,
                "device": device,
            }

            print("Successfully loaded capacitance model.")

        except Exception as e:
            print(f"Warning: Failed to initialize capacitance model: {e}")
            print("The environment will continue without capacitance prediction capabilities.")
            self.capacitance_model = None


    def _init_voltage_ranges(self, plunger_ground_truths, barrier_ground_truths):

        full_plunger_range_width = self.config['simulator']['full_plunger_range_width']
        full_barrier_range_width = self.config['simulator']['full_barrier_range_width']


        plunger_range = np.random.uniform(
            low=full_plunger_range_width['min'],
            high=full_plunger_range_width['max']
        )

        plunger_center = np.random.uniform(
            low=plunger_ground_truths - 0.5 * (plunger_range-2),
            high=plunger_ground_truths + 0.5 * (plunger_range-2), 
        )

        self.plunger_max = plunger_center + 0.5 * plunger_range
        self.plunger_min = plunger_center - 0.5 * plunger_range

        
        barrier_range = np.random.uniform(
            low=full_barrier_range_width['min'],
            high=full_barrier_range_width['max']
        )

        barrier_center = np.random.uniform(
                low=barrier_ground_truths - 0.5 * (barrier_range-1),
                high=barrier_ground_truths + 0.5 * (barrier_range-1),
            )

        self.barrier_max = barrier_center + 0.5 * barrier_range
        self.barrier_min = barrier_center - 0.5 * barrier_range


    def _starting_voltages(self):
        plunger_centers = np.random.uniform(
            low=self.plunger_min,
            high=self.plunger_max,
            size=self.num_plunger_voltages
        )

        if self.use_barriers:
            barrier_centers = np.random.uniform(
                low=self.barrier_min,
                high=self.barrier_max,
                size=self.num_barrier_voltages
            ) 
        else:
            barrier_centers = np.zeros(self.num_barrier_voltages)

        return plunger_centers, barrier_centers


    def _rescale_gate_voltages(self, obs):
        obs = (obs + 1) / 2 # [0, 1]

        if self.use_deltas:
            obs = obs * (self.plunger_delta_max - self.plunger_delta_min) + self.plunger_delta_min
            obs += self.device_state["current_gate_voltages"]
            obs = np.clip(obs, self.plunger_min, self.plunger_max)
        else:
            obs = obs * (self.plunger_max - self.plunger_min) + self.plunger_min

        return obs

    def _rescale_barrier_voltages(self, obs):
        obs = (obs + 1) / 2 # [0, 1]
        obs = obs * (self.barrier_max - self.barrier_min) + self.barrier_min
        return obs


    def _load_config(self, config_path):
        # Make config path relative to the env.py file directory
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as file:
            config = yaml.safe_load(file)

        return config

    def _render_frame(self, single_scan):
        self.array._render_frame(single_scan)

    def _cleanup(self):
        pass


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    env = QuantumDeviceEnv()
    env.reset()
    print(env.observation_space)
    print(env.action_space)
    print(env.device_state)
