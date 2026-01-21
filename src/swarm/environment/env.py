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

from swarm.capacitance_model import CapacitancePredictionModel, BayesianCapacitancePredictor, KrigingCapacitancePredictor, EmaCapacitancePredictor


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
        num_dots=None,
        use_barriers=None,
    ):

        super().__init__()

        # environment parameters
        self.config = self._load_config(config_path)
        self.training = training  # if we are training or not
        self.num_dots = num_dots if num_dots is not None else self.config['simulator']['num_dots']
        self.use_barriers = use_barriers if use_barriers is not None else self.config['simulator']['use_barriers']
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

        radial_noise_config = self.config['simulator']['radial_noise']

        self.array = QarrayBaseClass(
            num_dots=self.num_dots,
            use_barriers=self.use_barriers,
            obs_voltage_min=-self.window_delta,
            obs_voltage_max=self.window_delta,
            obs_image_size=self.resolution,
            radial_noise_config=radial_noise_config,
        )

        # Reset virtual gate matrix to identity (no virtualization knowledge)
        # This simulates starting without knowledge of crosstalk
        self.array._reset_virtual_gate_matrix_to_identity()

        if self.capacitance_model == "perfect":
            self.array._reset_virtual_gate_matrix_to_perfect()

        voltage_offset_range = self.config['simulator']['constant_voltage_offset']
        self.constant_voltage_offset = np.random.uniform(
            low=voltage_offset_range['min'],
            high=voltage_offset_range['max'],
            size=(self.num_dots,)
        )

        # Add the random offset to shift the voltage space
        self.array._update_virtual_gate_origin(np.concatenate([self.constant_voltage_offset, [0]]))

        plunger_ground_truth, barrier_ground_truth, _ = self.array.calculate_ground_truth()

        if barrier_ground_truth is None:
            assert not self.use_barriers, "Expected array for barrier_ground_truth, got None"
            barrier_ground_truth = np.zeros(self.num_barrier_voltages, dtype=np.float32)
        else:
            barrer_ground_truth = np.array(barrier_ground_truth).flatten().astype(np.float32)

        plunger_ground_truth = np.array(plunger_ground_truth).flatten().astype(np.float32)

        assert len(plunger_ground_truth) == self.num_plunger_voltages, f"Expected plunger ground truth to be of length {self.num_plunger_voltages}, got {len(plunger_ground_truth)}"
        assert len(barrier_ground_truth) == self.num_barrier_voltages, f"Expected plunger ground truth to be of length {self.num_barrier_voltages}, got {len(barrier_ground_truth)}"

        # Set ground truth on array for radial noise
        self.array.gate_ground_truth = plunger_ground_truth

        self._init_voltage_ranges(
            plunger_ground_truth,
            barrier_ground_truth
        )

        plungers, barriers = self._starting_voltages()
        #note this overrides the ideal ground truth calculated for zero charge occupation
        if self.use_barriers:
            barrier_ground_truth = self.array.calculate_barrier_ground_truth(plungers)

        self.device_state = {
            "gate_ground_truth": plunger_ground_truth,
            "barrier_ground_truth": barrier_ground_truth,
            "current_gate_voltages": plungers,
            "current_barrier_voltages": barriers,
            "virtual_gate_matrix": self.array.model.gate_voltage_composer.virtual_gate_matrix,
            "virtual_gate_origin": self.array.model.gate_voltage_composer.virtual_gate_origin,
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


    def step(self, action: dict, skip_obs: bool = False):
        """
        Updates the environment state based on the agent's action.

        This method is the core of the environment. It takes an action from the
        agent and calculates the next state, the reward, and whether the
        episode has ended.

        Args:
            action: An action provided by the agent.
            skip_obs: If True, skip expensive observation generation (CSD images).
                      Useful for benchmarking when only voltage dynamics are needed.

        Returns:
            observation (np.ndarray): The observation of the environment's state (None if skip_obs).
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

        gate_voltages = np.clip(gate_voltages, -1, 1)
        barrier_voltages = np.clip(barrier_voltages, -1, 1)

        # Rescale voltages from [-1, 1] to actual ranges
        gate_voltages = self._rescale_gate_voltages(gate_voltages)
        barrier_voltages = self._rescale_barrier_voltages(barrier_voltages)

        self.device_state["current_gate_voltages"] = gate_voltages
        self.device_state["current_barrier_voltages"] = barrier_voltages

        if self.use_barriers:
            self.device_state["barrier_ground_truth"] = self.array.calculate_barrier_ground_truth(gate_voltages)

        reward = self._get_reward()

        terminated = False
        truncated = False

        if self.current_step >= self.max_steps:
            truncated = True

        if skip_obs:
            observation = None
        else:
            raw_observation = self.array._get_obs(gate_voltages, barrier_voltages)
            observation = self._normalise_obs(raw_observation)

            self._update_virtual_gate_matrix(observation)
            self.device_state["virtual_gate_matrix"] = (
                self.array.model.gate_voltage_composer.virtual_gate_matrix
            )
            self._recalculate_ground_truth()

        info = self._get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def _recalculate_ground_truth(self):
        """
        Recalculate ground truth based on current VGM.

        This should be called after any VGM update to ensure the reward target
        reflects the current virtualization state. The optimal physical voltages
        (true target) remain constant, but their representation in virtual space
        changes as the VGM is updated.
        """
        plunger_ground_truth, barrier_ground_truth, _ = self.array.calculate_ground_truth()
        plunger_ground_truth = np.array(plunger_ground_truth).flatten().astype(np.float32)

        self.device_state["gate_ground_truth"] = plunger_ground_truth
        self.array.gate_ground_truth = plunger_ground_truth

        if self.use_barriers and barrier_ground_truth is not None:
            barrier_ground_truth = np.array(barrier_ground_truth).flatten().astype(np.float32)
            self.device_state["barrier_ground_truth"] = barrier_ground_truth


    def _get_reward(self):
        """
        Get the reward for the current state using a piecewise reward function.

        Gate rewards use a configurable piecewise function:
        - Region 1 (distance > ramp_start): reward = 0
        - Region 2 (quadratic_start < distance <= ramp_start): linear from 0 to 0.5
        - Region 3 (0 < distance <= quadratic_start): configurable curve (polynomial/exponential/linear) from 0.5 to 1.0
        - Region 4 (distance = 0): reward = 1.0

        Gate distances are scaled by CGD diagonal elements to reflect physical impact.

        Barrier rewards use a simple linear function from 0 (at barrier_ramp_start) to 1.0 (at ground truth).
        Barrier distances are scaled by alpha (tunnel coupling sensitivity) to reflect physical impact.
        """

        gate_ground_truth = self.device_state["gate_ground_truth"]
        current_gate_voltages = self.device_state["current_gate_voltages"]
        gate_distances = np.abs(gate_ground_truth - current_gate_voltages)

        # Scale gate distances by CGD diagonal (physical impact scaling)
        # Higher CGD diagonal means voltage errors have larger effect on dot potential
        # Use cgd_full for barriers model, cgd for non-barriers model
        if self.use_barriers:
            cgd = self.array.model.cgd_full
        else:
            cgd = self.array.model.cgd
        cgd_diagonal = np.abs([cgd[i, i] for i in range(self.num_dots)])
        gate_distances = gate_distances * cgd_diagonal

        barrier_ground_truth = self.device_state["barrier_ground_truth"]
        current_barrier_voltages = self.device_state["current_barrier_voltages"]
        barrier_distances = np.abs(barrier_ground_truth - current_barrier_voltages)

        # Scale barrier distances by alpha (tunnel coupling sensitivity)
        # Higher alpha means voltage errors have larger effect on tunnel coupling
        if self.use_barriers and self.array.barrier_alpha is not None:
            barrier_alpha = np.array(self.array.barrier_alpha)
            barrier_distances = barrier_distances * barrier_alpha

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
                elif self.gate_curve_type == "constant":
                    curve_value = 1
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

        if self.capacitance_model == "perfect":
            # update handled in initialisation
            return

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

        values_np = values.cpu().numpy()  # Shape: (num_dots-1, num outputs)
        log_vars_np = log_vars.cpu().numpy()  # Shape: (num_dots-1, num outputs)

        # cgd = self.array.model.cgd
        # true_values = np.zeros_like(values_np)
        # for scan_idx in range(true_values.shape[0]):
        #     # outputs are RL, LR couplings
        #     true_values[scan_idx, 0] = cgd[scan_idx, scan_idx + 1] # rows are the dots, columns are the gates
        #     true_values[scan_idx, 1] = cgd[scan_idx + 1, scan_idx]

        # errors = values_np - np.absolute(true_values)

        # # Log values and logvars to capacitance_values.log
        # # Use absolute path to avoid Ray working directory issues
        # log_file_path = Path("/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/inference/capacitance_values.log")

        # with open(log_file_path, "a") as f:
        #     f.write(f"values: {values_np.tolist()}\t\t\t")
        #     f.write(f"errors: {errors.tolist()}\t\t\t")
        #     f.write(f"log_vars: {log_vars_np.tolist()}\n")

        update_method = self.config["capacitance_model"]["update_method"]

        if update_method == "ema":
            # EMA method: use ML predictions directly (not deltas)
            if self.capacitance_model["nearest_neighbour"]:
                for i in range(self.num_dots - 1):
                    # For EMA, use absolute ML predictions directly
                    # The ML model predicts absolute capacitance values, not deltas
                    absolute_values = [
                        float(values_np[i, 0]),  # RL coupling
                        float(values_np[i, 1]),  # LR coupling
                    ]

                    # Update the EMA predictor with absolute values and log variances
                    ml_outputs = [(absolute_values[j], float(log_vars_np[i, j])) for j in range(2)]
                    self.capacitance_model["capacitance_predictor"].update_from_scan(left_dot=i, ml_outputs=ml_outputs)
            else:
                raise NotImplementedError("Capacitance update only supports nearest-neighbour mode for now")

            # Get updated capacitance matrix from EMA predictor (diagonal already set to 1)
            cgd_estimate = self.capacitance_model["capacitance_predictor"].get_full_matrix()

            self.array._update_virtual_gate_matrix(cgd_estimate)
            return

        # Bayesian and Kriging methods
        if self.capacitance_model["nearest_neighbour"]:
            for i in range(self.num_dots - 1):

                current_mean_RL, _ = self.capacitance_model["capacitance_predictor"].get_capacitance_stats(i+1, i)
                current_mean_LR, _ = self.capacitance_model["capacitance_predictor"].get_capacitance_stats(i, i+1)

                # Add predictions to current means (since scans are already partially virtualised)
                absolute_values = [
                    current_mean_RL + float(values_np[i, 0]),
                    current_mean_LR + float(values_np[i, 1]),
                ]

                ml_outputs = [(absolute_values[j], float(log_vars_np[i, j])) for j in range(2)]
                self.capacitance_model["capacitance_predictor"].update_from_scan(left_dot=i, ml_outputs=ml_outputs)

        else:
            raise NotImplementedError("Capacitance update only supports nearest-neighbour mode for now")

        # Get updated capacitance matrix and apply to quantum array
        cgd_estimate = self.capacitance_model["capacitance_predictor"].get_full_matrix()

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

            elif update_method in ["perfect", "fake"]:
                self.capacitance_model = update_method
                return

            # Determine device (GPU if available, otherwise CPU)
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Running capacitance model on {device}")
            else:
                device = torch.device("cpu")
                print("Warning: Failed to find available CUDA device, running on CPU")
            
            nearest_neighbour = self.config["capacitance_model"]["nearest_neighbour"]

            # Initialize the neural network model
            output_size = 2 if nearest_neighbour else 3
            ml_model = CapacitancePredictionModel(output_size=output_size)

            if "SWARM_PROJECT_ROOT" in os.environ:
                # Ray distributed mode: use environment variable set by training script
                swarm_dir = os.environ["SWARM_PROJECT_ROOT"]
            else:
                # Local development mode: find Swarm directory from current file
                swarm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            if self.use_barriers:
                weights_path = os.path.join(swarm_dir, "capacitance_model", "weights", "best_model_barriers.pth")
            else:
                weights_path = os.path.join(swarm_dir, "capacitance_model", "weights", "best_model_no_barriers.pth")

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
                capacitance_predictor = BayesianCapacitancePredictor(
                    n_dots=self.num_dots, nn=nearest_neighbour, prior_config=distance_prior
                )
            elif update_method == "kriging":
                # Initialize spatially aware predictor
                capacitance_predictor = KrigingCapacitancePredictor(
                    n_dots=self.num_dots, nn=nearest_neighbour, prior_config=distance_prior
                )
            elif update_method == "ema":
                # Initialize EMA predictor
                capacitance_predictor = EmaCapacitancePredictor(
                    n_dots=self.num_dots, nn=nearest_neighbour, prior_config=distance_prior
                )
            else:
                raise ValueError(f"Unknown update method: {update_method}")

            # Store both components in the capacitance model
            self.capacitance_model = {
                "ml_model": ml_model,
                "capacitance_predictor": capacitance_predictor,
                "device": device,
                "nearest_neighbour": nearest_neighbour
            }

            print("Successfully loaded capacitance model.")

        except Exception as e:
            raise RuntimeError(f"Error initialising capacitance model: {e}")
            # print(f"Warning: Failed to initialize capacitance model: {e}")
            # print("The environment will continue without capacitance prediction capabilities.")
            # self.capacitance_model = None


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
    import matplotlib.pyplot as plt

    #os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    env = QuantumDeviceEnv(num_dots=6)
    obs, info = env.reset()
    print(env.observation_space)
    print(env.action_space)
    print(env.device_state)

    # Get the image observation (scans)
    scans = obs["image"]  # Shape: (resolution, resolution, num_channels)
    num_channels = scans.shape[2]

    # Create a figure with subplots side by side
    fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))

    # Handle single channel case
    if num_channels == 1:
        axes = [axes]

    # Plot each scan side by side
    for i in range(num_channels):
        axes[i].imshow(scans[:, :, i], cmap='viridis', origin='lower')
        axes[i].set_title(f'Scan {i+1} (Dots {i}-{i+1})')
        axes[i].set_xlabel('Gate Voltage')
        axes[i].set_ylabel('Gate Voltage')
        axes[i].axis('on')

    plt.tight_layout()
    plt.savefig('scans.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved scans to scans.png")
    plt.close()
