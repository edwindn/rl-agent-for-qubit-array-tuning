import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import time

import matplotlib.pyplot as plt
import io
import fcntl
import json

class QuantumDeviceEnv(gym.Env):
    """
    Represents the device with its quantum dots 
    """
    #rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    @staticmethod
    def get_global_rollout_counter():
        QuantumDeviceEnv._init_counter_file()
        with open(QuantumDeviceEnv.COUNTER_FILE, 'r') as f:
            data = json.load(f)
            now = time.time()
            start_time = data.get("start_time", now)
            elapsed = now - start_time
            return data.get("total_rollouts", 0), elapsed

    _total_rollouts = 0  # Class-level counter shared across all instances
    _instance_count = 0
    
    def __init__(self, config_path='qarray_4dot_config.yaml', render_mode=None, counter_file=None, **kwargs):
        """
        constructor for the environment

        define action and observation spaces

        init state and variables
        """
        super().__init__()

        print('Initialising qarray env with 4 dots ...')

        QuantumDeviceEnv._instance_count += 1
        self._instance_id = QuantumDeviceEnv._instance_count
        self._local_rollouts = 0
        self.counter_file = counter_file

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.debug = self.config['training']['debug']
        self.seed = self.config['training']['seed']
        self.max_steps = self.config['env']['max_steps']
        self.current_step = 0
        self.tolerance = self.config['env']['tolerance']


        # --- Define Action and Observation Spaces ---
        self.num_voltages = 2
        self.action_voltage_min = self.config['env']['action_space']['voltage_range'][0]
        self.action_voltage_max = self.config['env']['action_space']['voltage_range'][1]

        matrix_shape = np.array(self.config['simulator']['model']['Cgd']).shape
        matrix_length = np.prod(matrix_shape)
        matrix_low = np.array(self.config['simulator']['model']['Cgd_low']).flatten()
        matrix_high = np.array(self.config['simulator']['model']['Cgd_high']).flatten()

        assert len(matrix_low) == len(matrix_high) == matrix_length

        self.capacitances = None
        self.capacitance_shape = matrix_shape
        self.cgd_max = matrix_high.max()
        self.cgd_min = matrix_low.min()
        self.max_cgd_dist = np.linalg.norm(np.ones_like(matrix_low)) # since we normalise the capacitances in get_reward

        # self.action_space = spaces.Dict({
        #     'action_voltages': spaces.Box(
        #         low=self.action_voltage_min,
        #         high=self.action_voltage_max,
        #         shape=(self.num_voltages,),
        #         dtype=np.float32
        #     ),
        #     'capacitances': spaces.Box(
        #         #low=matrix_low,
        #         #high=matrix_high,
        #         low=self.cgd_min,
        #         high=self.cgd_max,
        #         shape=(matrix_length,),
        #         dtype=np.float32
        #     )
        # })

        self.action_space = spaces.Box(
            low=self.action_voltage_min,
            high=self.action_voltage_max,
            shape=(self.num_voltages,),
            dtype=np.float32
        )

        # Observation space for quantum device state - multi-modal with image and voltages
        obs_config = self.config['env']['observation_space']
        self.obs_image_size = obs_config['image_size']
        self.obs_channels = 1
        self.obs_normalization_range = obs_config['normalization_range']
        self.obs_dtype = obs_config['dtype']
        
        # Define multi-modal observation space using Dict
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_image_size[0], self.obs_image_size[1], self.obs_channels),
                dtype=np.uint8
            ),
            'obs_voltages': spaces.Box(
                low=self.action_voltage_min,
                high=self.action_voltage_max,
                shape=(self.num_voltages,),
                dtype=np.float32
            )
        })

        self.obs_voltage_min = self.config['simulator']['measurement']['v_min']
        self.obs_voltage_max = self.config['simulator']['measurement']['v_max']

        # --- Initialize Model (one-time setup) ---
        self.model = self._load_model()
        
        # --- Initialize normalization parameters ---
        self._init_normalization_params()

        # --- For Rendering --- 
        self.render_fps = self.config['training']['render_fps'] #unused for now
        self.render_mode = render_mode or self.config['training']['render_mode']


    def _increment_global_counter(self):
        if self.counter_file is not None:
            with open(self.counter_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = json.load(f)
                data["total_rollouts"] += 1
                now = time.time()
                start_time = data.get("start_time", now)
                elapsed = now - start_time
                f.seek(0)
                json.dump(data, f)
                f.truncate()
                fcntl.flock(f, fcntl.LOCK_UN)
                return data["total_rollouts"], elapsed


    def _init_normalization_params(self):
        """
        Initialize adaptive normalization parameters.
        Start with conservative bounds and update them as new data is encountered.
        """
        if self.debug:
            print("Initializing adaptive normalization parameters...")
        
        # Start with conservative bounds based on typical charge sensor data ranges
        # These will be updated as we encounter actual data
        self.data_min = 0.13
        self.data_max = 0.16
        self.bounds_initialized = False
        
        # Track statistics for adaptive updates
        self.episode_min = float('inf')
        self.episode_max = float('-inf')
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.update_count = 0
        
        if self.debug:
            print(f"Initial normalization range: [{self.data_min:.4f}, {self.data_max:.4f}]")

    def _update_normalization_bounds(self, raw_data):
        """
        Update normalization bounds based on new data encountered.
        Uses adaptive approach to gradually expand bounds while maintaining stability.
        
        Args:
            raw_data (np.ndarray): Raw charge sensor data
        """
        # Update episode statistics
        self.episode_min = min(self.episode_min, np.min(raw_data))
        self.episode_max = max(self.episode_max, np.max(raw_data))
        
        # Update global statistics
        self.global_min = min(self.global_min, np.min(raw_data))
        self.global_max = max(self.global_max, np.max(raw_data))
        
        # Check if we need to expand bounds
        needs_update = False
        new_min = self.data_min
        new_max = self.data_max
        
        # Expand lower bound if needed (with safety margin)
        if self.global_min < self.data_min:
            safety_margin = (self.data_max - self.data_min) * 0.05  # 5% safety margin
            new_min = self.global_min - safety_margin
            needs_update = True
            
        # Expand upper bound if needed (with safety margin)
        if self.global_max > self.data_max:
            safety_margin = (self.data_max - self.data_min) * 0.05  # 5% safety margin
            new_max = self.global_max + safety_margin
            needs_update = True
        
        # Update bounds if needed
        if needs_update:
            self.data_min = new_min
            self.data_max = new_max
            self.update_count += 1
            
            if self.debug:
                print(f"Updated normalization bounds to [{self.data_min:.4f}, {self.data_max:.4f}] (update #{self.update_count})")

    def _normalize_observation(self, raw_data):
        """
        Normalize the raw charge sensor data to the observation space range.
        Uses adaptive bounds that update as new data is encountered.
        
        Args:
            raw_data (np.ndarray): Raw charge sensor data of shape (height, width)
            
        Returns:
            np.ndarray: Normalized observation of shape (height, width, 1)
        """
        # Ensure input is 2D
        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {raw_data.shape}")
        
        # Update bounds based on new data
        self._update_normalization_bounds(raw_data)
        
        # Normalize to [0, 1] range
        normalized = (raw_data - self.data_min) / (self.data_max - self.data_min)
        
        # Clip to ensure values are within bounds
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Scale to uint8 range [0, 255]
        normalized = (normalized * 255).astype(np.uint8)
        
        # Reshape to add channel dimension
        normalized = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)
        
        return normalized


    def _get_charge_sensor_data(self, voltages, gate1, gate2):
        """
        Get charge sensor data for given voltages.
        
        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """
        # z, _ = self.device_state["model"].charge_sensor_open(voltages)

        # vg_current = self.model.gate_voltage_composer.do2d(
        #     gate1, center[0]+self.obs_voltage_min, center[0]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
        #     gate2, center[1]+self.obs_voltage_min, center[1]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        # )

        z, _ = self.device_state["model"].do2d_open(
            gate1, voltages[0]+self.obs_voltage_min, voltages[0]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
            gate2, voltages[1]+self.obs_voltage_min, voltages[1]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        )
        return z


    def get_raw_observation(self):
        """
        Get raw (unnormalized) observation data for debugging purposes.
        
        Returns:
            np.ndarray: Raw charge sensor data of shape (height, width)
        """
        voltage_centers = self.device_state["voltage_centers"]
        z = self._get_charge_sensor_data(voltage_centers, gate1=2, gate2=3)
        return z[:, :, 0]

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
            info (dict): A dictionary with auxiliary diagnostic information.
        """

        self._increment_global_counter()

        # Handle seed if provided
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0
        
        # Reset episode-specific normalization statistics
        self.episode_min = float('inf')
        self.episode_max = float('-inf')


        # Initialize episode-specific voltage state
        #center of current window
        center = self._random_center()

        # #current window
        # vg_current = self.model.gate_voltage_composer.do2d(
        #     1, center[0]+self.obs_voltage_min, center[0]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
        #     2, center[1]+self.obs_voltage_min, center[1]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        # )

        optimal_VG_center = self.model.optimal_Vg(self.config['simulator']['measurement']['optimal_VG_center'])

        # Device state variables (episode-specific)
        self.device_state = {
            "model": self.model,
            "ground_truth_center": optimal_VG_center[1:3],
            "voltage_centers": center
        }


        # --- Return the initial observation ---
        observation = self._get_obs()
        info = self._get_info() 

        return observation, info

    def step(self, action):
        """
        Updates the environment state based on the agent's action.

        This method is the core of the environment. It takes an action from the
        agent and calculates the next state, the reward, and whether the
        episode has ended.

        Args:
            action: An action provided by the agent.

        Returns:
            observation (np.ndarray): The observation of the environment's state.
            reward (float): The amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., reached a goal).
            truncated (bool): Whether the episode was cut short (e.g., time limit).
            info (dict): A dictionary with auxiliary diagnostic information.
        """

        # --- Update the environment's state based on the action ---
        self.current_step += 1
        # action is now a numpy array of shape (num_voltages,) containing voltage values

        # voltages, capacitances = action['action_voltages'], action['capacitances']
        voltages = action

        self._apply_voltages(voltages) #this step will update the voltages stored in self.device_state
        # self._update_capacitances(capacitances)

        # --- Determine the reward ---
        reward = self._get_reward()  #will compare current state to target state
        if self.debug:
            print(f"reward: {reward}")

        # --- Check for termination or truncation conditions ---
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            if self.debug:
                print("Max steps reached")
        
        # Check if the centers of the voltage sweeps are aligned
        ground_truth_center = self.device_state["ground_truth_center"]

        # Get current voltage settings (what the agent controls)
        current_voltage_center = self.device_state["voltage_centers"]

        # Compare only the first num_voltages dimensions (ignoring last dimension)
        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        
        if at_target:
            terminated = True
            if self.debug:
                print("Target voltage sweep center reached")

        # --- Get the new observation and info ---
        observation = self._get_obs() #new state
        info = self._get_info() #diagnostic info
        
        return observation, reward, terminated, truncated, info
    
    def _random_center(self):
        """
        Randomly generate a center voltage for the voltage sweep.
        """
        return np.random.uniform(self.action_voltage_min-self.obs_voltage_min, self.action_voltage_max-self.obs_voltage_max, self.num_voltages)
    
    def _get_reward(self):
        """
        Get the reward for the current state.

        Reward is based on the distance from the target voltage sweep center, with maximum reward
        when the agent aligns the centers of the voltage sweeps. The reward is calculated
        as: max_possible_distance - current_distance, where max_possible_distance is the maximum
        possible distance in the 2D voltage space to ensure positive rewards.

        Only considers the first 2 dimensions (ignoring the third dimension).
        The reward is also penalized by the number of steps taken to encourage efficiency.
        """

        ground_truth_center = self.device_state["ground_truth_center"]
        current_voltage_center = self.device_state["voltage_centers"]
        
        distance = np.linalg.norm(ground_truth_center - current_voltage_center)
        
        # max_possible_distance = np.sqrt(self.num_voltages) * (self.obs_voltage_max - self.obs_voltage_min)
        max_possible_distance = np.sqrt(self.num_voltages) * (self.action_voltage_max - self.action_voltage_min)
        

        if self.current_step == self.max_steps:
            # reward = max(max_possible_distance - distance, 0)*0.01
            reward = (1 - distance / max_possible_distance) * 100
        else:
            reward = 0.0

        reward -= self.current_step * 0.1

        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        if at_target:
            reward += 200.0

        return reward

        # ---- #

        # Capacitance reward (not implemented in training yet)
        Cgd = np.array(self.config['simulator']['model']['Cgd'])
        cgd_max, cgd_min = self.cgd_max, self.cgd_min
        Cgd = (Cgd - cgd_min) / (cgd_max - cgd_min)  # Normalize capacitance matrix to [0, 1]
        if self.capacitances is not None:
            cap = (self.capacitances - cgd_min) / (cgd_max - cgd_min)
            cgd_dist = np.linalg.norm(cap - Cgd)
        else:
            raise RuntimeError("_get_reward called before model capacitance output was set")

        if at_target or self.current_step == self.max_steps:
            reward += 100 * (1 - cgd_dist/self.max_cgd_dist)
            if self.debug:
                print(f"Applied capacitance reward of {100 * (1 - cgd_dist/self.max_cgd_dist):.2f}")

        return reward


    def _load_model(self):
        """
        Load the model from the config file.
        """
        white_noise = WhiteNoise(amplitude=self.config['simulator']['model']['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**self.config['simulator']['model']['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = self.config['simulator']['model']['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None


        model = ChargeSensedDotArray(
            Cdd=self.config['simulator']['model']['Cdd'],
            Cgd=self.config['simulator']['model']['Cgd'],
            Cds=self.config['simulator']['model']['Cds'],
            Cgs=self.config['simulator']['model']['Cgs'],
            coulomb_peak_width=self.config['simulator']['model']['coulomb_peak_width'],
            T=self.config['simulator']['model']['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=self.config['simulator']['model']['algorithm'],
            implementation=self.config['simulator']['model']['implementation'],
            max_charge_carriers=self.config['simulator']['model']['max_charge_carriers'],
        )
        
        model.gate_voltage_composer.virtual_gate_matrix = self.config['simulator']['virtual_gate_matrix']


        return model

    def _get_obs(self):
        """
        Helper method to get the current observation of the environment.
        
        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """
        # Get current voltage configuration
        # current_voltages = self.device_state["current_voltages"]
        voltage_centers = self.device_state["voltage_centers"]
        
        # Get charge sensor data
        self.z = self._get_charge_sensor_data(voltage_centers, gate1=2, gate2=3)

        expected_voltage_shape = (self.num_voltages,)
        
        if voltage_centers.shape != expected_voltage_shape:
            raise ValueError(f"Voltage observation shape {voltage_centers.shape} does not match expected {expected_voltage_shape}")

        z = self.z
        # Extract first channel and normalize for image observation
        channel_data = z[:, :, 0]  # Shape: (height, width)
        image_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
            
        # Validate observation structure
        expected_image_shape = (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels)

        if image_obs.shape != expected_image_shape:
            raise ValueError(f"Image observation shape {image_obs.shape} does not match expected {expected_image_shape}")

        return {
            "image": image_obs, # image for only the middle voltage sweep
            "obs_voltages": voltage_centers
        }


    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        return {
            "device_state": self.device_state,
            "current_step": self.current_step,
            "normalization_range": [self.data_min, self.data_max],
            "normalization_updates": self.update_count,
            "global_data_range": [self.global_min, self.global_max],
            "episode_data_range": [self.episode_min, self.episode_max],
            "observation_structure": {
                "image_shape": (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels),
                "voltage_shape": (self.num_voltages,),
                "total_modalities": 2
            }
        }

    def _apply_voltages(self, voltages):
        """
        Apply voltage settings to the quantum device.
        
        Args:
            voltages (np.ndarray): Array of voltage values for each gate
        """

        voltages = np.array(voltages).flatten().astype(np.float32)
        assert len(voltages) == self.num_voltages, f"Expected voltages to be of size {self.num_voltages}, got {len(voltages)}"

        self.device_state["voltage_centers"] = voltages
        
        # # Create two grids centered around voltages[0] and voltages[1]
        # # Grid extent from obs_vmin to obs_vmax
        # x_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[1])
        # y_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[0])
        
        # # Create 2D grids centered around the voltage values
        # X, Y = np.meshgrid(x_grid + voltages[0], y_grid + voltages[1])

        # # Update the first two channels with the new grids
        # self.device_state["current_voltages"][:,:,0] = X
        # self.device_state["current_voltages"][:,:,1] = Y
        # self.device_state["voltage_centers"] = np.array(voltages)
        
        # # self.device_state["current_voltages"][:,:,2] remains as initialized

        # self.device_state["current_voltages"][:,:,:2] = np.clip(self.device_state["current_voltages"][:,:,:2], self.action_voltage_min, self.action_voltage_max)
    

    def _extract_voltage_centers(self, voltages):
        """
        Extract the voltage center values from the current voltage configuration.
        This inverses the process in _apply_voltages by finding the center of each 2D grid.
        
        Returns:
            tuple: (voltage[0], voltage[1]) representing the center values of the voltage grids
        """
        
        # Extract the first two channels (voltage grids)
        v1_grid = voltages[:, :, 0]  # First voltage grid
        v2_grid = voltages[:, :, 1]  # Second voltage grid
        
        # Calculate the center of each grid by taking the median
        # Since the grids are created by adding voltage centers to linspace grids,
        # the median of each grid gives us the voltage center
        voltage_center_1 = np.median(v1_grid)
        voltage_center_2 = np.median(v2_grid)
        
        return np.array([voltage_center_1, voltage_center_2])

    
    def _update_capacitances(self, capacitances):
        self.capacitances = capacitances.reshape(self.capacitance_shape)
         

    def render(self):
        """
        Render the environment state.
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            # For human mode, save the plot and return None
            self._render_frame()
            return None
        else:
            return None


    def _render_frame(self, inference_plot=False):
        """
        Internal method to create the render image.

        We plot the CSD between gate1 and its successor
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        z = self.z

        if inference_plot:
            vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)

            plt.figure(figsize=(5, 5))
            plt.imshow(z, extent=[vmin, vmax, vmin, vmax], origin='lower', aspect='auto', cmap='viridis')
            plt.xlabel('$Vx$')
            plt.ylabel('$Vy$')
            plt.title('$z$')
            plt.axis('equal')
            #plt.savefig('test_image.png')
            #plt.close()
            #return None

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            from PIL import Image
            img = Image.open(buf)
            img = np.array(img)
            return img

        # Get the normalized observation that the agent sees
        channel_data = z[:, :, 0]  # Shape: (height, width)
        normalized_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
        normalized_data = normalized_obs[:, :, 0]  # Remove channel dimension for plotting
        
        # Create figure and plot
        vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)
        num_ticks = 5
        tick_values = np.linspace(vmin, vmax, num_ticks)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(normalized_data, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        
        # Set x and y axis ticks to correspond to voltage range
        ax.set_xticks(np.linspace(0, z.shape[1]-1, num_ticks))
        ax.set_xticklabels([f'{v:.0f}' for v in tick_values])
        ax.set_yticks(np.linspace(0, z.shape[0]-1, num_ticks))
        ax.set_yticklabels([f'{v:.0f}' for v in tick_values])
        
        ax.set_xlabel("$\Delta$PL (V)")
        ax.set_ylabel("$\Delta$PR (V)")
        ax.set_title("Normalized $|S_{11}|$ (Agent Observation)")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar.get_ticks()])


        if self.render_mode == "human":
            # Save plot for human mode
            script_dir = os.path.dirname(os.path.abspath(__file__))
            plot_path = os.path.join(script_dir, 'quantum_dot_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved as '{plot_path}'")
            return None
        else:
            # Convert to RGB array for rgb_array mode
            fig.canvas.draw()
            
            # Simple approach: save to bytes and load as image
            try:
                # Save to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                # Load as image using PIL
                from PIL import Image
                img = Image.open(buf)
                data = np.array(img)
                
                # Convert RGBA to RGB if needed
                if data.shape[-1] == 4:
                    data = data[:, :, :3]
                    
            except Exception as e:
                print(f"Error getting RGB data from canvas: {e}")
                # Last resort: create a simple colored array
                height, width = normalized_data.shape
                data = np.zeros((height, width, 3), dtype=np.uint8)
                # Create a simple visualization based on the normalized data
                data[:, :, 0] = (normalized_data * 255).astype(np.uint8)  # Red channel
                data[:, :, 1] = (normalized_data * 255).astype(np.uint8)  # Green channel
                data[:, :, 2] = (normalized_data * 255).astype(np.uint8)  # Blue channel
            
            plt.close()
            return data
 


    def close(self):
        """
        Performs any necessary cleanup.
        """
  
        pass
    

    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config


if __name__ == "__main__":
    import sys
    env = QuantumDeviceEnv()
    env.reset()

    voltages = [-3.0, 1.0, 0.0, 0.0]
    env._apply_voltages(voltages)

    frame = env._render_frame(gate1=1, inference_plot=True)
    path = "quantum_dot_plot.png"
    plt.imsave(path, frame, cmap='viridis')
    # sample_action = np.array([-1, -1])
    # env.step(sample_action)
    # frame = env._render_frame(inference_plot=True)
    # path = "quantum_dot_plot_2.png"
    # plt.imsave(path, frame, cmap='viridis')
    # env.close()
