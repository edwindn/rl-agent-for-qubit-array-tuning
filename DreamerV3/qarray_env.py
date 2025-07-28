import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import json
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io

class QuantumDeviceEnv(gym.Env):
    """
    Represents the device with its quantum dots 
    """
    #rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path='qarray_config.yaml', render_mode=None, **kwargs):
        """
        constructor for the environment

        define action and observation spaces

        init state and variables
        """
        super().__init__()

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.debug = self.config['training']['debug']
        self.seed = self.config['training']['seed']
        self.max_steps = self.config['env']['max_steps']
        self.current_step = 0
        self.tolerance = self.config['env']['tolerance']

        # --- Define Action and Observation Spaces ---
        self.num_voltages = self.config['env']['action_space']['num_voltages']  # Default to 2 gate voltages 
        self.action_voltage_min = self.config['env']['action_space']['voltage_range'][0]  
        self.action_voltage_max = self.config['env']['action_space']['voltage_range'][1]   
        
        self.action_space = spaces.Box(
            low=self.action_voltage_min, 
            high=self.action_voltage_max, 
            shape=(self.num_voltages,), 
            dtype=np.float32
        )

        # Observation space for quantum device state - multi-modal with image and voltages
        obs_config = self.config['env']['observation_space']
        self.obs_image_size = obs_config['image_size']
        self.obs_channels = obs_config['channels']
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
            'voltages': spaces.Box(
                low=self.action_voltage_min,
                high=self.action_voltage_max,
                shape=(self.num_voltages,),
                dtype=np.float32
            )
        })

        self.obs_voltage_min = self.config['simulator']['measurement']['v_min']
        self.obs_voltage_max = self.config['simulator']['measurement']['v_max']

        # --- Initialize Model ---
        self.model = None
        
        # --- Initialize normalization parameters ---
        self._init_normalization_params()

        # --- For Rendering --- 
        self.render_fps = self.config['training']['render_fps'] #unused for now
        self.render_mode = render_mode or self.config['training']['render_mode']

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

    def _get_charge_sensor_data(self, voltages):
        """
        Get charge sensor data for given voltages.
        
        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """
        z, _ = self.device_state["model"].charge_sensor_open(voltages)
        return z

    def get_raw_observation(self):
        """
        Get raw (unnormalized) observation data for debugging purposes.
        
        Returns:
            np.ndarray: Raw charge sensor data of shape (height, width)
        """
        current_voltages = self.device_state["current_voltages"]
        z = self._get_charge_sensor_data(current_voltages)
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
        # Handle seed if provided
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0

        #initialize the voltage scaler
        self.scaler = np.random.uniform(0.33, 3.33)
        
        # Reset episode-specific normalization statistics
        self.episode_min = float('inf')
        self.episode_max = float('-inf')

        # --- Initialize episode-specific Model ---
        self.model = self._load_model()


        # Initialize episode-specific voltage state
        #center of current window
        center = self._random_center()

        #current window
        vg_current = self.model.gate_voltage_composer.do2d(
            "vP1", center[0]+self.obs_voltage_min, center[0]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
            "vP2", center[1]+self.obs_voltage_min, center[1]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        )

        optimal_VG_center = self.model.optimal_Vg(self.config['simulator']['measurement']['optimal_VG_center'])
        vg_current += optimal_VG_center

        # Device state variables (episode-specific)
        self.device_state = {
            "model": self.model,
            "current_voltages": vg_current,
            "ground_truth_center": optimal_VG_center,
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

        #scale the action
        action = action * self.scaler


        # --- Update the environment's state based on the action ---
        self.current_step += 1definitely
        # action is now a numpy array of shape (num_voltages,) containing voltage values

        self._apply_voltages(action) #this step will update the voltages stored in self.device_state

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
        
        # Check if the centers of the voltage sweeps are aligned (ignoring third dimension)
        ground_truth_center = self.device_state["ground_truth_center"][:2]

        # Get current voltage settings (what the agent controls)
        current_voltage_center = self._extract_voltage_centers(self.device_state["current_voltages"])

        # Compare only the first 2 dimensions (ignoring third dimension)
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
        return np.random.uniform(self.action_voltage_min-self.obs_voltage_min, self.action_voltage_max-self.obs_voltage_max, 2)
    
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

        ground_truth_center = self.device_state["ground_truth_center"][:2]
        
        # Get current voltage settings (what the agent controls)
        # We need to get this from the action that was just applied
        # Since action is passed to step(), we need to store it or calculate it
        current_voltage_center = self._extract_voltage_centers(self.device_state["current_voltages"])
        

        # Calculate distance between target and current (2D)
        distance = np.linalg.norm(ground_truth_center - current_voltage_center)
        
        # Calculate maximum possible distance in 2D voltage space
        max_possible_distance = np.sqrt(2) * (self.obs_voltage_max - self.obs_voltage_min)
        
        # Reward = max_possible_distance - current_distance
        # This gives maximum reward when distance = 0 (perfect alignment)
        # and minimum reward (0) when distance = max_possible_distance (worst case)
        reward = max(max_possible_distance - distance, 0)*0.1


        # Penalize for taking too many steps (small penalty to encourage efficiency)
        reward -= self.current_step * 0.1

        # ADDED (should be more? what is the maximal reward we can achieve without terminating)
        # could have a length-based reward, eg. 1000*exp(-current_step)
        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        if at_target:
            reward += 200.0

        return reward

    def _gen_random_qarray_params(self):
        """
        Generate random parameters for the quantum device.
        """
        rb = self.config['simulator']['model']
        
        latching = True 
        
        cdd_value = np.random.uniform(rb["Cdd"]["min"], rb["Cdd"]["max"])
        p_inter = [
            [0., np.random.uniform(rb["latching_model_parameters"]["p_inter"]["min"], rb["latching_model_parameters"]["p_inter"]["max"])],
            [np.random.uniform(rb["latching_model_parameters"]["p_inter"]["min"], rb["latching_model_parameters"]["p_inter"]["max"]), 0.],
        ]

        p01 = np.random.uniform(rb["telegraph_noise_parameters"]["p01"]["min"], rb["telegraph_noise_parameters"]["p01"]["max"])
        model_params = {
            "Cdd": [[1, cdd_value], [cdd_value, 1]],
            "Cgd": [
                [
                    np.random.uniform(*rb["Cgd"][0][0].values()),
                    np.random.uniform(*rb["Cgd"][0][1].values()),
                    np.random.uniform(*rb["Cgd"][0][2].values())
                ],
                [
                    np.random.uniform(*rb["Cgd"][1][0].values()),
                    np.random.uniform(*rb["Cgd"][1][1].values()),
                    np.random.uniform(*rb["Cgd"][1][2].values())
                ]
            ],
            "Cds": [[
                np.random.uniform(rb["Cds"][0]["min"], rb["Cds"][0]["max"]),
                np.random.uniform(rb["Cds"][1]["min"], rb["Cds"][1]["max"])
            ]],
            "Cgs": [[
                np.random.uniform(rb["Cgs"][0]["min"], rb["Cgs"][0]["max"]),
                np.random.uniform(rb["Cgs"][1]["min"], rb["Cgs"][1]["max"]),
                rb["Cgs"][2]["fixed"]
            ]],
            "white_noise_amplitude": np.random.uniform(rb["white_noise_amplitude"]["min"], rb["white_noise_amplitude"]["max"]),
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": np.random.uniform(rb["telegraph_noise_parameters"]["p10_factor"]["min"], rb["telegraph_noise_parameters"]["p10_factor"]["max"]) * p01,
                "amplitude": np.random.uniform(rb["telegraph_noise_parameters"]["amplitude"]["min"], rb["telegraph_noise_parameters"]["amplitude"]["max"]),
            },
            "latching_model_parameters": {
                "Exists": latching,
                "n_dots": 2,
                "p_leads": [
                    np.random.uniform(rb["latching_model_parameters"]["p_leads"]["min"], rb["latching_model_parameters"]["p_leads"]["max"]),
                    np.random.uniform(rb["latching_model_parameters"]["p_leads"]["min"], rb["latching_model_parameters"]["p_leads"]["max"])
                ],
                "p_inter": p_inter,
            },
            "T": np.random.uniform(rb["T"]["min"], rb["T"]["max"]),
            "coulomb_peak_width": np.random.uniform(rb["coulomb_peak_width"]["min"], rb["coulomb_peak_width"]["max"]),
            "algorithm": rb["algorithm"],
            "implementation": rb["implementation"],
            "max_charge_carriers": rb["max_charge_carriers"],
        }
        
        return model_params

    def _load_model(self):
        """
        Load the model from the config file.
        """
        # Generate random model parameters
        model_params = self._gen_random_qarray_params()

        # Load the model parameters
        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None

        model = ChargeSensedDotArray(
            Cdd=model_params['Cdd'],
            Cgd=model_params['Cgd'],
            Cds=model_params['Cds'],
            Cgs=model_params['Cgs'],
            coulomb_peak_width=model_params['coulomb_peak_width'],
            T=model_params['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params['algorithm'],
            implementation=model_params['implementation'],
            max_charge_carriers=model_params['max_charge_carriers'],
        )
        
        model.gate_voltage_composer.virtual_gate_matrix = np.array([[ 1, 0, 0], [0, 1, 0], [0, 0, 0]])


        return model

    def _get_obs(self):
        """
        Helper method to get the current observation of the environment.
        
        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """
        # Get current voltage configuration
        current_voltages = self.device_state["current_voltages"]
        
        # Get charge sensor data
        self.z = self._get_charge_sensor_data(current_voltages)
        z = self.z
        # Extract first channel and normalize for image observation
        channel_data = z[:, :, 0]  # Shape: (height, width)
        image_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
        
        # Extract voltage centers for voltage observation
        voltage_centers = self._extract_voltage_centers(current_voltages)  # Shape: (2,)
        
        # Create multi-modal observation dictionary with numpy arrays 
        observation = {
            'image': image_obs,  # numpy array
            'voltages': voltage_centers  # numpy array
        }
        
        # Validate observation structure
        expected_image_shape = (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels)
        expected_voltage_shape = (self.num_voltages,)
        
        if observation['image'].shape != expected_image_shape:
            raise ValueError(f"Image observation shape {observation['image'].shape} does not match expected {expected_image_shape}")
        
        if observation['voltages'].shape != expected_voltage_shape:
            raise ValueError(f"Voltage observation shape {observation['voltages'].shape} does not match expected {expected_voltage_shape}")
        
        return observation


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
        
        # Create two grids centered around voltages[0] and voltages[1]
        # Grid extent from obs_vmin to obs_vmax
        x_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[1])
        y_grid = np.linspace(self.obs_voltage_min, self.obs_voltage_max, self.obs_image_size[0])
        
        # Create 2D grids centered around the voltage values
        X, Y = np.meshgrid(x_grid + voltages[0], y_grid + voltages[1])

        # Update the first two channels with the new grids
        self.device_state["current_voltages"][:,:,0] = X
        self.device_state["current_voltages"][:,:,1] = Y
        
        # Keep the third channel unchanged (as per the TODO comment)
        # self.device_state["current_voltages"][:,:,2] remains as initialized

        self.device_state["current_voltages"][:,:,:2] = np.clip(self.device_state["current_voltages"][:,:,:2], self.action_voltage_min, self.action_voltage_max)
    

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
    env = QuantumDeviceEnv()
    env.reset()
    frame = env._render_frame(inference_plot=True)
    path = "quantum_dot_plot.png"
    plt.imsave(path, frame, cmap='viridis')
    sample_action = np.array([-1, -1])
    action=env.device_state["ground_truth_center"]
    print(action)
    env.step(action)
    frame = env._render_frame(inference_plot=True)
    path = "quantum_dot_plot_2.png"
    plt.imsave(path, frame, cmap='viridis')
    env.close()


    #charge sensor voltage, note this is being completely ignored for now, just kept as intialised

#update to include virtual gate randomization