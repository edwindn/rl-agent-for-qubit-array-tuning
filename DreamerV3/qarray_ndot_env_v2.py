import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
from qarray_base_class import QarrayBaseClass
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import time
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import io
import fcntl
import json

"""
Defines the main class for running n-1 dreamers in parallel on n dots.
do NOT train the model on this class
"""

class QuantumDeviceEnv(QarrayBaseClass):
    """
    Defines the quantum dot array class for multi-agent rollouts
    note: this class should not be used to train any models, use only at inference time
    """
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
    
    def __init__(self, config_path='qarray_4dot_config.yaml', render_mode=None, counter_file=None, ndots=4, **kwargs):
        """
        constructor for the environment

        define action and observation spaces

        init state and variables
        """

        assert ndots%4==0, "Currently we only support multiples of 4 dots."

        print(f'Initialising qarray env with {ndots} dots ...')

        super().__init__(num_dots=ndots, num_voltages=ndots, config_path=config_path, render_mode=render_mode, counter_file=counter_file, **kwargs)


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

        if seed is not None:
            gym.Env.reset(self, seed=seed)
        else:
            gym.Env.reset(self, seed=None)

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

        optimal_VG_center = self.model.optimal_Vg(self.optimal_VG_center)

        # Device state variables (episode-specific)
        self.device_state = {
            "model": self.model,
            # "current_voltages": vg_current,
            "ground_truth_center": optimal_VG_center,
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
        
        max_possible_distance = np.sqrt(self.num_voltages) * (self.obs_voltage_max - self.obs_voltage_min)
        # max_possible_distance = np.sqrt(self.num_voltages) * (self.action_voltage_max - self.action_voltage_min)
    
        if self.current_step == self.max_steps:
            reward = max(max_possible_distance - distance, 0)*0.01
            # reward = (1 - distance / max_possible_distance) * 100
        else:
            reward = 0.0

        reward -= self.current_step * 0.1

        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        if at_target:
            reward += 200.0
        
        # print(reward)
        return reward

        # ---- #

        # Capacitance reward
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

        Cdd_base = self.config['simulator']['model']['Cdd']
        Cgd_base = self.config['simulator']['model']['Cgd']
        Cds_base = self.config['simulator']['model']['Cds']
        Cgs_base = self.config['simulator']['model']['Cgs']

        model_mats = []
        for mat in [Cdd_base, Cgd_base]:
            block_size = np.array(mat).shape[0]
            num_blocks = self.num_dots // block_size
            out_mat = block_diag(*([mat]*num_blocks))
            model_mats.append(out_mat)

        Cdd, Cgd = model_mats
        Cds = [np.array(Cds_base).flatten().tolist() * (self.num_dots // 4)]
        Cgs = [np.array(Cgs_base).flatten().tolist() * (self.num_dots // 4)]

        # print(np.array(Cdd).shape)
        # print(np.array(Cgd).shape)
        # print(np.array(Cds).shape)
        # print(np.array(Cgs).shape)

        model = ChargeSensedDotArray(
            Cdd=Cdd,
            Cgd=Cgd,
            Cds=Cds,
            Cgs=Cgs,
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
        # self.z = self._get_charge_sensor_data(current_voltages, gate1, gate2)
        allgates = list(range(1, self.num_voltages+1))
        self.all_z = []
        for (gate1, gate2) in zip(allgates[:-1], allgates[1:]):
            z = self._get_charge_sensor_data(voltage_centers, gate1, gate2)
            self.all_z.append(z)


        all_images = []
        voltage_centers = self.device_state["voltage_centers"]

        expected_voltage_shape = (self.num_voltages,)
        
        if voltage_centers.shape != expected_voltage_shape:
            raise ValueError(f"Voltage observation shape {voltage_centers.shape} does not match expected {expected_voltage_shape}")


        for z in self.all_z:
            # Extract first channel and normalize for image observation
            channel_data = z[:, :, 0]  # Shape: (height, width)
            image_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
            
            # Create multi-modal observation dictionary with numpy arrays 
            all_images.append(image_obs)

        all_images = np.concatenate(all_images, axis=-1)
        # all_images = all_images.squeeze(-1).transpose(1, 2, 0)
            
        # Validate observation structure
        expected_image_shape = (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels)

        if all_images.shape != expected_image_shape:
            raise ValueError(f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}")

        return {
            "image": all_images, # creates a multi-channel image with each adjacent pair of voltage sweeps
            "obs_voltages": voltage_centers
        }


if __name__ == "__main__":
    import sys
    env = QuantumDeviceEnv(ndots=12)
    obs, _ = env.reset()
    print(obs['image'].shape)
    sys.exit(0)

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
