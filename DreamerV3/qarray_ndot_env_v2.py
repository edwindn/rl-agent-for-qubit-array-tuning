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

returns n-1 scans and expects n voltages for an update
"""

class QuantumDeviceEnv(QarrayBaseClass):
    """
    Defines the quantum dot array class for multi-agent rollouts
    """

    def __init__(self, config_path='qarray_base_config.yaml', render_mode=None, counter_file=None, ndots=4, **kwargs):
        
        assert ndots%4==0, "Currently we only support multiples of 4 dots."

        print(f'Initialising qarray env with {ndots} dots ...')

        super().__init__(num_dots=ndots, num_voltages=ndots, randomise_actions=False, config_path=config_path, render_mode=render_mode, counter_file=counter_file, **kwargs)


    def _load_model(self):
        """
        Load the model from the config file.
        """

        model_params = self._gen_random_qarray_params()

        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None

        Cdd_base = model_params['Cdd']
        Cgd_base = model_params['Cgd']
        Cds_base = model_params['Cds']
        Cgs_base = model_params['Cgs']

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
            coulomb_peak_width=model_params['coulomb_peak_width'],
            T=model_params['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params['algorithm'],
            implementation=model_params['implementation'],
            max_charge_carriers=model_params['max_charge_carriers'],
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
        #random actions scaling
        self._init_random_action_scaling()
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
            "ground_truth_center": optimal_VG_center,
            "voltage_centers": center
        }


        # --- Return the initial observation ---
        observation = self._get_obs()
        info = self._get_info() 

        return observation, info



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
