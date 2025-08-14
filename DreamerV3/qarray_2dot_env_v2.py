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
import matplotlib.pyplot as plt
import io
import fcntl
import json

from utils import sigmoid


"""
Defines the class for training a single agent with full randomisation (except for rotations)

do NOT use this class for inference
"""

class QuantumDeviceEnv(QarrayBaseClass):
    """
    Represents the device with its quantum dots
    """
    
    def __init__(self, config_path='qarray_base_config.yaml', render_mode=None, counter_file=None, **kwargs):

        print('Initialising 4-dot qarray env with 2 voltages ...')

        super().__init__(num_dots=4, num_voltages=2, randomise_actions=True, config_path=config_path, render_mode=render_mode, counter_file=counter_file, **kwargs)

        cdd_max = self.config['simulator']['model']['Cdd']['max']
        cdd_min = self.config['simulator']['model']['Cdd']['min']
        self.cdd_max = cdd_max
        self.cdd_min = cdd_min

        cgd_diag_max = self.config['simulator']['model']['Cgd']['diagonal']['max']
        cgd_diag_min = self.config['simulator']['model']['Cgd']['diagonal']['min']
        cgd_off_max = self.config['simulator']['model']['Cgd']['off_diagonal']['max']
        cgd_off_min = self.config['simulator']['model']['Cgd']['off_diagonal']['min']
        self.cgd_min = min(cgd_off_min, cgd_diag_min)
        self.cgd_max = max(cgd_off_max, cgd_diag_max)

        self.current_cdd = None
        self.current_cgd = None
        matrix_shape = (self.num_voltages, self.num_voltages)
        matrix_length = np.prod(matrix_shape)
        self.capacitance_shape = matrix_shape
        self.max_matrix_dist = np.linalg.norm(np.ones(matrix_shape)) # since we normalise the capacitances in get_reward
        
        model_cgd = np.array(self.model_cgd)
        middle_cgd = model_cgd[1:-1, 1:-1]
        model_cdd = np.array(self.model_cdd)
        middle_cdd = model_cdd[1:-1, 1:-1]
        assert middle_cdd.min() >= 0 and middle_cdd.max() <= 1, "Cdd is not normalised"
        assert middle_cgd.min() >= 0 and middle_cgd.max() <= 1, "Cgd is not normalised"
        self.cdd_ground_truth = middle_cdd
        self.cgd_ground_truth = middle_cgd

        self.action_space = spaces.Dict({
            'action_voltages': spaces.Box(
                low=self.action_voltage_min,
                high=self.action_voltage_max,
                shape=(self.num_voltages,),
                dtype=np.float32
            ),
            'done': spaces.Box(
                shape=(1,),
                low=float('-inf'),
                high=float('inf'),
                dtype=np.float32
            ),
            'cdd':spaces.Box(
                shape=(matrix_length,),
                low=self.cdd_min,
                high=self.cdd_max,
                dtype=np.float32
            ),
            'cgd': spaces.Box(
                shape=(matrix_length,),
                low=self.cgd_min,
                high=self.cgd_max,
                dtype=np.float32
            )
        })

        self.reward_debug = self.config['training']['reward_debug']

    
    def _update_capacitances(self, cdd, cgd):
        cdd = np.array(cdd).flatten().astype(np.float32)
        cgd = np.array(cgd).flatten().astype(np.float32)
        self.current_cdd = cdd.reshape(self.capacitance_shape)
        self.current_cgd = cgd.reshape(self.capacitance_shape)

    
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
        
        try:
            voltages, cdd, cgd, done = action['action_voltages'], action['cdd'], action['cgd'], action['done']
        except:
            voltages = action
            done = None
            cdd = None
            cgd = None
            if self.debug:
                print("Action unpacking failed, done and capacitances set to None")

        if self.debug:
            print(f'Raw voltage outputs: {voltages}')

        # apply random transformation
        voltages = np.array(voltages).flatten().astype(np.float32)
        voltages = self.action_scale_factor * voltages + self.action_offset
        if self.debug:
            print(f'Scaled voltage outputs: {voltages}')

        self._apply_voltages(voltages) #this step will update the voltages stored in self.device_state
        if cdd is not None:
            self._update_capacitances(cdd, cgd)

        # --- Determine the reward ---
        reward, distance_rew, capacitance_rew, done_rew = self._get_reward(done)
        if self.debug or self.reward_debug:
            print(f"reward: {reward}")
            print(f"distance_reward: {distance_rew}")
            print(f"capacitance_reward: {capacitance_rew}")
            print(f"done_reward: {done_rew}")

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

        rew_dict = {
            'reward': reward,
            'distance_reward': distance_rew,
            'capacitance_reward': capacitance_rew,
            'done_reward': done_rew
        }
        info.update(rew_dict)
        
        return observation, reward, terminated, truncated, info

    
    def _get_reward(self, done=None):
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
            distance_reward = (1 - distance / max_possible_distance) * 100
        else:
            distance_reward = 0.0

        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        if at_target:
            distance_reward += 200.0

        time_penalty = -self.current_step * 0.1

        done_reward = 0.0
        if done is not None:
            prob = sigmoid(done)
            if prob > self.done_threshold:
                if at_target:
                    done_reward = 100.0
                else:
                    done_reward = -2.0
            elif prob <= self.done_threshold:
                if at_target:
                    done_reward = -100.0

        # Capacitance reward
        
        if self.current_cdd is not None and self.current_cgd is not None:
            #cap = (self.capacitances - cgd_min) / (cgd_max - cgd_min)
            # we don't normalise as the limits in config are already between 0 and 1
            cdd_dist = np.linalg.norm(self.current_cdd - self.cdd_ground_truth)
            cgd_dist = np.linalg.norm(self.current_cgd - self.cgd_ground_truth)
            dist = (cdd_dist + cgd_dist) / 2
        else:
            raise RuntimeError("_get_reward called before model capacitance output was set")

        capacitance_reward = 0.0
        if at_target or self.current_step == self.max_steps:
            capacitance_reward = 100 * (1 - dist/self.max_matrix_dist)

        reward = distance_reward + time_penalty + done_reward + capacitance_reward
        return reward, distance_reward, capacitance_reward, done_reward


if __name__ == "__main__":
    import sys
    env = QuantumDeviceEnv()
    env.reset()

    voltages = [-3.0, 1.0]
    env.step(voltages)

    gt = env.model.optimal_Vg(env.optimal_VG_center)
    print(gt)
    print(env.device_state['voltage_centers'])
    env.step(gt[1:3])
    print(env.device_state['voltage_centers'])
    frame = env._render_frame(inference_plot=True)
    path = "quantum_dot_plot.png"
    plt.imsave(path, frame, cmap='viridis')
    # sample_action = np.array([-1, -1])
    # env.step(sample_action)
    # frame = env._render_frame(inference_plot=True)
    # path = "quantum_dot_plot_2.png"
    # plt.imsave(path, frame, cmap='viridis')
    # env.close()
