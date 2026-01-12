"""
This script generates random data samples on the fly
with varying levels of nosie and adjustable parameters
to assess the robustness of the capacitance prediction model
"""

import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt

# Add src directory to path for clean imports
from pathlib import Path
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
sys.path.insert(0, str(src_dir))

from swarm.environment.qarray_base_class import QarrayBaseClass
from swarm.capacitance_model.CapacitancePrediction import CapacitancePredictionModel
from swarm.capacitance_model.BayesianUpdater import CapacitancePredictor
from swarm.capacitance_model.dataloader import PercentileNormalize, get_channel_targets

class TestCapacitancePredictionModel:

    @staticmethod
    def _load_data(n_dots, seed_base=42, voltage_offset_range=0.5, reverse=True, use_barriers=False):
        if use_barriers:
            raise NotImplementedError
        else:
            print("WARNING: barriers disabled for this test")
            
        # create data of varying levels of noise to test confidence update
        
        # create data with increasing levels of noise
        param_override_list = [
            {"white_noise_amplitude": 0.0, "telegraph_noise_parameters.amplitude": 0.001},
            {"white_noise_amplitude": 0.0001, "telegraph_noise_parameters.amplitude": 0.001},
            {"white_noise_amplitude": 0.0005, "telegraph_noise_parameters.amplitude": 0.002},
            {"white_noise_amplitude": 0.001, "telegraph_noise_parameters.amplitude": 0.005},
            {"white_noise_amplitude": 0.002, "telegraph_noise_parameters.amplitude": 0.01},
            {"white_noise_amplitude": 0.005, "telegraph_noise_parameters.amplitude": 0.02},
            {"white_noise_amplitude": 0.01, "telegraph_noise_parameters.amplitude": 0.05},
        ]

        if reverse:
            param_override_list = list(reversed(param_override_list))

        observations = []

        for params in param_override_list:
            qarray = QarrayBaseClass(
                num_dots=n_dots,
                use_barriers=use_barriers,
                obs_voltage_min=-1.5,
                obs_voltage_max=1.5,
                obs_image_size=100,
            )
            
            gt_voltages, _, _ = qarray.calculate_ground_truth()

            rng = np.random.default_rng(seed_base)
            voltage_offset = rng.uniform(
                -voltage_offset_range,
                voltage_offset_range,
                size=len(gt_voltages)
            )
            gate_voltages = gt_voltages + voltage_offset

            # Create dummy barrier voltages (not used in current implementation)
            barrier_voltages = [0.0] * (n_dots - 1)

            obs = qarray._get_obs(gate_voltages, barrier_voltages)

            cgd_matrix = qarray.model.Cgd.copy()

            cgd_targets = get_channel_targets(channel_idx=1, cgd_matrix=cgd_matrix, num_dots=n_dots)
            # cgd_ground_truth = np.array([cgd_matrix[0,2], cgd_matrix[1,2], cgd_matrix[1,3]], dtype=np.float32)

            observation = {
                'image': obs['image'].astype(np.float32)[:, :, 1:2],  # get middle two dots
                'cgd_matrix': cgd_matrix.astype(np.float32),
                'cgd_ground_truth': cgd_targets,
                'ground_truth_voltages': gt_voltages.astype(np.float32),
                'gate_voltages': gate_voltages.astype(np.float32),
                'white_noise': params['white_noise_amplitude'],
                'telegraph_noise': params['telegraph_noise_parameters.amplitude']
            }

            observations.append(observation)

        # Create visualization of all images in a single plot
        fig, axes = plt.subplots(1, len(observations), figsize=(3 * len(observations), 4))
        if len(observations) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        for idx, obs in enumerate(observations):
            # Extract the 2D image (remove the channel dimension)
            image = obs['image'][:, :, 0]  # Take first channel since it's single channel
            
            # Plot with viridis colormap in grayscale-friendly format
            im = axes[idx].imshow(image, cmap='viridis', aspect='auto')
            
            # Add labels with noise parameters
            white_noise = obs['white_noise']
            telegraph_noise = obs['telegraph_noise']
            axes[idx].set_title(f'White: {white_noise:.4f}\nTelegraph: {telegraph_noise:.4f}', fontsize=10)
            axes[idx].set_xlabel('$V_g 1$')
            axes[idx].set_ylabel('$V_g 2$')
            
            # make images square
            axes[idx].set_aspect('equal')

        plt.tight_layout()
        plt.suptitle('Charge Stability Diagrams with Different Noise Levels', y=1.02)
        
        # Save the combined image
        output_path = os.path.join(os.path.dirname(__file__), 'noise_comparison_images.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined noise comparison images to: {output_path}")

        return observations
        
    
    def __init__(self, n_dots, weights_path, reverse=False, disable_nearest_neighbours=False):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        assert n_dots == 4, "Currently only runs with 4 dots (Some hardcoded indices)"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        checkpoint = torch.load(weights_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create model with correct output size
        output_size = 3 if disable_nearest_neighbours else 2
        self.ml_model = CapacitancePredictionModel(output_size)

        self.ml_model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
        self.ml_model.to(device)
        self.ml_model.eval()

        print("Sweeping dots 1 and 2 (zero-indexed)")

        prior_dict = {}
        for i in range(n_dots):
            for j in range(n_dots):
                if i == j:
                    prior_dict[(i, j)] = (1, 0.01)  # Self-capacitance
                elif abs(i - j) == 1:
                    prior_dict[(i, j)] = (0.40, 0.2)  # Nearest neighbors
                elif abs(i - j) == 2:
                    prior_dict[(i, j)] = (0.2, 0.1)   # Distant pairs
                else:
                    prior_dict[(i, j)] = (0., 0.1)   # Distant pairs

        self.bayesian_predictor = CapacitancePredictor(n_dots, prior_dict)

        self.n_dots = n_dots

        self.normalizer = PercentileNormalize()

    
    def _run_noise_test(self):
        data = self._load_data(n_dots=self.n_dots)

        print("Prior means:")
        print(self.bayesian_predictor.means)

        for i, obs in enumerate(data):
            print("\n")
            print("="*40)
            print(f"Observation {i+1}")
            image = self.normalizer(obs['image']) # apply same normalization as in training
            image = image.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            values, logvars = self.ml_model(image)
            values = values.detach().flatten().cpu().numpy()
            logvars = logvars.detach().flatten().cpu().numpy()

            print("Ground truth values: ", end="")
            print(obs['cgd_ground_truth'])

            print("Values: ", end="")
            print(values)
            print("Variances: ", end="")
            print(np.exp(logvars))

            self.bayesian_predictor.update_from_scan((1, 2), [(values[0], logvars[0]), (values[1], logvars[1]), (values[2], logvars[2])])

            print("Updated means:")
            print(self.bayesian_predictor.means)
            print("Ground truth:")
            print(obs['cgd_matrix'][:,:-1])


    def _run_zeros_test(self):
        zeros = torch.zeros(1, 1, 128, 128).to(torch.float32).to(self.device)
        values, logvars = self.ml_model(zeros)
        values = values.detach().flatten().cpu().numpy()
        logvars = logvars.detach().flatten().cpu().numpy()

        print("\nAll zeros")
        print("Values: ", end="")
        print(values)
        print("Variances: ", end="")
        print(np.exp(logvars))

        ones = torch.ones(1, 1, 128, 128).to(torch.float32).to(self.device)
        values, logvars = self.ml_model(ones)
        values = values.detach().flatten().cpu().numpy()
        logvars = logvars.detach().flatten().cpu().numpy()

        print("\nAll ones")
        print("Values: ", end="")
        print(values)
        print("Variances: ", end="")
        print(np.exp(logvars))

        rand = torch.randn(1, 1, 128, 128).to(torch.float32).to(self.device)
        values, logvars = self.ml_model(rand)
        values = values.detach().flatten().cpu().numpy()
        logvars = logvars.detach().flatten().cpu().numpy()

        print("\nWhite noise")
        print("Values: ", end="")
        print(values)
        print("Variances: ", end="")
        print(np.exp(logvars))


    def run_test(self, type):
        if type == "noise":
            self._run_noise_test()
        elif type == "zeros":
            self._run_zeros_test()
        else:
            raise ValueError(f"Unknown test type: {type}")


if __name__ == '__main__':
    np.set_printoptions(precision=10, suppress=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" # change as needed

    weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'best_model.pth')
    disable_nearest_neighbours = False  # Set to True if model was trained without nearest neighbours

    test = TestCapacitancePredictionModel(n_dots=4, weights_path=weights_path, reverse=True, disable_nearest_neighbours=disable_nearest_neighbours)
    test.run_test(type="noise")
    print("Test ran successfully")