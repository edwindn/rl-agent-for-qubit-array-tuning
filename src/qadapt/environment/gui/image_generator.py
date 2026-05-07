"""
Image generation wrapper for QarrayBaseClass to support GUI functionality.
Clean interface similar to env.py but focused on parameter exploration.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to path for clean imports
src_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_dir))

from qadapt.environment.qarray_base_class import QarrayBaseClass


class QarrayImageGenerator:
    """
    Clean wrapper around QarrayBaseClass for GUI parameter exploration.
    Similar to QuantumDeviceEnv but focused on visualization rather than RL.
    """

    def __init__(
        self,
        num_dots: int = 2,
        use_barriers: bool = False,
        obs_voltage_min: float = -1.0,
        obs_voltage_max: float = 1.0,
        obs_image_size: int = 128,
        config_path: str = "qarray_config.yaml",
        use_variable_peak_width: bool = False,
    ):
        """
        Initialize the image generator.

        Args:
            num_dots: Number of quantum dots
            use_barriers: Whether to use barrier voltages
            obs_voltage_min: Minimum voltage for observation window
            obs_voltage_max: Maximum voltage for observation window
            obs_image_size: Size of generated images (image_size x image_size)
            config_path: Path to qarray configuration file
        """
        self.num_dots = num_dots
        self.use_barriers = use_barriers
        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max
        self.obs_image_size = obs_image_size
        self.use_variable_peak_width = use_variable_peak_width

        # Initialize with default parameters
        self.reset_array()

    def reset_array(self, param_overrides: dict = None):
        """
        Create a new QarrayBaseClass instance with optional parameter overrides.

        Args:
            param_overrides: Dictionary of parameter overrides for the qarray config
        """
        # Debug: print what parameters we're receiving
        if param_overrides:
            print(f"🔧 Image Generator received {len(param_overrides)} parameter overrides:")
            for key, value in param_overrides.items():
                print(f"  {key}: {value}")
        else:
            print("🔧 Image Generator: No parameter overrides provided")

        # Get the absolute path to qarray_config.yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "qarray_config.yaml"
        )

        self.array = QarrayBaseClass(
            num_dots=self.num_dots,
            use_barriers=self.use_barriers,
            config_path=config_path,
            obs_voltage_min=self.obs_voltage_min,
            obs_voltage_max=self.obs_voltage_max,
            obs_image_size=self.obs_image_size,
            param_overrides=param_overrides,
            vary_peak_width=self.use_variable_peak_width,
        )

        # Calculate ground truth for reference
        if self.use_barriers:
            self.plunger_ground_truth, self.barrier_ground_truth, self.sensor_ground_truth = (
                self.array.calculate_ground_truth()
            )
            if self.barrier_ground_truth is None:
                self.barrier_ground_truth = np.zeros(self.num_dots - 1)
        else:
            ground_truth_result = self.array.calculate_ground_truth()
            if isinstance(ground_truth_result, tuple):
                self.plunger_ground_truth = ground_truth_result[0]
            else:
                self.plunger_ground_truth = ground_truth_result
            self.barrier_ground_truth = np.zeros(self.num_dots - 1)
            self.sensor_ground_truth = None

    def generate_image(
        self,
        gate_voltages: np.ndarray,
        barrier_voltages: np.ndarray = None,
        gate_pair_index: int = 0
    ) -> np.ndarray:
        """
        Generate a charge stability diagram for the specified gate pair.

        Args:
            gate_voltages: Array of gate voltages (length = num_dots)
            barrier_voltages: Array of barrier voltages (length = num_dots - 1),
                            optional if use_barriers=False
            gate_pair_index: Which adjacent gate pair to visualize (0 to num_dots-2)

        Returns:
            2D numpy array representing the charge stability diagram
        """
        if len(gate_voltages) != self.num_dots:
            raise ValueError(f"Expected {self.num_dots} gate voltages, got {len(gate_voltages)}")

        if self.use_barriers:
            if barrier_voltages is None:
                barrier_voltages = np.zeros(self.num_dots - 1)
            elif len(barrier_voltages) != self.num_dots - 1:
                raise ValueError(f"Expected {self.num_dots - 1} barrier voltages, got {len(barrier_voltages)}")
        else:
            barrier_voltages = None

        if gate_pair_index < 0 or gate_pair_index >= self.num_dots - 1:
            raise ValueError(f"Gate pair index must be between 0 and {self.num_dots - 2}")

        # Get the full observation
        obs = self.array._get_obs(gate_voltages, barrier_voltages)

        # Return the specified channel (gate pair)
        return obs["image"][:, :, gate_pair_index]

    def get_ground_truth_voltages(self) -> dict:
        """
        Get the calculated ground truth voltages.

        Returns:
            Dictionary containing ground truth voltages
        """
        result = {
            "plunger_voltages": self.plunger_ground_truth,
            "barrier_voltages": self.barrier_ground_truth,
        }

        if self.sensor_ground_truth is not None:
            result["sensor_voltage"] = self.sensor_ground_truth

        return result

    def sample_random_voltages(self, voltage_range: float = 2.0) -> dict:
        """
        Sample random voltages around ground truth for quick testing.

        Args:
            voltage_range: Range around ground truth to sample from

        Returns:
            Dictionary with sampled voltages
        """
        plunger_voltages = self.plunger_ground_truth + np.random.uniform(
            -voltage_range/2, voltage_range/2, size=self.num_dots
        )

        barrier_voltages = self.barrier_ground_truth + np.random.uniform(
            -voltage_range/2, voltage_range/2, size=self.num_dots - 1
        )

        return {
            "plunger_voltages": plunger_voltages,
            "barrier_voltages": barrier_voltages,
        }

    def get_available_gate_pairs(self) -> list:
        """
        Get list of available gate pairs for visualization.

        Returns:
            List of tuples (gate1_index, gate2_index, display_name)
        """
        pairs = []
        for i in range(self.num_dots - 1):
            pairs.append((i, i + 1, f"Gates {i+1}-{i+2}"))
        return pairs

    def get_model_info(self) -> dict:
        """
        Get information about the current model parameters.

        Returns:
            Dictionary with model information
        """
        return {
            "num_dots": self.num_dots,
            "use_barriers": self.use_barriers,
            "obs_voltage_range": (self.obs_voltage_min, self.obs_voltage_max),
            "image_size": self.obs_image_size,
            "ground_truth": self.get_ground_truth_voltages(),
        }