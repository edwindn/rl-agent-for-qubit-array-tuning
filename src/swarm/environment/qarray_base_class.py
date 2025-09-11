"""
This file contains the QarrayBaseClass which is used to create a quantum dot array simulator.
"""

import os
import sys

# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
import numpy as np
import yaml
from qarray import ChargeSensedDotArray, LatchingModel, TelegraphNoise, WhiteNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from swarm.environment.qarray_remap import QarrayRemapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: gates are zero indexed but qarray is one indexed

"""
todo:

add barrier voltages in _get_obs
"""


class QarrayBaseClass:

    def __init__(
        self,
        num_dots,
        config_path="qarray_config.yaml",
        obs_voltage_min=-1.0,
        obs_voltage_max=1.0,
        obs_image_size=128,
        remap=False,
        debug=False,
        **kwargs,
    ):

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.num_dots = num_dots
        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1

        self.obs_image_size = obs_image_size
        self.obs_channels = self.num_dots - 1

        # Set voltage scanning range for observations
        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max

        optimal_center_dots = self.config["simulator"]["measurement"]["optimal_VG_center"]["dots"]
        optimal_center_sensor = self.config["simulator"]["measurement"]["optimal_VG_center"][
            "sensor"
        ]
        self.optimal_VG_center = [optimal_center_dots] * num_dots + [
            optimal_center_sensor
        ]  # must always call model.optimal_Vg on this

        # --- Initialize Model ---
        self.model = self._load_model()

        # --- Initialise remapping parameters ---
        self.remap = remap
        self.qarray_patched = QarrayRemapper(model=self.model, num_dots=num_dots, has_barriers=False, optimal_VG_center=self.optimal_VG_center,
                                            obs_voltage_min=obs_voltage_min, obs_voltage_max=obs_voltage_max)


    def _get_charge_sensor_data(self, voltage1, voltage2, gate1, gate2):
        """
        Get charge sensor data for given voltages.

        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration

        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """

        z, _ = self.model.do2d_open(
            gate1,
            voltage1 + self.obs_voltage_min,
            voltage1 + self.obs_voltage_max,
            self.obs_image_size,
            gate2,
            voltage2 + self.obs_voltage_min,
            voltage2 + self.obs_voltage_max,
            self.obs_image_size,
        )
        return z


    def _get_obs(self, gate_voltages, barrier_voltages, force_remap=False):
        """
        Helper method to get the current observation of the environment.

        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """
        # TODO we are currently not using the barrier voltages

        assert (
            len(gate_voltages) == self.num_dots
        ), f"Incorrect gate voltage shape, expected {self.num_dots}, got {len(gate_voltages)}"
        assert (
            len(barrier_voltages) == self.num_dots - 1
        ), f"Incorrect barrier voltage shape, expected {self.num_dots - 1}, got {len(barrier_voltages)}"

        if self.remap or force_remap:
            print("Warning: called get obs with remap set to True")
            return self._get_obs_remap(gate_voltages, barrier_voltages)


        allgates = list(range(1, self.num_dots + 1))  # Gate numbers for qarray (1-indexed)
        all_z = []
        for i, (gate1, gate2) in enumerate(zip(allgates[:-1], allgates[1:])):
            voltage1 = gate_voltages[i]  # Use 0-based indexing for gate_voltages array
            voltage2 = gate_voltages[i + 1]  # Use 0-based indexing for gate_voltages array
            z = self._get_charge_sensor_data(voltage1, voltage2, gate1, gate2)
            all_z.append(z[:, :, 0])

        # Stack images along the channel dimension
        all_images = np.stack(all_z, axis=-1)

        # Validate observation structure
        expected_image_shape = (
            self.obs_image_size,
            self.obs_image_size,
            self.obs_channels,
        )

        if all_images.shape != expected_image_shape:
            raise ValueError(
                f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}"
            )

        return {
            "image": all_images,  # unnormalised image
            "obs_gate_voltages": gate_voltages,
            "obs_barrier_voltages": barrier_voltages,
        }

    def _get_obs_remap(self, gate_voltages, barrier_voltages):
        allgates = list(range(1, self.num_dots + 1))  # Gate numbers for qarray (1-indexed)
        all_z = []

        new_gate_voltages = []
        new_barrier_voltages = []

        for i, (gate1, gate2) in enumerate(zip(allgates[:-1], allgates[1:])):
            voltage1 = gate_voltages[i]  # Use 0-based indexing for gate_voltages array
            voltage2 = gate_voltages[i + 1]  # Use 0-based indexing for gate_voltages array
            # z = self._get_charge_sensor_data(voltage1, voltage2, gate1, gate2)
            z, gate_vs, barrier_v = self.qarray_patched.get_remapped_scan(gate1, gate2, voltage1, voltage2)
            # each call returns only the first mapped voltage (since each gets computed twice)
            all_z.append(z[:, :, 0])
            if i == len(allgates) - 2: # last iteration
                new_gate_voltages.extend(gate_vs)
            else:
                new_gate_voltages.append(gate_vs[0])
            new_barrier_voltages.append(barrier_v)


        all_images = np.stack(all_z, axis=-1)

        # Validate observation structure
        expected_image_shape = (
            self.obs_image_size,
            self.obs_image_size,
            self.obs_channels,
        )

        if all_images.shape != expected_image_shape:
            raise ValueError(
                f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}"
            )
        
        assert len(new_gate_voltages) == self.num_dots, f"Incorrect gate voltage shape, expected {self.num_dots}, got {len(new_gate_voltages)}"
        assert len(new_barrier_voltages) == self.num_dots - 1, f"Incorrect barrier voltage shape, expected {self.num_dots - 1}, got {len(new_barrier_voltages)}"

        return {
            "image": all_images,
            "obs_gate_voltages": np.array(new_gate_voltages, dtype=np.float32),
            "obs_barrier_voltages": np.array(new_barrier_voltages, dtype=np.float32),
        }


    def _sample_from_range(self, range_config: dict, rng: np.random.Generator) -> float:
        """Sample a random value from a min-max range configuration."""
        return rng.uniform(range_config["min"], range_config["max"])

    def _get_coupling_by_distance(self, coupling_config: dict, distance: int) -> dict:
        """Get coupling range configuration based on distance between elements."""
        if distance == 1:
            return coupling_config[1]
        elif distance == 2:
            return coupling_config[2]
        else:
            return coupling_config["3_plus"]

    def _create_symmetric_matrix(self, size: int, fill_func) -> np.ndarray:
        """Create a symmetric matrix by filling upper triangle and mirroring."""
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                value = fill_func(i, j)
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    def _generate_cdd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate dot-to-dot capacitance matrix with distance-based coupling."""
        cdd_config = config_ranges["Cdd"]
        diagonal_val = cdd_config["diagonal"]
        distance_coupling = cdd_config["distance_coupling"]

        def fill_cdd(i: int, j: int) -> float:
            distance = abs(i - j)
            if distance == 0:
                return diagonal_val
            else:
                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                return self._sample_from_range(coupling_range, rng)

        return self._create_symmetric_matrix(self.num_dots, fill_cdd)

    def _generate_cgd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate gate-to-dot capacitance matrix with distance-based coupling."""
        cgd_config = config_ranges["Cgd"]
        num_gates = self.num_dots + 1  # plunger gates + sensor gate

        Cgd = np.zeros((self.num_dots, num_gates))

        # Fill plunger gate couplings
        for dot_i in range(self.num_dots):
            for gate_j in range(self.num_dots):
                distance = abs(dot_i - gate_j)
                if distance == 0:
                    # Primary coupling
                    coupling_range = cgd_config["primary_coupling"]
                else:
                    # Cross coupling
                    coupling_range = self._get_coupling_by_distance(
                        cgd_config["cross_coupling"], distance
                    )
                Cgd[dot_i, gate_j] = self._sample_from_range(coupling_range, rng)

        # Make plunger gate submatrix symmetric
        for i in range(self.num_dots):
            for j in range(i + 1, self.num_dots):
                avg_coupling = (Cgd[i, j] + Cgd[j, i]) / 2
                Cgd[i, j] = avg_coupling
                Cgd[j, i] = avg_coupling

        # Fill sensor gate couplings (last column)
        sensor_coupling = cgd_config["sensor_coupling"]
        for dot_i in range(self.num_dots):
            Cgd[dot_i, self.num_dots] = self._sample_from_range(sensor_coupling, rng)

        return Cgd

    def _generate_sensor_capacitances(self, config_ranges: dict, rng: np.random.Generator) -> tuple:
        """Generate dot-to-sensor (Cds) and gate-to-sensor (Cgs) capacitances."""
        # Cds: dot-to-sensor capacitances
        cds_config = config_ranges["Cds"]["dots"]
        Cds = [[self._sample_from_range(cds_config, rng) for _ in range(self.num_dots)]]

        # Cgs: gate-to-sensor capacitances
        cgs_plunger = config_ranges["Cgs"]["plunger_gates"]
        cgs_sensor = config_ranges["Cgs"]["sensor_gate"]

        cgs_values = [self._sample_from_range(cgs_plunger, rng) for _ in range(self.num_dots)]
        cgs_values.append(self._sample_from_range(cgs_sensor, rng))
        Cgs = [cgs_values]

        return Cds, Cgs

    def _generate_noise_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate white noise and telegraph noise parameters."""
        # White noise
        white_noise_amp = self._sample_from_range(config_ranges["white_noise_amplitude"], rng)

        # Telegraph noise
        telegraph_config = config_ranges["telegraph_noise_parameters"]
        p01 = self._sample_from_range(telegraph_config["p01"], rng)
        p10_factor = self._sample_from_range(telegraph_config["p10_factor"], rng)
        amplitude = self._sample_from_range(telegraph_config["amplitude"], rng)

        return {
            "white_noise_amplitude": white_noise_amp,
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": p10_factor * p01,
                "amplitude": amplitude,
            },
        }

    def _generate_latching_parameters(self, config_ranges: dict, rng: np.random.Generator) -> dict:
        """Generate symmetric latching model parameters."""
        latching_config = config_ranges["latching_model_parameters"]

        # Generate p_inter matrix (symmetric, zero diagonal)
        p_inter_range = latching_config["p_inter"]

        def fill_p_inter(i: int, j: int) -> float:
            if i == j:
                return 0.0  # No self-interaction
            else:
                return self._sample_from_range(p_inter_range, rng)

        p_inter = self._create_symmetric_matrix(self.num_dots, fill_p_inter)

        # Generate p_leads array
        p_leads_range = latching_config["p_leads"]
        p_leads = [self._sample_from_range(p_leads_range, rng) for _ in range(self.num_dots)]

        return {
            "Exists": True,
            "n_dots": self.num_dots,
            "p_leads": p_leads,
            "p_inter": p_inter,
        }

    def _gen_random_qarray_params(self, rng: np.random.Generator = None) -> dict:
        """
        Generate random parameters for the quantum device using distance-based coupling rules.

        Returns:
            dict: Complete model parameters for qarray simulator
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract configuration ranges
        model_config = self.config["simulator"]["model"]
        measurement_config = self.config["simulator"]["measurement"]

        config_ranges = {
            "Cdd": model_config["Cdd"],
            "Cgd": model_config["Cgd"],
            "Cds": model_config["Cds"],
            "Cgs": model_config["Cgs"],
            "white_noise_amplitude": model_config["white_noise_amplitude"],
            "telegraph_noise_parameters": model_config["telegraph_noise_parameters"],
            "latching_model_parameters": model_config["latching_model_parameters"],
            "T": model_config["T"],
            "coulomb_peak_width": model_config["coulomb_peak_width"],
        }

        # Generate all matrix components using helper methods
        Cdd = self._generate_cdd_matrix(config_ranges, rng)
        Cgd = self._generate_cgd_matrix(config_ranges, rng)
        Cds, Cgs = self._generate_sensor_capacitances(config_ranges, rng)
        noise_params = self._generate_noise_parameters(config_ranges, rng)
        latching_params = self._generate_latching_parameters(config_ranges, rng)

        # Assemble final model parameters
        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "white_noise_amplitude": noise_params["white_noise_amplitude"],
            "telegraph_noise_parameters": noise_params["telegraph_noise_parameters"],
            "latching_model_parameters": latching_params,
            "T": self._sample_from_range(config_ranges["T"], rng),
            "coulomb_peak_width": self._sample_from_range(config_ranges["coulomb_peak_width"], rng),
            "algorithm": model_config["algorithm"],
            "implementation": model_config["implementation"],
            "max_charge_carriers": model_config["max_charge_carriers"],
            "optimal_VG_center": measurement_config["optimal_VG_center"],
        }

        return model_params

    def _load_model(self):
        """
        Load the model from the config file.
        """

        model_params = self._gen_random_qarray_params()

        white_noise = WhiteNoise(amplitude=model_params["white_noise_amplitude"])
        telegraph_noise = TelegraphNoise(**model_params["telegraph_noise_parameters"])
        noise_model = white_noise + telegraph_noise
        latching_params = model_params["latching_model_parameters"]
        latching_model = (
            LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"})
            if latching_params["Exists"]
            else None
        )

        Cdd = model_params["Cdd"]
        Cgd = model_params["Cgd"]
        Cds = model_params["Cds"]
        Cgs = model_params["Cgs"]

        model = ChargeSensedDotArray(
            Cdd=Cdd,
            Cgd=Cgd,
            Cds=Cds,
            Cgs=Cgs,
            coulomb_peak_width=model_params["coulomb_peak_width"],
            T=model_params["T"],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params["algorithm"],
            implementation=model_params["implementation"],
            max_charge_carriers=model_params["max_charge_carriers"],
        )
        return model

    def _update_virtual_gate_matrix(self, cgd_estimate):
        vgm = -np.linalg.pinv(np.linalg.inv(self.model.Cdd) @ cgd_estimate)

        self.model.gate_voltage_composer.virtual_gate_matrix = vgm

    def _render_frame(self, image, path="quantum_dot_plot"):
        """
        Internal method to create the render image.

        Returns the CSD scan between gate1 and the next gate

        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        z = image

        vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)

        plt.figure(figsize=(5, 5))
        plt.imshow(
            z,
            extent=[vmin, vmax, vmin, vmax],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        plt.xlabel("$\\Delta$PL (V)")
        plt.ylabel("$\\Delta$PR (V)")
        plt.title("$|S_{11}|$ (Charge Stability Diagram)")
        plt.axis("equal")

        plt.savefig(f"{path}.png")
        plt.show()
        plt.close()

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

        with open(config_path) as file:
            config = yaml.safe_load(file)

        return config

    def calculate_ground_truth(self):
        """
        Get the ground truth for the quantum dot array.
        """

        vg_optimal_physical = self.model.optimal_Vg(self.optimal_VG_center)
        perfect_virtual_matrix = self.model.compute_optimal_virtual_gate_matrix()

        # removed - self.model.virtual_gate_origin
        vg_optimal_virtual = np.linalg.inv(perfect_virtual_matrix) @ (vg_optimal_physical)

        vg_optimal_virtual = vg_optimal_virtual[:-1]

        return vg_optimal_virtual


if __name__ == "__main__":
    num_dots = 2

    experiment = QarrayBaseClass(num_dots=num_dots, remap=True)
    import time

    start = time.time()

    # os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force CPU-only execution
    # os.environ['JAX_PLATFORMS'] = 'cpu'  # Alternative JAX CPU-only setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    image = experiment._get_obs([0] * num_dots, [0] * (num_dots - 1))["image"][:, :, 0]
    print(time.time() - start)

    start = time.time()

    voltage = - 2.0

    image = experiment._get_obs([voltage] * num_dots, [0] * (num_dots - 1))["image"][:, :, 0]
    print(time.time() - start)

    experiment._render_frame(image)#, path=f"quantum_dot_plot_{str(voltage)}")
