"""
This file contains the QarrayBaseClass which is used to create a quantum dot array simulator.
"""

import os
import sys
import jax.numpy as jnp
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
import numpy as np
import yaml
from qarray import ChargeSensedDotArray, LatchingModel, TelegraphNoise, WhiteNoise
from qarray_latched.DotArrays.barrier_voltage_model import BarrierVoltageModel
from qarray_latched.DotArrays.TunnelCoupledChargeSensed import TunnelCoupledChargeSensed
from qarray_latched.DotArrays.voltage_dependent_capacitance import (
    create_linear_capacitance_model,
)

from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm.environment.utils import VaryPeakWidth

# NOTE: gates are zero indexed but qarray is one indexed


class QarrayBaseClass:

    def __init__(
        self,
        num_dots: int,
        use_barriers: bool,
        config_path: str = "qarray_config.yaml",
        obs_voltage_min: float = -1.0,
        obs_voltage_max: float = 1.0,
        obs_image_size: int = 128,
        param_overrides: dict = None,
        vary_peak_width: bool = False,
        peak_width_alpha: float = 0.01,
        **kwargs,
    ):

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.use_barriers = use_barriers
        self.num_dots = num_dots
        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1

        self.obs_image_size = obs_image_size
        self.obs_channels = self.num_dots - 1

        # Set voltage scanning range for observations
        self.obs_voltage_min = obs_voltage_min
        self.obs_voltage_max = obs_voltage_max

        self.optimal_tc = self.config["simulator"]["measurement"]["tc"]
        optimal_center_dots = self.config["simulator"]["measurement"]["optimal_VG_center"]["dots"]
        optimal_center_sensor = self.config["simulator"]["measurement"]["optimal_VG_center"][
            "sensor"
        ]
        self.optimal_VG_center = [optimal_center_dots] * num_dots + [
            optimal_center_sensor
        ]  # must always call model.optimal_Vg on this

        self.barrier_alpha = None
        self.barrier_tc_base = None

        # Store peak width settings for use in _load_model
        self.vary_peak_width = vary_peak_width
        self.peak_width_alpha = peak_width_alpha

        # --- Initialize Model ---
        self.model = self._load_model(param_overrides)

        # Peak width model is now initialized in _load_model methods




    def _get_charge_sensor_data(self, voltage1, voltage2, gate1, gate2, barrier_voltages=None):
        """
        Get charge sensor data for given voltages using TunnelCoupledChargeSensed interface.

        Args:
            voltage1 (float): Center voltage for first gate
            voltage2 (float): Center voltage for second gate
            gate1 (int): First gate index (1-indexed for qarray)
            gate2 (int): Second gate index (1-indexed for qarray)
            barrier_voltages (np.ndarray): Barrier voltage array

        Returns:
            np.ndarray: Charge sensor data of shape (height, width)
        """

        if not self.use_barriers:
            # assert barrier_voltages is None, "Barrier voltages provided but model is not configured for barriers"

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
            z = z.reshape(self.obs_image_size, self.obs_image_size)
            return z

        # Generate 2D virtual gate grid using gate_voltage_composer
        assert barrier_voltages is not None, "Barrier voltages must be provided for models with barriers"
        vg = self.model.gate_voltage_composer.do2d(
            f"P{gate1}",
            voltage1 + self.obs_voltage_min,
            voltage1 + self.obs_voltage_max,
            self.obs_image_size,
            f"P{gate2}",
            voltage2 + self.obs_voltage_min,
            voltage2 + self.obs_voltage_max,
            self.obs_image_size,
        )

        # Flatten the voltage grid to shape (obs_image_size^2, n_gates)
        vg_flat = vg.reshape(-1, vg.shape[-1])

        # Create barrier voltage array (same shape as flattened gate voltage array)
        vb = jnp.full((vg_flat.shape[0], self.num_barrier_voltages), jnp.array(barrier_voltages))

        # Get charge sensor data using the new interface
        z, _ = self.model.charge_sensor_open(vg_flat, vb)

        # Reshape to 2D image
        z = z.reshape(self.obs_image_size, self.obs_image_size)

        return z


    def _get_obs(self, gate_voltages, barrier_voltages=None):
        """
        Helper method to get the current observation of the environment.

        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """

        assert (
            len(gate_voltages) == self.num_dots
        ), f"Incorrect gate voltage shape, expected {self.num_dots}, got {len(gate_voltages)}"

        if barrier_voltages is not None:
            assert (
                len(barrier_voltages) == self.num_dots - 1
            ), f"Incorrect barrier voltage shape, expected {self.num_dots - 1}, got {len(barrier_voltages)}"


        allgates = list(range(1, self.num_dots + 1))  # Gate numbers for qarray (1-indexed)
        all_z = []
        for i, (gate1, gate2) in enumerate(zip(allgates[:-1], allgates[1:])):

            if self.peak_width_model is not None:
                print(gate_voltages[i], gate_voltages[i+1])
                print(gate_voltages)
                peak_width = self.peak_width_model.linearly_vary_peak_width(gate_voltages[i], gate_voltages[i+1])
                self.model.coulomb_peak_width = peak_width

            voltage1 = float(gate_voltages[i])  # Use 0-based indexing for gate_voltages array
            voltage2 = float(gate_voltages[i + 1])  # Use 0-based indexing for gate_voltages array
            z = self._get_charge_sensor_data(voltage1, voltage2, gate1, gate2, barrier_voltages)
            all_z.append(z)  # z is now 2D, no need to index [:, :, 0]

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

        return Cgd

    def _generate_cbd_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-dot capacitance matrix with distance-based coupling."""
        cbd_config = config_ranges["Cbd"]
        distance_coupling = cbd_config["distance_coupling"]

        # Cbd is (n_dot, n_barrier) matrix
        Cbd = np.zeros((self.num_dots, self.num_barrier_voltages))

        for dot_i in range(self.num_dots):
            for barrier_j in range(self.num_barrier_voltages):
                # Calculate distance between dot and barrier
                # Barriers are typically between adjacent dots
                barrier_center = barrier_j + 0.5  # Barrier j is between dots j and j+1
                distance = int(abs(dot_i - barrier_center))
                distance = max(1, distance)  # Minimum distance is 1

                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                Cbd[dot_i, barrier_j] = self._sample_from_range(coupling_range, rng)

        return Cbd

    def _generate_cbg_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-gate capacitance matrix with distance-based coupling."""
        cbg_config = config_ranges["Cbg"]
        distance_coupling = cbg_config["distance_coupling"]
        num_gates = self.num_dots + 1  # plunger gates + sensor gate

        # Cbg is (n_barrier, n_gate) matrix
        Cbg = np.zeros((self.num_barrier_voltages, num_gates))

        for barrier_i in range(self.num_barrier_voltages):
            for gate_j in range(num_gates):
                if gate_j < self.num_dots:  # Plunger gates
                    # Calculate distance between barrier and gate
                    barrier_center = barrier_i + 0.5  # Barrier i is between dots i and i+1
                    distance = int(abs(barrier_center - gate_j))
                    distance = max(1, distance)  # Minimum distance is 1

                    coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                    Cbg[barrier_i, gate_j] = self._sample_from_range(coupling_range, rng)
                else:  # Sensor gate
                    # Use distance 2 coupling for sensor gate
                    coupling_range = self._get_coupling_by_distance(distance_coupling, 2)
                    Cbg[barrier_i, gate_j] = self._sample_from_range(coupling_range, rng)

        return Cbg
    
    def _generate_cbs_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-sensor capacitance matrix."""
        cbs_config = config_ranges["Cbs"]
        coupling_range = cbs_config["coupling"]

        # Cbs is (n_sensor, n_barrier) matrix - typically (1, n_barrier)
        Cbs = np.zeros((1, self.num_barrier_voltages))

        for barrier_j in range(self.num_barrier_voltages):
            Cbs[0, barrier_j] = self._sample_from_range(coupling_range, rng)

        return Cbs

    def _generate_cbb_matrix(self, config_ranges: dict, rng: np.random.Generator) -> np.ndarray:
        """Generate barrier-to-barrier capacitance matrix with distance-based coupling."""
        cbb_config = config_ranges["Cbb"]
        diagonal_val = cbb_config["diagonal"]
        distance_coupling = cbb_config["distance_coupling"]

        def fill_cbb(i: int, j: int) -> float:
            distance = abs(i - j)
            if distance == 0:
                return diagonal_val
            else:
                coupling_range = self._get_coupling_by_distance(distance_coupling, distance)
                return self._sample_from_range(coupling_range, rng)

        return self._create_symmetric_matrix(self.num_barrier_voltages, fill_cbb)

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

    def _generate_barrier_model_parameters(
        self, config_ranges: dict, rng: np.random.Generator
    ) -> dict:
        """Generate barrier voltage model parameters."""
        barrier_config = config_ranges["barrier_model"]

        tc_base = self._sample_from_range(barrier_config["tc_base"], rng)
        # Generate alpha array with one value per barrier
        alpha = [
            self._sample_from_range(barrier_config["alpha_per_barrier"], rng)
            for _ in range(self.num_barrier_voltages)
        ]

        return {"tc_base": tc_base, "alpha": alpha}
    
    def _generate_voltage_capacitance_parameters(
        self, config_ranges: dict, rng: np.random.Generator
    ) -> dict:
        """Generate voltage-dependent capacitance model parameters."""
        vc_config = config_ranges["voltage_capacitance_model"]

        return {
            "alpha": self._sample_from_range(vc_config["alpha"], rng),
            "beta": self._sample_from_range(vc_config["beta"], rng),
        }

    def _generate_variable_peak_width_parameters(
        self, config_ranges: dict, rng: np.random.Generator
    ) -> dict:
        """Generate variable peak width model parameters."""
        vpw_config = config_ranges["variable_peak_width_model"]

        return {
            "alpha": self._sample_from_range(vpw_config["alpha"], rng),
        }


    def _gen_random_qarray_params_no_barriers(self, rng: np.random.Generator, param_overrides: dict) -> dict:

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
            "variable_peak_width_model": model_config["variable_peak_width_model"],
            "T": model_config["T"],
            "coulomb_peak_width": model_config["coulomb_peak_width"],
        }

        # Generate all matrix components using helper methods
        Cdd = self._generate_cdd_matrix(config_ranges, rng)
        Cgd = self._generate_cgd_matrix(config_ranges, rng)
        Cds, Cgs = self._generate_sensor_capacitances(config_ranges, rng)
        noise_params = self._generate_noise_parameters(config_ranges, rng)
        latching_params = self._generate_latching_parameters(config_ranges, rng)
        variable_peak_width_params = self._generate_variable_peak_width_parameters(config_ranges, rng)

        # Assemble final model parameters
        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "white_noise_amplitude": noise_params["white_noise_amplitude"],
            "telegraph_noise_parameters": noise_params["telegraph_noise_parameters"],
            "latching_model_parameters": latching_params,
            "variable_peak_width_parameters": variable_peak_width_params,
            "T": self._sample_from_range(config_ranges["T"], rng),
            "coulomb_peak_width": self._sample_from_range(config_ranges["coulomb_peak_width"], rng),
            "algorithm": model_config["algorithm"],
            "implementation": model_config["implementation"],
            "max_charge_carriers": model_config["max_charge_carriers"],
            "optimal_VG_center": measurement_config["optimal_VG_center"],
        }

        self._apply_param_overrides(model_params, param_overrides)

        return model_params

    def _gen_random_qarray_params_with_barriers(self, rng: np.random.Generator, param_overrides: dict) -> dict:
        """
        Generate random parameters for the quantum device using distance-based coupling rules.

        Returns:
            dict: Complete model parameters for qarray simulator
        """

        # Extract configuration ranges
        model_config = self.config["simulator"]["model"]
        measurement_config = self.config["simulator"]["measurement"]
        latched_config = self.config["simulator"]["latched_model"]

        config_ranges = {
            "Cdd": model_config["Cdd"],
            "Cgd": model_config["Cgd"],
            "Cds": model_config["Cds"],
            "Cgs": model_config["Cgs"],
            "Cbd": model_config["Cbd"],
            "Cbg": model_config["Cbg"],
            "Cbs": model_config["Cbs"],
            "Cbb": model_config["Cbb"],
            "white_noise_amplitude": model_config["white_noise_amplitude"],
            "telegraph_noise_parameters": model_config["telegraph_noise_parameters"],
            "latching_model_parameters": model_config["latching_model_parameters"],
            "barrier_model": model_config["barrier_model"],
            "voltage_capacitance_model": model_config["voltage_capacitance_model"],
            "variable_peak_width_model": model_config["variable_peak_width_model"],
            "T": model_config["T"],
            "coulomb_peak_width": model_config["coulomb_peak_width"],
            "tc": model_config["tc"],
        }

        # Generate all matrix components using helper methods
        Cdd = self._generate_cdd_matrix(config_ranges, rng)
        Cgd = self._generate_cgd_matrix(config_ranges, rng)
        Cds, Cgs = self._generate_sensor_capacitances(config_ranges, rng)

        # Generate barrier matrices
        Cbd = self._generate_cbd_matrix(config_ranges, rng)
        Cbg = self._generate_cbg_matrix(config_ranges, rng)
        Cbs = self._generate_cbs_matrix(config_ranges, rng)
        Cbb = self._generate_cbb_matrix(config_ranges, rng)

        # Generate model parameters
        noise_params = self._generate_noise_parameters(config_ranges, rng)
        latching_params = self._generate_latching_parameters(config_ranges, rng)
        barrier_model_params = self._generate_barrier_model_parameters(config_ranges, rng)
        voltage_capacitance_params = self._generate_voltage_capacitance_parameters(
            config_ranges, rng
        )
        variable_peak_width_params = self._generate_variable_peak_width_parameters(
            config_ranges, rng
        )

        # Assemble final model parameters
        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "Cbd": Cbd,
            "Cbg": Cbg,
            "Cbs": Cbs,
            "Cbb": Cbb,
            "white_noise_amplitude": noise_params["white_noise_amplitude"],
            "telegraph_noise_parameters": noise_params["telegraph_noise_parameters"],
            "latching_model_parameters": latching_params,
            "barrier_model_parameters": barrier_model_params,
            "voltage_capacitance_parameters": voltage_capacitance_params,
            "variable_peak_width_parameters": variable_peak_width_params,
            "T": self._sample_from_range(config_ranges["T"], rng),
            "coulomb_peak_width": self._sample_from_range(config_ranges["coulomb_peak_width"], rng),
            "tc": self._sample_from_range(config_ranges["tc"], rng),
            "algorithm": model_config["algorithm"],
            "implementation": model_config["implementation"],
            "max_charge_carriers": model_config["max_charge_carriers"],
            "optimal_VG_center": measurement_config["optimal_VG_center"],
            "use_sparse": latched_config["use_sparse"],
            "num_charge_states": latched_config["num_charge_states"],
            "charge_state_batch_size": latched_config["charge_state_batch_size"],
        }

        self._apply_param_overrides(model_params, param_overrides)

        return model_params

    
    def _apply_perfect_virtualisation(self):
        Cdd = np.zeros((self.num_dots, self.num_dots))
        Cgd = np.eye(self.num_dots + 1)[:-1, :]
        Cds = self.model.Cds.copy()
        Cgs = self.model.Cgs.copy()

        if self.use_barriers:
            raise NotImplementedError("Updating matrices in qarray-latched not yet implemented.")
            Cbd = np.zeros((self.num_dots, self.num_dots - 1))
            Cbg = np.zeros((self.num_dots - 1, self.num_dots + 1))
            Cbb = np.eye(self.num_dots - 1)
            Cbs = self.model.Cbs

            self.model.update_capacitance_matrices(Cdd, Cgd, Cds, Cgs, Cbd, Cbg, Cbs, Cbb)

        else:
            self.model.update_capacitance_matrices(Cdd, Cgd, Cds, Cgs)

    
    def _load_model(self, param_overrides: dict = None):

        if param_overrides is None:
            param_overrides = {}

        self._validate_param_overrides(param_overrides)

        if not self.use_barriers:
            return self._load_model_no_barriers(param_overrides)
        
        return self._load_model_with_barriers(param_overrides)


    def _load_model_no_barriers(self, param_overrides: dict, rng: np.random.Generator = None):
        """
        Load the model from the config file.
        Also not using voltage-dependent capacitance model
        """
        if rng is None:
            rng = np.random.default_rng()

        model_params = self._gen_random_qarray_params_no_barriers(rng=rng, param_overrides=param_overrides)

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

        # Initialize peak width model (same logic as barriers version)
        if self.vary_peak_width:
            vpw_params = model_params["variable_peak_width_parameters"]
            # Use override if provided, otherwise use generated parameter
            alpha = self.peak_width_alpha if self.peak_width_alpha != 0.01 else vpw_params["alpha"]
            self.peak_width_model = VaryPeakWidth(peak_width_0=model.coulomb_peak_width, alpha=alpha)
        else:
            self.peak_width_model = None

        return model

    def _load_model_with_barriers(self, param_overrides: dict, rng: np.random.Generator = None):
        """
        Load the TunnelCoupledChargeSensed model from the config file.
        """
        if rng is None:
            rng = np.random.default_rng()

        model_params = self._gen_random_qarray_params_with_barriers(rng=rng, param_overrides=param_overrides)

        # Extract noise models
        white_noise = WhiteNoise(amplitude=model_params["white_noise_amplitude"])
        telegraph_noise = TelegraphNoise(**model_params["telegraph_noise_parameters"])
        noise_model = white_noise + telegraph_noise

        # Extract latching model
        latching_params = model_params["latching_model_parameters"]
        latching_model = (
            LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"})
            if latching_params["Exists"]
            else None
        )

        # Extract capacitance matrices
        Cdd = model_params["Cdd"]
        Cgd = model_params["Cgd"]
        Cds = model_params["Cds"]
        Cgs = model_params["Cgs"]
        Cbd = model_params["Cbd"]
        Cbg = model_params["Cbg"]
        Cbs = model_params["Cbs"]
        Cbb = model_params["Cbb"]

        # Create barrier voltage model
        barrier_params = model_params["barrier_model_parameters"]
        barrier_model = BarrierVoltageModel(
            n_barrier=self.num_barrier_voltages,
            n_dot=self.num_dots,
            tc_base=barrier_params["tc_base"],
            alpha=barrier_params["alpha"],
        )

        self.barrier_alpha = barrier_params["alpha"]
        self.barrier_tc_base = barrier_params["tc_base"]

        # Create TunnelCoupledChargeSensed model
        model = TunnelCoupledChargeSensed(
            Cdd=Cdd,
            Cgd=Cgd,
            Cds=Cds,
            Cgs=Cgs,
            Cbd=Cbd,
            Cbg=Cbg,
            Cbs=Cbs,
            Cbb=Cbb,
            barrier_model=barrier_model,
            coulomb_peak_width=model_params["coulomb_peak_width"],
            T=model_params["T"],
            max_charge_carriers=model_params["max_charge_carriers"],
            tc=model_params["tc"],
            noise_model=noise_model,
            latching_model=latching_model,
            voltage_capacitance_model=None,  # Will be set below
            use_sparse=model_params["use_sparse"],
            num_charge_states=model_params["num_charge_states"],
            charge_state_batch_size=model_params["charge_state_batch_size"],
        )

        # Create and set voltage-dependent capacitance model
        vc_params = model_params["voltage_capacitance_parameters"]

        capacitance_model_type = self.config["simulator"]["voltage_capacitance_model"]["type"]
        if capacitance_model_type is not None:
            if capacitance_model_type == "linear":
                linear_voltage_capacitance_model = create_linear_capacitance_model(
                    cdd_0=jnp.array(model.cdd_full),
                    cgd_0=jnp.array(model.cgd_full),
                    alpha=vc_params["alpha"],
                    beta=vc_params["beta"],
                )
                model.voltage_capacitance_model = linear_voltage_capacitance_model
            else:
                raise ValueError(f"Capacitance model type '{capacitance_model_type}' does not exist")

        # Initialize peak width model
        if self.vary_peak_width:
            vpw_params = model_params["variable_peak_width_parameters"]
            # Use override if provided, otherwise use generated parameter
            alpha = self.peak_width_alpha if self.peak_width_alpha != 0.01 else vpw_params["alpha"]
            self.peak_width_model = VaryPeakWidth(peak_width_0=model.coulomb_peak_width, alpha=alpha)
        else:
            self.peak_width_model = None

        return model


    def _update_virtual_gate_matrix(self, cgd_estimate):
        vgm = -np.linalg.pinv(np.linalg.inv(self.model.Cdd) @ cgd_estimate)

        self.model.gate_voltage_composer.virtual_gate_matrix = vgm


    def _render_frame(self, image, path="quantum_dot_plot_pv"):
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

    
    def _validate_param_overrides(self, param_overrides: dict) -> None:
        """
        Validate that parameter override keys exist in the configuration.
        
        Args:
            param_overrides (dict): Dictionary of parameter overrides
            
        Raises:
            ValueError: If any override key doesn't exist in config
        """
        if not param_overrides:
            return
            
        model_config = self.config['simulator']['model']
        valid_params = set(model_config.keys())
        
        # Add nested parameter paths for complex structures
        nested_params = {
            'telegraph_noise_parameters.p01',
            'telegraph_noise_parameters.p10_factor',
            'telegraph_noise_parameters.amplitude',
            'latching_model_parameters.p_leads',
            'latching_model_parameters.p_inter',
            'barrier_model_parameters.tc_base',
            'barrier_model_parameters.alpha',
            'voltage_capacitance_parameters.alpha',
            'voltage_capacitance_parameters.beta',
            'variable_peak_width_model.alpha',
            'voltage_capacitance_model.type'
        }
        valid_params.update(nested_params)
        
        for param_key in param_overrides.keys():
            if param_key not in valid_params:
                available_params = sorted(valid_params)
                raise ValueError(
                    f"Invalid parameter override '{param_key}'. "
                    f"Available parameters: {available_params}"
                )

    def _apply_param_overrides(self, model_params: dict, param_overrides: dict) -> None:
        """
        Apply parameter overrides to the generated model parameters.

        Args:
            model_params (dict): Generated model parameters to modify
            param_overrides (dict): Dictionary of parameter overrides
        """
        for param_key, override_value in param_overrides.items():
            if '.' in param_key:
                # Handle nested parameters like 'telegraph_noise_parameters.p01'
                parent_key, child_key = param_key.split('.', 1)
                if parent_key in model_params and isinstance(model_params[parent_key], dict):
                    # Special handling for telegraph noise p10_factor conversion
                    if parent_key == 'telegraph_noise_parameters' and child_key == 'p10_factor':
                        # Convert p10_factor to p10 (p10 = p10_factor * p01)
                        p01 = model_params[parent_key].get('p01', 0.005)  # Use existing p01 or default
                        model_params[parent_key]['p10'] = override_value * p01
                    else:
                        model_params[parent_key][child_key] = override_value
            else:
                # Handle top-level parameters
                model_params[param_key] = override_value
                

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
    
    def calculate_barrier_ground_truth(self, vg_plunger):
        """
        Calculate barrier ground truth voltages given current plunger voltages.

        Args:
            vg_plunger: Current plunger gate voltages (n_dots,)

        Returns:
            vb_optimal: Optimal barrier voltages accounting for gate contributions (n_barrier,)
        """
        # Calculate base barrier voltages needed for optimal tunnel coupling
        # From barrier model: tc = tc_base * exp(-alpha * Vb)
        # Solving for Vb: Vb = -ln(tc / tc_base) / alpha
        tc_ratio = float(self.optimal_tc) / self.barrier_tc_base

        # Calculate barrier voltage for each barrier using its own alpha value
        vb_base = np.array([-np.log(tc_ratio) / alpha_i for alpha_i in self.barrier_alpha])

        # Account for gate contributions using only plunger gates (ignore sensor gate column)
        # Cbg is (n_barrier, n_gate) where n_gate = n_dots + 1
        # We only use the first n_dots columns (plunger gates)
        gate_contribution = self.model.Cbg[:, :self.num_dots] @ vg_plunger

        vb_optimal = vb_base - gate_contribution

        return vb_optimal


    def calculate_ground_truth(self, debug=False):
        """
        Get the ground truth for the quantum dot array.
        """
        if not self.use_barriers:
            # calculates virtualised optimal gate voltages
                
            vg_optimal_physical = self.model.optimal_Vg(self.optimal_VG_center)
            perfect_virtual_matrix = self.model.compute_optimal_virtual_gate_matrix()

            # removed - self.model.virtual_gate_origin
            vg_optimal_virtual = np.linalg.inv(perfect_virtual_matrix) @ (vg_optimal_physical)

            vg_optimal_virtual = vg_optimal_virtual[:-1]

            return vg_optimal_virtual, None, None

    
        ndot = self.num_dots
        nbarrier = self.num_barrier_voltages
        cgd = self.model.cgd_full
        # cdd = self.model.cdd_full  # Currently unused
        cdd_inv = self.model.cdd_inv_full

        if debug:
            print(f"Cgd: {cgd}")

        numsensor = 1

        # Assumes Vb = 0
        vg_optimal_physical = self.model.optimal_Vg(self.optimal_VG_center)

        # Assumes perfectly virtualised barriers (this is not the case but since ground truth for barriers is approximate we leave this)
        tc_ratio = float(self.optimal_tc) / self.barrier_tc_base

        if debug:
            print(tc_ratio)

        # Calculate barrier voltage for each barrier using its own alpha value
        vb_optimal = np.array([-np.log(tc_ratio) / alpha_i for alpha_i in self.barrier_alpha])

        # calculate change in dot potential due to barriers
        dot_potential = cgd[: ndot + numsensor, -nbarrier:] @ vb_optimal

        delta_vg = np.linalg.inv(cgd[: ndot + numsensor, : ndot + numsensor]) @ dot_potential

        vg_optimal_physical -= delta_vg
        sensor_optimal_physical = vg_optimal_physical[-numsensor:]
        plunger_optimal_physical = vg_optimal_physical[:-numsensor]

        # note ignoring virtual gate origins here
        plunger_optimal_virtual = (
            -cdd_inv[:ndot, :ndot] @ cgd[:ndot, :ndot] @ plunger_optimal_physical
        )

        if debug:
            print(f"Original Optimal VG: {self.model.optimal_Vg(self.optimal_VG_center)}")
            print(f"Plunger optimal virtual: {plunger_optimal_virtual}")
            print(f"Sensor optimal physical: {sensor_optimal_physical}")
            print(f"Plunger optimal physical: {plunger_optimal_physical}")
            print(f"VB optimal: {vb_optimal}")

        return plunger_optimal_virtual, vb_optimal, sensor_optimal_physical


if __name__ == "__main__":
    num_dots = 4

    use_barriers = True

    experiment = QarrayBaseClass(
        num_dots=num_dots,
        use_barriers=use_barriers,
        obs_voltage_min=-3,
        obs_voltage_max=1.5,
        vary_peak_width=False,
        obs_image_size=256,
    )
    import time

    #experiment._apply_perfect_virtualisation()

    start = time.time()

    # os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force CPU-only execution
    # os.environ['JAX_PLATFORMS'] = 'cpu'  # Alternative JAX CPU-only setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    vg, vb, _ = experiment.calculate_ground_truth()

    print("Ground truth:")
    print(vg)
    print(vb)

    if not use_barriers:
        image = experiment._get_obs([0] * num_dots)["image"][:, :, 0]
    else:
        image = experiment._get_obs([0] * num_dots, [0] * (num_dots - 1))["image"][:, :, 0]
    print(time.time() - start)

    start = time.time()

    voltage = - 2.0

    if not use_barriers:
        image = experiment._get_obs(vg)["image"][:, :, 0]
    else:
        #vg = np.array(vg)
        #vg += np.random.normal(0, .1, size=vg.shape)
        #vb = np.array(vb)
        #vb += np.random.normal(0, 1, size=vb.shape)
        #vb = np.ones_like(vb) * 10.

        image = experiment._get_obs(vg, vb)["image"][:, :, 0]
    print(time.time() - start)

    experiment._render_frame(image)
