
"""
A class that represents a charge sensed dot array. The class has methods to compute the ground state of the dot array
and the charge sensor output for both open and closed dot arrays.
"""

from dataclasses import dataclass

import numpy as np
from qarray.functions import compute_optimal_virtual_gate_matrix, optimal_Vg

from qarray_latched.DotArrays.GateVoltageComposer import GateVoltageComposer

from qarray_latched.DotArrays._helper_functions import check_algorithm_and_implementation, \
    check_and_warn_user, lorentzian, convert_to_maxwell, _convert_to_maxwell_with_sensor, _convert_to_maxwell_with_barriers_and_sensor
from qarray.latching_models import LatchingBaseModel
from qarray.noise_models import BaseNoiseModel
from qarray.python_implementations.helper_functions import free_energy
from qarray.qarray_types import CddNonMaxwell, CgdNonMaxwell, VectorList, CdsNonMaxwell, CgsNonMaxwell, Vector, \
    PositiveValuedMatrix

#from qarray_latched.DotArrays.ground_state_tunnel_coupled import _ground_state_open
#from qarray_latched.DotArrays.full_physics_informed_tunneling import _ground_state_open as _ground_state_open_full_physics_informed
from qarray_latched.DotArrays.ground_state import _ground_state_open
from qarray_latched.DotArrays.voltage_dependent_capacitance import VoltageDependendentCapacitanceModel
from qarray_latched.DotArrays.barrier_voltage_model import BarrierVoltageModel

@dataclass
class TunnelCoupledChargeSensed:
    """
    A class that represents a charge sensed dot array. The class has methods to compute the ground state of the dot array
    and the charge sensor output for both open and closed dot arrays.

    The class has the following attributes:

    - Cdd: an (n_dot, n_dot) array of the capacitive coupling between dots
    - Cgd: an (n_dot, n_gate) array of the capacitive coupling between gates and dots
    - Cds: an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    - Cgs: an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    - algorithm: the algorithm to use to compute the ground state
    - implementation: the implementation to use to compute the ground state
    - threshold: the threshold to use if the threshold algorithm is used
    - max_charge_carriers: the maximum number of charge carriers to use if the brute force algorithm is used
    - polish: a bool specifying whether to polish the result of the ground state computation by the default or thresholded algorithm

    - coulomb_peak_width: the width of the lorentzian peaks

    - noise_model: the noise model to use to add noise to the charge sensor output
    - voltage_capacitance_model: optional model for voltage-dependent capacitances


    """

    Cdd: CddNonMaxwell  # an (n_dot, n_dot) array of the capacitive coupling between dots
    Cgd: CgdNonMaxwell  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots

    Cds: CdsNonMaxwell  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    Cgs: CgsNonMaxwell  # an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    algorithm: str | None = 'brute-force'  # which algorithm to use
    implementation: str | None = 'jax'  # which implementation of the algorithm to use
    use_sparse: bool | None = False
    num_charge_states: int | None = None #if none uses all charge states
    charge_state_batch_size: int | None = None #batches if total charge state size is too big to hold in RAM
    tunneling_convention: str = "fermionic_negative"  # Options: "simple", "fermionic_positive", "fermionic_negative", "dynamiqs_fock", "dynamiqs_optimized"
    

    threshold: float | str = 1.  # if the threshold algorithm is used the user needs to pass the threshold
    max_charge_carriers: int | None = None  # if the brute force algorithm is used the user needs to pass the maximum number of charge carriers
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation
    max_charge_carriers: int | None = None  # need if using a brute_force algorithm
    batch_size: int | None = None  # needed if using jax implementation
    charge_carrier: str = 'h'

    T: float | int = 0.  # the temperature of the system in mK
    n_peak: int = 5

    coulomb_peak_width: float = 0.1  # the width of the lorentzian peaks
    noise_model: BaseNoiseModel | None = None
    latching_model: LatchingBaseModel | None = None
    tc: float = 0
    voltage_capacitance_model: VoltageDependendentCapacitanceModel | None = None

    constant_charge_shift : int | None = None

    # Barrier voltage parameters
    Cbd: CddNonMaxwell | None = None  # barrier-to-dot capacitance matrix
    Cbg: CgdNonMaxwell | None = None  # barrier-to-gate capacitance matrix
    Cbs: CdsNonMaxwell | None = None  # barrier-to-sensor capacitance matrix 
    Cbb: CddNonMaxwell | None = None  # barrier-to-barrier capacitance matrix
    barrier_model: BarrierVoltageModel | None = None

    def update_capacitance_matrices(self, Cdd: CddNonMaxwell, Cgd: CgdNonMaxwell, Cds: CdsNonMaxwell,
                                    Cgs: CgsNonMaxwell, Cbd=None, Cbg=None, Cbs=None, Cbb=None):

        self.Cdd = PositiveValuedMatrix(Cdd)
        self.Cgd = PositiveValuedMatrix(Cgd)
        self.Cds = PositiveValuedMatrix(Cds)
        self.Cgs = PositiveValuedMatrix(Cgs)

        # Handle barrier matrices
        self.Cbd = PositiveValuedMatrix(Cbd) if Cbd is not None else None
        self.Cbg = PositiveValuedMatrix(Cbg) if Cbg is not None else None
        self.Cbs = PositiveValuedMatrix(Cbs) if Cbs is not None else None
        self.Cbb = PositiveValuedMatrix(Cbb) if Cbb is not None else None

        self.n_dot = self.Cdd.shape[0]
        self.n_sensor = self.Cds.shape[0]
        self.n_gate = self.Cgd.shape[1]
        self.n_barrier = self.Cbd.shape[1] if self.Cbd is not None else 0
        
        
        self._assert_shape()

        # Choose Maxwell conversion based on whether barriers are present
        if any([self.Cbd is not None, self.Cbg is not None, self.Cbs is not None, self.Cbb is not None]):
            # Use barrier-aware conversion for full system
            self.cdd_full, self.cdd_inv_full, self.cgd_full = _convert_to_maxwell_with_barriers_and_sensor(
                self.Cdd, self.Cgd, self.Cds, self.Cgs,
                self.Cbd, self.Cbg, self.Cbs, self.Cbb
            )
        else:
            # Use standard sensor conversion
            self.cdd_full, self.cdd_inv_full, self.cgd_full = _convert_to_maxwell_with_sensor(
                self.Cdd, self.Cgd, self.Cds, self.Cgs
            )
        
        # Extract dot-only matrices for backward compatibility
        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.Cdd, self.Cgd)
        
        # Extract barrier matrices from full system if barriers exist
        if self.n_barrier > 0:
            # Barrier matrices are extracted from the full system
            # For now, set to None - this will be implemented in Tasks 2-4
            self.cbd = None
            self.cbg = None  
            self.cbs = None
            self.cbb = None

        self.cgs = self.Cgs
        self.cds = self.Cds


    def __post_init__(self):

        # converting to the non-maxwellian capacitance matrices to their respective type. This
        # is done to ensure that the capacitance matrices are of the correct type and the values are correct. Aka
        # the capacitance matrices are positive and the diagonal elements are zero.

        self.update_capacitance_matrices(self.Cdd, self.Cgd, self.Cds, self.Cgs, 
                                         self.Cbd, self.Cbg, self.Cbs, self.Cbb)

        # type casting the temperature to a float
        self.T = float(self.T)

        if self.algorithm == 'threshold':
            assert self.threshold is not None, 'The threshold must be specified when using the thresholded algorithm'

        if self.algorithm == 'brute_force':
            assert self.max_charge_carriers is not None, 'The maximum number of charge carriers must be specified'

        if self.noise_model is None:
            # this is the default noise model adds no noise
            self.noise_model = BaseNoiseModel()

        if self.latching_model is None:
            self.latching_model = LatchingBaseModel()

        if self.algorithm in ['thresholded', 'default']:
            check_and_warn_user(self)

        self.gate_voltage_composer = GateVoltageComposer(n_gate=self.n_gate, n_dot=self.n_dot, n_sensor=self.n_sensor)
        # Extract gate-only submatrix for virtual gate calculation (exclude barrier columns)
        cgd_gates_only = self.cgd_full[:, :self.n_gate]
        virtual_gate_matrix = -np.linalg.pinv(self.cdd_inv_full @ cgd_gates_only)

        # Apply charge carrier convention
        if self.charge_carrier == 'electrons':
            virtual_gate_matrix = -virtual_gate_matrix

        self.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix
        self.gate_voltage_composer.virtual_gate_origin = np.zeros(self.n_gate)

        if self.voltage_capacitance_model is not None:
            self.voltage_capacitance_model = self.voltage_capacitance_model
            
        # Validate and initialize barrier model
        if self.barrier_model is not None:
            self.barrier_model.validate_dimensions(self.n_gate, self.n_dot, self.n_sensor)

    def do1d_open(self, gate: int | str, min: float, max: float, points: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate in the open configuration

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param points: the number of res to sweep the gate over

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot) in the open configuration
        """

        vg = self.gate_voltage_composer.do1d(gate, min, max, points)
        return self.charge_sensor_open(vg)

    def gui(self, port=9001, run=True, print_compute_time=True, initial_dac_values=None, initial_virtual_gate_matrix=None):
        """
        A function to open the GUI for the ChargeSensedDotArray class
        """
        from qarray.gui.gui_charge_sensor import run_gui_charge_sensor

        run_gui_charge_sensor(self, port = port, run = run, print_compute_time = print_compute_time, initial_dac_values = initial_dac_values, initial_virtual_gate_matrix = initial_virtual_gate_matrix)

    def compute_optimal_virtual_gate_matrix(self):
        """
        Computes the optimal virtual gate matrix for the dot array and sets it as the virtual gate matrix
        in the gate voltage composer.

        The virtual gate matrix is computed as the pseudo inverse of the dot to dot capacitance matrix times the dot to gate capacitance matrix.

        returns np.ndarray: the virtual gate matrix
        """
        # Extract gate-only submatrix for virtual gate calculation (exclude barrier columns)
        cgd_gates_only = self.cgd_full[:, :self.n_gate]
        virtual_gate_matrix = compute_optimal_virtual_gate_matrix(self.cdd_inv_full, cgd_gates_only)

        # Apply charge carrier convention
        if self.charge_carrier == 'electrons':
            virtual_gate_matrix = -virtual_gate_matrix

        self.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix
        return virtual_gate_matrix

    def compute_optimal_sensor_virtual_gate_matrix(self):
        assert self.n_sensor == 1, 'currently not implemented for more than one sensor'
        # Extract gate-only submatrix for virtual gate calculation (exclude barrier columns)
        cgd_gates_only = self.cgd_full[:, :self.n_gate]
        virtual_gate_matrix = compute_optimal_virtual_gate_matrix(self.cdd_inv_full, cgd_gates_only)

        device_virtual_gates = virtual_gate_matrix[:-1, :-1]
        sensor_gates = virtual_gate_matrix[-1, :-1]

        inv = np.linalg.inv(device_virtual_gates)
        sensor_virtual_matrix = np.eye(self.n_gate)
        sensor_virtual_matrix[-1, :-1] = inv @ sensor_gates

        # Apply charge carrier convention
        if self.charge_carrier == 'electrons':
            sensor_virtual_matrix = -sensor_virtual_matrix

        self.gate_voltage_composer.virtual_gate_matrix = sensor_virtual_matrix
        return sensor_virtual_matrix


    def do1d_closed(self, gate: int | str, min: float, max: float, points: int, n_charge: int) -> np.ndarray:
        """
        Performs a 1D sweep of the dot array with the gate in the closed configuration

        :param gate: the gate to sweep
        :param min: the minimum value of the gate to sweep
        :param max: the maximum value of the gate to sweep
        :param points: the number of res to sweep the gate over

        returns the ground state of the dot array which is a np.ndarray of shape (res, n_dot) in the closed configuration
        """
        vg = self.gate_voltage_composer.do1d(gate, min, max, points)
        return self.charge_sensor_closed(vg, n_charge)

    def do2d_open(self, x_gate: int | str, x_min: float, x_max: float, x_points: int,
                  y_gate: int | str, y_min: float, y_max: float, y_points: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate in the open configuration

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_points: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_points: the number of res to sweep

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot) in the open
        configuration
        """

        vg = self.gate_voltage_composer.do2d(x_gate, x_min, x_max, x_points, y_gate, y_min, y_max, y_points)
        return self.charge_sensor_open(vg)

    def do2d_closed(self, x_gate: int | str, x_min: float, x_max: float, x_points: int,
                    y_gate: int | str, y_min: float, y_max: float, y_points: int, n_charge: int) -> np.ndarray:
        """
        Performs a 2D sweep of the dot array with the gates x_gate and y_gate in the open configuration

        :param x_gate: the gate to sweep in the x direction
        :param x_min: the minimum value of the gate to sweep
        :param x_max: the maximum value of the gate to sweep
        :param x_points: the number of res to sweep the gate over
        :param y_gate: the gate to sweep in the y direction
        :param y_min: the minimum value of the gate to sweep
        :param y_max: the maximum value of the gate to sweep
        :param y_points: the number of res to sweep

        returns the ground state of the dot array which is a np.ndarray of shape (x_res, y_res, n_dot)
        in the closed configuration
        """
        vg = self.gate_voltage_composer.do2d(x_gate, x_min, x_max, x_points, y_gate, y_min, y_max, y_points)
        return self.charge_sensor_closed(vg)


    def ground_state_open(self, vg: VectorList | np.ndarray, vb: VectorList | np.ndarray | None = None) -> np.ndarray:
        """
        Computes the ground state for an open dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :return: (..., n_dot) array of ground state charges
        """
        return _ground_state_open(self, vg, vb)

    def charge_sensor_open(self, vg: VectorList | np.ndarray, vb: VectorList | np.ndarray | None = None) -> np.ndarray:
        """
        Computes the charge sensor output for an open dot array.

        :param vg: (..., n_gate) array of gate voltages to compute the charge sensor output for
        :param vb: (..., n_barrier) array of barrier voltages (optional, requires barrier_model)
        :return: (..., n_sensor) array of the charge sensor output

        Noise is added according to the noise model passed to the ChargeSensedDotArray class.
        Barrier effects included when vb is provided and barrier_model exists.
        """

        n_open = self.ground_state_open(vg, vb)

        # Create extended voltage vector [vg, vb]
        if vb is not None and self.barrier_model is not None:
            v_extended = np.concatenate([vg, vb], axis=-1)  # (..., n_gate + n_barrier)
        else:

            v_extended = vg
        
        # Computing the continuous minimum charge state (open) - now includes all effects
        N_cont = np.einsum('ij, ...j', self.cgd_full, v_extended)
        if self.constant_charge_shift is not None:
            n0 = np.array([self.constant_charge_shift]*self.n_dot + [0]*self.n_sensor)
            N_cont = N_cont + n0
        #N_cont[:] += np.array([10]*5)
        #print(N_cont.shape)

        # computing the discrete state on the charge sensor
        # Extract only sensor portion from extended system [dots + sensors + barriers]
        N_sensor = np.round(N_cont[..., self.n_dot:self.n_dot + self.n_sensor])

        # computing the noise to be added to the charge sensor potential before it is used in as the input to the lorentzian
        input_noise = self.noise_model.sample_input_noise(N_sensor.shape)

        F = np.zeros(shape=(2 * self.n_peak + 1, *N_sensor.shape))
        for sensor in range(self.n_sensor):
            for i, n in enumerate(range(-self.n_peak, self.n_peak + 1)):
                perturbed_N_sensor = N_sensor.copy()
                perturbed_N_sensor[..., sensor] = perturbed_N_sensor[..., sensor] + n
                if self.barrier_model is not None:
                    # With corrected physics: only dots and sensors have charges (barriers are voltage sources)
                    N_full = np.concatenate([n_open, perturbed_N_sensor + input_noise], axis=-1)
                    
                else:
                    # Standard system: [dots, sensors]
                    N_full = np.concatenate([n_open, perturbed_N_sensor + input_noise], axis=-1)
                
                def free_energy_shifted(cdd_inv, cgd, vg, n, n0_shift):
                    v_dash = np.einsum('ij, ...j', cgd, vg)
                    if n0_shift is not None:
                        v_dash = v_dash + n0_shift
                    F = np.einsum('...i, ij, ...j', n - v_dash, cdd_inv, n - v_dash)
                    return F
                n0_shift = np.array([self.constant_charge_shift]*self.n_dot + [0]*self.n_sensor) if self.constant_charge_shift is not None else None
                F[i, ..., sensor] = free_energy_shifted(self.cdd_inv_full, self.cgd_full, v_extended, N_full, n0_shift)

        signal = lorentzian(np.diff(F, axis=0), 0, self.coulomb_peak_width).sum(axis=0)
        output_noise = self.noise_model.sample_output_noise(N_sensor.shape)
        return signal + output_noise, n_open

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: int) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charge)

    def charge_sensor_closed(self, vg: VectorList | np.ndarray, n_charge) -> np.ndarray:
        """
        Computes the charge sensor output for a closed dot array.

        :param vg: (..., n_gate) array of dot voltages to compute the charge sensor output for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_sensor) array of the charge sensor output

        Noise is added according to the noise model passed to the ChargeSensedDotArray class.
        """
        n_closed = self.ground_state_closed(vg, n_charge)

        # For consistency, define v_extended (closed systems typically don't use barriers, so just use vg)
        v_extended = vg
        
        # computing the continuous minimum charge state (open)
        N_cont = np.einsum('ij, ...j', self.cgd_full, v_extended)

        # computing the discrete state on the charge sensor
        N_sensor = np.round(N_cont[..., self.n_dot:self.n_dot + self.n_sensor])

        # computing the noise to be added to the charge sensor potential before it is used in as the input to the lorentzian
        input_noise = self.noise_model.sample_input_noise(N_sensor.shape)

        F = np.zeros(shape=(2 * self.n_peak + 1, *N_sensor.shape))
        for sensor in range(self.n_sensor):
            for i, n in enumerate(range(-self.n_peak, self.n_peak + 1)):
                perturbed_N_sensor = N_sensor.copy()
                perturbed_N_sensor[..., sensor] = perturbed_N_sensor[..., sensor] + n
                N_full = np.concatenate([n_closed, perturbed_N_sensor + input_noise], axis=-1)
                F[i, ..., sensor] = free_energy(self.cdd_inv_full, self.cgd_full, v_extended, N_full)

        signal = lorentzian(np.diff(F, axis=0), 0, self.coulomb_peak_width).sum(axis=0)
        output_noise = self.noise_model.sample_output_noise(N_sensor.shape)

        return signal + output_noise, n_closed

    def _assert_shape(self):
        """
        A function to assert the shape of the capacitance matrices.
        """

        # checking the shape of the cgd matrix
        assert self.Cgd.shape[0] == self.n_dot, f'Cgd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'
        assert self.Cgd.shape[1] == self.n_gate, f'Cdd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'

        # checking the shape of the cds matrix
        assert self.Cds.shape[0] == self.n_sensor, 'Cds must be of shape (n_sensor, n_dot)'
        assert self.Cds.shape[1] == self.n_dot, 'Cds must be of shape (n_sensor, n_dot)'

        # checking the shape of the cgs matrix
        assert self.Cgs.shape[0] == self.n_sensor, 'Cgs must be of shape (n_sensor, n_gate)'
        assert self.Cgs.shape[1] == self.n_gate, 'Cgs must be of shape (n_sensor, n_gate)'

    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_dot + n_sensor,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        n_charges = Vector(n_charges)
        assert n_charges.shape == (
            self.n_dot + self.n_sensor,), 'The n_charge vector must be of shape (n_dot + n_sensor)'
        
        # With corrected physics: cdd_inv_full and cgd_full already contain only dots+sensors
        # Barriers are voltage sources only, so we only need to extract gate columns from cgd
        cgd_gates_only = self.cgd_full[:, :self.n_gate]  # Extract gate columns only

        def optimal_Vg_cholesky(cdd_inv, cgd, n_charges, rcond=1e-3, constant_charge_shift=None):
            R = np.linalg.cholesky(cdd_inv).T
            M = np.linalg.pinv(R @ cgd, rcond=rcond) @ R

            n_eff = np.array(n_charges)
            if constant_charge_shift is not None:
                n0 = np.array([constant_charge_shift]*self.n_dot + [0]*self.n_sensor)
                n_eff = n_eff - n0

            return np.einsum('ij, ...j', M, n_eff)

        return optimal_Vg_cholesky(cdd_inv=self.cdd_inv_full, cgd=cgd_gates_only, n_charges=n_charges, rcond=rcond, constant_charge_shift=self.constant_charge_shift)
