"""
Physics-based objective function for quantum dot array tuning.

The objective measures the sum of squared deviations from locally optimal voltages,
where each voltage's optimal value depends on all other voltages through the
capacitance coupling model:

- Plunger optimal: from charge model n = cgd @ V, solve for V_i given target charge
- Barrier optimal: from tunnel coupling model tc = tc_base * exp(-alpha * V_eff)

The optimum is the fixed point where all local constraints are simultaneously satisfied.

Optional features:
- Cap: returns constant value when any voltage is far from its local optimal
- Noise: adds Gaussian noise to simulate measurement uncertainty
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ObjectiveConfig:
    """
    Configuration for objective function behavior.

    Attributes:
        noise_std: Standard deviation of Gaussian noise to add (default 0.0 = no noise)
        noise_seed: Random seed for reproducible noise (default None = random)
    """
    noise_std: float = 0.01
    noise_seed: Optional[int] = 42


@dataclass
class PhysicsParams:
    """
    Cached physics parameters extracted from environment.

    All matrices are extracted and reshaped at initialization to avoid
    repeated indexing during objective evaluation.

    Physics:
    - Charge on dots: n = cgd @ V (energy minimum when induced charge = target)
    - Tunnel coupling: tc = tc_base * exp(-alpha * V_eff) where V_eff depends on barriers and gates

    Virtual gate transformation:
    - The RL agent operates in virtual voltage space, not physical space
    - Optimal voltages must be transformed: V_virtual = VGM^-1 @ (V_physical - offset)
    """
    n_dots: int
    n_barriers: int

    # Target charge configuration (n_dots,)
    n_target: np.ndarray

    # Capacitance matrices for charge model (n = cgd @ V)
    cgd_plungers: np.ndarray      # (n_dots, n_dots) - effect of plungers on dot charge
    cgd_barriers: np.ndarray      # (n_dots, n_barriers) - effect of barriers on dot charge
    cgd_plunger_diag: np.ndarray  # (n_dots,) - diagonal of cgd_plungers for self-coupling

    # Barrier coupling matrices for tunnel coupling model
    Cbg: np.ndarray               # (n_barriers, n_dots+1) - all gates including sensor
    alpha: np.ndarray             # (n_barriers,) per-barrier alpha
    log_tc_ratio: float           # -ln(tc_target / tc_base)

    # Virtual gate matrix transformation (for matching env ground truth)
    vgm_inv: np.ndarray           # (n_dots+1, n_dots+1) - inverse of virtual gate matrix
    vg_offset: np.ndarray         # (n_dots+1,) - virtual gate origin offset
    optimal_vg_physical: np.ndarray  # (n_dots+1,) - optimal physical plunger voltages


def _extract_physics_params(env, eps: float = 1e-10) -> PhysicsParams:
    """
    Extract physics parameters from environment.

    Args:
        env: QuantumDeviceEnv with use_barriers=True
        eps: Minimum threshold for validation

    Returns:
        PhysicsParams dataclass

    Raises:
        ValueError: If parameters are invalid or barriers not enabled
    """
    if not env.use_barriers:
        raise ValueError("Physical objective requires use_barriers=True")

    array = env.array
    model = array.model

    n_dots = array.num_dots
    n_barriers = array.num_barrier_voltages
    n_gates = n_dots + 1  # plunger gates + sensor gate

    # Validate barrier parameters exist
    if not hasattr(array, 'barrier_alpha') or array.barrier_alpha is None:
        raise ValueError("Environment missing barrier_alpha parameter")
    if not hasattr(array, 'barrier_tc_base') or array.barrier_tc_base is None:
        raise ValueError("Environment missing barrier_tc_base parameter")
    if not hasattr(model, 'Cbg') or model.Cbg is None:
        raise ValueError("Model missing Cbg matrix")

    # Extract target charge configuration (dots only, not sensor)
    n_target = np.array(array.optimal_VG_center[:n_dots], dtype=np.float64)

    # Extract cgd matrices for charge model: n = cgd @ V
    # cgd_full columns: [plunger0, ..., plungerN, sensor_gate, barrier0, ..., barrierM]
    cgd_full = np.array(model.cgd_full, dtype=np.float64)
    cgd_plungers = cgd_full[:n_dots, :n_dots]           # (n_dots, n_dots)
    cgd_barriers = cgd_full[:n_dots, n_gates:]          # (n_dots, n_barriers)
    cgd_plunger_diag = np.diag(cgd_plungers)            # (n_dots,)

    # Extract barrier coupling matrix for tunnel coupling model
    Cbg = np.array(model.Cbg[:, :n_dots], dtype=np.float64)  # (n_barriers, n_dots)

    # Per-barrier parameters
    alpha = np.array(array.barrier_alpha, dtype=np.float64)
    tc_base = float(array.barrier_tc_base)
    tc_target = float(array.optimal_tc)

    # Validate tunnel coupling parameters
    if tc_base <= 0 or tc_target <= 0:
        raise ValueError(f"tc_base ({tc_base}) and tc_target ({tc_target}) must be positive")

    log_tc_ratio = -np.log(tc_target / tc_base)

    # Extract virtual gate matrix transformation (matching env's calculate_ground_truth)
    vgm = np.array(model.gate_voltage_composer.virtual_gate_matrix, dtype=np.float64)
    vgm_inv = np.linalg.inv(vgm)
    vg_offset = np.array(model.gate_voltage_composer.virtual_gate_origin, dtype=np.float64)

    # Get optimal physical plunger voltages (model.optimal_Vg returns physical voltages)
    optimal_vg_physical = np.array(model.optimal_Vg(array.optimal_VG_center), dtype=np.float64)

    # Get full Cbg matrix (including sensor gate for barrier calculation)
    Cbg_full = np.array(model.Cbg, dtype=np.float64)  # (n_barriers, n_dots+1)

    return PhysicsParams(
        n_dots=n_dots,
        n_barriers=n_barriers,
        n_target=n_target,
        cgd_plungers=cgd_plungers,
        cgd_barriers=cgd_barriers,
        cgd_plunger_diag=cgd_plunger_diag,
        Cbg=Cbg_full,
        alpha=alpha,
        log_tc_ratio=log_tc_ratio,
        vgm_inv=vgm_inv,
        vg_offset=vg_offset,
        optimal_vg_physical=optimal_vg_physical,
    )


class PhysicalObjective:
    """
    Physics-based objective function for quantum dot array tuning.

    Computes the sum of squared deviations from locally optimal voltages,
    where each voltage's optimal value depends on all other voltages through
    the capacitance coupling model.

    Physics basis:
    - Plunger optimal: from charge model n = cgd @ V, solve for V_i given target n_i
    - Barrier optimal: from tunnel coupling model tc = tc_base * exp(-alpha * V_eff)

    The optimum is the fixed point where all local constraints are satisfied.

    Virtual gate transformation:
    - The RL agent operates in virtual voltage space, not physical space
    - Optimal voltages are computed to match env's calculate_ground_truth
    - The VGM and offset are fetched fresh each call to handle capacitance model updates

    Optional config enables:
    - Cap: return constant when any voltage is far from local optimal
    - Noise: add Gaussian noise to simulate measurement uncertainty
    """

    def __init__(self, env, config: Optional[ObjectiveConfig] = None, eps: float = 1e-10):
        """
        Initialize from environment.

        Args:
            env: QuantumDeviceEnv instance with use_barriers=True
            config: Optional ObjectiveConfig for cap/noise behavior
            eps: Minimum denominator value for numerical stability
        """
        self.env = env  # Store env reference for fresh VGM access
        self.params = _extract_physics_params(env, eps)
        self.config = config or ObjectiveConfig()
        self.eps = eps

        # Initialize RNG for noise if configured
        if self.config.noise_std > 0:
            self._rng = np.random.default_rng(self.config.noise_seed)
        else:
            self._rng = None

    def __call__(
        self,
        voltages: np.ndarray,
        plungers: Optional[list] = None,
        barriers: Optional[list] = None,
        cap: Optional[float] = None,
    ) -> float:
        """
        Compute the physical objective value.

        Args:
            voltages: Concatenated array [plunger_voltages, barrier_voltages]
                     Shape: (n_dots + n_barriers,)
            plungers: List of plunger indices to include (None = all)
            barriers: List of barrier indices to include (None = all)
            cap: Cap value for total objective (None = no cap)

        Returns:
            Objective value (possibly capped and/or noisy)
        """
        p = self.params

        # Split voltage vector
        V_p = voltages[:p.n_dots]
        V_b = voltages[p.n_dots:]

        # Default to all gates if not specified
        if plungers is None:
            plungers = list(range(p.n_dots))
        if barriers is None:
            barriers = list(range(p.n_barriers))

        # Compute optimal voltages with joint solve for subset
        V_p_opt = self._compute_plunger_optimal(V_p, V_b, plunger_subset=plungers)
        V_b_opt = self._compute_barrier_optimal(V_p, V_b, barrier_subset=barriers)

        # Compute distances for selected gates only
        plunger_dists = np.abs(V_p[plungers] - V_p_opt[plungers]) if plungers else np.array([])
        barrier_dists = np.abs(V_b[barriers] - V_b_opt[barriers]) if barriers else np.array([])

        # Sum squared distances
        result = np.sum(plunger_dists ** 2) + np.sum(barrier_dists ** 2)

        # Cap total if specified
        if cap is not None:
            result = min(result, cap)

        # Add noise if configured
        if self._rng is not None:
            result += self._rng.normal(0, self.config.noise_std)

        return float(result)

    def _compute_plunger_optimal(
        self,
        V_p: np.ndarray,
        V_b: np.ndarray,
        plunger_subset: Optional[list] = None
    ) -> np.ndarray:
        """
        Compute optimal plunger voltages in VIRTUAL space.

        This matches the env's calculate_ground_truth which transforms physical
        optimal voltages to virtual space using the CURRENT VGM:
            V_virtual = VGM^-1 @ (V_physical - offset)

        The VGM is fetched fresh from the env to handle capacitance model updates.
        For non-RL algorithms, VGM = identity so virtual = physical.

        Note: plunger_subset is ignored in virtual-space mode since the
        transformation is applied globally.
        """
        _ = V_p  # Not used - we use pre-computed optimal physical voltages
        _ = V_b
        _ = plunger_subset
        p = self.params

        # Get CURRENT VGM and offset from env (may have been updated by capacitance model)
        model = self.env.array.model
        current_vgm = np.array(model.gate_voltage_composer.virtual_gate_matrix, dtype=np.float64)
        current_offset = np.array(model.gate_voltage_composer.virtual_gate_origin, dtype=np.float64)
        vgm_inv = np.linalg.inv(current_vgm)

        # Transform optimal physical voltages to virtual space
        # This matches env's: vg_optimal_virtual = np.linalg.inv(current_vgm) @ (vg_optimal_physical - plunger_offset)
        vg_optimal_virtual = vgm_inv @ (p.optimal_vg_physical - current_offset)

        # Return only plunger voltages (exclude sensor gate at the end)
        return vg_optimal_virtual[:p.n_dots]

    def _compute_barrier_optimal(
        self,
        V_p: np.ndarray,
        V_b: np.ndarray,
        barrier_subset: Optional[list] = None
    ) -> np.ndarray:
        """
        Compute optimal barrier voltages.

        Matches environment's calculate_ground_truth formula:
            vb_optimal = vb_optimal_base - cbg @ vg_optimal_physical

        Where:
            vb_optimal_base = -ln(tc_target / tc_base) / alpha
            cbg @ vg_optimal_physical is the gate contribution using PHYSICAL voltages
        """
        _ = V_p  # Not used - we use pre-computed optimal physical voltages
        _ = V_b
        _ = barrier_subset
        p = self.params

        # Base term from tunnel coupling target: -ln(tc_target/tc_base) / alpha_j
        base_term = p.log_tc_ratio / p.alpha

        # Gate contribution: Cbg @ optimal_physical_voltages (including sensor)
        gate_term = p.Cbg @ p.optimal_vg_physical

        return base_term - gate_term

    def get_optimal_voltages(self, voltages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the computed optimal voltages for diagnostic purposes.

        Args:
            voltages: Current voltage array [plungers, barriers]

        Returns:
            (V_plunger_optimal, V_barrier_optimal)
        """
        p = self.params
        V_p = voltages[:p.n_dots]
        V_b = voltages[p.n_dots:]
        return (
            self._compute_plunger_optimal(V_p, V_b),
            self._compute_barrier_optimal(V_p, V_b)
        )

def create_objective_fn(
    env,
    config: Optional[ObjectiveConfig] = None,
    eps: float = 1e-10
) -> PhysicalObjective:
    """
    Create physics-based objective function for environment.

    Args:
        env: QuantumDeviceEnv instance with use_barriers=True
        config: Optional ObjectiveConfig for cap/noise behavior
        eps: Numerical stability threshold

    Returns:
        PhysicalObjective callable
    """
    return PhysicalObjective(env, config=config, eps=eps)


def get_distances(voltages: np.ndarray, env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get individual distances from ground truth voltages.

    Uses env's ground truth directly for consistency with RL training.

    Args:
        voltages: Concatenated voltage array [plungers, barriers]
        env: QuantumDeviceEnv instance

    Returns:
        (plunger_distances, barrier_distances): Per-gate distance arrays
    """
    num_plungers = env.num_plunger_voltages
    plunger_v = voltages[:num_plungers]
    barrier_v = voltages[num_plungers:]

    plunger_gt = env.device_state["gate_ground_truth"]
    barrier_gt = env.device_state["barrier_ground_truth"]

    plunger_dists = np.abs(plunger_v - plunger_gt)
    barrier_dists = np.abs(barrier_v - barrier_gt)

    return plunger_dists, barrier_dists


def check_success(voltages: np.ndarray, env, threshold: float = 0.5) -> bool:
    """
    Check if all voltages are within threshold of optimal.

    Args:
        voltages: Concatenated voltage array
        env: QuantumDeviceEnv instance
        threshold: Success threshold in volts (default 0.5V)

    Returns:
        True if all voltages within threshold of local optimal
    """
    plunger_dists, barrier_dists = get_distances(voltages, env)

    all_plungers_ok = np.all(plunger_dists < threshold)
    all_barriers_ok = np.all(barrier_dists < threshold)

    return all_plungers_ok and all_barriers_ok
