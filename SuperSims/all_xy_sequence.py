import functools

import dynamiqs as dq
import jax
import jax.numpy as jnp

from parameter_generation import N, N_QUBITS
from hamiltonian_definitions import H_bare_all, s_I, s_Q, iX_op

"""
All-XY calibration protocol for N_QUBITS transmon qubits.

Each qubit is driven through the 21 two-gate sequences of the All-XY protocol
simultaneously with all other qubits. Qubit i receives cross-talk from qubit j
via the perturbative coupling lambda_[i][j].

The characteristic staircase P(|1⟩) ∈ {0, 0.5, 1} reveals amplitude, phase,
and DRAG calibration errors.

params layout: (N_QUBITS, 5) — columns [omega_01, omega_d, phi, Omega, beta]
"""


# ----- All-XY Gate Definitions ----- #

GATES = {
    'I':    {'amp_scale': 0.0, 'phase': 0.0},
    'Xpi':  {'amp_scale': 1.0, 'phase': 0.0},
    'Ypi':  {'amp_scale': 1.0, 'phase': jnp.pi / 2},
    'Xpi2': {'amp_scale': 0.5, 'phase': 0.0},
    'Ypi2': {'amp_scale': 0.5, 'phase': jnp.pi / 2},
}

# 21 two-gate sequences; ideal P(|1⟩): 5 at 0, 12 at 0.5, 4 at 1.
ALLXY_GATES = [
    # P = 0
    ('I',    'I'),
    ('Xpi',  'Xpi'),
    ('Ypi',  'Ypi'),
    ('Xpi',  'Ypi'),
    ('Ypi',  'Xpi'),
    # P = 0.5
    ('Xpi2', 'I'),
    ('Ypi2', 'I'),
    ('Xpi2', 'Ypi'),
    ('Ypi2', 'Xpi'),
    ('Xpi2', 'Ypi2'),
    ('Ypi2', 'Xpi2'),
    ('Xpi',  'Xpi2'),
    ('Xpi',  'Ypi2'),
    ('Ypi',  'Xpi2'),
    ('Ypi',  'Ypi2'),
    ('Xpi2', 'Xpi'),
    ('Ypi2', 'Ypi'),
    # P = 1
    ('Xpi2', 'Xpi2'),
    ('Ypi2', 'Ypi2'),
    ('Xpi',  'I'),
    ('Ypi',  'I'),
]
ALLXY_IDEAL = [0.0] * 5 + [0.5] * 12 + [1.0] * 4
N_ALLXY = len(ALLXY_GATES)   # 21

_amp_scales = jnp.array([[GATES[g1]['amp_scale'], GATES[g2]['amp_scale']] for g1, g2 in ALLXY_GATES])
_phases     = jnp.array([[GATES[g1]['phase'],     GATES[g2]['phase']]     for g1, g2 in ALLXY_GATES])

# ----- Precomputed Constants ----- #

_M_alone = jnp.eye(N_QUBITS)
psi0     = dq.basis(N, 0)


# ----- Simulation ----- #

def _build_sim_inputs(params, hw, t_g, alpha, lambda_, simultaneous):
    """
    Build the batched Hamiltonian and tsave for dq.sesolve.
    """
    M = (_M_alone + lambda_) if simultaneous else _M_alone

    omega_01_p = params[:, 0]
    omega_d_p  = params[:, 1]
    phi_p      = params[:, 2] + hw[:, 0]
    Omega_p    = params[:, 3] * hw[:, 2]
    beta_p     = params[:, 4]
    t_delay    = hw[:, 1]

    H_bare_allxy = jnp.broadcast_to(
        H_bare_all(omega_01_p, alpha)[:, None], (N_QUBITS, N_ALLXY, N, N)
    )

    def _coeff(t):
        tau1 = t - t_delay        # (N_QUBITS,) — per-qubit delayed envelope time
        tau2 = tau1 - t_g

        w1 = jnp.where(jnp.logical_and(tau1 >= 0.0, tau1 <= t_g), 1.0, 0.0)
        w2 = jnp.where(jnp.logical_and(tau2 >= 0.0, tau2 <= t_g), 1.0, 0.0)

        env1  = s_I(tau1, t_g);  denv1 = s_Q(tau1, beta_p, alpha, t_g)
        env2  = s_I(tau2, t_g);  denv2 = s_Q(tau2, beta_p, alpha, t_g)

        base = omega_d_p * t + phi_p
        ph1  = base[:, None] + _phases[None, :, 0]
        ph2  = base[:, None] + _phases[None, :, 1]

        v1 = env1[:, None] * jnp.cos(ph1) + denv1[:, None] * jnp.sin(ph1)
        v2 = env2[:, None] * jnp.cos(ph2) + denv2[:, None] * jnp.sin(ph2)

        contrib = (
            w1[:, None] * _amp_scales[None, :, 0] * Omega_p[:, None] * v1
            + w2[:, None] * _amp_scales[None, :, 1] * Omega_p[:, None] * v2
        )
        return M @ contrib

    t_start = jnp.minimum(0.0, jnp.min(t_delay))
    t_end   = 2.0 * t_g + jnp.maximum(0.0, jnp.max(t_delay))
    tsave   = jnp.stack([t_start, t_end])

    return H_bare_allxy + dq.modulated(_coeff, iX_op), tsave


@functools.partial(jax.jit, static_argnums=(5,))
def run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=True):
    """Run the All-XY simulation with hardware imperfections applied.

    All episode-varying quantities are passed explicitly so JIT compilation
    remains valid across episodes without stale captures.

    Args:
        params:      (N_QUBITS, 5) array with columns [omega_01, omega_d, phi, Omega, beta].
        hw:          (N_QUBITS, 3) hardware imperfections [phi_hw, t_delay, Omega_scale].
        t_g:         scalar gate duration [ns].
        alpha:       (N_QUBITS,) anharmonicities [rad/ns].
        lambda_:     (N_QUBITS, N_QUBITS) flux cross-talk matrix.
        simultaneous: if True (default), apply cross-talk mixing via lambda_.

    Returns:
        P1: (N_QUBITS, N_ALLXY) P(|1⟩) at the end of each sequence.
    """
    H, tsave = _build_sim_inputs(params, hw, t_g, alpha, lambda_, simultaneous)
    result = dq.sesolve(H, psi0, tsave)
    states = result.states.to_jax()
    return jnp.abs(states[:, :, -1, 1, 0]) ** 2


def _run_allxy_sim_for_jac(params, hw, t_g, alpha, lambda_):
    """simultaneous=True with gradient=Forward() for use inside jax.jacfwd.

    Not JIT-compiled here; compiled via build_compensation's @jax.jit boundary.
    gradient=Forward() is only needed when JAX evaluates JVPs (i.e. during
    jacfwd), so it is kept out of run_allxy_simulation to avoid any overhead
    on plain forward-pass evaluation calls.
    """
    H, tsave = _build_sim_inputs(params, hw, t_g, alpha, lambda_, True)
    result = dq.sesolve(H, psi0, tsave, gradient=dq.gradient.Forward())
    states = result.states.to_jax()
    return jnp.abs(states[:, :, -1, 1, 0]) ** 2
