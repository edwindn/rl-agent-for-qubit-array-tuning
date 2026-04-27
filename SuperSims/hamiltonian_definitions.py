import dynamiqs as dq
import jax.numpy as jnp

import parameter_generation as pg
from parameter_generation import N

"""
Hamiltonian, pulse envelopes, and operators for transmon qubit simulation.
Reference: arXiv:2603.11018

Each qubit is driven with a two-quadrature baseband envelope modulated onto a
lab-frame carrier at frequency ω_d.  The in-phase (I) and quadrature (Q)
envelopes for qubit j follow:
    s_I(t) = [1 − cos(2πt / t_g)] / 2,           peaks at 1, area = t_g/2
    s_Q(t) = −(β/α) · ds_I/dt
           = −(β/α) · (π/t_g) · sin(2πt / t_g)

where β is the dimensionless DRAG coefficient (β = 1 is optimal) and α the
anharmonicity [rad/ns] (negative for transmons).  The drive amplitude Ω is
applied externally so that hardware effects (Omega_scale, amp_scale) compose
cleanly.  β = 0 recovers a plain raised-cosine pulse.

π-pulse condition: Ω · t_g / 2 = π/2  →  Ω = π / t_g.

The bare Hamiltonian for qubit i in the Fock-space truncated to N levels is:
    H_bare(i) = ω₀₁[i] n̂ + (α[i]/2) n̂(n̂ − 1),

where n̂ is the photon-number operator. The drive couples capacitively via
    H_drive(t) = f(t) · iX,    iX = i(a† − a),

with f(t) the drive amplitude constructed in all_xy_sequence.py and
compensation_matrix.py.  No rotating-wave approximation (RWA) is made.
"""


# ----- Pulse Envelopes ----- #
# Raised-cosine:  s_I(t) = [1 − cos(2πt/t_g)] / 2,  peaks at 1
# DRAG:           s_Q(t) = −(β/α) · ds_I/dt = −(β/α) · (π/t_g) · sin(2πt/t_g)
# β = 1 is the standard optimal DRAG value (reduces to −(1/α) · ds_I/dt).
# Ω is applied externally by the caller.

def s_I(t, t_g):
    """In-phase raised-cosine envelope (normalised to peak 1). Accepts scalar or array t."""
    return (1 - jnp.cos(2 * jnp.pi * t / t_g)) / 2

def s_Q(t, beta_i, alpha_i, t_g):
    """DRAG Q-envelope: −(β/α) · ds_I/dt. Vanishes when beta_i = 0.

    Args:
        t:       time [ns] — scalar or array
        beta_i:  dimensionless DRAG coefficient — scalar or (N_QUBITS,)
        alpha_i: anharmonicity [rad/ns] — scalar or (N_QUBITS,)
        t_g:     gate duration [ns]
    """
    return -(beta_i / alpha_i) * jnp.pi / t_g * jnp.sin(2 * jnp.pi * t / t_g)


# ----- Operators ----- #

a    = dq.destroy(N).to_jax()   # lowering (annihilation) operator
adag = dq.create(N).to_jax()    # raising (creation) operator
n_op = dq.number(N).to_jax()    # photon-number operator
I_op = dq.eye(N).to_jax()       # identity
iX_op = 1j * (adag - a)         # capacitive drive coupling operator

def H_bare_all(omega_01_vec, alpha):
    """Build (N_QUBITS, N, N) bare Hamiltonians from omega_01_vec.

    JAX-traceable: safe to differentiate through with jacfwd.

    Args:
        omega_01_vec: (N_QUBITS,) transition frequencies [rad/ns].
        alpha:        (N_QUBITS,) anharmonicities [rad/ns].
    """
    H_bare   = omega_01_vec[:, None, None] * n_op[None]
    H_anharm = (alpha / 2.0)[:, None, None] * (n_op @ (n_op - I_op))[None]
    return H_bare + H_anharm

