import jax.numpy as jnp
import jax
from functools import partial

import sys
import os
# Ensure root path is available
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from qarray_latched.DotArrays.hamiltonian_build import (
    compute_tunneling_matrix_vectorized, 
    compute_tunneling_simple_operators,
    compute_tunneling_fermionic_positive,
    compute_tunneling_dynamiqs_fock,
    compute_tunneling_dynamiqs_optimized,
    choose_hamiltonian_convention
)

@partial(jax.jit, static_argnums=(4, 5))
def _jit_free_energy_unbatched(v_extended, cdd_inv_batch, cgd_batch, charge_states, n_dot, constant_charge_shift=0):
    """
    JIT-compiled free energy computation for voltage-dependent capacitances.

    Parameters:
    -----------
    v_extended : jnp.ndarray
        Extended voltages [gates, barriers], shape (..., n_gate + n_barrier)
    cdd_inv_batch : jnp.ndarray
        Voltage-dependent inverse capacitance matrices, shape (..., n_full, n_full)
    cgd_batch : jnp.ndarray
        Voltage-dependent gate-dot matrices, shape (..., n_full, n_gate + n_barrier)
    charge_states : jnp.ndarray
        Charge states, shape (M, n_dot)
    constant_charge_shift : int
        Constant charge shift to add. Pass 0 if no shift desired.

    Returns:
    --------
    jnp.ndarray
        Free energies for all charge states, shape (..., M)
    """
    # Extract dot effects from full system matrices
    gate_effect = jnp.einsum('...ij, ...j -> ...i', cgd_batch[..., :n_dot, :], v_extended)  # (..., n_dot)
    n0 = jnp.full(n_dot, constant_charge_shift)
    gate_effect += n0
    cdd_inv_dots = cdd_inv_batch[..., :n_dot, :n_dot]  # (..., n_dot, n_dot)

    # Compute free energy with voltage-dependent capacitances
    inner = charge_states - gate_effect[..., None, :]
    return jnp.einsum('...ni, ...ij, ...nj -> ...n', inner, cdd_inv_dots, inner)


def full_physics_informed_tunneling_hamiltonian_unbatched(tc_matrix_batch: jnp.ndarray, charge_states: jnp.ndarray, 
                                               max_electrons_per_dot: int = 4,
                                               convention: str = "fermionic_negative") -> jnp.ndarray:
    """
    Compute tunneling Hamiltonians for batched tunnel coupling matrices using modular implementation (unbatched version).
    
    Parameters:
    -----------
    tc_matrix_batch : jnp.ndarray
        Tunnel coupling matrices, shape (N_points, n_dot, n_dot)
    charge_states : jnp.ndarray
        Charge states, shape (M, n_dot)
    max_electrons_per_dot : int
        Maximum electrons per dot
    convention : str
        Hamiltonian convention to use
        
    Returns:
    --------
    jnp.ndarray
        Tunneling Hamiltonians, shape (N_points, M, M) or (M, M)
    """
    hamiltonian_func = choose_hamiltonian_convention(convention)
    compute_batch = jax.vmap(
            lambda tc: hamiltonian_func(tc, charge_states, max_electrons_per_dot),
            in_axes=0, out_axes=0
            )
    return compute_batch(tc_matrix_batch)

