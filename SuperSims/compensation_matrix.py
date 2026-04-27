import jax
import jax.numpy as jnp

from parameter_generation import N_QUBITS
from all_xy_sequence import _run_allxy_sim_for_jac
from normalisations import clip_params

"""
Jacobian blocks and compensation matrix for the All-XY calibration protocol.

The compensation matrix C has the structure:

    C[i, :, i, :] = I_5                                (identity — qubit i's own update passes through)
    C[i, :, j, :] = -pinv(J_self[i]) @ J_cross[i, j]  (cancels cross-talk on qubit i from qubit j)

where:
    J_self[i]    = J[i, :, i, :] — (N_ALLXY, 5) — qubit i's staircase sensitivity to its own params
    J_cross[i,j] = J[i, :, j, :] — (N_ALLXY, 5) — qubit i's staircase sensitivity to qubit j's params

This ensures that a raw parameter update delta_raw[i] is applied directly to qubit i, while
each other qubit j receives a compensating update that cancels the first-order cross-talk
perturbation on its All-XY staircase. Hardware imperfections are included throughout.

The full (N_QUBITS×N_ALLXY, N_QUBITS×5) Jacobian is computed in a single jacfwd pass over
run_allxy_simulation, batching all N_QUBITS×5 forward sensitivity equations into one augmented
ODE solve. Only N_QUBITS pseudoinverses of (N_ALLXY, 5) matrices are then required.
"""


_jacfwd_sim = jax.jacfwd(_run_allxy_sim_for_jac, argnums=0)


@jax.jit
def build_compensation(params, hw, t_g, alpha, lambda_):
    """Compute the full Jacobian and compensation tensor from nominal params and hardware.

    Args:
        params:   (N_QUBITS, 5) nominal parameter values.
        hw:       (N_QUBITS, 3) hardware imperfections [phi_hw, t_delay, Omega_scale].
        t_g:      scalar gate duration [ns].
        alpha:    (N_QUBITS,) anharmonicities [rad/ns].
        lambda_:  (N_QUBITS, N_QUBITS) flux cross-talk matrix.

    Returns:
        C_tensor: (N_QUBITS, 5, N_QUBITS, 5) updated compensation tensor.
        J_cols:   list of N_QUBITS arrays, each (N_QUBITS, N_ALLXY, 5) — Jacobian columns.
    """
    J_full = _jacfwd_sim(params, hw, t_g, alpha, lambda_)
    # J_full: (N_QUBITS, N_ALLXY, N_QUBITS, 5) — dP1[i,s]/dparams[j,k]
    J_cols = [J_full[:, :, j, :] for j in range(N_QUBITS)]

    n_params    = params.shape[1]
    pinv_J_self = [jnp.linalg.pinv(J_cols[i][i]) for i in range(N_QUBITS)]

    rows = []
    for i in range(N_QUBITS):
        row = jnp.stack([
            jnp.eye(n_params) if i == j else -pinv_J_self[i] @ J_cols[j][i]
            for j in range(N_QUBITS)
        ])   # (N_QUBITS, n_params, n_params)
        rows.append(row)
    C_out = jnp.stack(rows).transpose(0, 2, 1, 3)   # (N_QUBITS, n_params, N_QUBITS, n_params)
    return C_out, J_cols


def update_params(params, delta_raw, C_tensor, param_mins, param_maxs, clip_multiplier=2.0):
    """Apply a virtual update through C_tensor, then clip to relaxed safety bounds.

    C_tensor maps each agent's virtual action delta_raw[i] to physical updates
    across all qubits. The resulting physical params may exceed episode_bounds due
    to compensation offsets; clip_params provides a hard safety rail at
    clip_multiplier times the episode span, without restricting the agent's own
    action or the compensation contributions independently.

    Args:
        params:          (N_QUBITS, 5) current physical parameters.
        delta_raw:       (N_QUBITS, 5) intended raw update per qubit in physical units.
        C_tensor:        (N_QUBITS, 5, N_QUBITS, 5) compensation tensor.
        param_mins:      (N_QUBITS, 5) lower bounds from episode_bounds.
        param_maxs:      (N_QUBITS, 5) upper bounds from episode_bounds.
        clip_multiplier: extend each bound by (clip_multiplier - 1) × span (default 2.0).

    Returns:
        params: (N_QUBITS, 5) updated physical parameters, clipped to relaxed bounds.
    """
    delta_phys = jnp.einsum('ikjl,jl->ik', C_tensor, delta_raw)
    return clip_params(params + delta_phys, param_mins, param_maxs, clip_multiplier)
