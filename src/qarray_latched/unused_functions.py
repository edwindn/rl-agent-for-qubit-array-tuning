"""
Author: b-vanstraaten
Date: 01/04/2025
"""


def thermal_density_matrix(w, v, T):
    """
    Compute the thermal density matrix in the Fock basis using einsum.

    Parameters:
    -----------
    w : array (N,)
        Eigenvalues (energies) from diagonalising the Hamiltonian
    v : array (N, N)
        Eigenvectors (columns) in the Fock basis
    beta : float
        Inverse temperature (1 / k_B T)

    Returns:
    --------
    rho_th : array (N, N)
        Thermal density matrix in the Fock basis
    """
    if T == 0:
        v_min = v[:, 0]
        rho_th = jnp.einsum('i, j->ij', v_min, jnp.conj(v_min))

    else:
        w_shifted = w - jnp.min(w)  # shift spectrum to have min = 0

        weights = jnp.exp(- w_shifted / T)  # shape (N,)
        Z = jnp.sum(weights)
        probs = weights / Z  # shape (N,)

        # Construct rho_th = sum_k p_k * |ψ_k⟩⟨ψ_k| using einsum
        # v: shape (N, N), columns are eigenvectors
        # Output: rho_th[n, m] = sum_k p_k * v[n, k] * conj(v[m, k])
        rho_th = jnp.einsum('ik,k,jk->ij', v, probs, jnp.conj(v))
    return rho_th