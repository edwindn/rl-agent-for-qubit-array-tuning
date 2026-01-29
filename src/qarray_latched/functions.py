
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxopt import BoxOSQP
import jax

import matplotlib.pyplot as plt

qp = BoxOSQP(check_primal_dual_infeasability=False, verbose=False)

# tunnelling matrix 2 dot
# |00> |01> |10> |11>
t2d = jnp.array([
    [0, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 0]
])

X = jnp.array([
    [0, 1],
    [1, 0]
])





def free_energy(vg, cdd_inv, cgd, basis):
    # computing the remainder
    inner = basis - cgd @ vg
    # computing the free energy of the change configurations
    return jnp.einsum('...i, ij, ...j', inner, cdd_inv, inner)

def open_charge_configurations_jax(n_continuous):
    """
    Generates all possible charge configurations for an open array.
    :param n_continuous:
    :return: a tensor of shape (2 ** n_dot, n_dot) containing all possible charge configurations
    """
    n_dot = n_continuous.shape[-1]
    floor_values = jnp.floor(n_continuous)
    args = jnp.zeros((n_dot, 2)) + jnp.array([0, 1])
    number_of_configurations = 2 ** n_dot
    zero_one_combinations = jnp.stack(jnp.meshgrid(*args), axis=-1).reshape(number_of_configurations, n_dot)
    return zero_one_combinations + floor_values[..., jnp.newaxis, :]


def expectation_number_of_charges(basis, eigenstates, index = 0):
    """
    Computes the expectation value of the number of charges in the ground state.
    :param eigenstates: the eigenstates of the Hamiltonian
    :param index: the index of the eigenstate to compute the expectation value for
    :return: the expectation value of the number of charges
    """
    match eigenstates.ndim:
        case 1:
            return jnp.einsum('ij, i -> j', basis, jnp.abs(eigenstates) ** 2)
        case 2:
            return jnp.einsum('ij, ij -> j', basis, jnp.abs(eigenstates[:, index]) ** 2)
        case _:
            raise ValueError(f"eigenstates must be 1d or 2d, got {eigenstates.ndim}d")


def numerical_solver_open(vg, cdd_inv, cgd) -> jnp.ndarray:
    """
    Solve the quadratic program for the continuous charge distribution for an open array.
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to dot capacitance matrix
    :param vg: the dot voltage coordinate vector
    :return: the continuous charge distribution
    """
    n_dot = cdd_inv.shape[0]
    P = cdd_inv
    q = -cdd_inv @ cgd @ vg
    A = jnp.eye(n_dot)
    l = jnp.zeros(n_dot)
    u = jnp.full(n_dot, fill_value=jnp.inf)
    params = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    return jnp.clip(params.primal[0], 0, None)




def make_hamiltonian(basis, vg, cgd, cdd_inv, tc):
    F = free_energy(vg, cdd_inv, cgd, basis)
    H = jnp.diag(F) + t2d * tc
    return H

def evaluate_energy(H, state):
    """
    Computes the energy of the state in the Hamiltonian.
    :param H: the Hamiltonian
    :param state: the state to compute the energy for
    :return: the energy of the state
    """
    return jnp.einsum('i, ij, j -> ', jnp.conj(state), H, state)

def ground_state_0d(vg, cgd, cdd_inv, tc):
    """
    Computes the ground state for an open array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param T: the temperature
    :return: the continuous charge distribution
    """
    cont_solution = numerical_solver_open(vg, cdd_inv, cgd)
    basis = open_charge_configurations_jax(cont_solution)

    H = make_hamiltonian(basis, vg, cgd, cdd_inv, tc)
    eigen_energies, eigen_states = jnp.linalg.eigh(H)
    return basis, eigen_energies[0], eigen_states[:, 0]


def latched_csd_1d(vg, cgd, cdd_inv, tc):
    """
    Computes the ground state for an open array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param T: the temperature
    :return: the continuous charge distribution
    """
    assert vg.ndim == 2, "vg must be 2d, shape (n_points, n_gates)"

    basis, energy, state = ground_state_0d(vg[0, :], cgd = cgd, cdd_inv = cdd_inv, tc=tc)
    charge_states = [expectation_number_of_charges(basis, state)]

    for i in range(1, vg.shape[0]):

        cont_solution = numerical_solver_open(vg[i, :], cdd_inv, cgd)
        basis = open_charge_configurations_jax(cont_solution)

        H = make_hamiltonian(basis, vg[i, :], cgd, cdd_inv, tc)
        eigen_energies, eigen_states = jnp.linalg.eigh(H)
        charge_states.append(expectation_number_of_charges(basis, state))

    return charge_states


def latched_2d(vg, cgd, cdd_inv, tc):
    """
    Computes the ground state for an open array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param T: the temperature
    :return: the continuous charge distribution
    """
    assert vg.ndim == 3, "vg must be 2d, shape (ny_points, nx_points, n_gates)"

    charge_states = []
    for i in range(vg.shape[0]):
        charge_states.append(latched_csd_1d(vg[i, :, :], cgd = cgd, cdd_inv = cdd_inv, tc=tc))

    return jnp.array(charge_states)






def ground_state_nd(vg, cgd, cdd_inv, tc):
    vg_shape = vg.shape[:-1]
    vg = vg.reshape(-1, vg.shape[-1])
    batched_ground_state = vmap(ground_state_0d, in_axes=(0, None, None, None))
    return batched_ground_state(vg, cdd_inv, cgd, tc).reshape(*vg_shape, -1)

