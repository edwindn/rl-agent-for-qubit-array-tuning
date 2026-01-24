import numpy as np
import scipy.linalg
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Union

from qarray.qarray_types import (CddInv, Cgd_holes, Cdd, VectorList, Tetrad,
                           Vector)


def optimal_Vg(cdd_inv: CddInv, cgd: Cgd_holes, n_charges: VectorList, rcond: float = 1e-3):
    '''
    calculate voltage that minimises charge state's energy

    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to gate capacitance matrix
    :param n_charges: the charge state of the dots of shape (n_dot)
    :return:
    '''
    R = np.linalg.cholesky(cdd_inv).T
    M = np.linalg.pinv(R @ cgd, rcond=rcond) @ R
    return np.einsum('ij, ...j', M, n_charges)



def compute_optimal_virtual_gate_matrix(
        cdd_inv: CddInv, cgd: Cgd_holes, rcond: float = 1e-4) -> np.ndarray:
    """
    Function to compute the optimal virtual gate matrix.

    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to gate capacitance matrix
    :param rcond: the rcond parameter for the pseudo inverse
    :return: the optimal virtual gate matrix

    """
    n_dot = cdd_inv.shape[0]
    n_gate = cgd.shape[1]
    virtual_gate_matrix = -np.linalg.pinv(cdd_inv @ cgd, rcond=rcond)

    # if the number of dots is less than the number of gates then we pad with zeros
    if n_dot < n_gate:
        virtual_gate_matrix = np.pad(virtual_gate_matrix, ((0, 0), (0, n_gate - n_dot)), mode='constant')

    return virtual_gate_matrix


def optimal_Vg_with_barriers(
    model,
    target_charges: np.ndarray,
    target_tcs: Union[float, np.ndarray],
    weight_charges: float = 1.0,
    weight_tcs: float = 1.0,
    rcond: float = 1e-3,

) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find gate and barrier voltages to achieve target charges and tunnel couplings.
    
    Parameters
    ----------
    target_charges : array-like, shape (n_dot + n_sensor,)
        Target charge configuration
    target_tcs : float or array-like  
        Target tunnel coupling strength(s)
    model : TunnelCoupledChargeSensed
        Quantum dot model with barrier_model and capacitance matrices
    weight_charges, weight_tcs : float
        Relative weights for objectives
    rcond : float
        Regularization for pseudoinverse
    
    Returns
    -------
    vg_optimal, vb_optimal : np.ndarray
        Optimal gate and barrier voltages
    """

    # With corrected physics: matrices already contain only dots+sensors
    cdd_inv = model.cdd_inv_full
    # Extract gate-only columns (exclude barriers)
    cgd = model.cgd_full[:, :model.n_gate]

    target_charges = np.atleast_1d(target_charges)
    target_tcs = np.atleast_1d(target_tcs)


    # Initial guess: use the old optimal_Vg ignoring barriers
    R = np.linalg.cholesky(cdd_inv).T
    M = np.linalg.pinv(R @ cgd, rcond=rcond) @ R
    vg0 = M @ target_charges
    
    # Smart initial guess for barriers: tc_target = tc_base * exp(-alpha * |vb|)
    # So |vb| = -ln(tc_target / tc_base) / alpha
    tc_ratio = np.mean(target_tcs) / model.barrier_model.tc_base  
    vb_mag = max(0.1, -np.log(max(tc_ratio, 0.01)) / np.mean(model.barrier_model.alpha))
    vb0 = np.full(model.n_barrier, vb_mag)
    print(f"Initial barrier guess: vb_mag = {vb_mag:.3f}V (for tc_ratio = {tc_ratio:.3f})")
    
    x0 = np.concatenate([vg0, vb0])
    print(f"Initial guess: vg0={vg0}, vb0={vb0}")

    # Cost function = weighted squared error
    iteration_count = [0]
    def cost(x):
        iteration_count[0] += 1
        vg, vb = np.split(x, [model.n_gate])
        
        # Charge error
        charges_pred = np.einsum('ij,j->i', M, vg)  # linear prediction
        charge_err = np.sum((charges_pred - target_charges) ** 2)
        
        # Tunnel coupling error
        tc_matrix = model.barrier_model.compute_tunnel_coupling_strength(vg, vb, model) 
        # Extract nearest-neighbor couplings: [tc_01, tc_12] for 3-dot system
        tcs_pred = np.array([tc_matrix[0, 1], tc_matrix[1, 2]])
        tc_err = np.sum((tcs_pred - target_tcs) ** 2)
        
        total_cost = weight_charges * charge_err + weight_tcs * tc_err
        
        # Debug output for first few iterations
        if iteration_count[0] <= 5:
            print(f"Iter {iteration_count[0]}: vb={vb}, tcs_pred={tcs_pred}, tc_err={tc_err:.2e}, total_cost={total_cost:.2e}")
            if iteration_count[0] == 1:
                print(f"  target_tcs={target_tcs}")
                print(f"  weight_charges={weight_charges}, weight_tcs={weight_tcs}")
                print(f"  charge_err={charge_err:.2e}, tc_err={tc_err:.2e}")

        return total_cost

    # Add bounds to allow reasonable exploration
    bounds = [(-5, 5)] * model.n_gate + [(-5, 5)] * model.n_barrier  # Allow ±5V range
    
    # Use L-BFGS-B with better initial guess
    print("Running optimization...")
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds)
    print(f"Optimization result: success={res.success}, fun={res.fun:.2e}")

    vg_opt, vb_opt = np.split(res.x, [model.n_gate])
    return vg_opt, vb_opt