"""
Voltage-dependent capacitance models for quantum dot arrays.

Extracted and adapted from voltage_dependent_capacitance_boxosqp.py
for integration with tunnel coupling physics.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp


class VoltageDependendentCapacitanceModel:
    """
    Optimized voltage-dependent capacitance model that pre-vectorizes operations.
    """
    
    def __init__(self, 
                 cdd_func: Callable,
                 cgd_func: Callable,
                 n_dot: int,
                 n_gate: int):
        """
        Initialize the fast voltage-dependent capacitance model.
        
        Parameters:
        -----------
        cdd_func : Callable
            Function that takes voltage and returns the dot-to-dot capacitance matrix
        cgd_func : Callable  
            Function that takes voltage and returns the gate-to-dot capacitance matrix
        n_dot : int
            Number of quantum dots
        n_gate : int
            Number of gates
        """
        self.n_dot = n_dot
        self.n_gate = n_gate
        
        # Create vectorized versions of the capacitance functions
        self.cdd_func_vmap = jax.vmap(cdd_func)
        self.cgd_func_vmap = jax.vmap(cgd_func)
        
        # JIT compile the vectorized functions
        self.get_cdd_batch = jax.jit(self.cdd_func_vmap)
        self.get_cgd_batch = jax.jit(self.cgd_func_vmap)
        self.get_cdd_inv_batch = jax.jit(jax.vmap(jnp.linalg.inv))
        
        # For single voltage queries
        self.get_cdd = jax.jit(cdd_func)
        self.get_cgd = jax.jit(cgd_func)
        
    def compute_all_capacitances(self, vg_batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute all capacitance matrices for a batch of gate voltages.
        
        Parameters:
        -----------
        vg_batch : jnp.ndarray
            Array of gate voltages, shape (n_points, n_gates)
            
        Returns:
        --------
        cdd_batch : jnp.ndarray
            Dot-to-dot capacitance matrices, shape (n_points, n_dots, n_dots)
        cdd_inv_batch : jnp.ndarray
            Inverse of Cdd matrices, shape (n_points, n_dots, n_dots)
        cgd_batch : jnp.ndarray
            Gate-to-dot capacitance matrices, shape (n_points, n_dots, n_gates)
        """
        cdd_batch = self.get_cdd_batch(vg_batch)
        cdd_inv_batch = self.get_cdd_inv_batch(cdd_batch)
        cgd_batch = self.get_cgd_batch(vg_batch)
        return cdd_batch, cdd_inv_batch, cgd_batch


# Capacitance function implementations
def linear_voltage_dependent_cdd(vg: jnp.ndarray, 
                                cdd_0: jnp.ndarray,
                                alpha: float = 0.1) -> jnp.ndarray:
    """Linear voltage dependence for dot-dot capacitances."""
    v_scale = 1 + alpha * jnp.mean(jnp.abs(vg))
    return cdd_0 * v_scale


def linear_voltage_dependent_cgd(vg: jnp.ndarray,
                                cgd_0: jnp.ndarray,
                                beta: float = 0.01) -> jnp.ndarray:
    """Linear voltage dependence for gate-dot capacitances."""
    v_scale = 1 + beta * jnp.mean(jnp.abs(vg))
    return cgd_0 * v_scale


def quadratic_voltage_dependent_cdd(vg: jnp.ndarray,
                                   cdd_0: jnp.ndarray,
                                   gamma: float = 0.01) -> jnp.ndarray:
    """Quadratic voltage dependence for dot-dot capacitances."""
    v_scale = 1 + gamma * jnp.sum(vg**2)
    return cdd_0 * v_scale


def sigmoid_voltage_dependent_cdd(vg: jnp.ndarray,
                                 cdd_0: jnp.ndarray,
                                 v_char: float = 1.0,
                                 delta: float = 0.5) -> jnp.ndarray:
    """Sigmoid voltage dependence for dot-dot capacitances."""
    v_norm = jnp.linalg.norm(vg) / v_char
    v_scale = 1 + delta * jax.nn.sigmoid(v_norm - 1)
    return cdd_0 * v_scale


def gate_specific_voltage_dependent_cgd(vg: jnp.ndarray,
                                       cgd_0: jnp.ndarray,
                                       beta_gates: jnp.ndarray) -> jnp.ndarray:
    """
    Gate-specific voltage dependence where each gate has different sensitivity.
    
    C_gd[i,j](V) = C_gd^0[i,j] * (1 + beta_j * |V_j|)
    """
    v_scale = 1 + beta_gates * jnp.abs(vg)
    return cgd_0 * v_scale[None, :]  # Broadcasting over dot dimension


# Factory functions for creating common voltage-dependent models
def create_linear_capacitance_model(cdd_0: jnp.ndarray, 
                                   cgd_0: jnp.ndarray,
                                   alpha: float = 0.1, 
                                   beta: float = 0.01) -> VoltageDependendentCapacitanceModel:
    """Create a linear voltage-dependent capacitance model."""
    def cdd_func(vg):
        return linear_voltage_dependent_cdd(vg, cdd_0, alpha)
    
    def cgd_func(vg):
        return linear_voltage_dependent_cgd(vg, cgd_0, beta)
    
    n_dot, n_gate = cgd_0.shape
    return VoltageDependendentCapacitanceModel(cdd_func, cgd_func, n_dot, n_gate)


def create_quadratic_capacitance_model(cdd_0: jnp.ndarray, 
                                      cgd_0: jnp.ndarray,
                                      gamma: float = 0.01, 
                                      beta: float = 0.01) -> VoltageDependendentCapacitanceModel:
    """Create a quadratic voltage-dependent capacitance model."""
    def cdd_func(vg):
        return quadratic_voltage_dependent_cdd(vg, cdd_0, gamma)
    
    def cgd_func(vg):
        return linear_voltage_dependent_cgd(vg, cgd_0, beta)
    
    n_dot, n_gate = cgd_0.shape
    return VoltageDependendentCapacitanceModel(cdd_func, cgd_func, n_dot, n_gate)


def create_sigmoid_capacitance_model(cdd_0: jnp.ndarray, 
                                    cgd_0: jnp.ndarray,
                                    v_char: float = 1.0,
                                    delta: float = 0.5, 
                                    beta: float = 0.01) -> VoltageDependendentCapacitanceModel:
    """Create a sigmoid voltage-dependent capacitance model."""
    def cdd_func(vg):
        return sigmoid_voltage_dependent_cdd(vg, cdd_0, v_char, delta)
    
    def cgd_func(vg):
        return linear_voltage_dependent_cgd(vg, cgd_0, beta)
    
    n_dot, n_gate = cgd_0.shape
    return VoltageDependendentCapacitanceModel(cdd_func, cgd_func, n_dot, n_gate)