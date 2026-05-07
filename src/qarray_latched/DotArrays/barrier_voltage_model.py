"""
Barrier voltage model for quantum dot arrays with tunnel coupling.

Handles exponential voltage dependence of tunnel coupling and cross-capacitive effects
between barriers, dots, gates, and sensors.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, List


class BarrierVoltageModel:
    """
    Model for barrier voltage effects on tunnel coupling in quantum dot arrays.
    
    Focuses purely on voltage-dependent physics calculations while capacitance
    matrices are stored in the parent TunnelCoupledChargeSensed model.
    """
    
    def __init__(self, n_barrier: int, n_dot: int, tc_base: float = 0.1, alpha: Union[float, List[float], jnp.ndarray] = 1.0):
        """
        Initialize barrier voltage model.
        
        Parameters:
        -----------
        n_barrier : int
            Number of barrier voltages
        n_dot : int
            Number of quantum dots
        tc_base : float
            Base tunnel coupling strength (default 0.1)
        alpha : float, list, or jnp.ndarray
            Exponential scaling factor(s) for voltage dependence (default 1.0)
            If float: same scaling for all barriers
            If list/array: per-barrier scaling factors (must have length n_barrier)
        """
        self.n_barrier = n_barrier
        self.n_dot = n_dot
        self.tc_base = tc_base
        
        # Handle alpha parameter - convert to JAX array with proper shape
        if isinstance(alpha, (list, jnp.ndarray)):
            if len(alpha) != n_barrier:
                raise ValueError(f"Alpha list length ({len(alpha)}) must match n_barrier ({n_barrier})")
            self.alpha = jnp.array(alpha)
        else:
            # Single value - broadcast to all barriers
            self.alpha = jnp.full(n_barrier, alpha)
        
        # JAX-compiled functions for performance
        self.compute_tc_matrix = jax.jit(self._compute_tc_matrix)
        self.compute_tc_matrix_batch = jax.jit(jax.vmap(self._compute_tc_matrix, in_axes=(0,)))
        
    def _compute_tc_matrix(self, vb: jnp.ndarray) -> jnp.ndarray:
        """
        Compute tunnel coupling matrix from barrier voltages.
        
        Uses exponential dependence: tc_eff = tc_base * exp(-alpha[i] * |vb[i]|)
        Assumes nearest-neighbor linear topology: dot0-barrier0-dot1-barrier1-...-dot(n-1)
        
        Parameters:
        -----------
        vb : jnp.ndarray
            Barrier voltages, shape (n_barrier,)
            
        Returns:
        --------
        jnp.ndarray
            Tunnel coupling matrix, shape (n_dot, n_dot)
        """
        # Initialize coupling matrix
        tc_matrix = jnp.zeros((self.n_dot, self.n_dot))
        
        # General implementation for nearest-neighbor linear topology
        # Each barrier[i] controls coupling between dot[i] and dot[i+1]
        # For n dots, we need (n-1) barriers
        expected_barriers = self.n_dot - 1
        if self.n_barrier < expected_barriers:
            raise ValueError(f"Linear topology with {self.n_dot} dots requires {expected_barriers} barriers, got {self.n_barrier}")
        
        # Compute all effective tunnel couplings with per-barrier alpha scaling
        tc_effs = self.tc_base * jnp.exp(-self.alpha[:expected_barriers] * vb[:expected_barriers])
        
        # Vectorized assignment of nearest-neighbor couplings
        # Create indices for all nearest-neighbor pairs
        i_indices = jnp.arange(expected_barriers)  # [0, 1, 2, ...]
        j_indices = i_indices + 1                 # [1, 2, 3, ...]
        
        # Set symmetric coupling matrix elements
        tc_matrix = tc_matrix.at[i_indices, j_indices].set(tc_effs)  # Upper diagonal
        tc_matrix = tc_matrix.at[j_indices, i_indices].set(tc_effs)  # Lower diagonal (symmetric)
        
        return tc_matrix
    
    def compute_effective_barrier_potential(self, vg: jnp.ndarray, vb: jnp.ndarray, 
                                          parent_model) -> jnp.ndarray:
        """
        Compute effective barrier potential including cross-capacitive effects.
        
        V_barrier_eff = V_barrier_applied + Cbg@vg + Cbb_cross@V_applied
        
        Neglects dot charge contributions:
        1. Gate contributions: Cbg @ vg
        2. Cross-barrier coupling: Cbb_off_diag @ V_applied  
        3. Dot contributions are neglected (gates typically dominate)
        
        Parameters:
        -----------
        vg : jnp.ndarray
            Gate voltages, shape (..., n_gate)
        vb : jnp.ndarray  
            Applied barrier voltages, shape (..., n_barrier)
        parent_model : TunnelCoupledChargeSensed
            Parent model containing capacitance matrices
            
        Returns:
        --------
        jnp.ndarray
            Effective barrier potentials, shape (..., n_barrier)
        """
        # Start with applied barrier voltages
        V_direct = vb.copy() if isinstance(vb, jnp.ndarray) else jnp.array(vb)
        
        # Add gate contributions: Cbg @ vg
        if parent_model.Cbg is not None:
            if vg.ndim > 1:
                # Batched case: (..., n_barrier) += (n_barrier, n_gate) @ (..., n_gate)
                gate_contribution = jnp.einsum('bg, ...g -> ...b', parent_model.Cbg, vg)
            else:
                # Single point case
                gate_contribution = parent_model.Cbg @ vg
            V_direct = V_direct + gate_contribution
        
        # Add cross-barrier coupling using V_direct as estimate
        if parent_model.Cbb is not None:
            # Create off-diagonal Cbb matrix (exclude self-interaction)  
            Cbb_off_diag = parent_model.Cbb - jnp.diag(jnp.diag(parent_model.Cbb))
            
            if V_direct.ndim > 1:
                # Batched case
                cross_barrier_contribution = jnp.einsum('bb, ...b -> ...b', Cbb_off_diag, V_direct)
            else:
                # Single point case
                cross_barrier_contribution = Cbb_off_diag @ V_direct
            
            V_effective = V_direct + cross_barrier_contribution
        else:
            V_effective = V_direct
            
        return V_effective
    
    def compute_tunnel_coupling_strength(self, vg: jnp.ndarray, vb: jnp.ndarray, 
                                       parent_model) -> jnp.ndarray:
        """
        Compute effective tunnel coupling strength including all voltage effects.
        
        Parameters:
        -----------
        vg : jnp.ndarray
            Gate voltages, shape (..., n_gate) 
        vb : jnp.ndarray
            Barrier voltages, shape (..., n_barrier)
        parent_model : TunnelCoupledChargeSensed
            Parent model containing capacitance matrices
            
        Returns:
        --------
        jnp.ndarray or float
            Effective tunnel coupling strength(s)
        """
        # Compute effective barrier potential
        vb_eff = self.compute_effective_barrier_potential(vg, vb, parent_model)
        
        # Handle batched inputs
        if vb_eff.ndim > 1:
            # Batch processing
            tc_matrices = self.compute_tc_matrix_batch(vb_eff)
            # For backward compatibility, return scalar if only nearest-neighbor
            if self.n_dot == 2:
                return tc_matrices[..., 0, 1]  # Return (0,1) element
            else:
                return tc_matrices
        else:
            # Single point
            tc_matrix = self.compute_tc_matrix(vb_eff)
            if self.n_dot == 2:
                return tc_matrix[0, 1]  # Return scalar for compatibility
            else:
                return tc_matrix
    
    def validate_dimensions(self, n_gate: int, n_dot: int, n_sensor: int):
        """
        Validate that barrier model dimensions are consistent with parent model.
        
        Parameters:
        -----------
        n_gate, n_dot, n_sensor : int
            Dimensions from parent model
        """
        if self.n_dot != n_dot:
            raise ValueError(f"Barrier model n_dot={self.n_dot} doesn't match parent n_dot={n_dot}")
            
        # Basic sanity checks
        if self.n_barrier < 1:
            raise ValueError("Must have at least 1 barrier")
            
        # Check linear topology requirement: n_dot requires (n_dot-1) barriers
        expected_barriers = self.n_dot - 1
        if self.n_barrier < expected_barriers:
            raise ValueError(f"Linear topology with {self.n_dot} dots requires {expected_barriers} barriers, got {self.n_barrier}")
            
        # Check alpha array dimensions
        if len(self.alpha) != self.n_barrier:
            raise ValueError(f"Alpha array length ({len(self.alpha)}) must match n_barrier ({self.n_barrier})")


def create_basic_barrier_model(n_barrier: int, n_dot: int, 
                              tc_base: float = 0.1, alpha: Union[float, List[float], jnp.ndarray] = 1.0) -> BarrierVoltageModel:
    """
    Factory function to create a basic barrier voltage model.
    
    Parameters:
    -----------
    n_barrier : int
        Number of barrier voltages
    n_dot : int
        Number of quantum dots  
    tc_base : float
        Base tunnel coupling strength
    alpha : float, list, or jnp.ndarray
        Exponential scaling factor(s) for voltage dependence (default 1.0)
        If float: same scaling for all barriers
        If list/array: per-barrier scaling factors (must have length n_barrier)
        
    Returns:
    --------
    BarrierVoltageModel
        Configured barrier model
    """
    return BarrierVoltageModel(n_barrier, n_dot, tc_base, alpha)