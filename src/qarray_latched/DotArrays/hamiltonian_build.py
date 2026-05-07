import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
try:
    import dynamiqs as dq
    DYNAMIQS_AVAILABLE = True
except ImportError:
    DYNAMIQS_AVAILABLE = False


@partial(jax.jit, static_argnums=(4, 5))
def _jit_free_energy(v_extended, cdd_inv_batch, cgd_batch, charge_states, n_dot, constant_charge_shift=0):
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
    inner = charge_states[None, ...] - gate_effect[..., None, :]  # (..., M, n_dot)

    return jnp.einsum('...ni, ...ij, ...nj -> ...n', inner, cdd_inv_dots, inner)


def full_physics_informed_tunneling_hamiltonian(tc_matrix_batch: jnp.ndarray, charge_states: jnp.ndarray, 
                                               max_electrons_per_dot: int = 4, 
                                               convention: str = "fermionic_negative") -> jnp.ndarray:
    """
    Compute tunneling Hamiltonians for batched tunnel coupling matrices using modular implementation.
    
    Parameters:
    -----------
    tc_matrix_batch : jnp.ndarray
        Tunnel coupling matrices, shape (N_points, n_dot, n_dot)
    charge_states : jnp.ndarray
        Charge states, shape (num_charge_states, M, n_dot)
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
    compute_batch = jax.vmap(lambda tc, charge_state: hamiltonian_func(tc, charge_state, max_electrons_per_dot), in_axes=(0, 0), out_axes=0)
    return compute_batch(tc_matrix_batch, charge_states)


@partial(jax.jit, static_argnums=(2,))
def compute_tunneling_matrix_vectorized(tc_matrix: jnp.ndarray, charge_states: jnp.ndarray,
                                       max_electrons_per_dot: int = 4) -> jnp.ndarray:
    """
    Vectorized JAX implementation of tunneling Hamiltonian computation.
    
    
    The physics: Matrix elements are -t√(n_i)(n_j+1) for electron hopping
    from dot i to j, accounting for fermionic statistics.
    
    Args:
        tc_matrix: Tunnel coupling matrix, shape (n_dots, n_dots)
        charge_states: Array of charge states, shape (M, n_dots)
        max_electrons_per_dot: Maximum electrons per dot
        
    Returns:
        Tunneling Hamiltonian matrix, shape (M, M)
    """
    if charge_states.ndim == 2:
        n_states = charge_states.shape[0]
        n_dots = charge_states.shape[1]
    elif charge_states.ndim == 3:
        n_states = charge_states.shape[1]
        n_dots = charge_states.shape[2]
    
    # Create all pairs of states
    state_i = charge_states[:, None, :]  # Shape: (M, 1, n_dots)
    state_j = charge_states[None, :, :]  # Shape: (1, M, n_dots)
    
    # Compute differences
    diff = state_j - state_i  # Shape: (M, M, n_dots)
    
    # Initialize Hamiltonian
    H = jnp.zeros((n_states, n_states))
    
    # For each adjacent dot pair
    for dot_from in range(n_dots - 1):
        dot_to = dot_from + 1
        
        # Create expected difference pattern for this hopping
        expected_diff = jnp.zeros(n_dots)
        expected_diff = expected_diff.at[dot_from].set(-1)
        expected_diff = expected_diff.at[dot_to].set(1)
        
        # Find state pairs connected by this hopping
        # Shape: (M, M)
        is_valid_hop_forward = jnp.all(diff == expected_diff[None, None, :], axis=2)
        is_valid_hop_backward = jnp.all(diff == -expected_diff[None, None, :], axis=2)
        
        # Get electron numbers
        n_from_forward = state_i[:, :, dot_from]  # electrons at source dot
        n_to_forward = state_i[:, :, dot_to]      # electrons at target dot
        
        # Compute matrix elements for forward hopping (dot_from -> dot_to)
        t = tc_matrix[dot_from, dot_to]
        elements_forward = -t * jnp.sqrt(n_from_forward * (n_to_forward + 1))
        elements_backward = -t * jnp.sqrt(n_to_forward * (n_from_forward + 1))
        
        # Add to Hamiltonian
        H = H + is_valid_hop_forward * elements_forward
        H = H + is_valid_hop_backward * elements_backward
    
    return H


@partial(jax.jit, static_argnums=(2,))
def compute_tunneling_simple_operators(tc_matrix: jnp.ndarray, charge_states: jnp.ndarray,
                                      max_electrons_per_dot: int = 4) -> jnp.ndarray:
    """
    Compute tunneling Hamiltonian using the simplified operator formalism QDarts.
    
    From the image:
    - c_i†|n⟩ = |n + e_i⟩ (no sqrt factors)
    - c_i|n⟩ = |n - e_i⟩
    - H_tunneling = Σᵢⱼ t_ij(c_i†c_j + c_j†c_i)
    
    This gives matrix elements: ⟨n'|H|n⟩ = t_ij when n' = n - e_j + e_i
    
    Args:
        tc_matrix: Tunnel coupling matrix (can have non-zero elements for any i,j)
        charge_states: Array of charge states, shape (M, n_dots)
        max_electrons_per_dot: Maximum electrons per dot
        
    Returns:
        Tunneling Hamiltonian matrix, shape (M, M)
    """
    n_states = charge_states.shape[0]
    n_dots = charge_states.shape[1]
    
    # Create all pairs of states
    state_i = charge_states[:, None, :]  # Shape: (M, 1, n_dots)
    state_j = charge_states[None, :, :]  # Shape: (1, M, n_dots)
    
    # Initialize Hamiltonian
    H = jnp.zeros((n_states, n_states))
    
    # Check all dot pairs (not just nearest neighbors)
    for dot_from in range(n_dots):
        for dot_to in range(n_dots):
            if dot_from != dot_to:
                # Create expected difference pattern for hopping dot_from → dot_to
                expected_diff = jnp.zeros(n_dots)
                expected_diff = expected_diff.at[dot_from].set(-1)
                expected_diff = expected_diff.at[dot_to].set(1)
                
                # Find state pairs connected by this hopping
                diff = state_j - state_i  # Shape: (M, M, n_dots)
                is_valid_hop = jnp.all(diff == expected_diff[None, None, :], axis=2)
                
                # Check constraints: n_from > 0 and n_to < max
                n_from = state_i[:, :, dot_from]
                n_to = state_i[:, :, dot_to]
                constraints_met = (n_from > 0) & (n_to < max_electrons_per_dot)
                
                # Add matrix elements (no sqrt, no minus sign)
                t = tc_matrix[dot_from, dot_to]
                H = H + is_valid_hop * constraints_met * t
    
    return H


@partial(jax.jit, static_argnums=(2,))
def compute_tunneling_fermionic_positive(tc_matrix: jnp.ndarray, charge_states: jnp.ndarray,
                                        max_electrons_per_dot: int = 4) -> jnp.ndarray:
    """
    Compute tunneling Hamiltonian with fermionic sqrt factors but positive sign convention.
    
    This is our current physics but without the minus sign:
    - Matrix elements: +t√(n_from)√(n_to+1) instead of -t√(n_from)√(n_to+1)
    
    Args:
        tc_matrix: Tunnel coupling matrix
        charge_states: Array of charge states, shape (M, n_dots)
        max_electrons_per_dot: Maximum electrons per dot
        
    Returns:
        Tunneling Hamiltonian matrix, shape (M, M)
    """
    n_states = charge_states.shape[0]
    n_dots = charge_states.shape[1]
    
    state_i = charge_states[:, None, :]
    state_j = charge_states[None, :, :]
    diff = state_j - state_i
    
    H = jnp.zeros((n_states, n_states))
    
    # Check all dot pairs (or restrict to nearest neighbors as needed)
    for dot_from in range(n_dots):
        for dot_to in range(n_dots):
            if dot_from != dot_to:
                expected_diff = jnp.zeros(n_dots)
                expected_diff = expected_diff.at[dot_from].set(-1)
                expected_diff = expected_diff.at[dot_to].set(1)
                
                is_valid_hop = jnp.all(diff == expected_diff[None, None, :], axis=2)
                
                n_from = state_i[:, :, dot_from]
                n_to = state_i[:, :, dot_to]
                
                # Positive sign convention with fermionic factors
                t = tc_matrix[dot_from, dot_to]
                elements = t * jnp.sqrt(n_from * (n_to + 1))
                
                H = H + is_valid_hop * elements
    
    return H


def compute_tunneling_dynamiqs_fock(tc_matrix: jnp.ndarray, charge_states: jnp.ndarray,
                                   max_electrons_per_dot: int = 4) -> jnp.ndarray:
    """
    Compute tunneling Hamiltonian using dynamiqs in full Fock state basis.
    
    This implementation:
    - Works in the full Fock space (2^n_orbitals dimensional)
    - Uses proper fermionic creation/annihilation operators via Jordan-Wigner
    - Projects down to charge state basis at the end
    - Fully JAX-compatible through dynamiqs
    
    Note: Currently works correctly for single-orbital-per-dot systems.
    For multi-orbital systems, the charge state to Fock mapping may need refinement.
    
    Args:
        tc_matrix: Tunnel coupling matrix
        charge_states: Array of charge states, shape (M, n_dots)
        max_electrons_per_dot: Maximum electrons per dot
        
    Returns:
        Tunneling Hamiltonian matrix in charge state basis, shape (M, M)
    """
    if not DYNAMIQS_AVAILABLE:
        print("dynamiqs not available, falling back to JAX implementation")
        return compute_tunneling_matrix_vectorized(tc_matrix, charge_states, max_electrons_per_dot)
    
    n_dots = charge_states.shape[1]
    n_states = charge_states.shape[0]
    n_orbitals = n_dots * max_electrons_per_dot  # Total orbital levels
    
    # Create fermionic annihilation operators for each orbital
    # Using Jordan-Wigner transformation with dynamiqs
    a_ops = []
    
    # For a small number of orbitals, we can construct the operators directly
    if n_orbitals > 6:  # Limit to avoid exponentially large matrices
        raise ValueError(f"Too many orbitals ({n_orbitals}) for Fock space approach. Use optimized version.")
    
    for i in range(n_orbitals):
        # Build the full operator in the 2^n_orbitals dimensional space
        # Start with identity and build up the tensor product
        
        # For the i-th annihilation operator:
        # - sigma_z for all j < i (Jordan-Wigner string)
        # - sigma_minus for site i
        # - identity for all j > i
        
        if n_orbitals == 1:
            # Special case: single orbital
            a_i = dq.sigmam()
        else:
            # Build operator list
            ops = []
            for j in range(n_orbitals):
                if j < i:
                    ops.append(dq.sigmaz())
                elif j == i:
                    ops.append(dq.sigmam())
                else:
                    ops.append(dq.eye(2))
            
            # Compute tensor product
            a_i = ops[0]
            for op in ops[1:]:
                a_i = dq.tensor(a_i, op)
        
        a_ops.append(a_i)
    
    # Create the Hamiltonian in Fock space
    # Initialize as JAX array then convert to dynamiqs if needed
    fock_dim = 2**n_orbitals
    H_fock_jax = jnp.zeros((fock_dim, fock_dim), dtype=jnp.complex64)
    
    # Add tunneling terms between dots
    # Only iterate over upper triangle to avoid double counting
    for dot_from in range(n_dots):
        for dot_to in range(dot_from + 1, n_dots):
            if tc_matrix[dot_from, dot_to] != 0:
                t = tc_matrix[dot_from, dot_to]
                
                # For multi-orbital dots, we allow tunneling between corresponding orbitals
                for orbital in range(max_electrons_per_dot):
                    site_from = dot_from * max_electrons_per_dot + orbital
                    site_to = dot_to * max_electrons_per_dot + orbital
                    
                    # Get annihilation operators
                    a_from = a_ops[site_from].to_jax()
                    a_to = a_ops[site_to].to_jax()
                    
                    # Add hopping term: -t (c†_from c_to + c†_to c_from)
                    H_fock_jax = H_fock_jax - t * (jnp.conj(a_from.T) @ a_to + 
                                                   jnp.conj(a_to.T) @ a_from)
    
    # H_fock_jax is already a JAX array
    H_fock_array = H_fock_jax
    
    # Map charge states to Fock basis indices
    def charge_state_to_fock_index(charge_state):
        """Convert charge state to index in full Fock space."""
        index = 0
        for dot_idx, n_electrons in enumerate(charge_state):
            # Fill lowest orbitals first (ground state approximation)
            for orbital in range(min(n_electrons, max_electrons_per_dot)):
                site = dot_idx * max_electrons_per_dot + orbital
                index += 2 ** (n_orbitals - 1 - site)
        return index
    
    # Get Fock indices for our charge states
    fock_indices = jnp.array([charge_state_to_fock_index(state) for state in charge_states])
    
    # Project Hamiltonian to charge state subspace
    H_projected = H_fock_array[jnp.ix_(fock_indices, fock_indices)]
    
    return H_projected


def compute_tunneling_dynamiqs_optimized(tc_matrix: jnp.ndarray, charge_states: jnp.ndarray,
                                        max_electrons_per_dot: int = 4) -> jnp.ndarray:
    """
    Optimized dynamiqs implementation using sparse operators and JAX operations.
    
    This implementation:
    - Uses dynamiqs for operator construction 
    - Leverages sparsity for efficiency
    - Remains fully JAX-compatible
    - Works with reduced Hilbert space when possible
    
    Args:
        tc_matrix: Tunnel coupling matrix
        charge_states: Array of charge states, shape (M, n_dots)
        max_electrons_per_dot: Maximum electrons per dot
        
    Returns:
        Tunneling Hamiltonian matrix in charge state basis, shape (M, M)
    """
    if not DYNAMIQS_AVAILABLE:
        print("dynamiqs not available, falling back to JAX implementation")
        return compute_tunneling_matrix_vectorized(tc_matrix, charge_states, max_electrons_per_dot)
    
    n_dots = charge_states.shape[1]
    n_states = charge_states.shape[0]
    
    # For small systems, use full Fock space approach
    if n_dots <= 2 and max_electrons_per_dot <= 2:
        return compute_tunneling_dynamiqs_fock(tc_matrix, charge_states, max_electrons_per_dot)
    
    # For larger systems, use direct charge state basis construction
    # This avoids the exponentially large Fock space
    H = jnp.zeros((n_states, n_states), dtype=jnp.complex64)
    
    # Create tunneling operators directly in charge state basis
    # Vectorized approach: keep dot loops but vectorize state loops

    # Create all state pairs for vectorized operations
    states_i = charge_states[:, None, :]  # (n_states, 1, n_dots)
    states_j = charge_states[None, :, :]  # (1, n_states, n_dots)

    # Initialize Hamiltonian contributions
    H_total = jnp.zeros((n_states, n_states))

    # For each dot pair, vectorize over all state pairs
    for dot_from in range(n_dots):
        for dot_to in range(n_dots):
            if dot_from != dot_to:
                t = tc_matrix[dot_from, dot_to]

                # Vectorized transition check
                state_diff = states_j - states_i  # (n_states, n_states, n_dots)
                expected_diff = jnp.zeros(n_dots).at[dot_from].set(-1).at[dot_to].set(1)
                is_valid_transition = jnp.all(state_diff == expected_diff, axis=2)

                # Vectorized occupancy check
                can_hop_from = states_i[:, 0, dot_from] > 0  # (n_states,)
                can_hop_to = states_i[:, 0, dot_to] < max_electrons_per_dot  # (n_states,)
                valid_hops = is_valid_transition & can_hop_from[:, None] & can_hop_to[:, None]

                # Vectorized matrix elements
                n_from = states_i[:, 0, dot_from]  # (n_states,)
                n_to = states_i[:, 0, dot_to]     # (n_states,)
                matrix_elements = -t * jnp.sqrt(n_from[:, None] * (n_to[:, None] + 1))

                H_total += jnp.where(valid_hops, matrix_elements, 0.0)

    H = H_total
    
    return H


def choose_hamiltonian_convention(convention: str = "fermionic_negative"):
    """
    Choose which Hamiltonian convention to use.
    
    Options:
    - "simple": QDarts formalism (no sqrt factors, positive sign)
    - "fermionic_positive": With sqrt factors, positive sign  
    - "fermionic_negative": With sqrt factors, negative sign (current JAX implementation)
    - "dynamiqs_fock": Using dynamiqs with full Fock space and fermionic operators
    - "dynamiqs_optimized": Optimized dynamiqs implementation (auto-selects best approach)
    
    Returns:
        The appropriate Hamiltonian function
    """
    if convention == "simple":
        return compute_tunneling_simple_operators
    elif convention == "fermionic_positive":
        return compute_tunneling_fermionic_positive
    elif convention == "fermionic_negative":
        return compute_tunneling_matrix_vectorized
    elif convention == "dynamiqs_fock":
        return compute_tunneling_dynamiqs_fock
    elif convention == "dynamiqs_optimized":
        return compute_tunneling_dynamiqs_optimized
    else:
        raise ValueError(f"Unknown convention: {convention}")


def convert_free_energy_into_hamiltionian_form(F, charge_states):
    """
    Convert free energy values into Hamiltonian matrix form.
    
    Parameters:
    -----------
    F : jnp.ndarray
        Free energies, shape (..., M)
    charge_states : jnp.ndarray
        Charge states, shape (M, n_dot)
        
    Returns:
    --------
    jnp.ndarray
        Diagonal Hamiltonian matrices, shape (..., M, M)
    """
    # handle batched case where we extract the charge states prior to computing the hamiltonian
    if charge_states.ndim == 2:
        M = charge_states.shape[0]
    elif charge_states.ndim == 3:
        M = charge_states.shape[1]

    diag_mask = jnp.eye(M)
    return F[..., :, None] * diag_mask

