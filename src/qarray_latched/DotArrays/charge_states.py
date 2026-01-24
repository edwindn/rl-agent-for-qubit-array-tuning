import jax.numpy as jnp
import jax
from functools import partial

def create_full_charge_state_space(max_electrons_per_dot: int = 4, n_dot: int = 2) -> jnp.ndarray:
    """
    Create all possible charge state combinations for n quantum dots.
    
    Args:
        max_electrons_per_dot: Maximum electrons per dot (default 4, giving 5 options: 0,1,2,3,4)
        n_dot: Number of quantum dots
        
    Returns:
        Array of shape (num_states, n_dot) containing all possible charge state combinations
    """
    # More efficient JAX-native implementation
    # Total number of possible states
    num_states = (max_electrons_per_dot + 1) ** n_dot
    base = max_electrons_per_dot + 1
    
    # Create state indices (0 to num_states-1)
    state_indices = jnp.arange(num_states)
    
    # Vectorized base conversion using broadcasting
    # Create powers for each digit position: [base^(n_dot-1), base^(n_dot-2), ..., base^1, base^0]
    powers = base ** jnp.arange(n_dot - 1, -1, -1)
    
    # Extract all digits at once using vectorized operations
    # state_indices[:, None] creates shape (num_states, 1)
    # powers[None, :] creates shape (1, n_dot)  
    # // and % operations broadcast to (num_states, n_dot)
    states = (state_indices[:, None] // powers[None, :]) % base
    
    return states.astype(jnp.int32)

@partial(jax.jit, static_argnums=(3,))
def compute_continuous_ground_state_open(v_extended: jnp.ndarray,
                                         cdd_inv: jnp.ndarray,
                                         cgd: jnp.ndarray,
                                         n_dot: int
                                         ) -> jnp.ndarray:
    """
    Computes the continuous ground state for a single set of voltages and their corresponding capacitance matrices.
    Identical to the single-step solver in https://github.com/pranavjv/QArray/blob/voltage_dependence/qarray/jax_implementations/voltage_dependent_capacitance.py
    (as of 11/09/25)
    
    Parameters:
    -----------
    v_extended : jnp.ndarray
        Extended voltage vector [gates, barriers]
    cdd_inv : jnp.ndarray
        Inverse capacitance matrix (full system)
    cgd : jnp.ndarray
        Gate-dot capacitance matrix (full system)
    n_dot : int
        Number of dots (to extract dot-only portion)
    """
    # Extract only dot effects from full system matrices
    n_continuous = cgd[:n_dot, :] @ v_extended

    def analytical_valid():
        return n_continuous

    def numerical_solution():
        n = jnp.clip(cgd[:n_dot, :] @ v_extended, 0, None)
        lr = 0.1
        n_iter = 50

        def step(i, n):
            grad = cdd_inv[:n_dot, :n_dot] @ n - cdd_inv[:n_dot, :n_dot] @ (cgd[:n_dot, :] @ v_extended)
            n_new = n - lr * grad
            n_new = jnp.clip(n_new, 0, None)
            return n_new
        
        n_final = jax.lax.fori_loop(0, n_iter, step, n)
        return n_final

    n_continuous = jax.lax.cond(
        jnp.all(n_continuous >= 0),
        analytical_valid,
        numerical_solution
    )

    n_continuous = jnp.clip(n_continuous, 0, None)
    return n_continuous



@partial(jax.jit, static_argnums=(3, 4))
def _jit_extract_charge_state_candidates(v_extended, cdd_inv, cgd, num_states, n_dot):
    """
    Extracts a set of candidate charge states around the continuous ground state and selects those with the lowest energy.
    """

    n_continuous = compute_continuous_ground_state_open(v_extended, cdd_inv, cgd, n_dot)

    floor_values = jnp.floor(n_continuous)

    # create enough charge states to cover num_states
    possible_deltas = [-1, 0, 1, 2]

    n_deltas = len(possible_deltas)
    delta_array = jnp.array(possible_deltas)
    
    # Create meshgrid for all combinations of deltas across all dots
    args = [delta_array] * n_dot
    number_of_configurations = n_deltas ** n_dot
    delta_combinations = jnp.stack(jnp.meshgrid(*args, indexing='ij'), axis=-1).reshape(number_of_configurations, n_dot)

    # Add floor values to delta combinations to get charge configurations
    charge_configurations = delta_combinations + floor_values
    
    # Filter out any combinations with negative charge (JAX-compatible approach)
    valid_mask = jnp.all(charge_configurations >= 0, axis=-1)  # (n_configurations,)
    
    # Calculate energy for all configurations, but set invalid ones to high energy
    v_dash = cgd[:n_dot, :] @ v_extended
    F = jnp.einsum('...i, ij, ...j', charge_configurations - v_dash, cdd_inv[:n_dot, :n_dot], charge_configurations - v_dash)
    
    # Set energy of invalid configurations to a very high value so they're never selected
    F = jnp.where(valid_mask, F, jnp.inf)
    
    # Select the num_states states with the lowest energy (invalid ones will have inf energy)
    lowest_energy_indices = jnp.argsort(F)[:num_states]
    charge_states = charge_configurations[lowest_energy_indices]
    
    return charge_states.astype(int), n_continuous # (num_states, n_dot)



@partial(jax.jit, static_argnums=(3, 4, 5))
def _jit_extract_charge_state_candidates_memory_optimized(v_extended, cdd_inv, cgd, num_states, n_dot, chunk_size=1024):
    """
    Memory-optimized version for 8-dot systems using chunked energy computation.
    Generates all charge state combinations efficiently and computes energies in chunks.
    """

    n_continuous = compute_continuous_ground_state_open(v_extended, cdd_inv, cgd, n_dot)
    floor_values = jnp.floor(n_continuous)

    possible_deltas = [-1, 0, 1, 2] # Approximate delta range for ~98% coverage
    n_deltas = len(possible_deltas)
    delta_array = jnp.array(possible_deltas)
    
    # Calculate total combinations but don't store them all
    total_combinations = n_deltas ** n_dot  # 4^8 = 65,536 combinations
    n_chunks = (total_combinations + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Initialize tracking of best candidates
    v_dash = cgd[:n_dot, :] @ v_extended
    best_energies = jnp.full(num_states, jnp.inf)
    best_states = jnp.zeros((num_states, n_dot))
    
    # print(f'Total combinations: {total_combinations}, Chunk size: {chunk_size}, Chunks: {n_chunks}')
    
    def process_chunk(carry, chunk_idx):
        best_energies_curr, best_states_curr = carry
        
        # Generate indices for this chunk using dynamic operations
        start_idx = chunk_idx * chunk_size
        chunk_base_indices = jnp.arange(chunk_size) + start_idx
        
        # Handle out-of-bounds by using modulo and masking
        within_bounds = chunk_base_indices < total_combinations
        safe_indices = chunk_base_indices % total_combinations  # Safe from out-of-bounds
        
        # Convert indices to base-n_deltas representation for this chunk
        delta_indices = jnp.zeros((chunk_size, n_dot), dtype=jnp.int32)
        
        # Vectorized base conversion
        temp_indices = safe_indices
        for i in range(n_dot):
            delta_indices = delta_indices.at[:, n_dot-1-i].set(temp_indices % n_deltas)
            temp_indices = temp_indices // n_deltas
        
        # Convert to actual delta values and add floor values
        delta_values = delta_array[delta_indices]
        chunk_configs = delta_values + floor_values
        
        # Check validity (negative charges invalid, out-of-bounds also invalid)
        valid_charges = jnp.all(chunk_configs >= 0, axis=-1)
        chunk_valid_mask = within_bounds & valid_charges
        
        # Compute energies for this chunk
        chunk_energies = jnp.einsum('...i, ij, ...j', 
                                   chunk_configs - v_dash, 
                                   cdd_inv[:n_dot, :n_dot], 
                                   chunk_configs - v_dash)
        
        # Set invalid configurations to infinite energy
        chunk_energies = jnp.where(chunk_valid_mask, chunk_energies, jnp.inf)
        
        # Find best from current chunk (always use num_states for fixed shape)
        chunk_best_indices = jnp.argsort(chunk_energies)[:num_states]
        chunk_best_energies = chunk_energies[chunk_best_indices]
        chunk_best_states = chunk_configs[chunk_best_indices]
        
        # Combine with previous best
        combined_energies = jnp.concatenate([best_energies_curr, chunk_best_energies])
        combined_states = jnp.concatenate([best_states_curr, chunk_best_states], axis=0)
        
        # Select overall best
        final_best_indices = jnp.argsort(combined_energies)[:num_states]
        new_best_energies = combined_energies[final_best_indices]
        new_best_states = combined_states[final_best_indices]
        
        return (new_best_energies, new_best_states), None
    
    # Process all chunks
    chunk_indices = jnp.arange(n_chunks)
    (final_energies, final_states), _ = jax.lax.scan(
        process_chunk,
        (best_energies, best_states),
        chunk_indices
    )
    
    return final_states.astype(jnp.int32), n_continuous



def build_charge_states(truncate = False, num_charge_states = None, v_extended = None, cdd_inv_batch = None, cgd_batch = None, batchsize = None, max_charge_carriers=None, n_dot=None):

    #Truncates the charge states considered by electrostatic free energy
    if truncate:
        #For large dots the full charge state space is too large to hold in RAM
        if batchsize is not None:
            extract_charge_state_candidates_vmap = partial(_jit_extract_charge_state_candidates_memory_optimized, num_states=num_charge_states, n_dot=n_dot, chunk_size=batchsize)
        else:
            extract_charge_state_candidates_vmap = partial(_jit_extract_charge_state_candidates, num_states=num_charge_states, n_dot=n_dot)

        charge_states, ground_states = jax.vmap(
            extract_charge_state_candidates_vmap,
            in_axes=(0, 0, 0),
            out_axes=(0, 0)
            )(v_extended, cdd_inv_batch, cgd_batch)


    #Blindly selects all charge states
    else:
        charge_states = jnp.array(create_full_charge_state_space(max_charge_carriers, n_dot))
    
    return charge_states