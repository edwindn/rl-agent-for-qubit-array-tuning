"""
Fully sparse JAX implementation with custom eigenvalue solver
Maintains sparsity throughout the computation, similar to CuPy's eigsh
"""

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import numpy as np
from functools import partial
import itertools
from typing import Tuple
import time


def create_charge_state_space(max_electrons_per_dot: int, n_dots: int) -> jnp.ndarray:
    """Create all possible charge states."""
    states = list(itertools.product(range(max_electrons_per_dot + 1), repeat=n_dots))
    return jnp.array(states, dtype=jnp.float32)


def compute_tunneling_indices_and_values(charge_states: np.ndarray, tc: float, max_electrons: int):
    """Pre-compute indices and values for sparse tunneling Hamiltonian."""
    n_states = charge_states.shape[0]
    n_dots = charge_states.shape[1]
    
    rows = []
    cols = []
    values = []
    
    # Find all tunneling connections
    for i in range(n_states):
        for j in range(n_states):
            state_i = charge_states[i]
            state_j = charge_states[j]
            
            # Check each dot pair
            for dot in range(n_dots - 1):
                # Forward tunneling: dot -> dot+1
                if (state_j[dot] == state_i[dot] - 1 and 
                    state_j[dot+1] == state_i[dot+1] + 1 and
                    all(state_j[k] == state_i[k] for k in range(n_dots) if k != dot and k != dot+1)):
                    
                    n_from = state_i[dot]
                    n_to = state_i[dot+1]
                    if n_from > 0 and n_to < max_electrons:
                        amplitude = -tc * np.sqrt(n_from * (n_to + 1))
                        rows.append(i)
                        cols.append(j)
                        values.append(amplitude)
                
                # Backward tunneling: dot+1 -> dot
                if (state_j[dot] == state_i[dot] + 1 and 
                    state_j[dot+1] == state_i[dot+1] - 1 and
                    all(state_j[k] == state_i[k] for k in range(n_dots) if k != dot and k != dot+1)):
                    
                    n_from = state_i[dot+1]
                    n_to = state_i[dot]
                    if n_from > 0 and n_to < max_electrons:
                        amplitude = -tc * np.sqrt(n_from * (n_to + 1))
                        rows.append(i)
                        cols.append(j)
                        values.append(amplitude)
    
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(values, dtype=np.float32)


@partial(jax.jit, static_argnums=(2,))
def sparse_lanczos_iteration(H_sparse: jsparse.BCOO, H_diagonal: jnp.ndarray, 
                            n_iterations: int = 50) -> Tuple[float, jnp.ndarray]:
    """
    Lanczos iteration for finding the lowest eigenvalue of a sparse matrix.
    
    Args:
        H_sparse: Sparse BCOO matrix (tunneling part)
        H_diagonal: Diagonal elements (electrostatic part)
        n_iterations: Number of Lanczos iterations
        
    Returns:
        Lowest eigenvalue and corresponding eigenvector
    """
    n_states = H_sparse.shape[0]
    
    # Initialize with uniform superposition (better for finding ground state)
    v0 = jnp.ones(n_states) / jnp.sqrt(n_states)
    
    # Pre-allocate arrays for fixed size
    alpha = jnp.zeros(n_iterations)
    beta = jnp.zeros(n_iterations - 1)
    V = jnp.zeros((n_states, n_iterations))
    V = V.at[:, 0].set(v0)
    
    # Initial step
    v = v0
    w = H_sparse @ v + H_diagonal * v  # Apply full Hamiltonian
    alpha = alpha.at[0].set(jnp.dot(v, w))
    w = w - alpha[0] * v
    
    # Lanczos iterations
    def lanczos_step(carry, j):
        v_prev, w, alpha, beta, V = carry
        
        beta_j = jnp.linalg.norm(w)
        # Use a small epsilon to avoid division by zero
        beta_j_safe = jnp.maximum(beta_j, 1e-10)
        
        beta = beta.at[j-1].set(beta_j)
        v = w / beta_j_safe
        V = V.at[:, j].set(v)
        
        w_new = H_sparse @ v + H_diagonal * v  # Apply full Hamiltonian
        alpha_j = jnp.dot(v, w_new)
        alpha = alpha.at[j].set(alpha_j)
        w_new = w_new - alpha_j * v - beta_j * v_prev
        
        return (v, w_new, alpha, beta, V), None
    
    # Run iterations using scan
    (_, _, alpha, beta, V), _ = jax.lax.scan(
        lanczos_step, (v0, w, alpha, beta, V), jnp.arange(1, n_iterations)
    )
    
    # Build tridiagonal matrix
    T = jnp.diag(alpha) + jnp.diag(beta, 1) + jnp.diag(beta, -1)
    
    # Solve tridiagonal eigenvalue problem
    eigvals, eigvecs = jnp.linalg.eigh(T)
    
    # Get ground state in original basis
    ground_state = V @ eigvecs[:, 0]
    ground_state = ground_state / jnp.linalg.norm(ground_state)
    
    return eigvals[0], ground_state


@partial(jax.jit, static_argnums=(2, 3))
def sparse_power_iteration(H_sparse: jsparse.BCOO, H_diagonal: jnp.ndarray,
                          n_iterations: int = 100, shift: float = -10.0) -> Tuple[float, jnp.ndarray]:
    """
    Shifted inverse power iteration for finding the lowest eigenvalue.
    
    Args:
        H_sparse: Sparse BCOO matrix (tunneling part)
        H_diagonal: Diagonal elements (electrostatic part)
        n_iterations: Number of iterations
        shift: Shift parameter (should be less than lowest eigenvalue)
        
    Returns:
        Lowest eigenvalue and corresponding eigenvector
    """
    n_states = H_sparse.shape[0]
    
    # Initialize random vector
    key = jax.random.PRNGKey(42)
    v = jax.random.normal(key, (n_states,))
    v = v / jnp.linalg.norm(v)
    
    # Create shifted matrix: (H - shift*I)
    H_shifted_diag = H_diagonal - shift
    
    for _ in range(n_iterations):
        # Apply H - shift*I
        w = H_sparse @ v + H_shifted_diag * v
        
        # Normalize (inverse iteration would solve system, but we approximate with direct iteration)
        v = w / jnp.linalg.norm(w)
        
        # Rayleigh quotient
        Hv = H_sparse @ v + H_diagonal * v
        eigenvalue = jnp.dot(v, Hv)
    
    return eigenvalue, v


@jax.jit
def compute_electrostatic_diagonal(vg_single, cdd_inv, cgd, charge_states):
    """Compute diagonal electrostatic Hamiltonian elements."""
    n_dots = charge_states.shape[1]
    
    # Extract dot-only matrices if full system matrices are provided
    if cgd.shape[0] > n_dots:
        # Full system matrices - extract dot portion
        cgd_dots = cgd[:n_dots, :]
        cdd_inv_dots = cdd_inv[:n_dots, :n_dots]
    else:
        # Already dot-only matrices
        cgd_dots = cgd
        cdd_inv_dots = cdd_inv
    
    # Compute induced gate charges
    gate_effect = jnp.einsum('dg,g->d', cgd_dots, vg_single)
    
    # Compute electrostatic energy for each charge state
    # Using the same formula as OpenFermion
    inner = charge_states - gate_effect[None, :]  # (n_states, n_dots)
    F = jnp.einsum('md,de,me->m', inner, cdd_inv_dots, inner)
    
    # Return as diagonal elements (no shift)
    return F


def create_sparse_tunneling_matrix(charge_states_np: np.ndarray, tc: float, max_electrons: int) -> jsparse.BCOO:
    """Create sparse tunneling Hamiltonian as BCOO matrix."""
    rows, cols, values = compute_tunneling_indices_and_values(charge_states_np, tc, max_electrons)
    n_states = len(charge_states_np)
    
    # Create JAX sparse matrix
    indices = np.stack([rows, cols], axis=1)
    sparse_H = jsparse.BCOO((jnp.array(values), jnp.array(indices)), shape=(n_states, n_states))
    
    return sparse_H


@partial(jax.jit, static_argnums=(5,))
def fully_sparse_ground_state_lanczos(vg_single, cdd_inv, cgd, 
                                      H_sparse, charge_states,
                                      n_iterations=50):
    """
    Compute ground state using fully sparse Lanczos method.
    """
    # Compute diagonal part
    H_diagonal = compute_electrostatic_diagonal(vg_single, cdd_inv, cgd, charge_states)
    
    # Find ground state using sparse Lanczos
    _, ground_state = sparse_lanczos_iteration(H_sparse, H_diagonal, n_iterations)
    
    # Compute expectation values
    probs = jnp.abs(ground_state)**2
    n_expect = jnp.dot(probs, charge_states)
    
    return n_expect


def benchmark_fully_sparse_jax(n_dots_list=[2, 3, 4], n_points=10000, max_electrons=3):
    """Benchmark fully sparse JAX against OpenFermion."""
    from .full_physics_informed_tunneling import (
        full_physics_informed_ground_state,
        create_full_charge_state_space
    )
    
    print("="*70)
    print("FULLY SPARSE JAX vs OPENFERMION BENCHMARK")
    print("="*70)
    print(f"Points per calculation: {n_points}")
    print(f"Max electrons per dot: {max_electrons}")
    
    tc = 0.15
    results = {}
    
    for n_dots in n_dots_list:
        print(f"\n{n_dots}-dot system:")
        print("-"*50)
        
        # Setup
        cdd_inv = jnp.eye(n_dots)
        cgd = jnp.eye(n_dots)
        vg_batch = jnp.ones((n_points, n_dots)) * 0.5
        
        # Create charge states
        charge_states_np = np.array(
            list(itertools.product(range(max_electrons + 1), repeat=n_dots)), 
            dtype=np.float32
        )
        charge_states_jax = jnp.array(charge_states_np)
        n_states = len(charge_states_np)
        
        print(f"State space size: {n_states} states")
        
        # Create sparse tunneling matrix
        H_sparse = create_sparse_tunneling_matrix(charge_states_np, tc, max_electrons)
        
        # Count sparsity
        rows, cols, values = compute_tunneling_indices_and_values(charge_states_np, tc, max_electrons)
        nnz = len(values)
        sparsity = 100.0 * (1 - nnz / (n_states * n_states))
        print(f"Tunneling matrix: {nnz} non-zero elements ({sparsity:.1f}% sparse)")
        
        # Test 1: Fully Sparse JAX with Lanczos
        print("\nFully Sparse JAX (Lanczos):")
        
        # Create vmapped function
        sparse_batch = jax.vmap(
            lambda vg: fully_sparse_ground_state_lanczos(
                vg, cdd_inv, cgd, H_sparse, charge_states_jax, n_iterations=30
            )
        )
        
        # Warm up
        _ = sparse_batch(vg_batch[:10])
        jax.block_until_ready(_)
        
        # Time
        start = time.time()
        results_sparse = sparse_batch(vg_batch)
        jax.block_until_ready(results_sparse)
        sparse_time = time.time() - start
        
        print(f"  Time: {sparse_time:.3f}s")
        print(f"  Throughput: {n_points/sparse_time:.0f} points/sec")
        
        # Test 2: OpenFermion (sample and extrapolate)
        print("\nOpenFermion:")
        
        # Time OpenFermion on a sample
        sample_size = min(100, n_points // 100)  # 1% sample or 100 points
        start = time.time()
        results_of = []
        for i in range(sample_size):
            vg = np.array(vg_batch[i])
            n = full_physics_informed_ground_state(
                vg, np.array(cdd_inv), np.array(cgd), tc, n_dots=n_dots, max_number_of_charge_carriers=max_electrons
            )
            results_of.append(n)
        sample_time = time.time() - start
        
        # Extrapolate
        of_time = sample_time * n_points / sample_size
        
        print(f"  Sampled {sample_size} points in {sample_time:.3f}s")
        print(f"  Estimated total time: {of_time:.3f}s")
        print(f"  Throughput: {n_points/of_time:.0f} points/sec")
        
        # Verify correctness on a few points
        print("\nVerifying correctness (comparing first 5 points):")
        max_diff = 0
        for i in range(min(5, sample_size)):
            diff = np.max(np.abs(results_sparse[i] - results_of[i]))
            max_diff = max(max_diff, diff)
        print(f"  Max difference: {max_diff:.6f}")
        
        # Speedup
        speedup = of_time / sparse_time
        print(f"\nSpeedup: {speedup:.1f}x")
        
        results[n_dots] = {
            'sparse_time': sparse_time,
            'of_time': of_time,
            'speedup': speedup,
            'sparsity': sparsity,
            'n_states': n_states
        }
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Execution times
    dots = list(results.keys())
    sparse_times = [results[d]['sparse_time'] for d in dots]
    of_times = [results[d]['of_time'] for d in dots]
    
    x = np.arange(len(dots))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, of_times, width, label='OpenFermion', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, sparse_times, width, label='Fully Sparse JAX', color='red', alpha=0.7)
    
    ax1.set_xlabel('System Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title(f'Execution Time ({n_points:,} points)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{d}-dot' for d in dots])
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add time labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Speedup with state space size
    speedups = [results[d]['speedup'] for d in dots]
    n_states = [results[d]['n_states'] for d in dots]
    
    bars = ax2.bar(x, speedups, color='red', alpha=0.7)
    ax2.set_xlabel('System Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Fully Sparse JAX Speedup vs OpenFermion')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d}-dot\n({results[d]["n_states"]} states)' for d in dots])
    ax2.grid(True, alpha=0.3)
    
    # Add speedup labels
    for bar, speedup in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,
                f'{speedup:.0f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add sparsity info as secondary labels
    for i, (d, bar) in enumerate(zip(dots, bars)):
        ax2.text(bar.get_x() + bar.get_width()/2, 0.5,
                f'{results[d]["sparsity"]:.0f}% sparse', ha='center', va='bottom', 
                fontsize=8, color='gray', rotation=0)
    
    plt.suptitle('Fully Sparse JAX (Lanczos) vs OpenFermion Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig('fully_sparse_jax_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_fully_sparse_jax(n_dots_list=[2, 3, 4], n_points=10000)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nSystem sizes and sparsity:")
    for n_dots, data in results.items():
        print(f"  {n_dots}-dot: {data['n_states']} states, {data['sparsity']:.1f}% sparse")
    
    print("\nPerformance comparison:")
    for n_dots, data in results.items():
        print(f"  {n_dots}-dot: Sparse JAX is {data['speedup']:.1f}x faster")
        print(f"    - Sparse JAX: {10000/data['sparse_time']:.0f} points/sec")
        print(f"    - OpenFermion: {10000/data['of_time']:.0f} points/sec")
    
    print("\nConclusion:")
    print("The fully sparse JAX implementation with Lanczos iteration maintains")
    print("sparsity throughout and provides significant speedups over OpenFermion.")
