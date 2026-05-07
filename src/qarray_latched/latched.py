
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from qarray import GateVoltageComposer, charge_state_to_scalar, optimal_Vg

# ------------------------------------------------------------------------------
# Initialization: Capacitance matrices and virtual gate setup
# ------------------------------------------------------------------------------

# Define the capacitance matrices
cdd = np.array([[1, -0.2], [-0.2, 1]])  # Dot-dot capacitance
cgd = -np.array([[1, 0.1], [0.1, 1]])    # Gate-dot capacitance

# Inverse of dot-dot capacitance matrix
cdd_inv = np.linalg.inv(cdd)

# Setup virtual gate transformation
voltage_composer = GateVoltageComposer(n_gate=2, n_dot=2)
voltage_composer.virtual_gate_matrix = -np.linalg.pinv(cdd_inv @ cgd)
voltage_composer.virtual_gate_origin = np.zeros(2)

# Define scan range in virtual gate space
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Generate 2D virtual gate grid
virtual_gate_grid = voltage_composer.do2d('P1', x_min, x_max, 100,
                                          'P2', y_min, y_max, 100)

# Add optimal gate voltage offset
virtual_gate_grid += optimal_Vg(cdd_inv, -cgd, np.array([1, 1]))

# Generate all combinations of charge states (0 to 3 electrons per dot)
charge_states = jnp.stack(
    jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing='ij'), axis=-1
).reshape(-1, 2)

# ------------------------------------------------------------------------------
# Energy Computations
# ------------------------------------------------------------------------------

@jax.jit
def free_energy_grid(vg_flat, charge_states, cdd_inv, cgd):
    """Compute the electrostatic free energy of each charge state at each point."""
    v_dash = vg_flat @ cgd.T
    delta = charge_states[None, :, :] - v_dash[:, None, :]
    return jnp.einsum('nij,jk,nik->ni', delta, cdd_inv, delta)

@jax.jit
def single_energy(vg, n, cdd_inv, cgd):
    """Compute energy of a single charge state configuration."""
    v_dash = cgd @ vg
    delta = n - v_dash
    return delta @ cdd_inv @ delta

# ------------------------------------------------------------------------------
# Latched charge stability diagram simulation
# ------------------------------------------------------------------------------

@jax.jit
def latched_charge_stability_diagram(vg, cdd_inv, cgd, px=0.5, py=1.0, p_psb=0.5, N_charges = 5, seed=0):
    """
    Simulates the latched charge stability diagram.
    Includes probabilistic transitions and Pauli spin blockade behavior.
    Only accepts transitions to states with lower energy.
    """
    key = jax.random.PRNGKey(seed)
    nx, ny = vg.shape[:2]
    N = nx * ny
    vg_flat = vg.reshape(-1, 2)

    # Compute energy grid for all charge states
    energy = free_energy_grid(vg_flat, charge_states, cdd_inv, cgd)
    unlatched_states = charge_states[jnp.argmin(energy, axis=1)]

    def body_fn(carry, i):
        key, prev_state, latched_out = carry
        e_row = energy[i]
        args = jnp.argsort(e_row)
        candidate_0 = charge_states[args[0]]

        # Find index of previous state in charge_states to get its energy
        prev_energy_idx = jnp.argmax(jnp.all(charge_states == prev_state, axis=1))
        prev_energy = e_row[prev_energy_idx]

        def scan_loop(val):
            j, (accepted, best_state, key) = val
            candidate = charge_states[args[j]]
            delta = candidate - prev_state

            candidate_energy = e_row[args[j]]

            # Require energy to decrease
            energy_ok = candidate_energy < prev_energy

            is_same = jnp.all(delta == jnp.array([0, 0]))
            is_dot1 = jnp.all(jnp.abs(delta) == jnp.array([1, 0]))
            is_dot2 = jnp.all(jnp.abs(delta) == jnp.array([0, 1]))
            is_interdot = jnp.all(delta == jnp.array([1, -1])) | jnp.all(delta == jnp.array([-1, 1]))

            psb_transitions = jnp.array([
                [[1, 1], [2, 0]],
                [[1, 1], [0, 2]],
                [[3, 1], [2, 2]],
                [[3, 1], [4, 0]],
                [[1, 3], [2, 2]],
                [[1, 3], [0, 4]],
            ])

            def transition_case(prob):
                key1, subkey = jax.random.split(key)
                accept = jax.random.uniform(subkey) < prob
                return jax.lax.cond(
                    accept, lambda: (True, candidate, key1),
                    lambda: (False, prev_state, key1)
                )

            def interdot_case():
                is_psb = jnp.any(
                    jnp.logical_and(
                        jnp.all(psb_transitions[:, 0] == prev_state, axis=1),
                        jnp.all(psb_transitions[:, 1] == candidate, axis=1)
                    )
                )
                key1, subkey = jax.random.split(key)
                accept = jax.lax.cond(is_psb, lambda: jax.random.uniform(subkey) < p_psb, lambda: True)
                return jax.lax.cond(
                    accept,
                    lambda: (True, candidate, key1),
                    lambda: (False, prev_state, key1)
                )

            def choose():
                return jax.lax.cond(is_same, lambda: (True, candidate, key),
                        lambda: jax.lax.cond(is_dot1, lambda: transition_case(px),
                        lambda: jax.lax.cond(is_dot2, lambda: transition_case(py),
                        lambda: jax.lax.cond(is_interdot, interdot_case,
                                             lambda: (False, prev_state, key)))))

            result = jax.lax.cond(energy_ok, choose, lambda: (False, prev_state, key))
            return j + 1, result

        def scan_accept():
            _, result = jax.lax.while_loop(
                lambda val: (val[0] < charge_states.shape[0]) & (~val[1][0]),
                scan_loop,
                (0, (False, prev_state, key))
            )
            return result[1], result[2]

        new_state, new_key = jax.lax.cond(i % nx == 0, lambda: (candidate_0, key), scan_accept)
        latched_out = latched_out.at[i].set(new_state)
        return (new_key, new_state, latched_out), None

    initial_state = charge_states[jnp.argmin(energy[0])]
    output_states = jnp.zeros((N, 2), dtype=jnp.int32).at[0].set(initial_state)

    (_, _, latched_states), _ = jax.lax.scan(body_fn, (key, initial_state, output_states), jnp.arange(1, N))

    return unlatched_states.reshape(nx, ny, 2), latched_states.reshape(nx, ny, 2)

# ------------------------------------------------------------------------------
# Main execution and visualization
# ------------------------------------------------------------------------------

# Run simulation with fixed probabilities
t0 = time.time()
unlatched_states, latched_states = latched_charge_stability_diagram(
    virtual_gate_grid, cdd_inv, cgd, px=1, py=1, p_psb=0.5, seed=0
)
t1 = time.time()
print(f'Computing took {t1 - t0:.3f} seconds')

# Convert charge states to scalar values for plotting
z_unlatched = charge_state_to_scalar(unlatched_states)
z_latched = charge_state_to_scalar(latched_states)



# Plot results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

ax[0].imshow(z_unlatched, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
ax[0].set_title('Unlatched')

ax[1].imshow(z_latched, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
ax[1].set_title('Latched')

# plotting the pixels where the latched state is different from the unlatched state
ax[2].imshow(z_unlatched != z_latched, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
ax[2].set_title('Latched vs Unlatched')

plt.tight_layout()
plt.show()
