import jax
import jax.numpy as jnp
from all_xy_sequence import ALLXY_IDEAL

# Maximum possible deviation from each target level.
_targets = jnp.array(ALLXY_IDEAL)                   # (21,)
_max_devs = jnp.where(_targets == 0.5, 0.5, 1.0)    # (21,)


@jax.jit
def allxy_rewards(P1):
    """Batched reward over all qubits.

    Args:
        P1: (N_QUBITS, N_ALLXY) staircase results.

    Returns:
        rewards:    (N_QUBITS,) reward per qubit.
        deviations: (N_QUBITS, N_ALLXY) per-sequence normalised deviations.
    """
    deviations = jnp.abs(P1 - _targets[None]) / _max_devs[None]
    rewards    = 1 - jnp.mean(deviations, axis=1)
    return rewards, deviations