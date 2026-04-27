import json
import jax
import jax.numpy as jnp
from pathlib import Path

"""
Base parameters for operating N_QUBITS transmon qubits.
Device Selection and System Parameters should be fixed at all time
Per-qubit Parameters and Cross-talk Matrix vary between devices (episodes)
Drive/Pulse/Envelope Parameters are updated at each tuning step

Natural units are assumed, i.e. hbar = 1, time is in ns, angular frequencies in rad/ns

Ranges for all parameters are defined in parameter_config.json.
Frequencies with the _GHz suffix are multiplied by 2π on load (rad/ns).
"""

_cfg = json.loads((Path(__file__).parent / "parameter_config.json").read_text())

# ----- System Parameters ----- #
N_QUBITS = _cfg["system"]["N_QUBITS"]
N        = _cfg["system"]["N"]


def _u(key, r, shape=()):
    """Sample uniformly from range dict {"min": lo, "max": hi}."""
    return jax.random.uniform(key, shape, minval=r["min"], maxval=r["max"])


def sample_params(rng_key):
    """Sample all episode parameters (qubit, crosstalk, pulse) from uniform ranges.

    Returns:
        omega_01 : (N_QUBITS,)  bare |0>→|1> transition frequency [rad/ns]  — agent-tunable
        alpha    : (N_QUBITS,)  anharmonicity [rad/ns]
        lambda_  : (N_QUBITS, N_QUBITS)  flux cross-talk matrix, diagonal zero
        t_g      : scalar  gate duration [ns]
        omega_d  : (N_QUBITS,)  drive frequency [rad/ns]
        phi      : (N_QUBITS,)  drive phase [rad]
        Omega    : (N_QUBITS,)  π-pulse amplitude [rad/ns], factor 2 appears due to cosine envelope having half area
        beta     : (N_QUBITS,)  DRAG coefficient [ns]
    """
    q, c, p = _cfg["qubit"], _cfg["crosstalk"], _cfg["pulse"]
    keys = jax.random.split(rng_key, 8)

    omega_01 = 2*jnp.pi * _u(keys[0], q["omega_01_GHz"], (N_QUBITS,))
    alpha    = 2*jnp.pi * _u(keys[1], q["alpha_GHz"],    (N_QUBITS,))
    lambda_  = _u(keys[2], c["lambda_"], (N_QUBITS, N_QUBITS)) * (1 - jnp.eye(N_QUBITS))  # masked, diagonal = 0
    t_g      = _u(keys[3], p["t_g"])
    omega_d  = omega_01 + 2*jnp.pi * jax.random.normal(keys[4], (N_QUBITS,)) * p["omega_d_GHz"]["sigma_GHz"]
    phi      = _u(keys[5], p["phi"],   (N_QUBITS,))
    Omega    = (2*jnp.pi / t_g) * (1 + jax.random.normal(keys[6], (N_QUBITS,)) * p["Omega"]["sigma_frac"])
    beta     = _u(keys[7], p["beta"],  (N_QUBITS,))

    return omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta


def sample_hardware(rng_key):
    """Sample per-qubit hardware imperfections from uniform ranges.

    Returns:
        hw: (N_QUBITS, 3) array with columns [phi_hw, t_delay, Omega_scale].
    """
    h = _cfg["hardware"]
    keys = jax.random.split(rng_key, 3)

    phi_hw      = _u(keys[0], h["phi_hw"],      (N_QUBITS,))
    t_delay     = _u(keys[1], h["t_delay"],     (N_QUBITS,))
    Omega_scale = _u(keys[2], h["Omega_scale"], (N_QUBITS,))

    return jnp.column_stack([phi_hw, t_delay, Omega_scale])


def sample_all(rng_key):
    """Sample all episode and hardware parameters from a single key.

    Returns:
        Same as sample_episode plus hw: (N_QUBITS, 3) [phi_hw, t_delay, Omega_scale].
    """
    ep_key, hw_key = jax.random.split(rng_key)
    return (*sample_params(ep_key), sample_hardware(hw_key))


omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw = sample_all(jax.random.PRNGKey(0))
C_tensor = jnp.einsum('ij,kl->ikjl', jnp.eye(N_QUBITS), jnp.eye(5))  # Compensation matrix, block-identity init.


# Note: if CUDA-enabled jax is not installed do the following in your env:
# pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda13-plugin jax-cuda12-pjrt jax-cuda13-pjrt -y
# pip install "jax[cuda12]<0.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Higher jax versions are not compatible with dynamiqs