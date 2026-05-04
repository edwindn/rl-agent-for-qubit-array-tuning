import json
import jax.numpy as jnp
from pathlib import Path

"""
Normalisation utilities for the 5 agent-tunable parameters.

Parameter layout (N_QUBITS, 5): [omega_01, omega_d, phi, Omega, beta]

All bounds are in simulation units (rad/ns for frequencies, rad for phase).

Bound conventions:
  omega_01 : episode-fixed; upper bound at the initialised set-point, lower
             bound _OMEGA_01_DOWN below. Flux-tunable transmons tune downward
             from the flux sweet-spot.
  omega_d  : ±_OMEGA_D_WINDOW around the current omega_01; bounds are not
             stored — clip_params derives them dynamically from omega_01.
  phi      : fixed at [_PHI_MIN, _PHI_MAX].
  Omega    : episode-specific, ±_OMEGA_BOUND_FRAC around 2π/t_g.
  beta     : fixed, read from parameter_config.json.

Typical usage per episode:
    param_mins, param_maxs = norm.episode_bounds(omega_01_init, t_g)
    delta_scales           = norm.episode_delta_scales(t_g)   # precomputed once; only Omega span varies

Typical usage per step:
    delta_raw = norm.delta_to_physical(delta_norm, delta_scales)
    params    = update_params(params, delta_raw, C_tensor, param_mins, param_maxs)
    # Physical params may exceed episode_bounds due to compensation; clip_params
    # provides a hard safety rail at CLIP_MULTIPLIER times the episode span.
"""

_2pi = 2.0 * jnp.pi

# ----- All bounds from parameter_config.json ----- #
# Init ranges (sigma_GHz, sigma_frac, min/max) and tuning ranges
# (tuning_down_GHz, window_GHz, bound_frac, min/max) live side-by-side per
# parameter — see the _comment_* fields in parameter_config.json.
_cfg = json.loads((Path(__file__).parent / "parameter_config.json").read_text())

_BETA_MIN         = _cfg["pulse"]["beta"]["min"]
_BETA_MAX         = _cfg["pulse"]["beta"]["max"]
_PHI_MIN          = _cfg["pulse"]["phi"]["min"]
_PHI_MAX          = _cfg["pulse"]["phi"]["max"]
_OMEGA_01_DOWN    = _2pi * _cfg["qubit"]["omega_01_GHz"]["tuning_down_GHz"]
_OMEGA_D_WINDOW   = _2pi * _cfg["pulse"]["omega_d_GHz"]["window_GHz"]
_OMEGA_BOUND_FRAC = _cfg["pulse"]["Omega"]["bound_frac"]


def episode_bounds(omega_01_init, t_g):
    """Compute per-qubit parameter bounds at the start of an episode.

    Fixed bounds (phi, beta) are set from module-level constants and config.
    Episode-specific bounds (omega_01, Omega) depend on the sampled values.
    omega_d bounds are not stored; clip_params handles them from first principles.

    Args:
        omega_01_init: (N_QUBITS,) initialised transition frequencies [rad/ns].
        t_g:           scalar gate duration [ns].

    Returns:
        param_mins: (N_QUBITS, 5) lower bounds in physical units.
        param_maxs: (N_QUBITS, 5) upper bounds in physical units.
    """
    N = omega_01_init.shape[0]
    Omega_opt = _2pi / t_g
    param_mins = jnp.column_stack([
        omega_01_init - _OMEGA_01_DOWN,                       # omega_01
        jnp.zeros((N,)),                                      # omega_d — unused placeholder
        jnp.full((N,), _PHI_MIN),                             # phi
        jnp.full((N,), Omega_opt * (1 - _OMEGA_BOUND_FRAC)),  # Omega
        jnp.full((N,), _BETA_MIN),                            # beta
    ])
    param_maxs = jnp.column_stack([
        omega_01_init,                                        # omega_01
        jnp.zeros((N,)),                                      # omega_d — unused placeholder
        jnp.full((N,), _PHI_MAX),                             # phi
        jnp.full((N,), Omega_opt * (1 + _OMEGA_BOUND_FRAC)),  # Omega
        jnp.full((N,), _BETA_MAX),                            # beta
    ])
    return param_mins, param_maxs


def episode_delta_scales(t_g):
    """Precompute per-parameter physical delta scales for one episode.

    Four of the five spans are universal constants; only Omega's span depends
    on t_g (which is fixed for the episode). Compute once at episode start and
    pass to delta_to_physical each step.

    Args:
        t_g: scalar gate duration [ns].

    Returns:
        (5,) array of half-spans [physical units] — one per parameter column.
    """
    return jnp.array([
        _OMEGA_01_DOWN / 2.0,         # omega_01
        _OMEGA_D_WINDOW,              # omega_d  (half of 2·window)
        (_PHI_MAX - _PHI_MIN) / 2.0,  # phi
        _OMEGA_BOUND_FRAC * _2pi / t_g,  # Omega  (half of 2·frac·2π/t_g)
        (_BETA_MAX - _BETA_MIN) / 2.0,   # beta
    ])


def delta_to_physical(delta_norm, delta_scales):
    """Convert a normalised action delta to physical units.

    A delta_norm of ±1 corresponds to one half-span (delta_scales).

    Args:
        delta_norm:   (N_QUBITS, 5) normalised action in [-1, 1].
        delta_scales: (5,) half-spans from episode_delta_scales.

    Returns:
        (N_QUBITS, 5) delta in physical units.
    """
    return delta_norm * delta_scales


def clip_params(params, param_mins, param_maxs, multiplier=2.0):
    """Clip params to relaxed safety bounds and wrap phi to [-π, π].

    episode_bounds defines the nominal tuning range; compensation can push physical
    params outside that range. Each bound is extended by (multiplier - 1) × span
    outward as a hard safety rail. phi is wrapped rather than clipped since it is
    2π-periodic. omega_d is handled dynamically from the (already clipped) omega_01.

    Args:
        params:     (N_QUBITS, 5) physical parameters after compensation update.
        param_mins: (N_QUBITS, 5) lower bounds from episode_bounds (column 1 unused).
        param_maxs: (N_QUBITS, 5) upper bounds from episode_bounds (column 1 unused).
        multiplier: each bound is extended by (multiplier - 1) × span outward.

    Returns:
        (N_QUBITS, 5) clipped/wrapped parameters in physical units.
    """
    span  = param_maxs - param_mins
    mins  = param_mins - (multiplier - 1) * span
    maxs  = param_maxs + (multiplier - 1) * span

    omega_01 = jnp.clip(params[:, 0], mins[:, 0], maxs[:, 0])
    omega_d  = jnp.clip(params[:, 1],
                        omega_01 - multiplier * _OMEGA_D_WINDOW,
                        omega_01 + multiplier * _OMEGA_D_WINDOW)
    phi   = jnp.mod(params[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
    Omega = jnp.clip(params[:, 3], mins[:, 3], maxs[:, 3])
    beta  = jnp.clip(params[:, 4], mins[:, 4], maxs[:, 4])
    return jnp.column_stack([omega_01, omega_d, phi, Omega, beta])
