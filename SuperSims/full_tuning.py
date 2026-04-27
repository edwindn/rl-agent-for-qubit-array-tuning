import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# ----- Device Selection ----- #
_cfg = json.loads((Path(__file__).parent / "parameter_config.json").read_text())
if _cfg["device"]["USE_CPU"]:
    jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import parameter_generation as pg
from parameter_generation import N_QUBITS
from all_xy_sequence import ALLXY_GATES, ALLXY_IDEAL, N_ALLXY, run_allxy_simulation
from compensation_matrix import build_compensation, update_params
from reward import allxy_rewards
import normalisations as norm

"""
Full calibration tuning pipeline.

Outer loop: episodes — each resamples all qubit, crosstalk, pulse, and hardware
parameters from the ranges in parameter_config.json and resets C_tensor.

Inner loop: tuning steps — simulate → reward → compensation → print → plot →
update params.
"""

# ----- Config ----- #
N_EPISODES      = 3
N_STEPS         = 2
ALONE_ENABLED   = True    # set False to skip alone-driving simulation and plots
DELTA_SCALE     = 0.01    # fractional perturbation applied to each parameter per step
CLIP_MULTIPLIER = 2.0     # safety rail: allow params to drift up to this multiple of episode span
RANDOM_SEED     = None    # set to an integer for reproducible runs, None to randomise

PARAM_NAMES = ["ω₀₁", "ω_d", "φ", "Ω", "β"]


# ----- Helper Functions ----- #

def print_info(label, rewards_simul, rewards_alone, C_tensor, J_cols):
    # Compute all device-side values as JAX arrays first, then sync once
    conds       = jnp.stack([jnp.linalg.cond(J_cols[i][i]) for i in range(N_QUBITS)])
    norms_self  = jnp.stack([jnp.linalg.norm(J_cols[i][i]) for i in range(N_QUBITS)])
    norms_cross = jnp.stack([
        jnp.linalg.norm(jnp.stack([J_cols[j][i] for j in range(N_QUBITS) if j != i]))
        for i in range(N_QUBITS)
    ])
    to_sync = [conds, norms_self, norms_cross, rewards_simul]
    if rewards_alone is not None:
        to_sync.append(rewards_alone)
    synced = jnp.stack(to_sync).block_until_ready()   # single device→host transfer

    conds_v      = synced[0]
    norms_self_v = synced[1]
    norms_cross_v= synced[2]
    r_simul_v    = synced[3]
    r_alone_v    = synced[4] if rewards_alone is not None else None

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")

    print(f"\n  Per-qubit rewards:")
    hdr = f"    {'Qubit':<6}  {'simul':>8}"
    if rewards_alone is not None:
        hdr += f"  {'alone':>8}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for i in range(N_QUBITS):
        row = f"    Q{i:<5}  {float(r_simul_v[i]):>8.4f}"
        if rewards_alone is not None:
            row += f"  {float(r_alone_v[i]):>8.4f}"
        print(row)

    print(f"\n  Self-Jacobian diagnostics (shape {C_tensor.shape}):")
    print(f"    {'Qubit':<6}  {'cond(J_self)':>14}  {'‖J_self‖':>10}  {'‖J_cross‖':>10}")
    print("    " + "-" * 46)
    for i in range(N_QUBITS):
        print(f"    Q{i:<5}  {float(conds_v[i]):>14.3e}  {float(norms_self_v[i]):>10.4f}  {float(norms_cross_v[i]):>10.4f}")
    print()


def plot_staircase(label, params, P1_simul, rewards_simul, P1_alone=None):
    x          = list(range(N_ALLXY))
    seq_labels = [f"({g1},{g2})" for g1, g2 in ALLXY_GATES]
    ideal_arr  = jnp.array(ALLXY_IDEAL)

    fig, axes = plt.subplots(N_QUBITS, 1, figsize=(14, 3 * N_QUBITS),
                             sharex=True, sharey=True)
    if N_QUBITS == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        freq_ghz = float(params[i, 0]) / (2 * float(jnp.pi))
        r_simul  = rewards_simul[i]

        ax.axhspan(-0.1,  0.25, color="royalblue", alpha=0.05)
        ax.axhspan( 0.25, 0.75, color="gray",      alpha=0.05)
        ax.axhspan( 0.75, 1.15, color="firebrick", alpha=0.05)

        ax.step(x, ideal_arr, where="mid", color="black", lw=1.0, ls="--",
                alpha=0.45, label="ideal")

        if P1_alone is not None:
            ax.step(x, P1_alone[i, :], where="mid", color="#1f77b4", lw=1.4, ls=":",
                    label="alone")

        ax.step(x, P1_simul[i, :], where="mid", color="#d62728", lw=1.8,
                label=f"simultaneous  (reward = {float(r_simul):.3f})")

        ax.set_ylabel(r"$P(|1\rangle)$", fontsize=10)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.set_title(rf"Q{i}  ($\omega_{{01}}/2\pi = {freq_ghz:.3f}$ GHz)", fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Gate sequence", fontsize=10)
    fig.suptitle(f"All-XY calibration — {label}", fontsize=11)
    plt.tight_layout()
    plt.show()


# ----- Episode Loop ----- #

_seed = RANDOM_SEED if RANDOM_SEED is not None else int.from_bytes(__import__("os").urandom(4), "little")
rng   = jax.random.PRNGKey(_seed)
print(f"\n  RNG seed: {_seed}{'  (fixed)' if RANDOM_SEED is not None else '  (randomised)'}")

# ----- JIT Warmup ----- #
print("  Warming JIT caches...")
_wk = jax.random.fold_in(rng, 0xDEAD)   # derived key — does not consume rng
_w  = pg.sample_all(_wk)
_wp = jnp.column_stack([_w[0], _w[4], _w[5], _w[6], _w[7]])   # (N_QUBITS, 5)
_wa = (_wp, _w[8], _w[3], _w[1], _w[2])
run_allxy_simulation(*_wa, simultaneous=True).block_until_ready()
run_allxy_simulation(*_wa, simultaneous=False).block_until_ready()
jax.block_until_ready(build_compensation(*_wa))
print("  JIT warmup done.\n")

for episode in range(N_EPISODES):

    # Resample all parameters for this episode
    rng, ep_key = jax.random.split(rng)
    (pg.omega_01, pg.alpha, pg.lambda_, pg.t_g, pg.omega_d, pg.phi, pg.Omega, pg.beta, pg.hw) = pg.sample_all(ep_key)
    pg.C_tensor = jnp.einsum('ij,kl->ikjl', jnp.eye(N_QUBITS), jnp.eye(5))

    print(f"\n{'#'*64}")
    print(f"  Episode {episode}")
    print(f"{'#'*64}")

    print(f"\n  Hardware imperfections:")
    print(f"    {'Qubit':<6}  {'phi_hw [rad]':>13}  {'t_delay [ns]':>14}  {'Omega_scale':>12}")
    print("    " + "-" * 50)
    for i in range(N_QUBITS):
        print(f"    Q{i:<5}  {float(pg.hw[i, 0]):>13.4f}  {float(pg.hw[i, 1]):>14.4f}  {float(pg.hw[i, 2]):>12.4f}")

    # Single physical params array. C_tensor is the virtual→physical map; bounds
    # are enforced here on the physical params via a global scale factor α in update_params.
    params = jnp.column_stack([pg.omega_01, pg.omega_d, pg.phi, pg.Omega, pg.beta])

    # Episode-specific bounds on physical params
    param_mins, param_maxs = norm.episode_bounds(pg.omega_01, pg.t_g)
    delta_scales = norm.episode_delta_scales(pg.t_g)

    # ----- Tuning Steps ----- #
    for step in range(N_STEPS):
        label = f"Episode {episode} / Step {step}"

        # Simulate
        ep_args = (params, pg.hw, pg.t_g, pg.alpha, pg.lambda_)
        P1_alone = run_allxy_simulation(*ep_args, simultaneous=False) if ALONE_ENABLED else None
        P1_simul = run_allxy_simulation(*ep_args)

        # Rewards
        rewards_simul, _ = allxy_rewards(P1_simul)
        rewards_alone, _ = allxy_rewards(P1_alone) if P1_alone is not None else (None, None)

        # Compensation matrix
        C_tensor, J_cols = build_compensation(params, pg.hw, pg.t_g, pg.alpha, pg.lambda_)
        pg.C_tensor = C_tensor

        # Report
        #print_info(label, rewards_simul, rewards_alone, C_tensor, J_cols)
        #plot_staircase(label, params, P1_simul, rewards_simul, P1_alone)

        # Update: C_tensor maps virtual actions → physical update; α scales the full
        # physical step to keep params within bounds (α=1 = unconstrained).
        delta_norm = DELTA_SCALE * jnp.ones_like(params)
        delta_raw  = norm.delta_to_physical(delta_norm, delta_scales)
        params     = update_params(params, delta_raw, pg.C_tensor, param_mins, param_maxs,
                                   clip_multiplier=CLIP_MULTIPLIER)

        print(f"  delta_norm = {DELTA_SCALE:.3f} (uniform)  →  compensated update applied.")
        print(f"    {'Qubit':<6}  " + "  ".join(f"{n:>10}" for n in PARAM_NAMES))
        print("    " + "-" * (6 + 14 * 5))
        _delta_np = jax.device_get(delta_raw)
        for i in range(N_QUBITS):
            vals = "  ".join(f"{_delta_np[i, k]:>+10.5f}" for k in range(5))
            print(f"    Q{i:<5}  {vals}")


# Note: if CUDA-enabled jax is not installed do the following in your env:
# pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda13-plugin jax-cuda12-pjrt jax-cuda13-pjrt -y
# pip install "jax[cuda12]<0.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Higher jax versions are not compatible with dynamiqs