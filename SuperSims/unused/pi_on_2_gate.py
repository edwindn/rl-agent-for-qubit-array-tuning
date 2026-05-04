import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Note: if CUDA-enabled jax is not installed do the following in your env:
# pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda13-plugin jax-cuda12-pjrt jax-cuda13-pjrt -y
# pip install "jax[cuda12]<0.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Higher jax versions are not compatible with dynamiqs

"""
Simultaneous single-qubit gate simulation for N_QUBITS transmon qubits.
Each qubit is simulated independently in its own N-level Fock space.

For each qubit i, two scenarios are simulated:
1. driven alone -- only qubit i is driven; no cross-talk received
2. driven simultaneously -- all qubits driven; qubit i receives j's cross-talk via perturbative Hamiltonian
weighted by amplitude coefficient lambda_ij
"""

# ----- Device Selection ----- #
USE_CPU = True
if USE_CPU:
    jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# ----- System Parameters ----- #
N_QUBITS = 5   # total number of qubits
N  = 3   # Fock-space truncation per qubit: keep |0>, |1>, |2>
# assume natural units, i.e. hbar = 1, time is in ns, angular frequencies in rad/ns


# ----- Per-qubit parameters ----- #

omega_01 = [  # bare transition frequency |0> -> |1> [GHz]
    2 * jnp.pi * 5.000,
    2 * jnp.pi * 5.300,
    2 * jnp.pi * 5.600,
    2 * jnp.pi * 5.900,
    2 * jnp.pi * 6.200,
]

alpha = [  # anharmonicity (omega_12 - omega_01) [GHz]
    2 * jnp.pi * (-0.300),
    2 * jnp.pi * (-0.300),
    2 * jnp.pi * (-0.300),
    2 * jnp.pi * (-0.300),
    2 * jnp.pi * (-0.300),
]


# ----- Cross-talk coupling matrix ----- #

lambda_ = [
    [0.00,   0.05,   0.05,   0.05,   0.05],   # qubit 0 receives
    [0.05,   0.00,   0.05,   0.05,   0.05],   # qubit 1 receives
    [0.05,   0.05,   0.00,   0.05,   0.05],   # qubit 2 receives
    [0.05,   0.05,   0.05,   0.00,   0.05],   # qubit 3 receives
    [0.05,   0.05,   0.05,   0.05,   0.00],   # qubit 4 receives
]


# ----- Drive/Pulse/Envelope Parameters ----- #

t_g   = 20.0                 # gate duration [ns]
phi   = [0.0] * N_QUBITS     # drive phase [rad]
sigma = [5.0] * N_QUBITS     # Gaussian std-dev [ns] of pulse envelope, if Gaussian used
beta  = [0.0] * N_QUBITS     # DRAG coefficient [ns], set to zero to disable DRAG
ENVELOPE = "cosine"

omega_d = [  # drive frequency [GHz]
    2 * jnp.pi * 5.000,
    2 * jnp.pi * 5.300,
    2 * jnp.pi * 5.600,
    2 * jnp.pi * 5.900,
    2 * jnp.pi * 6.200,
]

Omega = [  # envelope peak amplitude, 12.53314 and 12.73239 for Gaussian/cosine baselines
    jnp.pi / 12.53314,
    jnp.pi / 12.53314,
    jnp.pi / 12.53314,
    jnp.pi / 12.53314,
    jnp.pi / 12.53314,
]


# ----- Pulse Envelopes ----- #

def s_I(t, i):
    """In-phase envelope for qubit i."""
    if ENVELOPE == "gaussian":
        return jnp.exp(-((t - t_g / 2) ** 2) / (2 * sigma[i] ** 2))
    elif ENVELOPE == "cosine":
        return jnp.cos(jnp.pi * (t - t_g / 2) / t_g)
    elif ENVELOPE == "raised_cosine":
        return 0.5 * (1 - jnp.cos(2 * jnp.pi * t / t_g))
    else:
        raise ValueError(
            f"Unknown envelope {ENVELOPE!r}. "
            "Choose one of: 'gaussian', 'cosine', 'raised_cosine'."
        )

def s_Q(t, i):
    """DRAG Q-envelope for qubit i: −β[i] · ds_I/dt."""
    if ENVELOPE == "gaussian":
        return beta[i] * (t - t_g / 2) / sigma[i] ** 2 * s_I(t, i)
    elif ENVELOPE == "cosine":
        return beta[i] * jnp.pi / t_g * jnp.sin(jnp.pi * (t - t_g / 2) / t_g)
    elif ENVELOPE == "raised_cosine":
        return -beta[i] * jnp.pi / t_g * jnp.sin(2 * jnp.pi * t / t_g)
    else:
        raise ValueError(
            f"Unknown envelope {ENVELOPE!r}. "
            "Choose one of: 'gaussian', 'cosine', 'raised_cosine'."
        )

def drive_voltage(t, i):
    """Full lab-frame drive voltage for qubit i [dimensionless]."""
    phase = omega_d[i] * t + phi[i]
    return s_I(t, i) * jnp.cos(phase) + s_Q(t, i) * jnp.sin(phase)


# def _envelope_area(i):
#     """∫₀^{t_g} s_I^i(t) dt for the selected envelope shape."""
#     if ENVELOPE == "gaussian":
#         return sigma[i] * jnp.sqrt(2 * jnp.pi)   # exact for infinite limits; good for σ ≪ t_g
#     elif ENVELOPE == "cosine":
#         return 2 * t_g / jnp.pi
#     elif ENVELOPE == "raised_cosine":
#         return t_g / 2.0
#     else:
#         raise ValueError(
#             f"Unknown envelope {ENVELOPE!r}. "
#             "Choose one of: 'gaussian', 'cosine', 'raised_cosine'."
#         )
#
# π-pulse calibrated voltage amplitude [V]: A_π = π / (g · ∫s_I dt)
# Omega  = [jnp.pi / ( _envelope_area(i)) for i in range(N_QUBITS)]
# To apply a rotation angle θ instead of π, use:
# Omega = [(theta / jnp.pi) * A_pi[i] for i in range(N_QUBITS)]


# ----- Hamiltonian Terms and Operators ----- #

# See arXiv:2603.11018 as a reference
# Operators as plain JAX arrays (dense) — used for the constant H_bare terms.
a    = dq.destroy(N).to_jax()   # lowering (annihilation) operator
adag = dq.create(N).to_jax()    # raising (creation) operator
n_op = dq.number(N).to_jax()    # photon-number operator
I_op = dq.eye(N).to_jax()       # identity
iX_op = 1j * (adag - a)         # capacitive drive coupling operator

def H_bare(i):
    """Constant bare Hamiltonian for qubit i: ω₀₁[i] n̂ + (α[i]/2) n̂(n̂ − 1)."""
    return omega_01[i] * n_op + (alpha[i] / 2.0) * n_op @ (n_op - I_op)

def _coeff_alone(t):
    """Self-drive amplitude for each qubit [rad/ns], driven individually."""
    return jnp.array([Omega[i] * drive_voltage(t, i) for i in range(N_QUBITS)])

def _coeff_simul(t):
    """Self-drive + cross-talk amplitude for each qubit [rad/ns], all driven simultaneously."""
    return jnp.array([
        Omega[i] * drive_voltage(t, i)
        + sum(lambda_[i][j] * Omega[j] * drive_voltage(t, j)
              for j in range(N_QUBITS) if j != i)
        for i in range(N_QUBITS)
    ])

# The full Hamiltonian has the form H(t) = H_bare + f(t) · iX_op
# The time dependence lives entirely in the scalar prefactor f(t); the operator iX_op is constant.
# This maps directly onto dq.modulated(f, iX_op), which constructs a ModulatedTimeQArray O(t) = f(t) · iX_op
# without rebuilding the full matrix at every solver step.
#
# Returning a (N_QUBITS,) array from f causes dq.modulated to produce a
# batched (N_QUBITS, N, N) TimeQArray — one simulation per qubit per call.
# The solver passes its internal clock t [ns] directly to f at each adaptive
# step; the same t flows into drive_voltage and cos(ω_d · t).


# ----- Simulation ----- #

psi0  = dq.basis(N, 0)
tsave = jnp.linspace(0.0, t_g, 500)

# Precompute constant bare Hamiltonians for all qubits as a single batched array.
H_bare_batch = jnp.stack([H_bare(i) for i in range(N_QUBITS)])   # (N_QUBITS, N, N)

result_alone = dq.sesolve(
    H_bare_batch + dq.modulated(_coeff_alone, iX_op),
    psi0, tsave,
)
result_simul = dq.sesolve(
    H_bare_batch + dq.modulated(_coeff_simul, iX_op),
    psi0, tsave,
)
# result_*.states has shape (N_QUBITS, ntsave, N, 1)

states_alone = result_alone.states.to_jax()   # (N_QUBITS, ntsave, N, 1)
states_simul  = result_simul.states.to_jax()  # (N_QUBITS, ntsave, N, 1)


# ----- Simulation Result Helpers ----- #

def level_populations(states_i):
    """Return array of shape (ntsave, N) with populations for each level.

    Args:
        states_i: state trajectory for one qubit, shape (ntsave, N, 1).
    """
    amps = states_i[:, :, 0]          # complex amplitudes, shape (ntsave, N)
    return jnp.abs(amps) ** 2

_psi_ideal = jnp.zeros(N, dtype=complex).at[1].set(1.0)    # |1⟩
_rho_ideal = jnp.outer(_psi_ideal, jnp.conj(_psi_ideal))   # |1⟩⟨1|

def process_fidelity(states_i):
    """F = Tr(ρ_ideal · ρ_actual) evaluated at the end of the gate (t = t_g).

    Args:
        states_i: state trajectory for one qubit, shape (ntsave, N, 1).
    """
    psi_final  = states_i[-1, :, 0]                               # (N,)
    rho_actual = jnp.outer(psi_final, jnp.conj(psi_final))        # |ψ(t_g)⟩⟨ψ(t_g)|
    return float(jnp.real(jnp.trace(_rho_ideal @ rho_actual)))

pop_alone = [level_populations(states_alone[i]) for i in range(N_QUBITS)]
pop_simul  = [level_populations(states_simul[i])  for i in range(N_QUBITS)]

fid_alone = [process_fidelity(states_alone[i]) for i in range(N_QUBITS)]
fid_simul  = [process_fidelity(states_simul[i])  for i in range(N_QUBITS)]

col_w = 10
print()
print(f"  Gate process fidelity  F = Tr(ρ_ideal · ρ_actual),  ρ_ideal = |1⟩⟨1|")
print(f"  {'Qubit':<8}{'Driven alone':>{col_w}}{'Simultaneous':>{col_w + 4}}")
print(f"  {'-'*8}{'-'*col_w}{'-'*(col_w + 4)}")
for i in range(N_QUBITS):
    freq = omega_01[i] / (2 * jnp.pi)
    print(
        f"  Q{i} ({freq:.2f} GHz)"
        f"  {fid_alone[i]:{col_w}.6f}"
        f"  {fid_simul[i]:{col_w}.6f}"
    )
print()


# ----- Plotting ----- #

COLORS = ["#1f77b4", "#d62728", "#2ca02c"]   # |0⟩ blue, |1⟩ red, |2⟩ green
LABELS = [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$"]

fig, axes = plt.subplots(N_QUBITS, 2, figsize=(12, 3 * N_QUBITS),
                         sharex=True, sharey=True)

for i in range(N_QUBITS):
    freq_str = rf"$\omega_{{01}}/2\pi={omega_01[i]/(2*jnp.pi):.2f}$ GHz"
    alpha_str = rf"$\alpha/2\pi={alpha[i]/(2*jnp.pi)*1e3:.0f}$ MHz"
    _sig = (rf"  $\sigma={sigma[i]:.1f}$ ns")
    qubit_info = rf"Q{i}: {freq_str},  {alpha_str}{_sig}"

    # Collect the unique off-diagonal lambda values for qubit i's row/col
    lambdas_onto_i = [lambda_[i][j] for j in range(N_QUBITS) if j != i]
    lambda_str = (
        rf"$\lambda={lambdas_onto_i[0]}$"
        if len(set(lambdas_onto_i)) == 1
        else r"$\lambda$ varied"
    )

    for col, (pop, suffix) in enumerate(
        [(pop_alone[i], "alone"), (pop_simul[i], rf"simultaneous  ({lambda_str})")]
    ):
        ax = axes[i, col]
        for k in range(N):
            ax.plot(tsave, pop[:, k], color=COLORS[k], label=LABELS[k], lw=1.8)
        ax.axvline(t_g / 2, color="gray", lw=0.7, ls=":", alpha=0.6,
                   label="pulse centre")
        ax.set_title(rf"{qubit_info} — {suffix}", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

for ax in axes[:, 0]:
    ax.set_ylabel("Population", fontsize=10)
for ax in axes[-1, :]:
    ax.set_xlabel("Time  [ns]", fontsize=10)

fig.suptitle(
    rf"$\pi$-pulse: individual vs simultaneous driving  (arXiv:2603.11018)",
    fontsize=11,
)
plt.tight_layout()
# plt.savefig("rabi_traces.png", dpi=150, bbox_inches="tight")
plt.show()
