import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

from parameters import (
    N, N_QUBITS,
    t_g, sigma, alpha, omega_01, lambda_
)
from hamiltonians import H_alone, H_simul

# ----- Simulation ----- #
psi0  = dq.basis(N, 0)
tsave = jnp.linspace(0.0, t_g, 2000)

result_alone = dq.sesolve(H_alone, psi0, tsave)
result_simul = dq.sesolve(H_simul,  psi0, tsave)

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
    rf"$\pi$-pulse: individual vs simultaneous driving",
    fontsize=11,
)
plt.tight_layout()
# plt.savefig("rabi_traces.png", dpi=150, bbox_inches="tight")
plt.show()
