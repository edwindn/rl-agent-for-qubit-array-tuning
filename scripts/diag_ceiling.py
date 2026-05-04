"""Diagnose the ~0.95 reward ceiling at perfect-pulse params with no hardware
errors and no cross-talk. Single-qubit, t_g=20 ns, alpha=-0.3 GHz*2pi (rad/ns).

Run via: CUDA_VISIBLE_DEVICES=2 uv run python scripts/diag_ceiling.py
"""
import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SuperSims"))

jax.config.update("jax_enable_x64", True)


_SUPERSIMS_MODS = {
    "parameter_generation", "all_xy_sequence", "compensation_matrix",
    "reward", "hamiltonian_definitions", "normalisations",
}


def _reset_modules():
    for mod in [m for m in list(sys.modules) if m in _SUPERSIMS_MODS]:
        del sys.modules[mod]


def _setup(N_QUBITS=1, N_FOCK=3):
    _reset_modules()
    import parameter_generation as pg
    cfg = json.loads((Path(pg.__file__).parent / "parameter_config.json").read_text())
    cfg["system"]["N_QUBITS"] = N_QUBITS
    cfg["system"]["N"] = N_FOCK
    pg._cfg = cfg
    pg.N_QUBITS = N_QUBITS
    pg.N = N_FOCK
    aseq = importlib.import_module("all_xy_sequence")
    rmod = importlib.import_module("reward")
    hdef = importlib.import_module("hamiltonian_definitions")
    return pg, aseq, rmod, hdef


def _params(omega_01, omega_d, phi, Omega, beta):
    return jnp.array([[omega_01, omega_d, phi, Omega, beta]])


def _zero_hw(N_QUBITS=1):
    hw = jnp.zeros((N_QUBITS, 3))
    return hw.at[:, 2].set(1.0)


def _zero_lambda(N_QUBITS=1):
    return jnp.zeros((N_QUBITS, N_QUBITS))


def beta_sweep(t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0):
    pg, aseq, rmod, _ = _setup(1, 3)
    omega_01 = 2 * jnp.pi * omega_01_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega = 2 * jnp.pi / t_g
    betas = jnp.arange(-1.0, 3.01, 0.1)
    rewards = []
    for b in betas:
        p = _params(omega_01, omega_01, 0.0, Omega, float(b))
        P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
        r, _ = rmod.allxy_rewards(P1)
        rewards.append(float(r[0]))
    rewards = np.array(rewards)
    betas_np = np.array(betas)
    i_opt = int(np.argmax(rewards))
    print(f"[1] beta sweep (1q, no hw, no lambda, t_g={t_g}, alpha={alpha_GHz} GHz)")
    print(f"    coarse: range [{betas_np[0]:.2f}, {betas_np[-1]:.2f}], step 0.1")
    i1 = int(np.argmin(np.abs(betas_np - 1.0)))
    print(f"    R(beta=1.0)   = {rewards[i1]:.6f}")
    print(f"    beta_opt(coarse) = {betas_np[i_opt]:+.3f}, R_max = {rewards[i_opt]:.6f}")
    fine_lo = max(-1.0, betas_np[i_opt] - 0.15)
    fine_hi = min(3.0, betas_np[i_opt] + 0.15)
    fine = jnp.arange(fine_lo, fine_hi + 1e-9, 0.005)
    fine_r = []
    for b in fine:
        p = _params(omega_01, omega_01, 0.0, Omega, float(b))
        P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
        r, _ = rmod.allxy_rewards(P1)
        fine_r.append(float(r[0]))
    fine_r = np.array(fine_r)
    fine_np = np.array(fine)
    j = int(np.argmax(fine_r))
    print(f"    fine [step 0.005]: beta_opt = {fine_np[j]:+.4f}, R_max = {fine_r[j]:.6f}")
    print("    Top 7 (beta, R):")
    for k in np.argsort(rewards)[::-1][:7]:
        print(f"      beta={betas_np[k]:+5.2f}  R={rewards[k]:.6f}")
    return float(fine_np[j]), float(fine_r[j])


def per_sequence(beta_val, t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0, label=""):
    pg, aseq, rmod, _ = _setup(1, 3)
    omega_01 = 2 * jnp.pi * omega_01_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega = 2 * jnp.pi / t_g
    p = _params(omega_01, omega_01, 0.0, Omega, beta_val)
    P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
    r, dev = rmod.allxy_rewards(P1)
    P1n = np.asarray(P1[0]); devn = np.asarray(dev[0])
    ideal = np.asarray(aseq.ALLXY_IDEAL)
    print(f"\n[2{label}] per-seq breakdown beta={beta_val:.4f}, R={float(r[0]):.6f}")
    print(f"    {'idx':>3}  {'gates':<14}  {'ideal':>5}  {'P1':>9}  {'dev':>9}")
    for i, (g1, g2) in enumerate(aseq.ALLXY_GATES):
        gate_str = f"{g1}, {g2}"
        print(f"    {i:>3}  {gate_str:<14}  {ideal[i]:>5.2f}  {P1n[i]:>9.6f}  {devn[i]:>9.6f}")
    order = np.argsort(devn)[::-1]
    print(f"    Top 5 by deviation:")
    for k in order[:5]:
        g1, g2 = aseq.ALLXY_GATES[k]
        print(f"      idx={k:>3}  ({g1}, {g2})  dev={devn[k]:.4f}  P1={P1n[k]:.4f}  ideal={ideal[k]:.2f}")


def leakage(beta_val, t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0, label=""):
    pg, aseq, rmod, _ = _setup(1, 3)
    import dynamiqs as dq
    omega_01 = 2 * jnp.pi * omega_01_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega = 2 * jnp.pi / t_g
    p = _params(omega_01, omega_01, 0.0, Omega, beta_val)
    H, tsave = aseq._build_sim_inputs(p, _zero_hw(), t_g, alpha, _zero_lambda(), True)
    res = dq.sesolve(H, aseq.psi0, tsave)
    states = res.states.to_jax()
    final = states[0, :, -1, :, 0]
    pop = jnp.abs(final) ** 2
    pop_np = np.asarray(pop)
    print(f"\n[3{label}] leakage |c2|^2 beta={beta_val:.4f}")
    print(f"    {'idx':>3}  {'gates':<14}  {'P0':>9}  {'P1':>9}  {'P2':>9}  {'sum':>6}")
    for i, (g1, g2) in enumerate(aseq.ALLXY_GATES):
        gate_str = f"{g1}, {g2}"
        s = pop_np[i].sum()
        print(f"    {i:>3}  {gate_str:<14}  {pop_np[i,0]:>9.6f}  {pop_np[i,1]:>9.6f}  {pop_np[i,2]:>9.6f}  {s:>6.4f}")
    print(f"    mean |c2|^2 = {pop_np[:,2].mean():.6e}, max = {pop_np[:,2].max():.6e}")


def rwa_test(beta_val, t_g=20.0, alpha_GHz=-0.3):
    pg, aseq, rmod, _ = _setup(1, 3)
    print(f"\n[4] RWA / counter-rotating diagnostic beta={beta_val:.4f}")
    print(f"    {'omega01_GHz':>11}  {'(O/w01)^2':>11}  {'reward':>10}")
    for w_GHz in (5.0, 20.0, 100.0, 500.0, 2000.0):
        omega_01 = 2 * jnp.pi * w_GHz
        alpha = jnp.array([2 * jnp.pi * alpha_GHz])
        Omega = 2 * jnp.pi / t_g
        ratio = (Omega / omega_01) ** 2
        p = _params(omega_01, omega_01, 0.0, Omega, beta_val)
        P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
        r, _ = rmod.allxy_rewards(P1)
        print(f"    {w_GHz:>11.1f}  {float(ratio):>11.3e}  {float(r[0]):>10.6f}")


def omega_sweep(beta_val, t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0):
    pg, aseq, rmod, _ = _setup(1, 3)
    omega_01 = 2 * jnp.pi * omega_01_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega0 = 2 * jnp.pi / t_g
    fracs = jnp.arange(-0.20, 0.20 + 1e-9, 0.01)
    rewards = []
    for f in fracs:
        Omega = Omega0 * (1.0 + f)
        p = _params(omega_01, omega_01, 0.0, Omega, beta_val)
        P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
        r, _ = rmod.allxy_rewards(P1)
        rewards.append(float(r[0]))
    rewards = np.array(rewards)
    fracs_np = np.array(fracs)
    i_opt = int(np.argmax(rewards))
    i0 = int(np.argmin(np.abs(fracs_np)))
    print(f"\n[5] Omega sweep beta={beta_val:.4f}")
    print(f"    R at Omega = 2pi/t_g       : {rewards[i0]:.6f}")
    print(f"    Omega_opt = (1{fracs_np[i_opt]:+.3f}) * 2pi/t_g  R_max = {rewards[i_opt]:.6f}")


def fock_test(beta_val, t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0):
    print(f"\n[6] Fock truncation sensitivity beta={beta_val:.4f}")
    print(f"    {'N_FOCK':>7}  {'reward':>10}")
    for N_F in (3, 4, 5, 6):
        pg, aseq, rmod, _ = _setup(1, N_F)
        omega_01 = 2 * jnp.pi * omega_01_GHz
        alpha = jnp.array([2 * jnp.pi * alpha_GHz])
        Omega = 2 * jnp.pi / t_g
        p = _params(omega_01, omega_01, 0.0, Omega, beta_val)
        P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
        r, _ = rmod.allxy_rewards(P1)
        print(f"    {N_F:>7d}  {float(r[0]):>10.6f}")


def joint_beta_omega_sweep(t_g=20.0, alpha_GHz=-0.3, omega_01_GHz=5.0):
    pg, aseq, rmod, _ = _setup(1, 3)
    omega_01 = 2 * jnp.pi * omega_01_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega0 = 2 * jnp.pi / t_g
    betas = jnp.arange(0.0, 2.51, 0.1)
    fracs = jnp.arange(-0.05, 0.05 + 1e-9, 0.01)
    best = (-1.0, None, None)
    for b in betas:
        for f in fracs:
            Omega = Omega0 * (1.0 + f)
            p = _params(omega_01, omega_01, 0.0, Omega, float(b))
            P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
            r, _ = rmod.allxy_rewards(P1)
            v = float(r[0])
            if v > best[0]:
                best = (v, float(b), float(f))
    print(f"\n[7] Joint (beta, Omega) sweep")
    print(f"    Best: beta={best[1]:+.3f}  Omega=(1{best[2]:+.4f})*2pi/t_g  R={best[0]:.6f}")


def main():
    print("=" * 70)
    print("SuperSims All-XY reward ceiling diagnostic")
    print("=" * 70)
    beta_opt, R_opt = beta_sweep()
    per_sequence(1.0, label="a")
    per_sequence(beta_opt, label="b")
    leakage(1.0, label="a")
    leakage(beta_opt, label="b")
    rwa_test(1.0)
    rwa_test(beta_opt)
    omega_sweep(1.0)
    omega_sweep(beta_opt)
    fock_test(1.0)
    fock_test(beta_opt)
    joint_beta_omega_sweep()
    print("\n" + "=" * 70)
    print(f"Summary: beta_opt={beta_opt:+.4f}  R_opt={R_opt:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
