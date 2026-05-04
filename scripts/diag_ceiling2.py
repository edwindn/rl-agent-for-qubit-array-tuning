"""Diagnostic continuation: omega sweep, fock test, joint sweep, hi-omega RWA."""
import importlib, json, sys, os
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SuperSims"))
jax.config.update("jax_enable_x64", True)

import dynamiqs as dq
dq.set_progress_meter(False)

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
    return pg, aseq, rmod


def _params(omega_01, omega_d, phi, Omega, beta):
    return jnp.array([[omega_01, omega_d, phi, Omega, beta]])


def _zero_hw(N_QUBITS=1):
    return jnp.zeros((N_QUBITS, 3)).at[:, 2].set(1.0)


def _zero_lambda(N_QUBITS=1):
    return jnp.zeros((N_QUBITS, N_QUBITS))


def eval_one(aseq, rmod, omega_01, omega_d, phi, Omega, beta, t_g, alpha):
    p = _params(omega_01, omega_d, phi, Omega, beta)
    P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
    r, _ = rmod.allxy_rewards(P1)
    return float(r[0])


def main():
    t_g = 20.0; alpha_GHz = -0.3; w_GHz = 5.0
    pg, aseq, rmod = _setup(1, 3)
    omega_01 = 2 * jnp.pi * w_GHz
    alpha = jnp.array([2 * jnp.pi * alpha_GHz])
    Omega0 = 2 * jnp.pi / t_g

    # 5. Omega sweep at beta=1.0 and beta=0.5
    print("[5] Omega sweep")
    for beta_val in (1.0, 0.5):
        rewards = []
        fracs = jnp.arange(-0.20, 0.20 + 1e-9, 0.01)
        for f in fracs:
            Omega = Omega0 * (1.0 + f)
            rewards.append(eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega, beta_val, t_g, alpha))
        rewards = np.array(rewards); fracs_np = np.array(fracs)
        i_opt = int(np.argmax(rewards)); i0 = int(np.argmin(np.abs(fracs_np)))
        print(f"  beta={beta_val:.2f}: R(2pi/t_g)={rewards[i0]:.6f}  Omega_opt=(1{fracs_np[i_opt]:+.3f})*2pi/t_g  R_max={rewards[i_opt]:.6f}")

    # 6. Fock truncation
    print("\n[6] Fock truncation sensitivity (perfect Omega=2pi/t_g)")
    for beta_val in (1.0, 0.5):
        line = f"  beta={beta_val:.2f}: "
        for N_F in (3, 4, 5, 6):
            pg, aseq, rmod = _setup(1, N_F)
            r = eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega0, beta_val, t_g, alpha)
            line += f"N={N_F}:R={r:.6f}  "
        print(line)

    # 7. RWA at very high omega — keep Omega = 2pi/t_g (so rotation angle constant)
    print("\n[7] RWA / counter-rotating diagnostic")
    pg, aseq, rmod = _setup(1, 3)
    for beta_val in (1.0, 0.5):
        print(f"  beta={beta_val:.2f}:")
        for w in (5.0, 20.0, 100.0, 500.0):
            try:
                o = 2*jnp.pi*w
                r = eval_one(aseq, rmod, o, o, 0.0, 2*jnp.pi/t_g, beta_val, t_g, alpha)
                ratio = (2*np.pi/t_g/(2*np.pi*w))**2
                print(f"    w_01_GHz={w:>6.1f}  (O/w01)^2={ratio:.3e}  R={r:.6f}")
            except Exception as e:
                print(f"    w_01_GHz={w}: ERROR {type(e).__name__}: {e}")

    # 8. Joint (beta, Omega) coarse sweep
    print("\n[8] Joint (beta, Omega) sweep")
    pg, aseq, rmod = _setup(1, 3)
    betas = jnp.arange(0.0, 2.51, 0.05)
    fracs = jnp.arange(-0.05, 0.05 + 1e-9, 0.005)
    best = (-1.0, None, None)
    for b in betas:
        for f in fracs:
            r = eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega0*(1+f), float(b), t_g, alpha)
            if r > best[0]:
                best = (r, float(b), float(f))
    print(f"  Best: beta={best[1]:+.3f}  Omega=(1{best[2]:+.4f})*2pi/t_g  R={best[0]:.6f}")

    # 9. Refined fine beta sweep around 0.5 with higher Fock
    print("\n[9] beta fine sweep with higher Fock truncation (perfect Omega)")
    for N_F in (3, 4, 5):
        pg, aseq, rmod = _setup(1, N_F)
        betas = jnp.arange(0.30, 0.71, 0.005)
        rs = []
        for b in betas:
            rs.append(eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega0, float(b), t_g, alpha))
        rs = np.array(rs); bs = np.array(betas)
        j = int(np.argmax(rs))
        print(f"  N_FOCK={N_F}: beta_opt={bs[j]:.4f}  R={rs[j]:.6f}")

    # 10. Fully unconstrained: vary t_g too at beta=0.5 to see absolute ceiling.
    print("\n[10] Fixed beta=0.5: vary t_g")
    pg, aseq, rmod = _setup(1, 3)
    for tg_val in (16.0, 20.0, 24.0, 32.0, 40.0):
        Omega = 2*jnp.pi/tg_val
        r = eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega, 0.5, tg_val, alpha)
        print(f"  t_g={tg_val:5.1f}  R={r:.6f}")


if __name__ == "__main__":
    main()
