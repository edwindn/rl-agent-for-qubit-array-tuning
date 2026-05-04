"""Fast continuation diagnostic. Uses unbuffered prints."""
import importlib, json, sys, os
from pathlib import Path

# unbuffered output
sys.stdout.reconfigure(line_buffering=True)

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


def _zero_hw():
    return jnp.zeros((1, 3)).at[:, 2].set(1.0)


def _zero_lambda():
    return jnp.zeros((1, 1))


def eval_one(aseq, rmod, omega_01, omega_d, phi, Omega, beta, t_g, alpha):
    p = _params(omega_01, omega_d, phi, Omega, beta)
    P1 = aseq.run_allxy_simulation(p, _zero_hw(), t_g, alpha, _zero_lambda())
    r, _ = rmod.allxy_rewards(P1)
    return float(r[0])


def main():
    t_g = 20.0
    alpha = jnp.array([2 * jnp.pi * (-0.3)])
    omega_01 = 2 * jnp.pi * 5.0
    Omega0 = 2 * jnp.pi / t_g

    print("[5] Omega sweep at beta=1 and beta=0.5", flush=True)
    pg, aseq, rmod = _setup(1, 3)
    for beta_val in (1.0, 0.5):
        rewards = []
        fracs = jnp.arange(-0.20, 0.20 + 1e-9, 0.01)
        for f in fracs:
            Omega = Omega0 * (1.0 + f)
            rewards.append(eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega, beta_val, t_g, alpha))
        rewards = np.array(rewards); fracs_np = np.array(fracs)
        i_opt = int(np.argmax(rewards)); i0 = int(np.argmin(np.abs(fracs_np)))
        print(f"  beta={beta_val:.2f}: R(2pi/t_g)={rewards[i0]:.6f}  "
              f"Omega_opt=(1{fracs_np[i_opt]:+.3f})*2pi/t_g  R_max={rewards[i_opt]:.6f}",
              flush=True)

    print("\n[6] Fock truncation sensitivity (Omega=2pi/t_g)", flush=True)
    for beta_val in (1.0, 0.5):
        line = f"  beta={beta_val:.2f}: "
        for N_F in (3, 4, 5):
            pg, aseq, rmod = _setup(1, N_F)
            r = eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega0, beta_val, t_g, alpha)
            line += f"N={N_F}:R={r:.6f}  "
        print(line, flush=True)

    print("\n[7] RWA / counter-rotating diagnostic (high omega_01)", flush=True)
    pg, aseq, rmod = _setup(1, 3)
    for beta_val in (1.0, 0.5):
        print(f"  beta={beta_val:.2f}:", flush=True)
        for w in (5.0, 20.0, 100.0):
            o = 2*jnp.pi*w
            r = eval_one(aseq, rmod, o, o, 0.0, Omega0, beta_val, t_g, alpha)
            ratio = (Omega0/o)**2
            print(f"    w_01_GHz={w:>6.1f}  (O/w01)^2={float(ratio):.3e}  R={r:.6f}", flush=True)

    print("\n[8] Joint (beta, Omega) coarse sweep", flush=True)
    pg, aseq, rmod = _setup(1, 3)
    betas = jnp.arange(0.0, 2.01, 0.1)
    fracs = jnp.arange(-0.05, 0.05 + 1e-9, 0.01)
    best = (-1.0, None, None)
    for b in betas:
        for f in fracs:
            r = eval_one(aseq, rmod, omega_01, omega_01, 0.0, Omega0*(1+f), float(b), t_g, alpha)
            if r > best[0]:
                best = (r, float(b), float(f))
    print(f"  Best: beta={best[1]:+.3f}  Omega=(1{best[2]:+.4f})*2pi/t_g  R={best[0]:.6f}", flush=True)


if __name__ == "__main__":
    main()
