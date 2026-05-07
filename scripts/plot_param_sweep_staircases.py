"""
Sweep one of the 5 transmon params across its full range while pinning the other
4 at ground truth, render the resulting 21-point All-XY staircase per sweep point,
and report reward.

Use this to visually verify:
  (a) at "all params at GT" the staircase matches the ideal step pattern
      ([0]*5 + [0.5]*12 + [1.0]*4) and reward ≈ 0.998
  (b) staircases off-GT degrade smoothly and predictably.

Hardware imperfections and cross-talk are zeroed so we see the param's own effect
in isolation. Single-qubit view (qubit 0) — the other 3 qubits are fixed at GT
parallel to qubit 0 with lambda_=0, so they don't influence qubit 0 anyway.

Usage:
    uv run python scripts/plot_param_sweep_staircases.py [--n-points 9]
                                                        [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SuperSims"))

jax.config.update("jax_enable_x64", True)


PARAM_NAMES = ["omega_01", "omega_d", "phi", "Omega", "beta"]


def build_gt_params(omega_01, t_g, beta_default=0.5):
    """(N_QUBITS, 5) physical params at the perfect-pulse setpoint."""
    Omega_opt = 2 * jnp.pi / t_g
    N = omega_01.shape[0]
    return jnp.column_stack([
        omega_01,
        omega_01,                        # omega_d = omega_01 (no detuning)
        jnp.zeros((N,)),                  # phi = 0
        jnp.full((N,), Omega_opt),        # Omega = 2π/t_g
        jnp.full((N,), beta_default),     # beta = 0.5 (raised-cosine DRAG optimum)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=9, help="Sweep points per param.")
    ap.add_argument("--out-dir", type=str, default="plots_supersims_diagnostic")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(exist_ok=True)
    print(f"Writing plots to {out_dir}")

    import parameter_generation as pg
    import all_xy_sequence as axy
    from reward import allxy_rewards

    N_QUBITS = pg.N_QUBITS
    # Canonical scalars for a representative episode.
    t_g = 20.0
    alpha = 2 * jnp.pi * jnp.full((N_QUBITS,), -0.3)        # canonical anharmonicity
    omega_01_init = 2 * jnp.pi * jnp.full((N_QUBITS,), 5.0)  # canonical 5 GHz
    hw = jnp.zeros((N_QUBITS, 3)).at[:, 2].set(1.0)          # phi_hw=0, t_delay=0, Omega_scale=1
    lambda_ = jnp.zeros((N_QUBITS, N_QUBITS))               # no cross-talk

    gt_params = build_gt_params(omega_01_init, t_g)
    Omega_opt = 2 * jnp.pi / t_g

    # ----- Reward at GT (sanity check) ----- #
    P1_gt = axy.run_allxy_simulation(gt_params, hw, t_g, alpha, lambda_)
    rewards_gt, _ = allxy_rewards(P1_gt)
    print(f"\nReward at GT (everything perfect, hw=0, λ=0):")
    print(f"  Per-qubit: {np.array(rewards_gt)}")
    print(f"  Mean:      {float(jnp.mean(rewards_gt)):.4f}\n")

    # ----- Sweep ranges (FULL config bounds) per param ----- #
    sweep_ranges = {
        "omega_01": (omega_01_init[0] - 2 * jnp.pi * 0.15, omega_01_init[0]),  # downward only, 150 MHz (post 2026-05-02)
        "omega_d":  (omega_01_init[0] - 2 * jnp.pi * 0.05,
                     omega_01_init[0] + 2 * jnp.pi * 0.05),
        "phi":      (-jnp.pi, jnp.pi),
        "Omega":    (Omega_opt * 0.7, Omega_opt * 1.3),
        "beta":     (0.0, 1.5),  # widened post 2026-05-02 — optimum=0.5 sits in mid-range
    }

    # ----- One plot per param ----- #
    ideal = np.array(axy.ALLXY_IDEAL)
    seq_labels = [f"({g1},{g2})" for g1, g2 in axy.ALLXY_GATES]
    x = np.arange(len(ideal))

    # Display each param in physical units in the legend (no rad/ns or π).
    #   omega_01, omega_d, Omega: rad/ns → GHz via /(2π).
    #   phi: rad → degrees.
    #   beta: dimensionless, shown as-is.
    def _display(pname, val):
        if pname in ("omega_01", "omega_d", "Omega"):
            return float(val) / (2 * float(jnp.pi)), "GHz"
        if pname == "phi":
            return float(val) * 180.0 / float(jnp.pi), "deg"
        return float(val), ""

    for param_idx, pname in enumerate(PARAM_NAMES):
        lo, hi = sweep_ranges[pname]
        # phi: nudge endpoints inward so n_points=9 gives a step of (160-(-160))/8 = 40°,
        # avoiding the π/4-multiple landings (-180,-135,...,180) of the literal ±π sweep.
        if pname == "phi":
            lo, hi = -8 * jnp.pi / 9, 8 * jnp.pi / 9   # ±160°, step = 40°
        sweep_vals = jnp.linspace(float(lo), float(hi), args.n_points)
        gt_val = float(gt_params[0, param_idx])
        gt_disp, unit = _display(pname, gt_val)

        fig, ax = plt.subplots(figsize=(13, 5.5))

        # Reference: ideal pattern + GT staircase (qubit 0).
        ax.step(x, ideal, where="mid", color="black", lw=1.2, ls="--", alpha=0.6,
                label=f"ideal (target)")
        ax.step(x, np.array(P1_gt[0]), where="mid", color="green", lw=2.4,
                label=f"GT @ qubit 0  (reward={float(rewards_gt[0]):.3f})")

        # Sweep: colour gradient from cool→warm; mark GT value distinctly if in range.
        cmap = plt.cm.coolwarm
        for s, val in enumerate(sweep_vals):
            params_q0 = gt_params.at[0, param_idx].set(val)
            P1 = axy.run_allxy_simulation(params_q0, hw, t_g, alpha, lambda_)
            rewards, _ = allxy_rewards(P1)
            r0 = float(rewards[0])
            color = cmap(s / max(args.n_points - 1, 1))
            disp_val, _u = _display(pname, val)
            label = f"{pname}={disp_val:.3f}{(' ' + _u) if _u else ''}  r={r0:.3f}"
            ax.step(x, np.array(P1[0]), where="mid", color=color, lw=1.2,
                    alpha=0.8, label=label)

        # Reward bands.
        ax.axhspan(-0.1, 0.25, color="royalblue", alpha=0.05)
        ax.axhspan(0.25, 0.75, color="gray", alpha=0.05)
        ax.axhspan(0.75, 1.15, color="firebrick", alpha=0.05)

        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(r"$P(|1\rangle)$ — qubit 0", fontsize=10)
        ax.set_xlabel("All-XY sequence", fontsize=10)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(alpha=0.25)
        ax.set_title(f"Sweep {pname} across full range, others pinned at GT  "
                     f"(GT: {pname}={gt_disp:.3f}{(' ' + unit) if unit else ''})",
                     fontsize=11)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
        plt.tight_layout()

        out_path = out_dir / f"sweep_{pname}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_path.name}")


if __name__ == "__main__":
    main()
