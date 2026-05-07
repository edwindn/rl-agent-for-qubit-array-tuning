"""
Render staircase plots for the SuperSims env at three regimes:
  (a) initial sampled state
  (b) after 20 steps of zero action  (no change → identical to (a))
  (c) after 20 steps of random Gaussian action with std=1
      — this is the policy state at iter 0 with the original (uncapped) PPO

Across the three sampling configs (full / medium / narrow) so we can eyeball
how easy or hard each regime is. PNGs go to plots_supersims_diagnostic/.

Usage:
    uv run python scripts/supersims_init_plots.py [--seed 0] [--out-dir DIR]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SuperSims"))

jax.config.update("jax_enable_x64", True)


CONFIGS = ["full", "medium", "narrow"]
CONFIG_FILE = {
    "full":   "parameter_config.json",
    "medium": "parameter_config_medium.json",
    "narrow": "parameter_config_narrow.json",
}


def _reload_supersims(cfg_file: str):
    for mod in [m for m in list(sys.modules) if m in {
        "parameter_generation", "all_xy_sequence", "compensation_matrix",
        "reward", "hamiltonian_definitions", "normalisations",
    }]:
        del sys.modules[mod]
    pg = importlib.import_module("parameter_generation")
    pg._cfg = json.loads((Path(pg.__file__).parent / cfg_file).read_text())
    return (
        pg,
        importlib.import_module("all_xy_sequence"),
        importlib.import_module("compensation_matrix"),
        importlib.import_module("reward"),
        importlib.import_module("normalisations"),
    )


def _plot_three_regimes(cfg_name: str, seed: int, out_path: Path):
    """One figure per config: rows = qubits, cols = (init, zero-action, random-action)."""
    pg, axy, comp, rwd, norm = _reload_supersims(CONFIG_FILE[cfg_name])
    N_QUBITS = pg.N_QUBITS

    key = jax.random.PRNGKey(seed)
    omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw = pg.sample_all(key)

    params0 = jnp.column_stack([omega_01, omega_d, phi, Omega, beta])
    param_mins, param_maxs = norm.episode_bounds(omega_01, t_g)
    delta_scales = norm.episode_delta_scales(t_g)

    def run(params):
        P1 = axy.run_allxy_simulation(params, hw, t_g, alpha, lambda_)
        rew, _ = rwd.allxy_rewards(P1)
        return np.asarray(P1), np.asarray(rew)

    # (a) initial
    P1_init, r_init = run(params0)

    # (b) 20 zero-action steps. Mathematically identical to init (delta=0 → no
    # update), but we still simulate to confirm.
    params_zero = params0
    for _ in range(20):
        delta_raw = norm.delta_to_physical(jnp.zeros_like(params_zero), delta_scales)
        C, _ = comp.build_compensation(params_zero, hw, t_g, alpha, lambda_)
        params_zero = comp.update_params(params_zero, delta_raw, C, param_mins, param_maxs)
    P1_zero, r_zero = run(params_zero)

    # (c) 20 random-action steps with std=1 (the untrained PPO regime).
    rng = np.random.default_rng(seed)
    params_rand = params0
    for _ in range(20):
        a = jnp.asarray(rng.normal(0.0, 1.0, params_rand.shape).clip(-1, 1), dtype=jnp.float64)
        delta_raw = norm.delta_to_physical(a, delta_scales)
        C, _ = comp.build_compensation(params_rand, hw, t_g, alpha, lambda_)
        params_rand = comp.update_params(params_rand, delta_raw, C, param_mins, param_maxs)
    P1_rand, r_rand = run(params_rand)

    # ----- Plot ----- #
    ideal = np.array(axy.ALLXY_IDEAL)
    seq_labels = [f"({g1},{g2})" for g1, g2 in axy.ALLXY_GATES]
    x = np.arange(len(ideal))

    fig, axes = plt.subplots(N_QUBITS, 3, figsize=(15, 2.6 * N_QUBITS),
                             sharex=True, sharey=True, squeeze=False)

    columns = [
        ("(a) sampled init",       P1_init, r_init),
        ("(b) zero action × 20",   P1_zero, r_zero),
        ("(c) std=1 random × 20",  P1_rand, r_rand),
    ]
    for col, (title, P1, rewards) in enumerate(columns):
        for i in range(N_QUBITS):
            ax = axes[i, col]
            ax.axhspan(-0.1, 0.25, color="royalblue", alpha=0.05)
            ax.axhspan(0.25, 0.75, color="gray",      alpha=0.05)
            ax.axhspan(0.75, 1.15, color="firebrick", alpha=0.05)
            ax.step(x, ideal, where="mid", color="black", lw=1.0, ls="--",
                    alpha=0.45, label="ideal")
            ax.step(x, P1[i, :], where="mid", color="#d62728", lw=1.5,
                    label=f"reward={float(rewards[i]):.3f}")
            ax.set_ylim(-0.1, 1.15)
            ax.set_yticks([0, 0.5, 1])
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
            if col == 0:
                ax.set_ylabel(f"Q{i}\n$P(|1\\rangle)$", fontsize=9)
            if i == 0:
                ax.set_title(title, fontsize=11)

    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("All-XY sequence", fontsize=9)

    fig.suptitle(f"SuperSims — config: {cfg_name}  (seed={seed},  mean rewards: "
                 f"init={float(r_init.mean()):.3f}, zero={float(r_zero.mean()):.3f}, "
                 f"rand={float(r_rand.mean()):.3f})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path.name}  "
          f"[init={r_init.mean():.3f}  zero={r_zero.mean():.3f}  rand={r_rand.mean():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="plots_supersims_diagnostic")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(exist_ok=True)
    print(f"Writing plots to {out_dir}")

    for cfg_name in CONFIGS:
        _plot_three_regimes(cfg_name, args.seed, out_dir / f"staircase_{cfg_name}_seed{args.seed}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
