"""
Sanity checks for the SuperSims All-XY env.

Two questions this script answers:

(1) Is the "ground truth" (perfect-pulse params) actually reachable in the
    sampling space? For each sampled episode we replace the agent-tunable
    columns with their physically-ideal values:
        omega_d = omega_01_init   (no detuning)
        phi     = 0               (no phase error)
        Omega   = 2π / t_g        (perfect π-pulse area for raised-cosine)
        beta    = 1.0             (canonical DRAG)
    omega_01 stays at its sampled value (it's an episode-fixed physical
    quantity, not really "tunable to GT" — and the agent's bound on
    omega_01 is [omega_01_init - 0.3 GHz, omega_01_init], with the upper
    bound at GT, so the optimal action on this column is also "do nothing").
    We compare the resulting reward to (a) the sampled init reward and
    (b) the same GT-params reward but with hardware imperfections zeroed.

(2) How does the reset reward distribution change across the three sampling
    configs (full / medium / narrow)? If the canonical "full" range gives a
    reset reward already near 1.0, then 0.92 baseline at "medium" isn't a
    config artefact — it's the underlying difficulty of the task. If "full"
    gives much lower reset reward, then medium is an artificially-easy
    starting point (current state of run8).

Usage:
    uv run python scripts/supersims_gt_verify.py [--n-seeds 16]
"""
import argparse
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


CONFIGS = {
    "full":   "parameter_config.json",
    "medium": "parameter_config_medium.json",
    "narrow": "parameter_config_narrow.json",
}


def _load_cfg_into_pg(pg, cfg_name: str):
    """Monkey-patch parameter_generation._cfg with the named config file."""
    path = Path(pg.__file__).parent / cfg_name
    pg._cfg = json.loads(path.read_text())


def _gt_params(omega_01, t_g, beta_default: float = 0.5) -> jnp.ndarray:
    """Construct (N_QUBITS, 5) physical params at the perfect-pulse setpoint.

    beta_default = 0.5 is the optimum for the raised-cosine envelope used in
    SuperSims/hamiltonian_definitions.py (NOT 1.0 — the docstring there claims 1.0
    is "standard optimal" but that's the Gaussian-envelope result; for raised cosine
    the empirical optimum is 0.5, confirmed by sweep in scripts/diag_ceiling.py).
    """
    Omega_opt = 2 * jnp.pi / t_g
    N = omega_01.shape[0]
    return jnp.column_stack([
        omega_01,                         # omega_01: episode-fixed
        omega_01,                         # omega_d  = omega_01 (no detuning)
        jnp.zeros((N,)),                  # phi = 0
        jnp.full((N,), Omega_opt),        # Omega = 2π/t_g
        jnp.full((N,), beta_default),     # beta = 1
    ])


def _eval_reward(params, hw, t_g, alpha, lambda_, run_allxy_simulation, allxy_rewards):
    P1 = run_allxy_simulation(params, hw, t_g, alpha, lambda_)
    rewards, _ = allxy_rewards(P1)
    return float(jnp.mean(rewards)), np.asarray(rewards, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=16)
    args = ap.parse_args()

    rows = []
    for cfg_name, cfg_file in CONFIGS.items():
        # parameter_generation reads _cfg at import time AND on each sample call,
        # but N_QUBITS / N are module constants. We need a fresh import for each
        # config so the rest of the SuperSims modules pick up consistent state.
        for mod in [m for m in list(sys.modules) if m in {
            "parameter_generation", "all_xy_sequence", "compensation_matrix",
            "reward", "hamiltonian_definitions", "normalisations",
        }]:
            del sys.modules[mod]

        pg = importlib.import_module("parameter_generation")
        _load_cfg_into_pg(pg, cfg_file)
        all_xy_sequence = importlib.import_module("all_xy_sequence")
        reward_mod = importlib.import_module("reward")

        run_allxy_simulation = all_xy_sequence.run_allxy_simulation
        allxy_rewards = reward_mod.allxy_rewards

        reset_means, gt_means, gt_nohw_means, gt_ideal_means = [], [], [], []
        for s in range(args.n_seeds):
            key = jax.random.PRNGKey(s)
            (omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw) = pg.sample_all(key)

            init_params = jnp.column_stack([omega_01, omega_d, phi, Omega, beta])
            r_reset, _ = _eval_reward(init_params, hw, t_g, alpha, lambda_,
                                      run_allxy_simulation, allxy_rewards)

            gt_params = _gt_params(omega_01, t_g)
            r_gt, _ = _eval_reward(gt_params, hw, t_g, alpha, lambda_,
                                   run_allxy_simulation, allxy_rewards)

            hw_zero = jnp.zeros_like(hw).at[:, 2].set(1.0)  # Omega_scale=1, others=0
            r_gt_nohw, _ = _eval_reward(gt_params, hw_zero, t_g, alpha, lambda_,
                                        run_allxy_simulation, allxy_rewards)

            # GT + zero hw + zero crosstalk → should be the true ceiling.
            # If this is NOT ≈ 1.0, the perfect-pulse setpoint itself is wrong.
            lambda_zero = jnp.zeros_like(lambda_)
            r_gt_ideal, _ = _eval_reward(gt_params, hw_zero, t_g, alpha, lambda_zero,
                                         run_allxy_simulation, allxy_rewards)

            reset_means.append(r_reset)
            gt_means.append(r_gt)
            gt_nohw_means.append(r_gt_nohw)
            gt_ideal_means.append(r_gt_ideal)

        rows.append((cfg_name, np.array(reset_means), np.array(gt_means),
                     np.array(gt_nohw_means), np.array(gt_ideal_means)))

    print(f"\nReward summaries over {args.n_seeds} sampled episodes per config.\n")
    header = f"{'config':<8}  {'reset':<22}  {'GT':<22}  {'GT + hw=0':<22}  {'GT + hw=0 + λ=0':<22}"
    print(header)
    print("-" * len(header))
    for cfg_name, reset, gt, gt_nohw, gt_ideal in rows:
        def fmt(a):
            return f"{a.mean():.3f} ± {a.std():.3f}"
        print(f"{cfg_name:<8}  {fmt(reset):<22}  {fmt(gt):<22}  {fmt(gt_nohw):<22}  {fmt(gt_ideal):<22}")

    print("\nInterpretation:")
    print("  - GT + hw=0 + λ=0 should be ≈ 1.0. If not, the perfect-pulse setpoint is wrong.")
    print("  - GT + hw=0 (sampled λ) tells us how much cross-talk degrades naive perfect-pulse params.")
    print("  - GT (sampled hw, sampled λ) is the agent's ceiling without C_tensor compensation.")
    print("  - The agent's *real* ceiling (with C_tensor cross-talk compensation) is higher than 'GT'.")
    print("  - Gap (GT - reset) is a *lower bound* on the improvement budget.")


if __name__ == "__main__":
    main()
