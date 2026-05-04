"""
Empirical audit of what the agent sees and emits, per (qubit, param) per_param agent.

Inputs (observation):
  - staircase: 21 floats, P(|1⟩) per All-XY sequence. Naturally in [0, 1].
  - params:    5 floats per qubit, in raw physical units (rad/ns, rad, dimensionless).

Outputs (action → physical delta):
  - action: 1-d in [-1, 1]
  - physical delta = action × delta_scale × delta_scale_factor

Reports per-column empirical [mean, std, min, max] across N sampled episodes,
plus the analytical action→physical mapping.

No normalisation is applied anywhere — that's the point of this audit.

Usage:
    uv run python scripts/audit_obs_action_ranges.py [--n-episodes 50] [--config FILE]
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


PARAM_NAMES = ["omega_01", "omega_d", "phi", "Omega", "beta"]


def _reload_supersims(cfg_file: str):
    for mod in [m for m in list(sys.modules) if m in {
        "parameter_generation", "all_xy_sequence", "compensation_matrix",
        "reward", "hamiltonian_definitions", "normalisations",
    }]:
        del sys.modules[mod]
    pg = importlib.import_module("parameter_generation")
    if cfg_file:
        pg._cfg = json.loads((Path(pg.__file__).parent / cfg_file).read_text())
    return (
        pg,
        importlib.import_module("all_xy_sequence"),
        importlib.import_module("normalisations"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--config", type=str, default="parameter_config_medium.json",
                    help="parameter_config_*.json under SuperSims/. Use 'parameter_config.json' for full.")
    ap.add_argument("--delta-scale-factor", type=float, default=0.33,
                    help="Multiplier applied to delta_scales (matches run8: 0.33).")
    args = ap.parse_args()

    pg, axy, norm = _reload_supersims(args.config)
    print(f"Config: {args.config}   N_QUBITS={pg.N_QUBITS}   N (Fock dim)={pg.N}")
    print(f"delta_scale_factor: {args.delta_scale_factor}\n")

    # Sample N episodes; collect staircases (B, 21), params (B, 5), and t_g (B,).
    all_staircase = []
    all_params = []
    all_tg = []
    all_delta_scales = []
    for s in range(args.n_episodes):
        key = jax.random.PRNGKey(s)
        omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw = pg.sample_all(key)
        params = jnp.column_stack([omega_01, omega_d, phi, Omega, beta])
        P1 = axy.run_allxy_simulation(params, hw, t_g, alpha, lambda_)
        all_staircase.append(np.asarray(P1))                         # (n_qubits, 21)
        all_params.append(np.asarray(params))                        # (n_qubits, 5)
        all_tg.append(float(t_g))
        all_delta_scales.append(np.asarray(norm.episode_delta_scales(t_g)))

    # Reshape: (B = n_episodes × n_qubits, 21) and (B, 5) so each row is one agent's obs.
    staircase = np.concatenate(all_staircase, axis=0)
    params = np.concatenate(all_params, axis=0)
    delta_scales = np.array(all_delta_scales)            # (n_episodes, 5)
    print(f"Collected {staircase.shape[0]} (qubit) observations from {args.n_episodes} episodes.\n")

    # ========== Observation: staircase ========== #
    print("=" * 90)
    print("OBSERVATION — staircase (21 floats, P(|1⟩) per All-XY sequence)")
    print("=" * 90)
    print(f"{'idx':>3}  {'gate seq':<14}  {'ideal':>6}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    seqs = [f"({g1},{g2})" for g1, g2 in axy.ALLXY_GATES]
    for i in range(staircase.shape[1]):
        col = staircase[:, i]
        print(f"{i:>3}  {seqs[i]:<14}  {axy.ALLXY_IDEAL[i]:>6.1f}  "
              f"{col.mean():>8.3f}  {col.std():>8.3f}  {col.min():>8.3f}  {col.max():>8.3f}")
    print(f"\nStaircase global: range = [{staircase.min():.3f}, {staircase.max():.3f}]   "
          f"(natural [0, 1], no normalisation needed)\n")

    # ========== Observation: params ========== #
    print("=" * 90)
    print("OBSERVATION — params (5 floats per qubit, raw physical units, NO normalisation)")
    print("=" * 90)
    print(f"{'col':<12}  {'units':<14}  {'mean':>10}  {'std':>10}  "
          f"{'min':>10}  {'max':>10}  {'span':>10}")
    units = ["rad/ns", "rad/ns", "rad", "rad/ns", "dimensionless"]
    for k, name in enumerate(PARAM_NAMES):
        col = params[:, k]
        print(f"{name:<12}  {units[k]:<14}  {col.mean():>10.4f}  {col.std():>10.4f}  "
              f"{col.min():>10.4f}  {col.max():>10.4f}  {col.max() - col.min():>10.4f}")

    # Highlight the scale mismatch.
    print("\n  Per-column |mean| ratio (max / min): "
          f"{np.abs(params).mean(axis=0).max() / np.abs(params).mean(axis=0).min():.0f}×")
    print("  → the encoder's first Linear layer sees inputs spanning >100× in magnitude.\n")

    # ========== Action ranges ========== #
    print("=" * 90)
    print("ACTION — Box(-1, 1) per per_param agent (1-d), then multiplied by delta_scale × factor")
    print("=" * 90)
    delta_scales_mean = delta_scales.mean(axis=0)
    delta_scales_std  = delta_scales.std(axis=0)
    print(f"{'param':<12}  {'half-span (raw)':>18}  "
          f"{'×factor (effective)':>22}  {'units':>14}")
    for k, name in enumerate(PARAM_NAMES):
        raw = delta_scales_mean[k]
        eff = raw * args.delta_scale_factor
        std = delta_scales_std[k]
        s_indicator = " (varies w/ t_g)" if std > 1e-6 else ""
        print(f"{name:<12}  {raw:>18.4f}  {eff:>22.4f}  {units[k]:>14}{s_indicator}")
    print()
    print("  Interpretation: action = +1 on `phi`     → +1.0 × 3.14 × 0.33 = 1.04 rad shift in phi")
    print("                  action = +1 on `omega_d` → +1.0 × 0.31 × 0.33 = 0.10 rad/ns drive freq shift")
    print()

    # ========== Suggested normalisation ========== #
    print("=" * 90)
    print("SUGGESTED NORMALISATION → all obs to [-1, 1] (or close to it)")
    print("=" * 90)
    print("Staircase: leave alone (already [0, 1]; could optionally rescale to [-1, 1] via 2x-1).\n")
    print("Params: divide each column by half-span around its midpoint, i.e.")
    print("  p_norm[k] = (p[k] - midpoint[k]) / half_span[k]")
    print()
    print("Empirical midpoints / half-spans (medium config):")
    midpoints = (params.max(axis=0) + params.min(axis=0)) / 2
    half_spans = (params.max(axis=0) - params.min(axis=0)) / 2
    for k, name in enumerate(PARAM_NAMES):
        print(f"  {name:<10}  midpoint={midpoints[k]:>10.4f}   half_span={half_spans[k]:>10.4f}")
    print()
    print("After normalisation each params column would be in approximately [-1, 1].")


if __name__ == "__main__":
    main()
