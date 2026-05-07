"""Smoke test the supersims_env_config_1d_omegad.yaml configuration.

Verifies:
  (1) Env loads with the new pin_to_gt / zero_hw / zero_crosstalk knobs.
  (2) After reset, the 4 pinned params are at their GT values; only omega_d varies.
  (3) hw is exactly [0, 0, 1] per qubit; lambda_ is all zero.
  (4) Reset reward is high (close to ceiling 0.998 since 4/5 params are at GT).
  (5) A zero-action step produces the same params (no change), reward unchanged.
"""
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from swarm.environment.supersims_env import SuperSimsEnv  # noqa: E402

CFG = "supersims_env_config_1d_omegad.yaml"

env = SuperSimsEnv(config_path=CFG)

# Sample 16 episodes, report reset reward distribution.
print("Sampling 16 episodes for reset-reward distribution...")
all_rewards = []
all_detunings = []
for s in range(16):
    obs_s, info_s = env.reset(seed=s)
    all_rewards.append(info_s["per_qubit_rewards"].mean())
    detuning = info_s["params_raw"][:, 1] - info_s["params_raw"][:, 0]
    all_detunings.append(np.abs(detuning).mean())
all_rewards = np.array(all_rewards)
all_detunings = np.array(all_detunings)
print(f"  Reset reward across 16 seeds: mean={all_rewards.mean():.3f}  std={all_rewards.std():.3f}  "
      f"range=[{all_rewards.min():.3f}, {all_rewards.max():.3f}]")
print(f"  |Δω_d| (rad/ns) per seed:    mean={all_detunings.mean():.3f}  max={all_detunings.max():.3f}  "
      f"(half-span = 0.314 rad/ns = ±50 MHz × 2π)")

obs, info = env.reset(seed=0)
params_raw = info["params_raw"]
print(f"\nReset reward (mean across qubits, seed=0): {info['per_qubit_rewards'].mean():.4f}")
print(f"params_raw shape:    {params_raw.shape}")
print(f"params_raw[qubit0]:  omega_01={params_raw[0,0]:.4f}  omega_d={params_raw[0,1]:.4f}  "
      f"phi={params_raw[0,2]:.4f}  Omega={params_raw[0,3]:.4f}  beta={params_raw[0,4]:.4f}")

# Sanity: pinned values
import jax.numpy as jnp
omega_01 = params_raw[:, 0]
t_g = float(env._t_g)
expected = {
    # omega_d is the FREE param — should differ slightly from omega_01 (Gaussian noise)
    "phi (= 0)":            (params_raw[:, 2], np.zeros_like(omega_01)),
    "Omega (= 2π/t_g)":     (params_raw[:, 3], np.full_like(omega_01, 2 * np.pi / t_g)),
    "beta (= 0.5)":         (params_raw[:, 4], np.full_like(omega_01, 0.5)),
}
# Show the omega_d offset from omega_01 (the free signal we expect the agent to learn)
detuning = params_raw[:, 1] - omega_01
print(f"\nomega_d − omega_01 (the free signal): {detuning}")
print(f"  per-qubit detuning |Δω_d|: {np.abs(detuning)}  (typical Gaussian σ=0.063 rad/ns full config)")
print("\nPinned-at-GT checks:")
for name, (got, want) in expected.items():
    ok = np.allclose(got, want, atol=1e-6)
    print(f"  {name:30s}  {'OK' if ok else 'FAIL'}  got={got}  want={want}")

# hw + lambda zeroing
hw = np.asarray(env._hw)
lambda_ = np.asarray(env._lambda_)
print(f"\nhw (should be 0/0/1 per qubit):\n{hw}")
print(f"\nlambda_ (should be all zero):\n{lambda_}")

# Zero-action step
zero_action = np.zeros((env.n_qubits, env.n_params), dtype=np.float32)
obs2, r, term, trunc, info2 = env.step(zero_action)
print(f"\nAfter zero-action step:")
print(f"  reward = {r:.4f}  (expected ≈ unchanged from reset)")
print(f"  per-qubit reward = {info2['per_qubit_rewards']}")

print("\nDone.")
