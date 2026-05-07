import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import parameter_generation as pg
from parameter_generation import N_QUBITS
from all_xy_sequence import run_allxy_simulation, N_ALLXY
from compensation_matrix import build_compensation
from reward import allxy_rewards
import normalisations as norm

print(f"\n  N_QUBITS={N_QUBITS}, N_ALLXY={N_ALLXY}, JAX backend: {jax.default_backend()}")

key = jax.random.PRNGKey(0)
omega_01, alpha, lambda_, t_g, omega_d, phi, Omega, beta, hw = pg.sample_all(key)
params = jnp.column_stack([omega_01, omega_d, phi, Omega, beta])

# Warmup
print("\n  Warming JIT...")
_ = run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=True).block_until_ready()
_ = run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=False).block_until_ready()
C, J_cols = build_compensation(params, hw, t_g, alpha, lambda_)
jax.block_until_ready(C)

# Timed
def t_op(name, fn, n=10):
    fn().block_until_ready() if hasattr(fn(), "block_until_ready") else None  # warmup once
    s = time.perf_counter()
    for _ in range(n):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        elif isinstance(out, tuple):
            jax.block_until_ready(out)
    e = time.perf_counter()
    print(f"  {name:<40} {(e-s)/n*1e3:>8.2f} ms")

print("\n  Op timings (avg over 10 calls):")
t_op("run_allxy_simulation(simul=True)",  lambda: run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=True))
t_op("run_allxy_simulation(simul=False)", lambda: run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=False))
t_op("build_compensation (jacfwd)",       lambda: build_compensation(params, hw, t_g, alpha, lambda_))
t_op("allxy_rewards",                     lambda: allxy_rewards(run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=True)))

# Show shapes
P1 = run_allxy_simulation(params, hw, t_g, alpha, lambda_, simultaneous=True)
rewards, deviations = allxy_rewards(P1)
print(f"\n  Shapes:")
print(f"    params       : {params.shape}    [omega_01, omega_d, phi, Omega, beta]")
print(f"    P1 (obs?)    : {P1.shape}        per-qubit {N_ALLXY}-vector staircase")
print(f"    rewards      : {rewards.shape}        per-qubit reward in [0,1]")
print(f"    C_tensor     : {C.shape}    (i,k)<-(j,l) virtual->physical map")
print(f"    hw           : {hw.shape}        (phi_hw, t_delay, Omega_scale)")
print(f"    lambda_      : {lambda_.shape}    crosstalk matrix")

print(f"\n  Reward sample (random params): {rewards}")
print(f"  Mean reward: {float(jnp.mean(rewards)):.4f}")
