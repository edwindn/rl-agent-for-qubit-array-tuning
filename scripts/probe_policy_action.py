"""
Direct probe of the omegad_policy: feed synthetic obs with varying detuning, all
other env state held fixed, observe the policy's mean output.

This bypasses the rollout loop and asks: "what action does the policy assign
to obs(detuning=X)?" — answering whether the policy *can* produce strong
correction actions, or whether it's saturated/timid by design.

Two probes:
  (A) Hold staircase fixed at "GT staircase" and only vary the params[:, 1]
      (normalized detuning) channel. This isolates the params signal.
  (B) For each test detuning, run the simulator forward to get the actual
      staircase that detuning produces (with other params at GT, hw=0, λ=0),
      then query the policy. This is the realistic probe.

Outputs a plot: x = detuning (MHz), y = policy mean action (in [-1,1]).
The ideal policy is a near-step function: action ≈ +1 for negative detuning
(needs to add to omega_d), ≈ -1 for positive detuning.

Usage:
  CUDA_VISIBLE_DEVICES=2 uv run python scripts/probe_policy_action.py \\
    --ckpt checkpoints_supersims_1d_omegad_freelogstd/iteration_17
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from swarm.environment.supersims_env import SuperSimsEnv  # noqa: E402
from swarm.inference.eval_supersims import load_modules_from_checkpoint  # noqa: E402


@torch.no_grad()
def policy_mean(module, obs_batch):
    from ray.rllib.core.columns import Columns
    out = module._forward({"obs": obs_batch})
    logits = out[Columns.ACTION_DIST_INPUTS]
    half = logits.shape[-1] // 2
    return logits[:, :half].cpu().numpy()  # (B, action_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints_supersims_1d_omegad_freelogstd/iteration_17")
    ap.add_argument("--config", type=str, default="supersims_env_config_1d_omegad.yaml")
    ap.add_argument("--n-detunings", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = (_REPO / args.ckpt).resolve()
    print(f"Loading checkpoint: {ckpt}")
    split, modules = load_modules_from_checkpoint(ckpt)
    omegad_pi = modules["omegad_policy"]

    # Build the env to get a representative reset state (GT staircase + GT params).
    env = SuperSimsEnv(config_path=args.config)
    obs0, info0 = env.reset(seed=args.seed)
    n_qubits = env.n_qubits
    omega_01 = info0["params_raw"][:, 0]            # (N,) — sampled
    half_span_omegad = env._param_half_spans[1]      # 50 MHz in rad/ns

    # Detuning sweep, in MHz. Convert to rad/ns for env.
    detunings_MHz = np.linspace(-50.0, 50.0, args.n_detunings, dtype=np.float64)
    detunings_radns = detunings_MHz / 1000.0 * 2 * np.pi   # (D,)

    # ----- Probe (B): realistic. For each detuning, set omega_d = omega_01 + Δ,
    # keep other params at GT, run sim → staircase. Build obs and query policy. ----- #
    print(f"\nProbe (B) realistic — detuning sweep, simulate the staircase each time:")
    print(f"  {'detuning (MHz)':>15s}  {'policy mean (q0)':>18s}  "
          f"{'policy mean (mean)':>20s}  {'norm detuning obs':>20s}")
    means = []
    for d_radns in detunings_radns:
        env.reset(seed=args.seed)
        # Override omega_d to omega_01 + d_radns for all qubits.
        new_omega_d = env._params[:, 0] + d_radns
        env._params = env._params.at[:, 1].set(new_omega_d)
        # Re-run the sim to get the new staircase
        P1 = env._run_sim()
        # Build obs
        obs = env._make_obs(P1)
        # Per-qubit obs batch (input to policy is per-qubit)
        obs_batch = {
            "staircase": torch.from_numpy(obs["staircase"].astype(np.float32)),  # (N, 21)
            "params":    torch.from_numpy(obs["params"].astype(np.float32)),     # (N, 5)
        }
        action_means = policy_mean(omegad_pi, obs_batch).flatten()  # (N,)
        means.append(action_means)
        print(f"  {float(d_radns/(2*np.pi)*1000):>15.2f}  "
              f"{action_means[0]:>18.4f}  {action_means.mean():>20.4f}  "
              f"{float(obs['params'][0, 1]):>20.4f}")
    means = np.array(means)  # (D, N)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for q in range(n_qubits):
        ax.plot(detunings_MHz, means[:, q], "-o", ms=4, lw=1.5, alpha=0.8,
                label=f"qubit {q}")
    ax.plot(detunings_MHz, means.mean(axis=1), "k--", lw=2, label="mean")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    # Reference: ideal corrective response — output -sign(detuning) at full magnitude.
    ax.plot(detunings_MHz, -np.sign(detunings_MHz), "g:", lw=1.5,
            label="ideal corrective (sign-step)")
    ax.set_xlabel("detuning omega_d - omega_01 (MHz)")
    ax.set_ylabel("policy mean action (in [-1, 1])")
    ax.set_title(f"omegad_policy mean action vs detuning (ckpt: {ckpt.name})\n"
                 f"per-qubit (4 lines) + mean — ideal: action = -sign(detuning)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out_path = _REPO / "plots_supersims_diagnostic" / f"policy_action_vs_detuning_{ckpt.name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
