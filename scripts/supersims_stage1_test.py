"""
Stage 1 visual verification for SuperSimsEnv.

Compares two trajectories on the same episode:
  - "no_tune":  zero action every step (rewards should hover near baseline).
  - "oracle":   gradient-following policy that knows the targets (rewards should
                climb monotonically toward 1.0). Validates that the compensation-
                tensor virtualisation is wired correctly — if C is wrong, the
                oracle won't converge.

Also renders the All-XY staircase before & after the oracle episode to confirm
the agent's parameter updates actually move P1 toward the ideal pattern.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/supersims_stage1_test.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "SuperSims"))

from swarm.environment.supersims_env import SuperSimsEnv  # noqa: E402
from all_xy_sequence import ALLXY_GATES, ALLXY_IDEAL  # noqa: E402
from compensation_matrix import build_compensation  # noqa: E402
from reward import allxy_rewards  # noqa: E402

_TARGETS = jnp.array(ALLXY_IDEAL)


def _oracle_action(env, lr: float = 0.5) -> np.ndarray:
    """Newton-step oracle.

    For each qubit i, take a damped Newton step toward the ideal staircase using the
    self-Jacobian J_self[i] = ∂P1[i]/∂params[i]. The step in physical units is

        delta_phys[i] = -lr · pinv(J_self[i]) @ (P1[i] - target),

    converted to normalised action by dividing by `delta_scales`.

    Verifies that the env's compensation tensor wiring + parameter update path is
    sign-correct — if either is flipped, this Newton step diverges instead of
    converging.
    """
    # Recompute the Jacobian from current env state (the env stores params/hw/etc).
    _, J_cols = build_compensation(env._params, env._hw, env._t_g, env._alpha, env._lambda_)
    P1 = np.asarray(env._run_sim())
    targets = np.asarray(_TARGETS)

    n_qubits, n_params = env.n_qubits, env.n_params
    action = np.zeros((n_qubits, n_params), dtype=np.float32)
    delta_scales = np.asarray(env._delta_scales)

    for i in range(n_qubits):
        J_self = np.asarray(J_cols[i][i])             # (21, 5)
        residual = P1[i] - targets                    # (21,)
        # Damped pseudo-inverse: drop singular values below 1% of the largest.
        # Without damping, ill-conditioned J_self at random init blows up the step.
        delta_phys = -lr * (np.linalg.pinv(J_self, rcond=1e-2) @ residual)  # (5,)
        action[i] = delta_phys / delta_scales
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_episode(env, policy_fn, seed: int):
    """Run one episode. policy_fn takes the env (so it can peek at internal state for the
    oracle's Jacobian). Returns per-step rewards (n_steps+1, n_qubits) and (P1_first, P1_last)."""
    obs, info = env.reset(seed=seed)
    P1_first = obs["staircase"].copy()
    rewards = [info["per_qubit_rewards"].copy()]
    for _ in range(env.max_steps):
        action = policy_fn(env)
        obs, _, terminated, _, info = env.step(action)
        rewards.append(info["per_qubit_rewards"].copy())
        if terminated:
            break
    P1_last = obs["staircase"].copy()
    return np.asarray(rewards), P1_first, P1_last


def plot_reward_curves(rewards_no_tune, rewards_oracle, out_path):
    n_steps, n_qubits = rewards_no_tune.shape
    fig, axes = plt.subplots(n_qubits, 1, figsize=(8, 2.2 * n_qubits), sharex=True)
    if n_qubits == 1:
        axes = [axes]
    x = np.arange(n_steps)
    for i, ax in enumerate(axes):
        ax.plot(x, rewards_no_tune[:, i], color="#888", lw=1.4, label="zero action")
        ax.plot(x, rewards_oracle[:, i], color="#d62728", lw=1.8, label="oracle")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Q{i} reward")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Stage 1: per-qubit reward — zero-action vs oracle", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reward curves → {out_path}")


def plot_staircase_before_after(P1_first, P1_last, out_path):
    n_qubits, n_allxy = P1_first.shape
    seq_labels = [f"({g1},{g2})" for g1, g2 in ALLXY_GATES]
    ideal = np.asarray(ALLXY_IDEAL)
    fig, axes = plt.subplots(n_qubits, 1, figsize=(13, 2.6 * n_qubits), sharex=True, sharey=True)
    if n_qubits == 1:
        axes = [axes]
    x = np.arange(n_allxy)
    for i, ax in enumerate(axes):
        ax.axhspan(-0.1, 0.25, color="royalblue", alpha=0.05)
        ax.axhspan(0.25, 0.75, color="gray", alpha=0.05)
        ax.axhspan(0.75, 1.15, color="firebrick", alpha=0.05)
        ax.step(x, ideal, where="mid", color="black", lw=1.0, ls="--", alpha=0.5, label="ideal")
        ax.step(x, P1_first[i], where="mid", color="#888", lw=1.4, label="step 0")
        ax.step(x, P1_last[i], where="mid", color="#d62728", lw=1.8, label="final")
        ax.set_ylim(-0.1, 1.15)
        ax.set_ylabel(f"Q{i}  P(|1⟩)")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Gate sequence")
    fig.suptitle("Stage 1: All-XY staircase before/after oracle episode", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved staircase plot → {out_path}")


def main():
    out_dir = _REPO_ROOT / "scripts" / "supersims_stage1_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building env (this triggers JIT warmup, ~30s)...")
    env = SuperSimsEnv()

    SEED = 0
    print(f"\nRunning zero-action episode (seed={SEED})...")
    rewards_zero, P1_first_z, P1_last_z = run_episode(
        env, lambda e: np.zeros((e.n_qubits, e.n_params), dtype=np.float32), seed=SEED
    )
    print(f"  zero-action: mean reward step 0 = {rewards_zero[0].mean():.4f}, final = {rewards_zero[-1].mean():.4f}")

    LR = 0.1
    print(f"\nRunning oracle episode (same seed={SEED}, lr={LR})...")
    rewards_oracle, P1_first_o, P1_last_o = run_episode(
        env, lambda e: _oracle_action(e, lr=LR), seed=SEED,
    )
    print(f"  oracle:      mean reward step 0 = {rewards_oracle[0].mean():.4f}, final = {rewards_oracle[-1].mean():.4f}")

    # Sanity: same seed → same P1 at step 0
    assert np.allclose(P1_first_z, P1_first_o), "different reset given same seed!"
    print("  determinism check: ✓ (zero & oracle reset to identical staircase)")

    plot_reward_curves(rewards_zero, rewards_oracle, out_dir / "reward_curves.png")
    plot_staircase_before_after(P1_first_o, P1_last_o, out_dir / "staircase_before_after.png")

    print("\nStage 1 verification complete. Inspect plots in:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
