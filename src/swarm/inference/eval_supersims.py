"""
Stage 4 evaluation + visualisation for SuperSims agents.

Loads a checkpoint, runs one episode under the trained policy (greedy = mean of the
Gaussian), and produces three plots:

  1. Per-qubit reward trajectory across the episode.
  2. All-XY staircase before vs after the episode (target overlay).
  3. Per-qubit parameter trajectories for ω₀₁, ω_d, φ, Ω, β.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python src/swarm/inference/eval_supersims.py \
    --checkpoint checkpoints_supersims/iteration_50

Defaults to the most recent iteration in `checkpoints_supersims/`.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "SuperSims"))

from swarm.environment.supersims_env import SuperSimsEnv  # noqa: E402
from swarm.voltage_model.create_rl_module import create_rl_module_spec  # noqa: E402
from all_xy_sequence import ALLXY_GATES, ALLXY_IDEAL  # noqa: E402

PARAM_NAMES = [r"$\omega_{01}$", r"$\omega_d$", r"$\phi$", r"$\Omega$", r"$\beta$"]


def find_latest_checkpoint(root: Path) -> Path:
    candidates = list(root.glob("iteration_*"))
    if not candidates:
        raise FileNotFoundError(f"No iteration_* checkpoints in {root}")
    return max(candidates, key=lambda p: int(p.name.split("_")[-1]))


_PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]


def load_modules_from_checkpoint(checkpoint_dir: Path):
    """Returns (policy_split, modules_dict).

    Uses Ray's MultiRLModule.from_checkpoint, which reads the layout and ctor args
    from class_and_ctor_args.pkl alongside module_state.pkl, so we don't have to
    rebuild the spec ourselves and the encoder layout (shared vs split) auto-matches.
    """
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

    rl_module_dir = next(checkpoint_dir.rglob("rl_module"), None)
    if rl_module_dir is None:
        raise FileNotFoundError(f"No rl_module directory under {checkpoint_dir}")
    multi_module = MultiRLModule.from_checkpoint(str(rl_module_dir))

    policy_ids = list(multi_module.keys())
    print(f"  Loaded MultiRLModule from {rl_module_dir.relative_to(checkpoint_dir)} "
          f"with policies {policy_ids}")
    if "qubit_policy" in policy_ids and len(policy_ids) == 1:
        policy_split = "per_qubit"
    elif set(policy_ids) == {f"{p}_policy" for p in _PARAM_NAMES}:
        policy_split = "per_param"
    elif set(policy_ids) == {"freq_policy", "env_policy"}:
        policy_split = "grouped"
    else:
        raise RuntimeError(f"Unrecognised policy set {policy_ids}")

    modules = {pid: multi_module[pid] for pid in policy_ids}
    for m in modules.values():
        m.eval()
    return policy_split, modules


# Back-compat alias used by existing callers.
def load_module_from_checkpoint(checkpoint_dir: Path) -> torch.nn.Module:
    split, modules = load_modules_from_checkpoint(checkpoint_dir)
    if split != "per_qubit":
        raise RuntimeError(
            f"load_module_from_checkpoint only supports per_qubit; got {split}. "
            f"Use load_modules_from_checkpoint instead."
        )
    return modules["qubit_policy"]


@torch.no_grad()
def _greedy_mean(module, obs_batch) -> np.ndarray:
    """Greedy = mean of the Gaussian. Returns (N, action_dim) numpy array."""
    from ray.rllib.core.columns import Columns
    out = module._forward({"obs": obs_batch})
    logits = out[Columns.ACTION_DIST_INPUTS]
    mean = logits[:, : logits.shape[-1] // 2]
    return np.clip(mean.cpu().numpy(), -1.0, 1.0).astype(np.float32)


@torch.no_grad()
def greedy_action(policy_split: str, modules: dict,
                  staircase_np: np.ndarray, params_np: np.ndarray) -> np.ndarray:
    """Returns (N_QUBITS, 5) greedy action stacked across qubits."""
    obs_batch = {
        "staircase": torch.from_numpy(staircase_np.astype(np.float32)),
        "params":    torch.from_numpy(params_np.astype(np.float32)),
    }
    if policy_split == "per_qubit":
        return _greedy_mean(modules["qubit_policy"], obs_batch)
    if policy_split == "grouped":
        # freq_policy → action cols [0, 1, 2]; env_policy → cols [3, 4].
        freq = _greedy_mean(modules["freq_policy"], obs_batch)  # (N, 3)
        env_ = _greedy_mean(modules["env_policy"], obs_batch)   # (N, 2)
        return np.concatenate([freq, env_], axis=1).astype(np.float32)
    # per_param: query each policy with the same per-qubit obs batch and stack columns.
    cols = []
    for pname in _PARAM_NAMES:
        col = _greedy_mean(modules[f"{pname}_policy"], obs_batch)  # shape (N, 1)
        cols.append(col)
    return np.concatenate(cols, axis=1).astype(np.float32)  # (N, 5)


def run_eval_episode(env: SuperSimsEnv, policy_split: str, modules: dict, seed: int):
    obs, info = env.reset(seed=seed)
    # obs["staircase"] is now normalised to [-1, 1] (= 2·P1 − 1). For plotting we
    # want raw P(|1⟩) ∈ [0, 1]; reverse the normalisation.
    P1_first = ((obs["staircase"] + 1.0) / 2.0).copy()
    rewards = [info["per_qubit_rewards"].copy()]
    params_traj = [info["params_raw"].copy()]
    for _ in range(env.max_steps):
        action = greedy_action(policy_split, modules, obs["staircase"], obs["params"])
        obs, _, terminated, _, info = env.step(action)
        rewards.append(info["per_qubit_rewards"].copy())
        params_traj.append(info["params_raw"].copy())
        if terminated:
            break
    P1_last = ((obs["staircase"] + 1.0) / 2.0).copy()
    return np.asarray(rewards), np.asarray(params_traj), P1_first, P1_last


def plot_reward_curve(rewards: np.ndarray, out_path: Path):
    n_steps, n_qubits = rewards.shape
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_steps)
    for i in range(n_qubits):
        ax.plot(x, rewards[:, i], lw=1.5, label=f"Q{i}")
    ax.plot(x, rewards.mean(axis=1), color="black", lw=2.0, ls="--", label="mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Per-qubit reward trajectory (eval episode, greedy policy)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_staircase_before_after(P1_first: np.ndarray, P1_last: np.ndarray, out_path: Path):
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
        ax.step(x, P1_first[i], where="mid", color="#888", lw=1.4, label="step 0 (random init)")
        ax.step(x, P1_last[i], where="mid", color="#d62728", lw=1.8, label="final (after eval)")
        ax.set_ylim(-0.1, 1.15)
        ax.set_ylabel(f"Q{i}  P(|1⟩)")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Gate sequence")
    fig.suptitle("All-XY staircase before/after eval episode", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_param_trajectories(params_traj: np.ndarray, out_path: Path):
    """params_traj: (n_steps+1, n_qubits, 5)."""
    n_steps, n_qubits, n_params = params_traj.shape
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2.0 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    x = np.arange(n_steps)
    for k, ax in enumerate(axes):
        for i in range(n_qubits):
            ax.plot(x, params_traj[:, i, k], lw=1.3, label=f"Q{i}")
        ax.set_ylabel(PARAM_NAMES[k])
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8, ncols=n_qubits)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Per-qubit parameter trajectories under greedy policy", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific iteration_X dir. Default: latest in checkpoints_supersims/")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the eval episode.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir for plots (default: <ckpt>/eval_outputs)")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
    else:
        checkpoint_path = find_latest_checkpoint(_REPO_ROOT / "checkpoints_supersims")

    out_dir = Path(args.out) if args.out else checkpoint_path / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Out dir:    {out_dir}")

    print("\nLoading policy modules from checkpoint...")
    policy_split, modules = load_modules_from_checkpoint(checkpoint_path)

    print("\nBuilding env (JIT warmup ~30s)...")
    env = SuperSimsEnv()

    print(f"\nRunning eval episode (seed={args.seed}, policy_split={policy_split})...")
    rewards, params_traj, P1_first, P1_last = run_eval_episode(env, policy_split, modules, args.seed)

    print(f"  Step 0 mean reward: {rewards[0].mean():.4f}")
    print(f"  Final mean reward:  {rewards[-1].mean():.4f}")
    print(f"  Per-qubit final:    {rewards[-1]}")

    plot_reward_curve(rewards, out_dir / "eval_reward_curve.png")
    plot_staircase_before_after(P1_first, P1_last, out_dir / "eval_staircase.png")
    plot_param_trajectories(params_traj, out_dir / "eval_params.png")

    print(f"\nWrote 3 plots to {out_dir}:")
    print("  eval_reward_curve.png   eval_staircase.png   eval_params.png")


if __name__ == "__main__":
    main()
