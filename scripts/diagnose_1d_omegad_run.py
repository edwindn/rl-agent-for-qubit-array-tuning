"""
Two-part diagnostic for the 1D omega_d pinned training run.

(1) For every saved iter_<N> in checkpoints_supersims_1d_omegad_pinned/:
    - Extract each per-param policy's final-layer log_std *bias* (the
      orthogonal-init constant offset on log_std output) and the empirical
      log_std_predicted_at_obs (mean over a fixed bank of obs from N reset
      seeds — captures both bias + obs-dependent contribution).
    - Plot 5 policies × 30 iters of mean predicted std vs iter.
    - Overlay the iteration-mean reward (from the training log).

(2) Roll out iter_30 greedy-policy on M seeds:
    - For each seed, plot |omega_d - omega_01| over the 20 episode steps.
    - Plot per-qubit reward over the same steps.
    - Overlay a zero-action rollout for comparison.

Outputs PNGs to plots_supersims_diagnostic/.

Usage:
  CUDA_VISIBLE_DEVICES=5 uv run python scripts/diagnose_1d_omegad_run.py
"""
import argparse
import re
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
from swarm.inference.eval_supersims import (  # noqa: E402
    load_modules_from_checkpoint, greedy_action,
)

PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]
PARAM_LATEX = [r"$\omega_{01}$", r"$\omega_d$", r"$\phi$", r"$\Omega$", r"$\beta$"]

# Force matplotlib to NOT use system LaTeX (avoid Unicode-in-LaTeX errors).
matplotlib.rcParams["text.usetex"] = False


def _final_layer(module):
    pi = module.pi
    seq = pi.net if hasattr(pi, "net") else (pi.mlp if hasattr(pi, "mlp") else pi)
    last_linear = None
    for m in seq.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No nn.Linear found in policy head")
    return last_linear


@torch.no_grad()
def _policy_logits(module, obs_batch):
    from ray.rllib.core.columns import Columns
    out = module._forward({"obs": obs_batch})
    return out[Columns.ACTION_DIST_INPUTS]  # (B, 2)  — [mean, log_std]


def collect_obs_bank(env: SuperSimsEnv, n_episodes: int = 16):
    staircases, params = [], []
    for s in range(n_episodes):
        obs, _ = env.reset(seed=s)
        staircases.append(obs["staircase"])
        params.append(obs["params"])
    return {
        "staircase": torch.from_numpy(np.concatenate(staircases, axis=0).astype(np.float32)),
        "params":    torch.from_numpy(np.concatenate(params, axis=0).astype(np.float32)),
    }


def parse_reward_log(log_path: Path) -> np.ndarray:
    """Return per-step-per-agent mean reward per iter (episode_return / 400)."""
    rewards = []
    pattern = re.compile(r"Episode Returns \| Mean:\s+([\-\d\.]+)")
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            rewards.append(float(m.group(1)) / 400.0)
    return np.array(rewards)


def diagnostic_1_logstd_curve(
    checkpoint_root: Path,
    log_path: Path,
    config_path: str,
    out_dir: Path,
    n_obs_eps: int = 16,
):
    print(f"\n[Diagnostic 1] Scanning {checkpoint_root} ...")
    iter_dirs = sorted(checkpoint_root.glob("iteration_*"),
                       key=lambda p: int(p.name.split("_")[-1]))
    print(f"  Found {len(iter_dirs)} iteration checkpoints.")

    env = SuperSimsEnv(config_path=config_path)
    obs_batch = collect_obs_bank(env, n_episodes=n_obs_eps)
    print(f"  Built obs bank from {n_obs_eps} reset seeds × {env.n_qubits} qubits "
          f"= {obs_batch['staircase'].shape[0]} obs.")

    iters = []
    pred_logstd = {p: [] for p in PARAM_NAMES}        # mean over obs bank of predicted log_std

    for ckpt_dir in iter_dirs:
        it = int(ckpt_dir.name.split("_")[-1])
        try:
            split, modules = load_modules_from_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"  iter {it}: load failed ({e}); skipping")
            continue
        iters.append(it)
        for pname in PARAM_NAMES:
            m = modules[f"{pname}_policy"]
            # Either free_log_std (param exists) or state-dependent (forward + read second half).
            if hasattr(m.pi, "log_std_param"):
                pred_logstd[pname].append(float(m.pi.log_std_param.detach().mean().item()))
            else:
                logits = _policy_logits(m, obs_batch).cpu().numpy()     # (B, 2)
                pred_logstd[pname].append(float(logits[:, 1].mean()))

    iters = np.array(iters)
    rewards = parse_reward_log(log_path)
    n = min(len(iters), len(rewards))
    iters = iters[:n]
    rewards = rewards[:n]
    print(f"  Aligned {n} iters with reward log.")

    # --- Plot ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax_top.plot(iters, rewards, "k-o", lw=2, ms=4, label="mean reward (per-step-per-agent)")
    ax_top.axhline(0.643, color="grey", ls=":", lw=1.5, label="reset baseline (no-learn)")
    ax_top.axhline(0.998, color="green", ls=":", lw=1.5, label="optimal ceiling")
    ax_top.set_ylabel("episode-mean reward")
    ax_top.set_title("1D omega_d pinned run: reward + per-policy log_std vs iter")
    ax_top.legend(loc="lower right", fontsize=8)
    ax_top.set_ylim(0.5, 1.0)
    ax_top.grid(True, alpha=0.3)

    colors = plt.cm.tab10.colors
    for i, pname in enumerate(PARAM_NAMES):
        std_pred = np.exp(np.array(pred_logstd[pname][:n]))
        ax_bot.plot(iters, std_pred, color=colors[i], lw=2, marker="o", ms=4,
                    label=f"{PARAM_LATEX[i]} {'(pinned)' if pname != 'omegad' else '(FREE)'}")
    ax_bot.axhline(np.exp(-3), color="grey", ls=":", lw=1.0, label="upper bound exp(-3)=0.05")
    ax_bot.axhline(np.exp(-10), color="grey", ls="--", lw=1.0, label="lower bound exp(-10) ~ 4.5e-5")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("training iter")
    ax_bot.set_ylabel("predicted std (mean over obs bank)")
    ax_bot.legend(loc="best", fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3, which="both")

    out_path = out_dir / "1d_omegad_pinned_reward_and_logstd.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")

    # Also print the table for the last iter so we can read it from stdout.
    print(f"\n  log_std at iter {iters[-1]} (averaged over {obs_batch['staircase'].shape[0]} obs):")
    for pname in PARAM_NAMES:
        ls = pred_logstd[pname][-1]
        print(f"    {pname:8s}  log_std={ls:+.3f}  std={np.exp(ls):.4f}")


def diagnostic_2_eval_rollout(
    checkpoint_dir: Path,
    config_path: str,
    out_dir: Path,
    n_seeds: int = 8,
):
    print(f"\n[Diagnostic 2] Rolling out {checkpoint_dir.name} on {n_seeds} seeds ...")
    split, modules = load_modules_from_checkpoint(checkpoint_dir)
    env = SuperSimsEnv(config_path=config_path)
    n_qubits = env.n_qubits
    n_steps = env.max_steps

    # Containers: (n_seeds, n_steps+1, n_qubits)
    detuning_trained = np.zeros((n_seeds, n_steps + 1, n_qubits))
    rewards_trained  = np.zeros((n_seeds, n_steps + 1, n_qubits))
    detuning_zero    = np.zeros((n_seeds, n_steps + 1, n_qubits))
    rewards_zero     = np.zeros((n_seeds, n_steps + 1, n_qubits))
    omega_d_trained  = np.zeros((n_seeds, n_steps + 1, n_qubits))
    omega_01_init    = np.zeros((n_seeds, n_qubits))

    zero_action = np.zeros((n_qubits, env.n_params), dtype=np.float32)

    for s in range(n_seeds):
        # Trained rollout
        obs, info = env.reset(seed=s)
        omega_01_init[s] = info["params_raw"][:, 0]
        for t in range(n_steps + 1):
            params_raw = info["params_raw"]
            omega_d_trained[s, t] = params_raw[:, 1]
            detuning_trained[s, t] = params_raw[:, 1] - params_raw[:, 0]
            rewards_trained[s, t]  = info["per_qubit_rewards"]
            if t == n_steps:
                break
            action = greedy_action(split, modules, obs["staircase"], obs["params"])
            obs, _, _, _, info = env.step(action)

        # Zero-action rollout (same seed)
        obs, info = env.reset(seed=s)
        for t in range(n_steps + 1):
            params_raw = info["params_raw"]
            detuning_zero[s, t] = params_raw[:, 1] - params_raw[:, 0]
            rewards_zero[s, t]  = info["per_qubit_rewards"]
            if t == n_steps:
                break
            obs, _, _, _, info = env.step(zero_action)

    # --- Plot per-seed: detuning + reward, with trained vs zero-action overlaid ---
    fig, axes = plt.subplots(n_seeds, 2, figsize=(14, 3 * n_seeds), squeeze=False)
    for s in range(n_seeds):
        ax_l = axes[s, 0]
        ax_r = axes[s, 1]
        time = np.arange(n_steps + 1)
        # Detuning, in MHz
        det_t = detuning_trained[s] * 1000.0 / (2 * np.pi)
        det_0 = detuning_zero[s]    * 1000.0 / (2 * np.pi)
        for q in range(n_qubits):
            ax_l.plot(time, det_t[:, q], color=plt.cm.tab10(q), lw=2,
                      label=f"q{q} trained" if s == 0 else None)
            ax_l.plot(time, det_0[:, q], color=plt.cm.tab10(q), lw=1, ls=":",
                      alpha=0.6, label=f"q{q} zero" if s == 0 else None)
        ax_l.axhline(0, color="grey", lw=0.6)
        ax_l.axhline( 50, color="red", ls="--", lw=0.6, alpha=0.5)
        ax_l.axhline(-50, color="red", ls="--", lw=0.6, alpha=0.5)
        ax_l.set_ylabel(f"seed {s}\ndetuning (MHz)")
        ax_l.grid(True, alpha=0.3)

        ax_r.plot(time, rewards_trained[s].mean(axis=1), "k-", lw=2, label="trained mean")
        ax_r.plot(time, rewards_zero[s].mean(axis=1),    "k:", lw=1.5, label="zero-action mean")
        for q in range(n_qubits):
            ax_r.plot(time, rewards_trained[s, :, q], color=plt.cm.tab10(q), lw=0.8, alpha=0.5)
        ax_r.set_ylim(0.4, 1.01)
        ax_r.set_ylabel("reward")
        ax_r.grid(True, alpha=0.3)
        if s == 0:
            ax_l.legend(loc="best", fontsize=7, ncol=2)
            ax_r.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("episode step")
    axes[-1, 1].set_xlabel("episode step")
    plt.suptitle(f"{checkpoint_dir.name}: per-seed rollout — detuning (left) and reward (right)\n"
                 f"trained-greedy (solid, thick) vs zero-action (dotted)", y=1.0)
    out_path = out_dir / f"1d_omegad_rollout_{checkpoint_dir.name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")

    # Print summary
    print(f"\n  Per-seed summary (mean across qubits):")
    print(f"    {'seed':>4}  {'init|Δω_d| MHz':>16}  {'init reward':>12}  "
          f"{'final|Δω_d| MHz':>16}  {'final reward':>13}  {'Δreward (vs zero)':>18}")
    for s in range(n_seeds):
        init_d = np.abs(detuning_trained[s, 0]).mean() * 1000.0 / (2 * np.pi)
        init_r = rewards_trained[s, 0].mean()
        fin_d  = np.abs(detuning_trained[s, -1]).mean() * 1000.0 / (2 * np.pi)
        fin_r  = rewards_trained[s, -1].mean()
        zero_r = rewards_zero[s, -1].mean()
        print(f"    {s:>4}  {init_d:>16.2f}  {init_r:>12.3f}  "
              f"{fin_d:>16.2f}  {fin_r:>13.3f}  {fin_r - zero_r:>+18.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=str,
                    default="checkpoints_supersims_1d_omegad_pinned")
    ap.add_argument("--log", type=str, default="/tmp/supersims_1d_omegad_pinned.log")
    ap.add_argument("--config", type=str,
                    default="supersims_env_config_1d_omegad.yaml")
    ap.add_argument("--out-dir", type=str, default="plots_supersims_diagnostic")
    ap.add_argument("--rollout-iter", type=int, default=30,
                    help="Iter checkpoint to roll out for Diagnostic 2.")
    ap.add_argument("--n-seeds", type=int, default=8)
    args = ap.parse_args()

    ckpt_root = (_REPO / args.ckpt_root).resolve()
    log_path  = Path(args.log)
    out_dir   = (_REPO / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_1_logstd_curve(
        checkpoint_root=ckpt_root,
        log_path=log_path,
        config_path=args.config,
        out_dir=out_dir,
    )

    rollout_ckpt = ckpt_root / f"iteration_{args.rollout_iter}"
    if not rollout_ckpt.exists():
        rollout_ckpt = max(ckpt_root.glob("iteration_*"),
                           key=lambda p: int(p.name.split("_")[-1]))
        print(f"\n  iter_{args.rollout_iter} not found — using latest: {rollout_ckpt.name}")

    diagnostic_2_eval_rollout(
        checkpoint_dir=rollout_ckpt,
        config_path=args.config,
        out_dir=out_dir,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    main()
