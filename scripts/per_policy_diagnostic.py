"""
Per-policy activity readout for SuperSims per-param runs.

For each of the 5 per-param policies (omega01, omegad, phi, drive, beta), reports
the magnitude of greedy actions emitted across episodes — i.e. how much each
policy is actually trying to tune its parameter, averaged over qubits.

Since all 5 param-agents on a qubit share the same per-qubit reward, "per-policy
reward" is uninformative (identical across policies). Instead we look at:
    mean |a|       ← how much the policy is pushing on its param
    final action   ← what action is emitted at the end of the episode (greedy)

Usage:
  uv run python scripts/per_policy_diagnostic.py [--ckpt PATH] [--n-episodes 5]
                                                 [--narrow] [--delta-scale-factor F]

If --ckpt is omitted, picks the latest iteration_* under checkpoints_supersims/.
--narrow uses parameter_config_narrow.json. --delta-scale-factor 1.0 disables the
action shrink (use 0.1 to evaluate run7+ ckpts trained with the shrink).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from qadapt.environment.supersims_env import SuperSimsEnv  # noqa: E402
from qadapt.inference.eval_supersims import load_modules_from_checkpoint, greedy_action  # noqa: E402

PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]


def find_latest_ckpt(root: Path) -> Path:
    cands = list(root.glob("iteration_*"))
    if not cands:
        raise FileNotFoundError(f"No iteration_* under {root}")
    return max(cands, key=lambda p: int(p.name.split("_")[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--narrow", action="store_true",
                    help="Use parameter_config_narrow.json (matches run6/run7 setup).")
    ap.add_argument("--config", type=str, default=None,
                    help="Override parameter_config_filename (e.g. parameter_config_medium.json).")
    ap.add_argument("--delta-scale-factor", type=float, default=None,
                    help="Override env's delta_scale_factor. Use 1.0 for run5/run6, 0.1 for run7+.")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve() if args.ckpt else find_latest_ckpt(_REPO / "checkpoints_supersims")
    print(f"Checkpoint: {ckpt}")

    # Resolve the parameter-config filename: explicit --config wins, then --narrow,
    # else None (meaning: use the canonical parameter_config.json).
    cfg_filename = args.config or ("parameter_config_narrow.json" if args.narrow else None)
    if cfg_filename:
        import parameter_generation as pg
        pg._cfg = json.loads((Path(pg.__file__).parent / cfg_filename).read_text())
        print(f"Using parameter sampling from: {cfg_filename}")

    # Build env config matching the run.
    cfg = {
        "simulator": {"max_steps": 20, "alone_enabled": False},
        "env_type": "supersims",
        "policy_split": "per_param",
    }
    if cfg_filename:
        cfg["parameter_config_filename"] = cfg_filename
    if args.delta_scale_factor is not None:
        cfg["delta_scale_factor"] = float(args.delta_scale_factor)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        cfg_path = f.name
    env = SuperSimsEnv(config_path=cfg_path)
    print(f"Env max_steps={env.max_steps}, n_qubits={env.n_qubits}, "
          f"delta_scale_factor={env._delta_scale_factor}")

    split, modules = load_modules_from_checkpoint(ckpt)
    if split != "per_param":
        print(f"WARNING: ckpt is {split}; this diagnostic targets per_param.")

    # Aggregate per-policy stats across episodes × steps × qubits.
    all_actions = {p: [] for p in PARAM_NAMES}  # list of |a| values
    all_rewards = []  # mean reward per step

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=ep)
        ep_rewards = [info["per_qubit_rewards"].mean()]
        for t in range(env.max_steps):
            action = greedy_action(split, modules, obs["staircase"], obs["params"])  # (N, 5)
            for k, pname in enumerate(PARAM_NAMES):
                all_actions[pname].extend(np.abs(action[:, k]).tolist())
            obs, _, terminated, _, info = env.step(action)
            ep_rewards.append(info["per_qubit_rewards"].mean())
            if terminated:
                break
        all_rewards.append(ep_rewards)

    print(f"\nPer-policy greedy action magnitude (averaged over {args.n_episodes} episodes × 20 steps × {env.n_qubits} qubits):")
    print(f"{'policy':12s}  {'mean |a|':>10s}  {'max |a|':>10s}  {'std |a|':>10s}  comment")
    for pname in PARAM_NAMES:
        a = np.array(all_actions[pname])
        comment = ""
        if a.mean() < 0.01:
            comment = "← essentially idle"
        elif a.mean() < 0.05:
            comment = "← very mild tuning"
        elif a.mean() > 0.5:
            comment = "← saturating ±1"
        print(f"  {pname:10s}  {a.mean():>10.4f}  {a.max():>10.4f}  {a.std():>10.4f}  {comment}")

    print(f"\nMean reward trajectory across {args.n_episodes} seeds:")
    rewards = np.array(all_rewards)  # (n_eps, max_steps+1)
    means = rewards.mean(axis=0)
    print(f"  step 0: {means[0]:.3f}  (random init)")
    print(f"  step {len(means)-1}: {means[-1]:.3f}  (final)")
    print(f"  full curve: {' → '.join(f'{r:.2f}' for r in means)}")


if __name__ == "__main__":
    main()
