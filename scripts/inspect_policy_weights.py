"""
Inspect a SuperSims per-param checkpoint to determine whether the policy has
collapsed to a constant (output dominated by final-layer bias) or whether it
genuinely depends on the observation.

Three diagnostics, run on each of the 5 per-param policies:

  (1) Final-layer bias and weight magnitudes — the structural answer.
      If `|bias.mean_head| >> |W.mean_head|.norm() × E[hidden]`, the policy
      output is bias-dominated *by construction*.

  (2) Output variability across realistic observations — the empirical answer.
      Reset 20 episodes (different seeds), record (obs, mean_output) pairs.
      Report std(mean_output) / |bias|. If close to 0, policy ≈ constant.

  (3) Bias vs ΔObs decomposition — pin one obs as reference, then vary the
      other 19 obs. Compute the change in mean output relative to bias.
      If `Δmean / bias < 1%`, policy is *effectively* a constant.

Optional (4): inspect the encoder's first layer weight magnitudes per input
column to see if the encoder is killing the params channel.

Usage:
  CUDA_VISIBLE_DEVICES=3 uv run python scripts/inspect_policy_weights.py \\
      [--ckpt PATH] [--n-episodes 20] [--config parameter_config_medium.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "SuperSims"))

from qadapt.environment.supersims_env import SuperSimsEnv  # noqa: E402
from qadapt.inference.eval_supersims import (  # noqa: E402
    load_modules_from_checkpoint, find_latest_checkpoint,
)

PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]


def _final_layer(module):
    """Return the policy head's final nn.Linear (the layer that produces
    [mean, log_std] for a Gaussian output)."""
    pi = module.pi
    if hasattr(pi, "net"):
        seq = pi.net
    elif hasattr(pi, "mlp"):
        seq = pi.mlp
    else:
        # Fall through: search children for the last Linear.
        seq = pi
    last_linear = None
    for m in seq.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError(f"No nn.Linear found in policy head of {module}")
    return last_linear


def _first_encoder_linear(module):
    """First nn.Linear in the encoder — receives the concatenated [staircase, params] obs."""
    enc = module.encoder
    for m in enc.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    raise RuntimeError("No nn.Linear in encoder")


@torch.no_grad()
def _policy_mean(module, obs_batch):
    """Forward a batch through the policy, return only the mean half of the Gaussian
    [mean, log_std]. obs_batch is a dict of tensors with leading batch dim."""
    from ray.rllib.core.columns import Columns
    out = module._forward({"obs": obs_batch})
    logits = out[Columns.ACTION_DIST_INPUTS]
    return logits[:, : logits.shape[-1] // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--config", type=str, default=None,
                    help="parameter_config_*.json filename. Default: canonical (full).")
    args = ap.parse_args()

    if args.config:
        import parameter_generation as pg
        pg._cfg = json.loads((Path(pg.__file__).parent / args.config).read_text())
        print(f"[inspect] Using parameter sampling: {args.config}")

    ckpt = Path(args.ckpt).resolve() if args.ckpt else find_latest_checkpoint(_REPO / "checkpoints_supersims")
    print(f"Checkpoint: {ckpt}")

    split, modules = load_modules_from_checkpoint(ckpt)
    if split != "per_param":
        raise SystemExit(f"This diagnostic targets per_param; got {split}.")

    # ----- Build env at the desired distribution and collect a batch of obs ----- #
    env_cfg = {
        "simulator": {"max_steps": 20, "alone_enabled": False},
        "env_type": "supersims",
        "policy_split": "per_param",
    }
    if args.config:
        env_cfg["parameter_config_filename"] = args.config
    import tempfile, yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(env_cfg, f)
        cfg_path = f.name
    env = SuperSimsEnv(config_path=cfg_path)
    print(f"Env n_qubits={env.n_qubits}, params_dim={env.n_params}, staircase_dim={env.n_allxy}")

    # Reset N episodes; for each, collect per-qubit obs (staircase, params).
    staircases, params = [], []
    for s in range(args.n_episodes):
        obs, _ = env.reset(seed=s)
        staircases.append(obs["staircase"])
        params.append(obs["params"])
    staircases = np.concatenate(staircases, axis=0).astype(np.float32)  # (N_eps × n_qubits, 21)
    params = np.concatenate(params, axis=0).astype(np.float32)          # (N_eps × n_qubits, 5)
    print(f"Collected {staircases.shape[0]} per-qubit observations.")
    print(f"  params per-column std: {params.std(axis=0)}")
    print(f"  params per-column mean: {params.mean(axis=0)}")
    print(f"  staircase column-0 std: {staircases.std(axis=0)[0]:.4f}, mean: {staircases.mean(axis=0)[0]:.4f}")

    obs_batch = {
        "staircase": torch.from_numpy(staircases),
        "params":    torch.from_numpy(params),
    }

    # ----- Per-policy diagnostics ----- #
    print("\n=== Final-layer structural diagnostics ===\n")
    print(f"{'policy':10s}  {'bias[mean]':>12s}  {'bias[logstd]':>14s}  "
          f"{'|W[mean,:]|':>12s}  {'|W[logstd,:]|':>14s}")
    print("-" * 75)
    for pname in PARAM_NAMES:
        m = modules[f"{pname}_policy"]
        last = _final_layer(m)
        b = last.bias.detach().cpu().numpy()         # (2,) for 1-d action
        W = last.weight.detach().cpu().numpy()       # (2, hidden_dim)
        b_mean, b_logstd = b[0], b[1]
        Wmean_norm = np.linalg.norm(W[0])
        Wlogstd_norm = np.linalg.norm(W[1])
        print(f"  {pname:8s}  {b_mean:>12.5f}  {b_logstd:>14.5f}  "
              f"{Wmean_norm:>12.5f}  {Wlogstd_norm:>14.5f}")

    # ----- Output variability across realistic obs ----- #
    print("\n=== Empirical output variability (mean head only) ===\n")
    print(f"{'policy':10s}  {'mean(out)':>10s}  {'std(out)':>10s}  "
          f"{'std/bias':>10s}  {'min(out)':>10s}  {'max(out)':>10s}")
    print("-" * 70)
    for pname in PARAM_NAMES:
        m = modules[f"{pname}_policy"]
        means = _policy_mean(m, obs_batch).cpu().numpy().flatten()
        b_mean = _final_layer(m).bias.detach().cpu().numpy()[0]
        ratio = means.std() / abs(b_mean) if abs(b_mean) > 1e-9 else float("inf")
        print(f"  {pname:8s}  {means.mean():>10.5f}  {means.std():>10.5f}  "
              f"{ratio:>10.4f}  {means.min():>10.5f}  {means.max():>10.5f}")
    print("\n  (std/bias < 0.01 ⇒ policy is effectively a constant; obs has ~no influence.)")

    # ----- Bias vs ΔObs decomposition ----- #
    print("\n=== Output decomposition: bias-only vs full forward ===\n")
    print(f"{'policy':10s}  {'bias[mean]':>12s}  {'mean(out)':>10s}  "
          f"{'mean(out) − bias':>16s}  {'(out−bias)/bias':>16s}")
    print("-" * 80)
    for pname in PARAM_NAMES:
        m = modules[f"{pname}_policy"]
        means = _policy_mean(m, obs_batch).cpu().numpy().flatten()
        b_mean = _final_layer(m).bias.detach().cpu().numpy()[0]
        delta_mean = means.mean() - b_mean
        ratio = delta_mean / b_mean if abs(b_mean) > 1e-9 else float("inf")
        print(f"  {pname:8s}  {b_mean:>12.5f}  {means.mean():>10.5f}  "
              f"{delta_mean:>16.6f}  {ratio:>16.5f}")
    print("\n  If 'mean(out) ≈ bias', the obs-dependent contribution averaged over "
          "realistic obs is ~0. (Doesn't *prove* zero per-obs contribution but is "
          "a strong signal combined with std/bias above.)")

    # ----- Encoder first-layer weight column norms ----- #
    print("\n=== Encoder first-layer weight column norms (per input dim) ===\n")
    print("Looking for: are some input columns getting much weaker weights "
          "than others? (Could indicate the encoder is ignoring those features.)\n")
    for pname in PARAM_NAMES:
        m = modules[f"{pname}_policy"]
        first = _first_encoder_linear(m)
        W = first.weight.detach().cpu().numpy()  # (hidden, in_dim)
        col_norms = np.linalg.norm(W, axis=0)    # (in_dim,)
        # Layout: probably [21 staircase, 5 params] = 26-d concatenated.
        if col_norms.shape[0] == 26:
            stair_norms = col_norms[:21]
            params_norms = col_norms[21:]
            print(f"  {pname:8s}  staircase-cols (21):  mean={stair_norms.mean():.4f}  "
                  f"std={stair_norms.std():.4f}  range=[{stair_norms.min():.4f}, {stair_norms.max():.4f}]")
            print(f"  {pname:8s}  params-cols (5):      "
                  f"{', '.join(f'{x:.4f}' for x in params_norms)}")
            print(f"  {pname:8s}  ratio(params/stair):  {params_norms.mean() / stair_norms.mean():.3f}")
        else:
            print(f"  {pname:8s}  unexpected in_dim={col_norms.shape[0]}, dumping first 8: {col_norms[:8]}")
        print()


if __name__ == "__main__":
    main()
