"""
Task 6 diagnostics — validates that wandb logging + per-group-mean metrics +
artifact upload all work end-to-end.

Usage:

  1) Kick off a short training run with wandb enabled (offline is fine for local
     validation; online works too if you're logged in):

     WANDB_MODE=offline uv run --extra facmac python benchmarks/facmac/train.py \\
         --config=facmac_quantum_smoke_grouped --env-config=env_quantum_smoke \\
         with use_cuda=True use_wandb=True t_max=200 save_model_interval=200 \\
              batch_size=4 buffer_size=8 test_interval=100

  2) Run this diagnostic. It finds the latest wandb run under benchmarks/facmac/wandb/
     and writes artifacts to diagnostics/task_6/.

Produces:
    summary.txt         — presence checks + sanity ratios + artifact verification
    key_metrics.png     — wandb-logged training curves (the key ones)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent.parent
WANDB_ROOT = _BENCH_DIR / "wandb"
OUT_DIR = _BENCH_DIR / "diagnostics" / "task_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Required keys in wandb history — these are what the wandb UI will show.
REQUIRED_KEYS = [
    "return_mean",
    "plunger_return_mean",
    "barrier_return_mean",
    "plunger_return_avg_mean",
    "barrier_return_avg_mean",
    "plunger_return_avg",          # PPO-compatible alias
    "barrier_return_avg",          # PPO-compatible alias
    "critic_loss",
    "pg_loss",
    "return_mean_ema",
    "plunger_return_avg_mean_ema",
]


def _latest_wandb_run() -> Path:
    candidates = [p for p in WANDB_ROOT.iterdir()
                  if p.is_dir() and (p.name.startswith("run-") or p.name.startswith("offline-run-"))]
    if not candidates:
        raise FileNotFoundError(
            f"No wandb runs under {WANDB_ROOT}. "
            "Did you run training with `use_wandb=True`?"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_run_id(run_dir: Path) -> str:
    """Extracts the 8-char run id from wandb dir names like 'run-<ts>-<id>' or 'offline-run-<ts>-<id>'."""
    return run_dir.name.rsplit("-", 1)[-1]


def _default_entity_project() -> tuple[str, str]:
    """Read defaults from the smoke config YAML so the diagnostic stays in sync with it."""
    import yaml
    cfg_path = _BENCH_DIR / "configs" / "facmac_quantum_smoke_grouped.yaml"
    cfg = yaml.safe_load(cfg_path.open())
    return cfg.get("wandb_entity"), cfg.get("wandb_project")


def _read_history(run_dir: Path) -> list[dict]:
    """
    History retrieval:
      - offline run: parse local files/wandb-history.jsonl
      - online run: use wandb.Api() against the cloud (local dir doesn't carry history)
    """
    history_file = run_dir / "files" / "wandb-history.jsonl"
    if history_file.exists():
        rows = []
        with history_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # Online path: query via wandb.Api
    import wandb
    api = wandb.Api()
    entity, project = _default_entity_project()
    run_id = _parse_run_id(run_dir)
    path = f"{entity}/{project}/{run_id}"
    run = api.run(path)
    history_df = run.history(samples=10000, pandas=False)
    return list(history_df)


def _series_from_history(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    ts, vs = [], []
    for row in history:
        if key in row:
            value = row[key]
            if isinstance(value, (int, float)):
                ts.append(float(row.get("_step", len(ts))))
                vs.append(float(value))
    return ts, vs


def _locate_artifact(run_dir: Path) -> str | None:
    """
    Verify an artifact was logged for this run.
      - offline: look for files/artifacts/*.json manifests.
      - online: query wandb.Api().run(...).logged_artifacts().
    Returns a human-readable hit string, or None.
    """
    manifest_dir = run_dir / "files" / "artifacts"
    if manifest_dir.exists():
        for p in manifest_dir.rglob("*.json"):
            return str(p)

    try:
        import wandb
        api = wandb.Api()
        entity, project = _default_entity_project()
        run_id = _parse_run_id(run_dir)
        run = api.run(f"{entity}/{project}/{run_id}")
        artifacts = list(run.logged_artifacts())
        if artifacts:
            a = artifacts[0]
            return f"wandb.Api: {a.type}/{a.name} (size={a.size})"
    except Exception as e:
        return f"wandb.Api error: {e}"

    logs = run_dir / "logs" / "debug.log"
    if logs.exists() and "log_artifact" in logs.read_text():
        return str(logs)
    return None


def main() -> None:
    run_dir = _latest_wandb_run()
    print(f"wandb run: {run_dir}")

    summary: list[str] = [f"wandb run dir: {run_dir}", ""]

    history = _read_history(run_dir)
    summary.append(f"History rows: {len(history)}")
    all_keys = sorted({k for row in history for k in row})
    summary.append(f"Sample logged keys (first 30): {all_keys[:30]}")
    summary.append("")

    summary.append("=== required-key presence ===")
    missing = []
    for k in REQUIRED_KEYS:
        ts, vs = _series_from_history(history, k)
        status = f"n={len(vs)}  last={vs[-1]:+.3e}" if vs else "MISSING"
        if not vs:
            missing.append(k)
        summary.append(f"  {k:<40s} {status}")
    summary.append("  => " + ("ALL PRESENT" if not missing else f"MISSING: {missing}"))
    summary.append("")

    summary.append("=== sanity: plunger_return_avg_mean * n_plungers ≈ plunger_return_mean ===")
    _, pa = _series_from_history(history, "plunger_return_avg_mean")
    _, pm = _series_from_history(history, "plunger_return_mean")
    if pa and pm:
        n_plungers_est = np.mean([m / a for m, a in zip(pm, pa) if abs(a) > 1e-8])
        summary.append(f"  inferred n_plungers from ratio: {n_plungers_est:.2f}  (expect 4 for 4-dot)")
        summary.append(f"  plunger_return_mean last = {pm[-1]:+.3f}")
        summary.append(f"  plunger_return_avg_mean last = {pa[-1]:+.3f}")
        summary.append(f"  ratio = {pm[-1] / pa[-1]:.2f}" if abs(pa[-1]) > 1e-8 else "  (avg is zero)")
    else:
        summary.append("  SKIPPED — missing at least one metric")
    summary.append("")

    summary.append("=== artifact upload ===")
    artifact = _locate_artifact(run_dir)
    if artifact:
        summary.append(f"  [found] {artifact}")
    else:
        summary.append("  NOT FOUND — expected an artifact manifest under files/artifacts/*.json")
    summary.append("")

    # Plot key metrics
    plot_keys = [
        ("return_mean", "return_mean_ema", "team return"),
        ("plunger_return_avg_mean", "plunger_return_avg_mean_ema", "plunger avg reward"),
        ("barrier_return_avg_mean", "barrier_return_avg_mean_ema", "barrier avg reward"),
        ("critic_loss", None, "critic loss"),
        ("pg_loss", None, "pg loss"),
        ("critic_grad_norm", None, "critic grad norm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for ax, (k, ema_k, title) in zip(axes.flat, plot_keys):
        ax.set_title(title)
        ax.set_xlabel("wandb step")
        ts, vs = _series_from_history(history, k)
        if ts:
            ax.plot(ts, vs, label=k, linewidth=1.2, marker="o", markersize=3)
        if ema_k:
            tse, vse = _series_from_history(history, ema_k)
            if tse:
                ax.plot(tse, vse, label=ema_k, linewidth=1.2, linestyle="--")
        if ts or (ema_k and _series_from_history(history, ema_k)[0]):
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.grid(alpha=0.3)
    fig.suptitle("Task 6 — wandb-logged metrics (validation run)", fontsize=11)
    plt.tight_layout()
    fig_out = OUT_DIR / "key_metrics.png"
    fig.savefig(fig_out, dpi=120)
    plt.close(fig)
    summary.append(f"  [written] {fig_out}")

    out = OUT_DIR / "summary.txt"
    out.write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n  [written] {out}")


if __name__ == "__main__":
    main()
