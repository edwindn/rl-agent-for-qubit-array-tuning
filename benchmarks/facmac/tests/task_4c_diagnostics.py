"""
Task 4c diagnostics — post-training analysis of the GroupedMAC + CNN smoke run,
plus overlay comparison against task 3's stock-MLP run.

Usage:
    uv run --extra facmac python benchmarks/facmac/tests/task_4c_diagnostics.py

Produces under benchmarks/facmac/diagnostics/task_4c/:

    1. summary.txt         — metric inventory, drift magnitudes, convergence deltas
    2. loss_curves.png     — critic_loss / grad_norms / pg_loss over t_env (CNN only)
    3. reward_overlay.png  — train & test return_mean: task-3 (MLP) vs task-4c (CNN)
    4. weight_drift.png    — per-group per-layer Δweight histograms (plunger + barrier)

The MLP run is auto-discovered by `name == facmac_quantum_smoke`; the CNN run by
`name == facmac_quantum_smoke_grouped`. If multiple runs match, the most recent
by stop_time (or start_time fallback) is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import matplotlib.pyplot as plt
import numpy as np
import torch as th

jsonpickle_numpy.register_handlers()

_BENCH_DIR = Path(__file__).resolve().parent.parent
_VENDOR_DIR = _BENCH_DIR / "vendor"
for p in (_VENDOR_DIR, _BENCH_DIR):
    sys.path.insert(0, str(p))

OUT_DIR = _BENCH_DIR / "diagnostics" / "task_4c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SACRED_DIR = _BENCH_DIR / "results" / "sacred"
MODELS_DIR = _BENCH_DIR / "results" / "models"

MLP_NAME = "facmac_quantum_smoke"
CNN_NAME = "facmac_quantum_smoke_grouped"


def _coerce(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        if "value" in v and isinstance(v["value"], (int, float)):
            return float(v["value"])
        if "py/reduce" in v or "py/object" in v:
            try:
                obj = jsonpickle.decode(json.dumps(v))
                if hasattr(obj, "item"):
                    return float(obj.item())
                return float(obj)
            except Exception:
                return None
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _series(info: dict, key: str) -> tuple[list[float], list[float]]:
    if key not in info or f"{key}_T" not in info:
        return [], []
    ts, vs = [], []
    for t, v in zip(info[f"{key}_T"], info[key]):
        coerced = _coerce(v)
        if coerced is not None:
            ts.append(float(t))
            vs.append(coerced)
    return ts, vs


def _find_run_by_name(name: str) -> Path | None:
    candidates = []
    for d in SACRED_DIR.iterdir():
        if not (d.is_dir() and d.name.isdigit()):
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.load(cfg_path.open())
        if cfg.get("name") == name:
            run_path = d / "run.json"
            run = json.load(run_path.open()) if run_path.exists() else {}
            key = run.get("stop_time") or run.get("start_time") or ""
            candidates.append((key, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _find_models_by_name(name: str) -> Path | None:
    candidates = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith(name + "__")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_loss_curves(info: dict) -> list[str]:
    keys = ["critic_loss", "critic_grad_norm", "agent_grad_norm", "pg_loss"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, squeeze=False)
    lines = []
    for ax, k in zip(axes.flat, keys):
        ax.set_title(k)
        ax.set_xlabel("t_env")
        ts, vs = _series(info, k)
        if not ts:
            ax.text(0.5, 0.5, f"(no data for {k})", ha="center", va="center", transform=ax.transAxes)
            lines.append(f"    - {k}: NO DATA")
            continue
        ax.plot(ts, vs, linewidth=1.2, marker="o", markersize=4)
        vs_np = np.asarray(vs)
        ax.set_ylabel(f"value  (finite: {bool(np.isfinite(vs_np).all())})")
        lines.append(
            f"    - {k}: n={len(vs)}  first={vs_np[0]:+.3e}  last={vs_np[-1]:+.3e}"
        )
    fig.suptitle("Task 4c loss curves — GroupedMAC + CNN + QMix (res=48, t_max=1500)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "loss_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(0, f"  [written] {out}")
    return lines


def plot_reward_overlay(cnn_info: dict, mlp_info: dict | None) -> list[str]:
    """
    3x2 grid — rows: (team, plunger-group, barrier-group), cols: (train, test).
    Per-group curves (plunger_return_mean, barrier_return_mean) are only
    present in runs whose env_wrapper logs them (added after first smoke runs);
    cells fall back to "not in log" if absent.
    """
    group_keys = [
        ("team",     "return_mean",          "test_return_mean"),
        ("plunger",  "plunger_return_mean",  "test_plunger_return_mean"),
        ("barrier",  "barrier_return_mean",  "test_barrier_return_mean"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    lines = []

    for row, (label, train_key, test_key) in enumerate(group_keys):
        for col, (key, subtitle) in enumerate(((train_key, "train"), (test_key, "test"))):
            ax = axes[row, col]
            ax.set_title(f"{label} — {subtitle}  ({key})", fontsize=10)
            ax.set_xlabel("t_env")
            ax.grid(alpha=0.3)

            plotted_any = False
            if mlp_info is not None:
                ts, vs = _series(mlp_info, key)
                if ts:
                    ax.plot(ts, vs, label="task-3 MLP", color="C0", linewidth=1.5, marker="s", markersize=5)
                    lines.append(f"    - MLP {key}: n={len(vs)} first={vs[0]:+.2f} last={vs[-1]:+.2f}")
                    plotted_any = True

            ts, vs = _series(cnn_info, key)
            if ts:
                ax.plot(ts, vs, label="task-4c CNN + Grouped", color="C1", linewidth=1.5, marker="o", markersize=5)
                lines.append(f"    - CNN {key}: n={len(vs)} first={vs[0]:+.2f} last={vs[-1]:+.2f}")
                plotted_any = True

            if not plotted_any:
                ax.text(0.5, 0.5, "(not in log)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            else:
                ax.legend(loc="best", fontsize=8)

        axes[row, 0].set_ylabel(f"{label} return (sum per ep)")

    fig.suptitle(
        "Reward overlay: stock MLP (task-3) vs CNN + GroupedMAC (task-4c)\n"
        "rows = team / plunger-group / barrier-group    cols = train / test",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "reward_overlay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(0, f"  [written] {out}")
    return lines


def _step_dirs(model_dir: Path) -> list[Path]:
    return sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def plot_weight_drift(model_dir: Path) -> list[str]:
    step_dirs = _step_dirs(model_dir)
    if len(step_dirs) < 2:
        return [f"  WARNING: only {len(step_dirs)} checkpoints in {model_dir}"]

    init_dir = step_dirs[0]
    final_dir = step_dirs[-1]

    groups = []
    for group in ("plunger", "barrier"):
        init_path = init_dir / f"agent_{group}.th"
        final_path = final_dir / f"agent_{group}.th"
        if not (init_path.exists() and final_path.exists()):
            continue
        w_init = th.load(init_path, map_location="cpu")
        w_final = th.load(final_path, map_location="cpu")
        shared = [k for k in w_init if k in w_final]
        groups.append((group, w_init, w_final, shared))

    if not groups:
        return [f"  WARNING: no per-group agent checkpoints found in {init_dir} / {final_dir}"]

    n_cols = max(len(s) for _, _, _, s in groups)
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3.0 * n_rows), squeeze=False)

    lines = [f"  [written] weight_drift.png  (init={init_dir.name}, final={final_dir.name})"]
    for row, (group, w_init, w_final, shared) in enumerate(groups):
        lines.append(f"    Group '{group}':")
        for col, k in enumerate(shared):
            delta = (w_final[k] - w_init[k]).detach().cpu().float().numpy().ravel()
            ax = axes[row, col]
            ax.hist(delta, bins=50, color=f"C{row}", edgecolor="none")
            ax.axvline(0, color="red", linewidth=0.6)
            ax.set_title(f"{group}.{k}", fontsize=8)
            ax.set_xlabel("Δweight", fontsize=7)
            mean_abs = float(np.mean(np.abs(delta)))
            max_abs = float(np.max(np.abs(delta)))
            ax.text(
                0.02, 0.95,
                f"mean|Δ|={mean_abs:.2e}\nmax|Δ|={max_abs:.2e}",
                transform=ax.transAxes, fontsize=7, va="top",
            )
            lines.append(f"      {k:<25s} mean|Δ|={mean_abs:.2e}  max|Δ|={max_abs:.2e}")
        for col in range(len(shared), n_cols):
            axes[row, col].axis("off")

    fig.suptitle("Task 4c per-group per-layer weight drift", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "weight_drift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return lines


def main() -> None:
    print(f"Writing diagnostics to {OUT_DIR}\n")
    summary: list[str] = []

    cnn_run = _find_run_by_name(CNN_NAME)
    mlp_run = _find_run_by_name(MLP_NAME)
    summary.append(f"CNN (task-4c) sacred run: {cnn_run}")
    summary.append(f"MLP (task-3) sacred run:  {mlp_run}")

    if cnn_run is None:
        summary.append(f"ERROR: no sacred run found with name={CNN_NAME!r}. Did you run task-4c training?")
        (OUT_DIR / "summary.txt").write_text("\n".join(summary))
        print("\n".join(summary))
        return

    cnn_info = json.load((cnn_run / "info.json").open())
    cnn_cfg = json.load((cnn_run / "config.json").open())
    summary.append(f"CNN config: t_max={cnn_cfg['t_max']} mixer={cnn_cfg['mixer']} mixing_embed_dim={cnn_cfg['mixing_embed_dim']} seed={cnn_cfg['seed']}")

    mlp_info = json.load((mlp_run / "info.json").open()) if mlp_run else None
    summary.append("")

    print("[1/3] loss curves")
    summary.append("=== loss curves (task-4c only) ===")
    summary.extend(plot_loss_curves(cnn_info))
    summary.append("")

    print("[2/3] reward overlay")
    summary.append("=== reward overlay ===")
    summary.extend(plot_reward_overlay(cnn_info, mlp_info))
    summary.append("")

    print("[3/3] weight drift")
    summary.append("=== weight drift (task-4c per-group checkpoints) ===")
    model_dir = _find_models_by_name(CNN_NAME)
    if model_dir is None:
        summary.append(f"  WARNING: no model dir matching {CNN_NAME}__*")
    else:
        summary.append(f"  model dir: {model_dir}")
        summary.extend(plot_weight_drift(model_dir))

    out = OUT_DIR / "summary.txt"
    out.write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n  [written] {out}")


if __name__ == "__main__":
    main()
