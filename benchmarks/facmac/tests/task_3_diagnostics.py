"""
Task 3 diagnostics — post-hoc analysis of the stock-MLP smoke training run.

Run task-3 training FIRST:

    uv run --extra facmac python benchmarks/facmac/train.py \\
        --config=facmac_quantum_smoke --env-config=env_quantum_smoke

Then this script:

    uv run --extra facmac python benchmarks/facmac/tests/task_3_diagnostics.py

Produces under benchmarks/facmac/diagnostics/task_3/:

    1. summary.txt      — pass/fail checks (finiteness of losses, weight drift > 0,
                          save_0 and save_{t_max} both present)
    2. loss_curves.png  — critic_loss, pg_loss, critic_grad_norm, agent_grad_norm
    3. reward_curve.png — train / test return_mean vs t_env
    4. weight_drift.png — histogram per actor layer of (w_final - w_init)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import torch as th

# jsonpickle needs to know about numpy + torch handlers for decode to work.
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

_BENCH_DIR = Path(__file__).resolve().parent.parent
_VENDOR_DIR = _BENCH_DIR / "vendor"
for p in (_VENDOR_DIR, _BENCH_DIR):
    sys.path.insert(0, str(p))

OUT_DIR = _BENCH_DIR / "diagnostics" / "task_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SACRED_DIR = _BENCH_DIR / "results" / "sacred"
MODELS_DIR = _BENCH_DIR / "results" / "models"


def _coerce(v):
    """
    Sacred serialises values via jsonpickle. Three formats show up in practice:
      - plain float  →  1.23
      - numpy scalar →  {"dtype": "float64", "py/object": "numpy.float64", "value": 1.23}
      - torch tensor →  {"py/reduce": [..., b64 binary, ...]}  (vendor forgot .item())
    We handle all three and return None for anything else.
    """
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
    raw_vs = info[key]
    raw_ts = info[f"{key}_T"]
    ts, vs = [], []
    for t, v in zip(raw_ts, raw_vs):
        coerced = _coerce(v)
        if coerced is not None:
            ts.append(float(t))
            vs.append(coerced)
    return ts, vs


def _latest_sacred_run() -> Path:
    candidates = [p for p in SACRED_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No sacred runs under {SACRED_DIR}. Did task-3 training run?")
    return max(candidates, key=lambda p: int(p.name))


def _latest_model_dir() -> Path:
    candidates = [p for p in MODELS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints under {MODELS_DIR}. Was save_model: True?")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_loss_curves(info: dict) -> list[str]:
    keys = [
        "critic_loss",
        "critic_grad_norm",
        "agent_grad_norm",
        "pg_loss",
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), sharex=True, squeeze=False
    )
    lines = []
    for ax, k in zip(axes.flat, keys):
        ax.set_title(k)
        ax.set_xlabel("t_env")
        ts, vs = _series(info, k)
        if not ts:
            ax.text(0.5, 0.5, f"(no data for {k})", ha="center", va="center", transform=ax.transAxes)
            lines.append(f"    - {k}: NO DATA")
            continue
        ax.plot(ts, vs, linewidth=1.0)
        vs_np = np.asarray(vs)
        finite = bool(np.isfinite(vs_np).all())
        ax.set_ylabel(f"value  (all finite: {finite})")
        lines.append(
            f"    - {k}: n={len(vs)}  finite={finite}  "
            f"min={vs_np.min():+.3e}  max={vs_np.max():+.3e}  mean={vs_np.mean():+.3e}"
        )

    fig.suptitle("Task 3 loss curves (stock MLP + QMix, resolution=48)", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "loss_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(0, f"  [written] {out}")
    return lines


def plot_reward_curve(info: dict) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"return_mean": "C0", "test_return_mean": "C1"}

    lines = []
    for k in ("return_mean", "test_return_mean"):
        ts, vs = _series(info, k)
        if ts:
            ax.plot(ts, vs, label=k, linewidth=1.2, color=colors[k])
            vs_np = np.asarray(vs)
            lines.append(f"    - {k}: n={len(vs)}  first={vs_np[0]:+.3f}  last={vs_np[-1]:+.3f}  delta={vs_np[-1]-vs_np[0]:+.3f}")

    ax.set_xlabel("t_env")
    ax.set_ylabel("episode return (team reward, 30-step episodes)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Task 3 reward over time — stock MLP cannot exploit spatial structure, expect flat-ish.")
    out = OUT_DIR / "reward_curve.png"
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(0, f"  [written] {out}")
    return lines


def plot_weight_drift(model_dir: Path) -> list[str]:
    """Compares the earliest and latest saved agent.th checkpoints."""
    step_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if len(step_dirs) < 2:
        return [f"  WARNING: only {len(step_dirs)} checkpoints in {model_dir}; expected ≥2 (step 0 + step t_max)"]

    init_path = step_dirs[0] / "agent.th"
    final_path = step_dirs[-1] / "agent.th"
    w_init = th.load(init_path, map_location="cpu")
    w_final = th.load(final_path, map_location="cpu")

    shared_keys = [k for k in w_init if k in w_final]
    fig, axes = plt.subplots(1, len(shared_keys), figsize=(3 * len(shared_keys), 3.5), squeeze=False)
    lines = [f"  [written] weight_drift.png  (init={step_dirs[0].name}, final={step_dirs[-1].name})"]

    all_deltas_max = 0.0
    for ax, k in zip(axes.flat, shared_keys):
        delta = (w_final[k] - w_init[k]).detach().cpu().float().numpy().ravel()
        ax.hist(delta, bins=60, color="C0", edgecolor="none")
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("Δweight")
        ax.axvline(0, color="red", linewidth=0.7)
        mean = float(np.mean(np.abs(delta)))
        mx = float(np.max(np.abs(delta)))
        all_deltas_max = max(all_deltas_max, mx)
        ax.text(
            0.02, 0.95,
            f"mean|Δ|={mean:.2e}\nmax|Δ|={mx:.2e}",
            transform=ax.transAxes, fontsize=8, va="top",
        )
        lines.append(f"    - {k}: mean|Δ|={mean:.2e}  max|Δ|={mx:.2e}")

    fig.suptitle("Task 3 actor weight drift  (init vs final checkpoint)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "weight_drift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    lines.insert(1, f"    max |Δweight| across all layers = {all_deltas_max:.2e}")
    return lines


def main() -> None:
    print(f"Writing diagnostics to {OUT_DIR}\n")
    summary: list[str] = []

    run_dir = _latest_sacred_run()
    info_path = run_dir / "info.json"
    config_path = run_dir / "config.json"
    summary.append(f"Sacred run: {run_dir}")

    with info_path.open() as f:
        info = json.load(f)
    with config_path.open() as f:
        config = json.load(f)

    summary.append(f"Config: name={config['name']}  t_max={config['t_max']}  mixer={config['mixer']}  agent={config['agent']}  seed={config['seed']}")
    summary.append(f"Metric keys in info.json (sampled): {sorted([k for k in info if not k.endswith('_T')])[:12]}")
    summary.append("")

    print("[1/3] loss curves")
    summary.append("=== loss curves ===")
    summary.extend(plot_loss_curves(info))
    summary.append("")

    print("[2/3] reward curve")
    summary.append("=== reward curve ===")
    summary.extend(plot_reward_curve(info))
    summary.append("")

    print("[3/3] weight drift")
    summary.append("=== weight drift ===")
    try:
        model_dir = _latest_model_dir()
        summary.append(f"  model dir: {model_dir}")
        summary.extend(plot_weight_drift(model_dir))
    except FileNotFoundError as e:
        summary.append(f"  {e}")

    out = OUT_DIR / "summary.txt"
    out.write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n  [written] {out}")
    print(f"\nDone. Inspect artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
