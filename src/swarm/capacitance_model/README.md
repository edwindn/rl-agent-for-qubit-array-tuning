# Capacitance Model: Paper Figures & Inference Pipeline

## Paper Figure 3 (4 panels)

| Panel | File | Script |
|-------|------|--------|
| **3a** | `kalman_calibration_nn.svg` | `plot_kalman_calibration.py` |
| **3b** | `kalman_calibration_nnn.svg` | `plot_kalman_calibration.py` |
| **3c** | `model_convergence_paper.svg` | `plot_kalman_calibration.py` |
| **3d** | `scaling_paper.svg` | `plot_scaling.py` |

### Reproducing panels (a), (b), (c)

These use `data/episode_data_1000ep.npy` — 1000+ explore-mode episodes on a 4-dot array with fixed capacitances (NN=0.7, NNN=0.3).

**Step 1 — configure environment** (if regenerating data):

Edit `src/swarm/environment/qarray_config.yaml`:
```yaml
Cgd:
  cross_coupling:
    1: {"min": 0.7, "max": 0.7}   # fixed NN
    2: {"min": 0.3, "max": 0.3}   # fixed NNN
```

Verify `src/swarm/environment/env.py` has:
- NNN sign fix (lines ~622–632): NNN outputs **not** negated, only NN is
- `prior_variance_nnn=0.03` in the Kalman init (line ~804)

**Step 2 — collect data** (parallelized across 8 GPUs, ~1 hour):
```bash
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu uv run python src/swarm/capacitance_model/collect_episode_data.py \
    --checkpoint artifacts/rl_checkpoint_best:v3482 \
    --episodes 125 --explore \
    --output data/episode_data_1000ep_gpu${gpu}.npy &
done
wait

# Merge GPU outputs
uv run python -c "
import numpy as np
all_samples = []
for gpu in range(8):
    data = list(np.load(f'data/episode_data_1000ep_gpu{gpu}.npy', allow_pickle=True))
    max_ep = max(s['episode'] for s in all_samples) + 1 if all_samples else 0
    for s in data:
        s['episode'] += max_ep
    all_samples.extend(data)
np.save('data/episode_data_1000ep.npy', np.array(all_samples, dtype=object), allow_pickle=True)
"
```

**Step 3 — generate plots:**
```bash
uv run python src/swarm/capacitance_model/plot_kalman_calibration.py \
  --data data/episode_data_1000ep.npy \
  --output-dir .
```

### Reproducing panel (d)

Panel (d) uses two data sources:
- **2/4/6/8 dots**: benchmark JSON files at `benchmarks/results/final_{N}dot/ppo_{N}dots.json`
- **10/12 dots**: wandb artifacts (plunger-only, no barriers)

**Download 10/12 dot artifacts:**
```bash
uv run python -c "
import wandb, os

artifacts_10dot = [
    'rl_agents_for_tuning/RLModel/plunger_data_iter_5:v12',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_5:v11',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_5:v10',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v6',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v8',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_5:v9',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v7',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v5',
]
artifacts_12dot = [
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v4',
    'rl_agents_for_tuning/RLModel/plunger_data_iter_6:v3',
]
os.makedirs('data/scaling_10dot', exist_ok=True)
os.makedirs('data/scaling_12dot', exist_ok=True)
run = wandb.init(entity='rl_agents_for_tuning', project='RLModel')
for i, name in enumerate(artifacts_10dot):
    run.use_artifact(name, type='plunger_distances').download(root=f'data/scaling_10dot/artifact_{i}')
for i, name in enumerate(artifacts_12dot):
    run.use_artifact(name, type='plunger_distances').download(root=f'data/scaling_12dot/artifact_{i}')
wandb.finish()
"
```

**Generate the plot:**
```bash
uv run python src/swarm/capacitance_model/plot_scaling.py --output scaling_paper.svg
```

### One-shot: regenerate all 4 paper figures
```bash
./reproduce_paper_figures.sh
```

---

## Scripts

### Paper figure scripts (keep these)

| Script | Purpose |
|--------|---------|
| `collect_episode_data.py` | Collect per-step data via `algo.evaluate()`. Saves `.npy` |
| `plot_kalman_calibration.py` | Paper Figure 3 (a, b, c) |
| `plot_scaling.py` | Paper Figure 3 (d) |

### Supporting / exploratory scripts

| Script | Purpose |
|--------|---------|
| `plot_calibration_grid.py` | 3×4 grid + Kalman variance reconstruction. Used by `plot_kalman_calibration.py` |
| `test_variances.py` | 2-panel calibration + convergence (earlier version) |

### Core model code

| File | Purpose |
|------|---------|
| `CapacitancePrediction.py` | MobileNet CNN that predicts (values, log_vars) per scan |
| `KalmanUpdater.py` | Kalman filter for fusing ML predictions into capacitance estimate |
| `capacitance_utils.py` | `get_targets_with_nnn()` — extract CGD elements per scan |

---

## Key Code Changes Documented

These are **required** for reproducing the paper figures:

### 1. NNN sign fix (`src/swarm/environment/env.py` lines 622–632)

The Kalman update path negates NN outputs but **not** NNN outputs. This is because the ML model's NNN outputs have an opposite effective sign convention relative to the Kalman residual (see rebuttal/paper for details). Without this fix, NNN estimates diverge.

```python
ml_outputs = [
    (-float(values_np[i, 0]), float(log_vars_np[i, 0])),  # NN (negated)
    ( float(values_np[i, 1]), float(log_vars_np[i, 1])),  # NNN_right (NOT negated)
    ( float(values_np[i, 2]), float(log_vars_np[i, 2])),  # NNN_left (NOT negated)
]
```

### 2. NNN prior variance (`src/swarm/environment/env.py` line ~804)

NNN couplings are smaller in magnitude (range [0.01, 0.3] vs NN's [0.3, 0.7]), so the prior variance should be proportionally smaller.

```python
KalmanCapacitanceUpdater(
    ...
    prior_variance=0.1,      # NN
    prior_variance_nnn=0.03, # NNN (~3x smaller)
)
```

### 3. `KalmanUpdater.py`: added `prior_variance_nnn` parameter

---

## Data

### `data/episode_data_1000ep.npy`

1008 explore-mode episodes, each 50 steps, 3 scans per step → ~150k samples. Each sample dict has:
- `episode`, `step` — identifiers
- `current_estimate` — pre-update Kalman estimate for this scan's 3 targets
- `model_values` — ML delta (sign handled per convention)
- `model_log_vars` — ML log-variance
- `capacitance` — ground truth targets from `model.Cgd`
- `estimated_vgm`, `true_vgm` — virtual gate matrices
- `plunger_distance`, `barrier_distance` — voltage distance to ground truth

### `data/scaling_10dot/`, `data/scaling_12dot/`

Wandb artifacts for 10 and 12 dot scaling. Each artifact has subdirs `plunger_0/ ... plunger_{N-1}/` each containing `0001_*.npy ... 000N_*.npy` per-episode files. Each file is a (99,) signed distance array. No barrier data (hence Fig 3d notes "preliminary" for 10/12).

### `benchmarks/results/final_{N}dot/ppo_{N}dots.json`

Benchmark runs for 2/4/6/8 dots with both plunger + barrier distance histories.

---

## Configs

### `src/swarm/environment/qarray_config.yaml`

Default ranges are restored (NN [0.3, 0.7], NNN [0.01, 0.3]). Comments note the fixed values used for Fig 3(a–c).

### `checkpoints/env_config.yaml`

```yaml
capacitance_model:
  update_method: "kalman"
  nearest_neighbour: false      # NNN mode
  variance_threshold: 0.05
  process_noise: 0.0
```

### Model weights

- **RL checkpoint**: `artifacts/rl_checkpoint_best:v3482` (download from wandb)
- **Capacitance model**: `src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth`
