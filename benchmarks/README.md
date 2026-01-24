# Quantum Dot Tuning Benchmarks

Benchmarking suite for comparing optimization methods on quantum dot array voltage tuning.

## Methods Implemented

| Method | Directory | Description |
|--------|-----------|-------------|
| Nelder-Mead | `nelder_mead/` | Derivative-free simplex optimization |
| L-BFGS-B | `lbfgs/` | Quasi-Newton with bounds |
| Bayesian | `bayesian/` | Gaussian Process + Expected Improvement (BoTorch) |
| Random | `random/` | Uniform random sampling baseline |
| DreamerV3 | `dreamer/` | RL agent evaluation (requires trained checkpoint) |

Each method supports two modes:
- **joint**: Optimize all voltages simultaneously
- **pairwise**: Sliding window over adjacent dot pairs (more realistic for hardware)

## Quick Start

```bash
cd benchmarks

# Run a single method
uv run python nelder_mead/run.py --num_dots 2 --num_trials 10 --mode pairwise

# Plot results
uv run python plot_results.py --dir results/
```

## Common Interface

All runners share the same CLI pattern:

```bash
uv run python <method>/run.py \
    --num_dots 2 \
    --num_trials 10 \
    --max_scans 1000 \
    --threshold 0.5 \
    --seed 42 \
    --output results/my_run.json
```

Results are saved to `benchmarks/results/` by default (auto-generated filename), or specify `--output path/to/file.json`.

## Key Concepts

### Scan Counting

A **scan** = one measurement of a single dot pair's charge stability diagram.

| Mode | Scans per Evaluation |
|------|---------------------|
| Joint/Random | `num_dots - 1` (all pairs measured simultaneously) |
| Pairwise | `1` (one pair per function eval) |

This ensures fair comparison: 1000 scans means the same experimental cost across methods.

### Convergence Tracking

Each trial records:
- `scan_numbers`: Which scan number each measurement occurred at
- `plunger_distance_history`: Sum of |plunger_i - optimal_i| at each record
- `barrier_distance_history`: Sum of |barrier_i - optimal_i| at each record
- `plunger_range`, `barrier_range`: From config, for normalization

Normalization formula:
```
max_distance = plunger_range * num_plungers + barrier_range * num_barriers
score = 1 - total_distance / max_distance  # 0=worst, 1=converged
```

### Environment Config

All benchmarks use a **centralized config path** defined in `env_init.py`:

```python
ENV_CONFIG_PATH = src_dir / "swarm" / "environment" / "env_config.yaml"
```

To use a different config, change this single line. The config controls:
- `full_plunger_range_width`: ~90V (plungers have large search space)
- `full_barrier_range_width`: ~8V (barriers are more constrained)
- Other simulator parameters (num_dots default, noise levels, etc.)

## Gotchas

1. **Pairwise mode parameters matter**: `cap_per_plunger`, `threshold_per_plunger`, `simplex_step_plunger` etc. significantly affect convergence. Defaults are tuned for typical use.

2. **Bayesian optimization is slow**: GP fitting scales O(n³). Use `--max_scans` to limit, or `--batch_size > 1` for parallel evaluation.

3. **DreamerV3 requires checkpoint**: Point `--checkpoint` to a training logdir with `ckpt/` subdirectory.

4. **Random baseline needs many samples**: Random search in high dimensions is inefficient. Expect 10,000+ scans for convergence.

## File Structure

```
benchmarks/
├── convergence_tracker.py  # Common dataclass for distance tracking
├── env_init.py             # Environment creation helpers
├── objective.py            # Objective function and distance calculations
├── utils.py                # TrialResult, BenchmarkResult, save/load
├── plot_results.py         # Plotting utilities
├── nelder_mead/run.py
├── lbfgs/run.py
├── bayesian/run.py
├── random/run.py
├── dreamer/run.py
└── results/                # Output JSON files
```

## Plotting

```bash
# Both plots
uv run python plot_results.py --dir results/

# Just convergence curves for 4-dot results
uv run python plot_results.py --dir results/ --plot convergence --num-dots 4 --max-scans 2000
```

Plots:
1. **Scans to threshold**: X=num_dots, Y=scans to reach threshold (converged trials only)
2. **Convergence curves**: X=scan number, Y=normalized score (median + IQR)
