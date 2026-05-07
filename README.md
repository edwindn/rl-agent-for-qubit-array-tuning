# QADAPT

**Action-Factored Multi-Agent RL for Scalable Quantum Device Tuning.**

QADAPT formulates quantum-dot-array tuning as a cooperative Dec-POMDP. An online
Kalman-filtered virtualization model estimates gate-to-dot cross-capacitances
and constructs a per-step factored action basis — decoupling agents,
preconditioning the local control problem, and enabling zero-shot generalization
across array sizes via gate-type parameter sharing.

## Layout

```
src/
  qadapt/                  main package — env, training, voltage + capacitance models
  qadapt_for_supersim/     parallel package: SuperSims-specific extensions (appendix)
benchmarks/
  domain/                  non-RL baselines: Bayesian, L-BFGS, Nelder-Mead, random, DreamerV3
  MARL/                    MARL baselines: MAPPO, IPPO/FACMAC/MADDPG (PyMARL), single-agent PPO/SAC
  Ablations/               ablation eval pipeline (run_ablation.py, compute_table.py)
SuperSims/                 superconducting-qubit simulator (Hamiltonian + All-XY)
scripts/                   paper-figure plotters + paper-data generators
paper_plots/               output SVG/PNG figures, plus paper_plots/data/ inputs (npz)
modal_scripts/             Modal cloud entry points for hero runs
```

## Setup

```bash
uv sync
```

Python 3.11. Major deps: PyTorch 2.8, Ray RLlib 2.51, JAX 0.7.2 (CUDA 12),
QArray 1.6, dynamiqs 0.3.4.

## Reproducing paper results

Table 1 (convergence + ablations, 2/5/10 V radii):

```bash
uv run python benchmarks/Ablations/ablation/compute_table.py --markdown table.md
```

Figures (output to `paper_plots/`):

```bash
# Training reward curves (pulls runs from wandb)
uv run python scripts/plot_reward_curves.py

# SuperSims appendix figures (read paper_plots/data/staircase_scan_N{2,4,6,8}.npz)
uv run python scripts/plot_allxy_violins.py
uv run python scripts/plot_convergence_multiN.py

# Capacitance model (need data/episode_data_1000ep.npy)
uv run python scripts/plot_capacitance_convergence.py
uv run python scripts/plot_kalman_calibration.py

# Domain benchmarks: per-N convergence panels (need benchmarks/results/final_{N}dot/*.json)
for n in 2 4 6 8; do
  uv run python benchmarks/domain/plot_results.py --plot convergence --num-dots $n
done

# QADAPT scaling vs num dots
uv run python benchmarks/domain/plot_scaling.py
```

Inputs not committed (data is large): wandb run histories, `data/episode_data_1000ep.npy`
(capacitance episode rollouts), `benchmarks/results/final_{N}dot/*.json` (domain
method outputs). The two SuperSims `.npz` files needed for figures A/B *are*
committed under `paper_plots/data/`.

## Citation

> De Nicolo, E., Marchand, R., Carlsson, C., Vaidhyanathan, P., Ares, N.
> *Action-Factored Multi-Agent Reinforcement Learning for Scalable Quantum Device Tuning.*
> NeurIPS 2026.
