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

This is a code release; large training artefacts (checkpoints, ~13 GB of domain-benchmark
result JSONs, ~15 GB of capacitance episode rollouts, wandb run histories) are **not**
committed. Reviewers can run a subset out of the box; the rest needs retraining or
data regeneration.

**Runnable from a fresh clone (no GPU, no auth):**

```bash
# SuperSims appendix figures — inputs are committed under paper_plots/data/
uv run python scripts/plot_allxy_violins.py        # appendix_supersim_violin.svg
uv run python scripts/plot_convergence_multiN.py   # appendix_supersim_convergence.svg
```

**Requires retraining or running a method (GPU, several hours each):**

| Output | Pipeline |
|---|---|
| Table 1 (`compute_table.py`) | Train every variant in `benchmarks/Ablations/ablation/ablation_config.yaml`, then `run_ablation.py` to populate `benchmarks/Ablations/collected_data/`. |
| Training reward curves (`scripts/plot_reward_curves.py`) | Retrain each variant and configure `wandb` logging; the script pulls run histories from a wandb project. |
| `paper_plots/data/staircase_scan_N{2,4,6,8}.npz` | Train a QADAPT policy on the SuperSims env, then `scripts/eval_multi_N.py` rolls out 100 seeds × N qubits. |
| Capacitance plots (`plot_capacitance_convergence.py`, `plot_kalman_calibration.py`) | Run `qadapt.capacitance_model.collect_episode_data` against a trained virtualization model to produce `data/episode_data_1000ep.npy`. |
| Domain benchmark plots (`benchmarks/domain/plot_results.py`, `plot_scaling.py`) | Run each `benchmarks/domain/{bayesian,lbfgs,nelder_mead,random,dreamer}/run.py` to populate `benchmarks/results/final_{N}dot/`. |

Training entry points: `src/qadapt/training/train.py` (QADAPT + ablations on the
QArray env), `src/qadapt_for_supersim/` (SuperSims env), `benchmarks/MARL/*/train.py`
(MARL baselines). Modal cloud entry points: `modal_scripts/`.

## Citation

> De Nicolo, E., Marchand, R., Carlsson, C., Vaidhyanathan, P., Ares, N.
> *Action-Factored Multi-Agent Reinforcement Learning for Scalable Quantum Device Tuning.*
> 2026.
