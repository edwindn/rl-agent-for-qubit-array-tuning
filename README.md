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

Large training artefacts (checkpoints, ~13 GB of domain-benchmark JSONs, ~15 GB
of capacitance episode rollouts, wandb run histories) are not committed.
Reproducing the numbers and figures requires retraining and regenerating data.

### Table 1 — QADAPT, ablations, MARL baselines

1. Train each variant listed in `benchmarks/Ablations/ablation/ablation_config.yaml`.
   QADAPT and the architecture/training ablations train via
   `src/qadapt/training/train.py` with the matching config; MARL baselines via
   their own entry points under `benchmarks/MARL/`.
2. Run `benchmarks/Ablations/ablation/run_ablation.py --algo <name>` for each
   trained checkpoint. This writes per-trial JSONs into
   `benchmarks/Ablations/collected_data/<timestamp>_<algo>/`.
3. Aggregate:

   ```bash
   uv run python benchmarks/Ablations/ablation/compute_table.py --markdown table.md
   ```

### Training reward curves (Fig. appendix)

Retrain each QADAPT-family variant with wandb logging on, then point the
`QADAPT_FAMILY` list at the top of `scripts/plot_reward_curves.py` at your runs
and execute the script.

### Domain benchmarks — per-N convergence panels + scaling

1. Run each method to populate `benchmarks/results/final_{2,4,6,8}dot/`:

   ```bash
   for d in benchmarks/domain/{bayesian,lbfgs,nelder_mead,random,dreamer}; do
     uv run python $d/run.py
   done
   ```

2. Render:

   ```bash
   for n in 2 4 6 8; do
     uv run python benchmarks/domain/plot_results.py --plot convergence --num-dots $n
   done
   uv run python benchmarks/domain/plot_scaling.py
   ```

### Capacitance plots

1. Train the virtualisation CNN with
   `src/qadapt/capacitance_model/train_capacitance_model.py`.
2. Roll out a tuning episode against the trained CNN, dumping per-step samples
   to `data/episode_data_1000ep.npy`.
3. Render:

   ```bash
   uv run python scripts/plot_capacitance_convergence.py
   uv run python scripts/plot_kalman_calibration.py
   ```

### SuperSims appendix figures

The committed `paper_plots/data/staircase_scan_N{2,4,6,8}.npz` rollouts let the
two appendix figures be re-rendered immediately:

```bash
uv run python scripts/plot_allxy_violins.py
uv run python scripts/plot_convergence_multiN.py
```

To regenerate the rollouts from scratch, train QADAPT on the SuperSims env using
`src/qadapt_for_supersim/training_config.yaml`, then run
`bash scripts/run_all_N.sh` (wraps `scripts/eval_multi_N.py`) for 100 seeds × N
qubits, N ∈ {2, 4, 6, 8}.

Modal cloud entry points for hero training runs live in `modal_scripts/`.

## Citation

> Anonymous Authors.
> *Action-Factored Multi-Agent Reinforcement Learning for Scalable Quantum Device Tuning.*
> 2026.
