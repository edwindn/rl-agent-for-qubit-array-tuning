#!/usr/bin/env python3
"""
Clean metrics extraction and logging for single-agent Ray RLlib training.
Provides console output and wandb logging.
"""
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import psutil
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ema_state = {
    "return_ema": None,
    "ema_period": 20,
    "ema_alpha": None,
}


def initialize_ema(ema_period: int = 20) -> None:
    """Initialize EMA tracking for sweep optimization."""
    _ema_state["ema_period"] = ema_period
    _ema_state["ema_alpha"] = 2.0 / (ema_period + 1)
    _ema_state["return_ema"] = None


def update_ema(current_value: float) -> float:
    """Update and return the EMA value."""
    if _ema_state["ema_alpha"] is None:
        initialize_ema()

    if _ema_state["return_ema"] is None:
        _ema_state["return_ema"] = current_value
    else:
        _ema_state["return_ema"] = (
            _ema_state["ema_alpha"] * current_value
            + (1 - _ema_state["ema_alpha"]) * _ema_state["return_ema"]
        )

    return _ema_state["return_ema"]


def _select_policy_metrics(learners: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(learners, dict) or not learners:
        return "default_policy", {}
    if "default_policy" in learners:
        return "default_policy", learners["default_policy"]
    policy_id = next(iter(learners.keys()))
    return policy_id, learners[policy_id]


def extract_training_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key training metrics from RLlib result dictionary."""
    metrics: Dict[str, Any] = {}

    metrics["iteration"] = result.get("training_iteration", 0)
    metrics["total_time"] = result.get("time_total_s", 0)
    metrics["iter_time"] = result.get("time_this_iter_s", 0)

    env_runners = result.get("env_runners", {})
    metrics["env_steps"] = env_runners.get("num_env_steps_sampled_lifetime", 0)

    metrics["episode_return_mean"] = env_runners.get("episode_return_mean")
    metrics["episode_return_min"] = env_runners.get("episode_return_min")
    metrics["episode_return_max"] = env_runners.get("episode_return_max")

    policy_id, policy = _select_policy_metrics(result.get("learners", {}))
    metrics["policy_id"] = policy_id

    metrics["policy_metrics"] = {
        "policy_loss": policy.get("policy_loss"),
        "vf_loss": policy.get("vf_loss"),
        "vf_loss_unclipped": policy.get("vf_loss_unclipped"),
        "vf_explained_var": policy.get("vf_explained_var"),
        "entropy": policy.get("entropy"),
        "mean_kl": policy.get("mean_kl_loss"),
        "advantage_mean": policy.get("advantage_mean"),
        "advantage_variance": policy.get("advantage_variance"),
        "grad_norm": policy.get("gradients_default_optimizer_global_norm"),
        "vf_predictions_mean": policy.get("vf_predictions_mean"),
        "vf_predictions_variance": policy.get("vf_predictions_variance"),
        "lr": policy.get("default_optimizer_learning_rate"),
        "qf_loss": policy.get("qf_loss"),
        "qf_twin_loss": policy.get("qf_twin_loss"),
        "alpha_loss": policy.get("alpha_loss"),
        "alpha_value": policy.get("alpha_value"),
        "qf_mean": policy.get("qf_mean"),
        "qf_min": policy.get("qf_min"),
        "qf_max": policy.get("qf_max"),
        "td_error_mean": policy.get("td_error_mean"),
    }

    metrics["memory_percent"] = psutil.virtual_memory().percent
    metrics["cpu_percent"] = psutil.cpu_percent()

    return metrics


def print_training_progress(result: Dict[str, Any], iteration: int, start_time: float) -> None:
    """Print clean training progress to console."""
    metrics = extract_training_metrics(result)
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(
        f"ITERATION {iteration + 1:3d} | Elapsed: {elapsed:6.1f}s | Iter Time: {metrics['iter_time']:6.1f}s"
    )
    print(f"{'='*80}")

    if metrics["episode_return_mean"] is not None:
        print(
            f"Episode Returns | Mean:    {metrics['episode_return_mean']:8.3f} | "
            f"Min:     {metrics['episode_return_min']:8.3f} | "
            f"Max: {metrics['episode_return_max']:8.3f}"
        )
    else:
        print("Episode Returns | No completed episodes yet")

    p_metrics = metrics["policy_metrics"]
    print("Policy Loss     |", end="")
    if p_metrics["policy_loss"] is not None:
        print(f" {p_metrics['policy_loss']:8.4f} |", end="")
    print()

    print("Value Loss      |", end="")
    if p_metrics["vf_loss"] is not None:
        print(f" {p_metrics['vf_loss']:8.4f} |", end="")
    print()

    print(
        f"System Usage    | Memory: {metrics['memory_percent']:5.1f}%    | "
        f"CPU:       {metrics['cpu_percent']:5.1f}% | "
        f"Steps: {metrics['env_steps']:,}"
    )

    print(f"{'='*80}\n")


def log_to_wandb(result: Dict[str, Any], iteration: int, distance_data_dir: Optional[str] = None) -> None:
    """Log metrics to wandb."""
    if not wandb.run:
        return

    metrics = extract_training_metrics(result)

    log_dict = {
        "iteration": iteration + 1,
        "total_time": metrics["total_time"],
    }

    if metrics["episode_return_mean"] is not None:
        log_dict.update(
            {
                "episode_return_mean": metrics["episode_return_mean"],
                "episode_return_min": metrics["episode_return_min"],
                "episode_return_max": metrics["episode_return_max"],
            }
        )

        return_ema = update_ema(metrics["episode_return_mean"])
        log_dict["return_ema"] = return_ema

    p_metrics = metrics["policy_metrics"]
    if p_metrics["policy_loss"] is not None:
        log_dict["policy_loss"] = p_metrics["policy_loss"]
    if p_metrics["vf_loss"] is not None:
        log_dict["vf_loss"] = p_metrics["vf_loss"]
    if p_metrics["vf_explained_var"] is not None:
        log_dict["vf_explained_var"] = p_metrics["vf_explained_var"]
    if p_metrics["entropy"] is not None:
        log_dict["entropy"] = p_metrics["entropy"]
    if p_metrics["mean_kl"] is not None:
        log_dict["mean_kl"] = p_metrics["mean_kl"]
    if p_metrics["advantage_mean"] is not None:
        log_dict["advantage_mean"] = p_metrics["advantage_mean"]
    if p_metrics["advantage_variance"] is not None:
        log_dict["advantage_variance"] = p_metrics["advantage_variance"]
    if p_metrics["grad_norm"] is not None:
        log_dict["grad_norm"] = p_metrics["grad_norm"]
    if p_metrics["vf_predictions_mean"] is not None:
        log_dict["vf_predictions_mean"] = p_metrics["vf_predictions_mean"]
    if p_metrics["vf_predictions_variance"] is not None:
        log_dict["vf_predictions_variance"] = p_metrics["vf_predictions_variance"]
    if p_metrics["lr"] is not None:
        log_dict["lr"] = p_metrics["lr"]

    if p_metrics["qf_loss"] is not None:
        log_dict["qf_loss"] = p_metrics["qf_loss"]
    if p_metrics["qf_twin_loss"] is not None:
        log_dict["qf_twin_loss"] = p_metrics["qf_twin_loss"]
    if p_metrics["alpha_loss"] is not None:
        log_dict["alpha_loss"] = p_metrics["alpha_loss"]
    if p_metrics["alpha_value"] is not None:
        log_dict["alpha_value"] = p_metrics["alpha_value"]
    if p_metrics["qf_mean"] is not None:
        log_dict["qf_mean"] = p_metrics["qf_mean"]
    if p_metrics["qf_min"] is not None:
        log_dict["qf_min"] = p_metrics["qf_min"]
    if p_metrics["qf_max"] is not None:
        log_dict["qf_max"] = p_metrics["qf_max"]
    if p_metrics["td_error_mean"] is not None:
        log_dict["td_error_mean"] = p_metrics["td_error_mean"]

    if distance_data_dir is not None:
        try:
            from pathlib import Path
            import glob

            distance_data_path = Path(distance_data_dir)
            plunger_folders = sorted([f for f in distance_data_path.iterdir() if f.is_dir() and f.name.startswith("plunger_")])
            barrier_folders = sorted([f for f in distance_data_path.iterdir() if f.is_dir() and f.name.startswith("barrier_")])

            plunger_distances_all = []
            if plunger_folders:
                fig, ax = plt.subplots(figsize=(10, 6))

                for agent_folder in plunger_folders:
                    npy_files = glob.glob(str(agent_folder / "*.npy"))
                    if not npy_files:
                        continue
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        distances = np.load(latest_file)
                        if not np.isfinite(distances).all():
                            raise ValueError(f"Distance data contains non-finite values: {latest_file}")
                        if distances.size == 0:
                            continue
                        plunger_distances_all.extend(distances)
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

                if plunger_distances_all:
                    ax.set_xlabel("Episode Step")
                    ax.set_ylabel("Distance from Ground Truth")
                    ax.set_title("Plunger Distances")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    wandb.log({"agent_vision/plunger_distances": wandb.Image(fig)})
                plt.close(fig)

            if plunger_distances_all:
                mean_distance_magnitude = np.mean(np.abs(plunger_distances_all))
                log_dict["plunger_mean_distance_magnitude"] = float(mean_distance_magnitude)

            if barrier_folders:
                fig, ax = plt.subplots(figsize=(10, 6))

                for agent_folder in barrier_folders:
                    npy_files = glob.glob(str(agent_folder / "*.npy"))
                    if not npy_files:
                        continue
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        distances = np.load(latest_file)
                        if not np.isfinite(distances).all():
                            raise ValueError(f"Distance data contains non-finite values: {latest_file}")
                        if distances.size == 0:
                            continue
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

                if barrier_folders:
                    ax.set_xlabel("Episode Step")
                    ax.set_ylabel("Distance from Ground Truth")
                    ax.set_title("Barrier Distances")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    wandb.log({"agent_vision/barrier_distances": wandb.Image(fig)})
                plt.close(fig)

        except Exception as e:
            print(f"Error plotting distance data: {e}")
            raise

    wandb.log(log_dict)

    if metrics["episode_return_mean"] is not None:
        if not hasattr(wandb.run, "summary") or wandb.run.summary.get("best_episode_return") is None:
            wandb.run.summary["best_episode_return"] = metrics["episode_return_mean"]
        else:
            if metrics["episode_return_mean"] > wandb.run.summary.get("best_episode_return", 0):
                wandb.run.summary["best_episode_return"] = metrics["episode_return_mean"]


def upload_checkpoint_artifact(checkpoint_path: str, iteration: int, reward: float) -> None:
    """Upload checkpoint as wandb artifact when performance improves."""
    if not wandb.run:
        return

    try:
        artifact = wandb.Artifact(
            name="rl_checkpoint_best",
            type="model_checkpoint",
            description=f"Best performing checkpoint at iteration {iteration} (reward: {reward:.4f})",
        )
        artifact.add_dir(checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded checkpoint artifact for iteration {iteration} (reward: {reward:.4f})")
    except Exception as e:
        print(f"Failed to upload checkpoint artifact: {e}")


def setup_wandb_metrics(ema_period: int = 20) -> None:
    """Setup wandb metric definitions for better visualization."""
    if not wandb.run:
        return

    initialize_ema(ema_period)

    wandb.define_metric("iteration")
    wandb.define_metric("episode_return_mean", step_metric="iteration", summary="max")
    wandb.define_metric("return_ema", step_metric="iteration", summary="max")
    wandb.define_metric("policy_loss", step_metric="iteration", summary="min")
    wandb.define_metric("vf_loss", step_metric="iteration", summary="min")
    wandb.define_metric("lr", step_metric="iteration")
