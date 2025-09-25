#!/usr/bin/env python3
"""
Clean metrics extraction and logging for Ray RLlib training.
Provides professional console output and comprehensive wandb logging.
"""
import time
from typing import Any, Dict, Optional

import numpy as np
import psutil
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Global EMA state for sweep optimization
_ema_state = {
    'plunger_return_ema': None,
    'ema_period': 20,
    'ema_alpha': None
}


def initialize_ema(ema_period: int = 20):
    """Initialize EMA tracking for sweep optimization."""
    global _ema_state
    _ema_state['ema_period'] = ema_period
    _ema_state['ema_alpha'] = 2.0 / (ema_period + 1)
    _ema_state['plunger_return_ema'] = None


def update_ema(current_value: float) -> float:
    """Update and return the EMA value."""
    global _ema_state
    
    if _ema_state['ema_alpha'] is None:
        initialize_ema()
    
    if _ema_state['plunger_return_ema'] is None:
        _ema_state['plunger_return_ema'] = current_value
    else:
        _ema_state['plunger_return_ema'] = (
            _ema_state['ema_alpha'] * current_value + 
            (1 - _ema_state['ema_alpha']) * _ema_state['plunger_return_ema']
        )
    
    return _ema_state['plunger_return_ema']


def extract_training_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key training metrics from RLlib result dictionary.

    Args:
        result: Training result dictionary from algorithm.train()

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    # Basic iteration info
    metrics["iteration"] = result.get("training_iteration", 0)
    metrics["total_time"] = result.get("time_total_s", 0)
    metrics["iter_time"] = result.get("time_this_iter_s", 0)

    # Environment sampling metrics
    env_runners = result.get("env_runners", {})
    metrics["env_steps"] = env_runners.get("num_env_steps_sampled_lifetime", 0)

    # Episode return metrics (new API)
    metrics["episode_return_mean"] = env_runners.get("episode_return_mean", None)
    metrics["episode_return_min"] = env_runners.get("episode_return_min", None)
    metrics["episode_return_max"] = env_runners.get("episode_return_max", None)

    # Multi-agent episode returns
    agent_returns = env_runners.get("agent_episode_returns_mean", {})
    metrics["agent_returns"] = {}

    # Extract plunger and barrier policy returns
    for agent_id, return_val in agent_returns.items():
        if "plunger" in agent_id:
            metrics["agent_returns"]["plunger"] = return_val
        elif "barrier" in agent_id:
            metrics["agent_returns"]["barrier"] = return_val

    # Get average returns per policy type if available
    plunger_returns = [v for k, v in agent_returns.items() if "plunger" in k]
    barrier_returns = [v for k, v in agent_returns.items() if "barrier" in k]

    if plunger_returns:
        metrics["agent_returns"]["plunger_avg"] = sum(plunger_returns) / len(plunger_returns)
    if barrier_returns:
        metrics["agent_returns"]["barrier_avg"] = sum(barrier_returns) / len(barrier_returns)

    # Learning metrics
    learners = result.get("learners", {})

    # Extract policy-specific learning metrics
    plunger_policy = learners.get("plunger_policy", {})
    barrier_policy = learners.get("barrier_policy", {})

    metrics["plunger_metrics"] = {
        "policy_loss": plunger_policy.get("policy_loss", None),
        "vf_loss": plunger_policy.get("vf_loss", None),
        "vf_loss_unclipped": plunger_policy.get("vf_loss_unclipped", None),
        "vf_explained_var": plunger_policy.get("vf_explained_var", None),
        "entropy": plunger_policy.get("entropy", None),
        "mean_kl": plunger_policy.get("mean_kl_loss", None),
        "advantage_mean": plunger_policy.get("advantage_mean", None),
        "advantage_variance": plunger_policy.get("advantage_variance", None),
        # Gradient norms (from Ray's built-in gradient logging)
        "grad_norm": plunger_policy.get("gradients_default_optimizer_global_norm", None),
        # Value function prediction statistics (from custom learner)
        "vf_predictions_mean": plunger_policy.get("vf_predictions_mean", None),
        "vf_predictions_variance": plunger_policy.get("vf_predictions_variance", None),
    }

    metrics["barrier_metrics"] = {
        "policy_loss": barrier_policy.get("policy_loss", None),
        "vf_loss": barrier_policy.get("vf_loss", None),
        "vf_loss_unclipped": barrier_policy.get("vf_loss_unclipped", None),
        "vf_explained_var": barrier_policy.get("vf_explained_var", None),
        "entropy": barrier_policy.get("entropy", None),
        "mean_kl": barrier_policy.get("mean_kl_loss", None),
        "advantage_mean": barrier_policy.get("advantage_mean", None),
        "advantage_variance": barrier_policy.get("advantage_variance", None),
        # Gradient norms (from Ray's built-in gradient logging)
        "grad_norm": barrier_policy.get("gradients_default_optimizer_global_norm", None),
    }

    # System metrics
    metrics["memory_percent"] = psutil.virtual_memory().percent
    metrics["cpu_percent"] = psutil.cpu_percent()

    return metrics


def print_training_progress(result: Dict[str, Any], iteration: int, start_time: float):
    """
    Print clean, professional training progress to console.

    Args:
        result: Training result dictionary
        iteration: Current iteration number
        start_time: Training start time for elapsed calculation
    """
    metrics = extract_training_metrics(result)
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(
        f"ITERATION {iteration + 1:3d} | Elapsed: {elapsed:6.1f}s | Iter Time: {metrics['iter_time']:6.1f}s"
    )
    print(f"{'='*80}")

    # Episode Returns
    if metrics["episode_return_mean"] is not None:
        print(
            f"Episode Returns | Mean:    {metrics['episode_return_mean']:8.3f} | "
            f"Min:     {metrics['episode_return_min']:8.3f} | "
            f"Max: {metrics['episode_return_max']:8.3f}"
        )
    else:
        print("Episode Returns | No completed episodes yet")

    # Multi-agent returns
    if metrics["agent_returns"]:
        print("Agent Returns   |", end="")
        if "plunger_avg" in metrics["agent_returns"]:
            print(f" Plunger: {metrics['agent_returns']['plunger_avg']:8.3f} |", end="")
        if "barrier_avg" in metrics["agent_returns"]:
            print(f" Barrier: {metrics['agent_returns']['barrier_avg']:8.3f} |", end="")
        print()

    # Learning Progress
    print("Policy Loss     |", end="")
    p_loss = metrics["plunger_metrics"]["policy_loss"]
    b_loss = metrics["barrier_metrics"]["policy_loss"]
    if p_loss is not None:
        print(f" Plunger: {p_loss:8.4f} |", end="")
    if b_loss is not None:
        print(f" Barrier: {b_loss:8.4f} |", end="")
    print()

    print("Value Loss      |", end="")
    p_vf = metrics["plunger_metrics"]["vf_loss"]
    b_vf = metrics["barrier_metrics"]["vf_loss"]
    if p_vf is not None:
        print(f" Plunger: {p_vf:8.4f} |", end="")
    if b_vf is not None:
        print(f" Barrier: {b_vf:8.4f} |", end="")
    print()

    # System Stats
    print(
        f"System Usage    | Memory: {metrics['memory_percent']:5.1f}%    | "
        f"CPU:       {metrics['cpu_percent']:5.1f}% | "
        f"Steps: {metrics['env_steps']:,}"
    )

    print(f"{'='*80}\n")


def log_to_wandb(result: Dict[str, Any], iteration: int):
    """
    Log comprehensive metrics to wandb using latest API.

    Args:
        result: Training result dictionary
        iteration: Current iteration number
    """
    if not wandb.run:
        return

    metrics = extract_training_metrics(result)

    # Prepare logging dictionary
    log_dict = {
        "iteration": iteration + 1,  # Log iteration for step metric
        "total_time": metrics["total_time"],
    }

    # Episode returns
    if metrics["episode_return_mean"] is not None:
        log_dict.update(
            {
                "episode_return_mean": metrics["episode_return_mean"],
                "episode_return_min": metrics["episode_return_min"],
                "episode_return_max": metrics["episode_return_max"],
            }
        )

    # Multi-agent returns
    if metrics["agent_returns"]:
        if "plunger_avg" in metrics["agent_returns"]:
            plunger_return = metrics["agent_returns"]["plunger_avg"]
            log_dict["plunger_return_avg"] = plunger_return
            
            # Update and log EMA for sweep optimization
            plunger_return_ema = update_ema(plunger_return)
            log_dict["plunger_return_ema"] = plunger_return_ema
            
        if "barrier_avg" in metrics["agent_returns"]:
            log_dict["barrier_return_avg"] = metrics["agent_returns"]["barrier_avg"]

    # Plunger policy metrics
    p_metrics = metrics["plunger_metrics"]
    if p_metrics["policy_loss"] is not None:
        log_dict.update(
            {
                "plunger_policy_loss": p_metrics["policy_loss"],
                "plunger_vf_loss": p_metrics["vf_loss"],
                "plunger_vf_explained_var": p_metrics["vf_explained_var"],
                "plunger_entropy": p_metrics["entropy"],
                "plunger_mean_kl": p_metrics["mean_kl"],
            }
        )
    
    # Add advantage metrics if available
    if p_metrics["advantage_mean"] is not None:
        log_dict.update(
            {
                "plunger_advantage_mean": p_metrics["advantage_mean"],
                "plunger_advantage_variance": p_metrics["advantage_variance"],
            }
        )
    
    # Add gradient norm if available
    if p_metrics["grad_norm"] is not None:
        log_dict["plunger_grad_norm"] = p_metrics["grad_norm"]
    
    # Add value function prediction statistics if available
    if p_metrics["vf_predictions_mean"] is not None:
        log_dict["plunger_vf_predictions_mean"] = p_metrics["vf_predictions_mean"]
    if p_metrics["vf_predictions_variance"] is not None:
        log_dict["plunger_vf_predictions_variance"] = p_metrics["vf_predictions_variance"]

    # Barrier policy metrics
    b_metrics = metrics["barrier_metrics"]
    if b_metrics["policy_loss"] is not None:
        log_dict.update(
            {
                "barrier_policy_loss": b_metrics["policy_loss"],
                "barrier_vf_loss": b_metrics["vf_loss"],
                "barrier_vf_explained_var": b_metrics["vf_explained_var"],
                "barrier_entropy": b_metrics["entropy"],
                "barrier_mean_kl": b_metrics["mean_kl"],
            }
        )
    
    # Add advantage metrics if available
    if b_metrics["advantage_mean"] is not None:
        log_dict.update(
            {
                "barrier_advantage_mean": b_metrics["advantage_mean"],
                "barrier_advantage_variance": b_metrics["advantage_variance"],
            }
        )
    
    # Add gradient norm if available
    if b_metrics["grad_norm"] is not None:
        log_dict["barrier_grad_norm"] = b_metrics["grad_norm"]

    # Policy log std metrics from callbacks
    learners = result.get("learners", {})
    plunger_policy = learners.get("plunger_policy", {})
    barrier_policy = learners.get("barrier_policy", {})

    # Log std metrics for plunger policy
    if "plunger_log_std_mean" in plunger_policy:
        log_dict.update({
            "plunger_log_std_mean": plunger_policy["plunger_log_std_mean"],
            "plunger_log_std_min": plunger_policy.get("plunger_log_std_min"),
            "plunger_log_std_max": plunger_policy.get("plunger_log_std_max"),
            "plunger_policy_variance_mean": float(np.exp(plunger_policy["plunger_log_std_mean"] * 2))  # Convert log_std to variance
        })

    # Log std metrics for barrier policy  
    if "barrier_log_std_mean" in barrier_policy:
        log_dict.update({
            "barrier_log_std_mean": barrier_policy["barrier_log_std_mean"],
            "barrier_log_std_min": barrier_policy.get("barrier_log_std_min"),
            "barrier_log_std_max": barrier_policy.get("barrier_log_std_max"),
            "barrier_policy_variance_mean": float(np.exp(barrier_policy["barrier_log_std_mean"] * 2))  # Convert log_std to variance
        })

    # Check for scan images in custom metrics
    env_runners = result.get("env_runners", {})
    custom_metrics = env_runners.get("custom_metrics", {})
    
    if "scan_images" in custom_metrics and "scan_episode_count" in custom_metrics:
        scan_images = custom_metrics["scan_images"]
        episode_count = custom_metrics["scan_episode_count"]
        
        # Log scan images to wandb
        try:
            log_scans_to_wandb(scan_images, episode_count)
        except Exception as e:
            print(f"Error logging scans to wandb: {e}")

    # Log to wandb (don't specify step since we're logging iteration explicitly)
    wandb.log(log_dict)

    # Update summary metrics for best performance tracking
    if metrics["episode_return_mean"] is not None:
        if (
            not hasattr(wandb.run, "summary")
            or wandb.run.summary.get("best_episode_return") is None
        ):
            wandb.run.summary["best_episode_return"] = metrics["episode_return_mean"]
        else:
            if metrics["episode_return_mean"] > wandb.run.summary.get("best_episode_return", 0):
                wandb.run.summary["best_episode_return"] = metrics["episode_return_mean"]


def upload_checkpoint_artifact(checkpoint_path: str, iteration: int, reward: float):
    """
    Upload checkpoint as wandb artifact when performance improves.

    Args:
        checkpoint_path: Path to the Ray RLlib checkpoint directory
        iteration: Current training iteration
        reward: Episode return mean that triggered this upload
    """
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


def setup_wandb_metrics(ema_period: int = 20):
    """
    Setup wandb metric definitions for better visualization.
    Call this after wandb.init().
    
    Args:
        ema_period: Period for EMA calculation (default: 20)
    """
    if not wandb.run:
        return
    
    # Initialize EMA with the specified period
    initialize_ema(ema_period)

    # Define custom metrics for better tracking
    wandb.define_metric("iteration")
    wandb.define_metric("episode_return_mean", step_metric="iteration", summary="max")
    wandb.define_metric("plunger_return_avg", step_metric="iteration", summary="max")
    wandb.define_metric("plunger_return_ema", step_metric="iteration", summary="max")
    wandb.define_metric("barrier_return_avg", step_metric="iteration", summary="max")
    wandb.define_metric("plunger_policy_loss", step_metric="iteration", summary="min")
    wandb.define_metric("barrier_policy_loss", step_metric="iteration", summary="min")


def log_scans_to_wandb(scan_images, iteration: int):
    """
    Log scan images directly to wandb.
    
    Args:
        scan_images: numpy array of shape (H, W, N-1) containing scan images
        iteration: Current training iteration
    """
    if not wandb.run:
        return
        
    try:
        num_channels = scan_images.shape[2]
        fig, axes = plt.subplots(1, num_channels, figsize=(4*num_channels, 4))
        if num_channels == 1:
            axes = [axes]
        
        scan_artifacts = {}
        for ch in range(num_channels):
            axes[ch].imshow(scan_images[:, :, ch], cmap='viridis')
            axes[ch].set_title(f'Scan Channel {ch}')
            axes[ch].axis('off')
            
            scan_artifacts[f'scan_channel_{ch}'] = wandb.Image(scan_images[:, :, ch])
        
        plt.tight_layout()
        
        wandb.log({
            'scans/combined_view': wandb.Image(fig),
            **{f'scans/{k}': v for k, v in scan_artifacts.items()},
            'iteration': iteration
        })
        
        plt.close(fig)
        print(f"Logged {num_channels} scan images to wandb at iteration {iteration}")
        
    except Exception as e:
        print(f"Error logging scans to wandb: {e}")
