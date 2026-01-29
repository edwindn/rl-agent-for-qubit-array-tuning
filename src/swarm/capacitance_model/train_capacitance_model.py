#!/usr/bin/env python3
"""
Training script for capacitance prediction model.

Trains a MobileNetV3-based model to predict Cgd[1,2], Cgd[1,3], and Cgd[0,2]
from single-channel capacitance images with uncertainty estimation.
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.train.torch import prepare_model, prepare_data_loader
from ray.air import ScalingConfig

# Local imports
from CapacitancePrediction import create_model, create_loss_function
from dataloader import create_data_loaders, get_transforms


def setup_device() -> torch.device:
    """Setup GPU device if available"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def calculate_percentage_errors(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate percentage errors for each target."""
    abs_errors = torch.abs(predictions - targets)
    # Avoid division by zero by adding small epsilon
    percentage_errors = 100 * abs_errors / (torch.abs(targets) + 1e-8)
    
    target_names = ['Cgd_1_2', 'Cgd_1_3', 'Cgd_0_2']
    metrics = {}
    
    # Overall percentage error
    metrics['mean_percentage_error'] = percentage_errors.mean().item()
    metrics['median_percentage_error'] = percentage_errors.median().item()
    
    # Per-target percentage errors
    for i, name in enumerate(target_names):
        target_pe = percentage_errors[:, i]
        metrics[f'{name}_mean_pe'] = target_pe.mean().item()
        metrics[f'{name}_median_pe'] = target_pe.median().item()
    
    return metrics


def calculate_uncertainty_calibration(predictions: Tuple[torch.Tensor, torch.Tensor], 
                                    targets: torch.Tensor) -> Dict[str, float]:
    """Calculate uncertainty calibration metrics."""
    values, log_vars = predictions
    uncertainties = torch.exp(0.5 * log_vars)
    errors = torch.abs(values - targets)
    
    # Convert to numpy for easier computation
    uncertainties_np = uncertainties.detach().cpu().numpy()
    errors_np = errors.detach().cpu().numpy()
    
    # Flatten for overall calibration
    uncertainties_flat = uncertainties_np.flatten()
    errors_flat = errors_np.flatten()
    
    # Calculate correlation between predicted uncertainty and actual error
    if len(uncertainties_flat) > 1:
        correlation, p_value = stats.pearsonr(uncertainties_flat, errors_flat)
        spearman_corr, _ = stats.spearmanr(uncertainties_flat, errors_flat)
    else:
        correlation, p_value, spearman_corr = 0.0, 1.0, 0.0
    
    # Calibration bins analysis
    n_bins = 10
    bin_boundaries = np.linspace(0, np.percentile(uncertainties_flat, 95), n_bins + 1)
    bin_indices = np.digitize(uncertainties_flat, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_error = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_uncertainty = np.mean(uncertainties_flat[mask])
            bin_error = np.mean(errors_flat[mask])
            calibration_error += np.abs(bin_uncertainty - bin_error) * np.sum(mask)
    
    calibration_error /= len(uncertainties_flat)
    
    return {
        'uncertainty_error_correlation': correlation,
        'uncertainty_error_spearman': spearman_corr,
        'uncertainty_calibration_error': calibration_error,
        'mean_uncertainty': uncertainties.mean().item(),
        'std_uncertainty': uncertainties.std().item()
    }


def save_config(config: dict, save_dir: Path):
    """Save training configuration to JSON file"""
    config_path = save_dir / 'config.json'
    
    # Create a copy of config and convert non-serializable objects
    config_to_save = config.copy()
    
    # Convert Path objects to strings
    for key, value in config_to_save.items():
        if isinstance(value, Path):
            config_to_save[key] = str(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Path):
            config_to_save[key] = [str(p) for p in value]
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Saved configuration to {config_path}")


def create_scatter_plots(predictions: Tuple[torch.Tensor, torch.Tensor], 
                        targets: torch.Tensor, 
                        epoch: int,
                        max_samples: int = 500) -> Dict[str, plt.Figure]:
    """Create scatter plots for predictions vs actual values and reliability diagrams."""
    values, log_vars = predictions
    values = values.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    uncertainties = torch.exp(0.5 * log_vars).detach().cpu().numpy()
    
    target_names = ['Cgd[1,2]', 'Cgd[1,3]', 'Cgd[0,2]']
    figures = {}
    
    # Sample data if too many points
    n_samples = min(len(values), max_samples)
    if len(values) > max_samples:
        indices = np.random.choice(len(values), max_samples, replace=False)
        values = values[indices]
        targets = targets[indices]
        uncertainties = uncertainties[indices]
    
    # Create individual plots for each target
    for i, name in enumerate(target_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: predicted vs actual
        ax1.scatter(targets[:, i], values[:, i], alpha=0.6, s=20, color='steelblue')
        
        # Perfect prediction line
        min_val = min(targets[:, i].min(), values[:, i].min())
        max_val = max(targets[:, i].max(), values[:, i].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel(f'Actual {name}', fontsize=12)
        ax1.set_ylabel(f'Predicted {name}', fontsize=12)
        ax1.set_title(f'{name} - Predictions vs Actual (Epoch {epoch})', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Calculate R² and RMSE
        r2 = 1 - np.sum((targets[:, i] - values[:, i])**2) / np.sum((targets[:, i] - np.mean(targets[:, i]))**2)
        rmse = np.sqrt(np.mean((targets[:, i] - values[:, i])**2))
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=11)
        
        # Enhanced Reliability Diagram
        errors = targets[:, i] - values[:, i]
        abs_errors = np.abs(errors)
        standardized_errors = errors / (uncertainties[:, i] + 1e-8)
        
        # Use quantile-based binning for more balanced bins
        n_bins = 10
        # Create bins based on uncertainty quantiles for better distribution
        uncertainty_quantiles = np.linspace(0, 1, n_bins + 1)
        uncertainty_bins = np.quantile(uncertainties[:, i], uncertainty_quantiles)
        # Ensure unique bin edges
        uncertainty_bins = np.unique(uncertainty_bins)
        n_bins = len(uncertainty_bins) - 1
        
        bin_indices = np.digitize(uncertainties[:, i], uncertainty_bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate calibration metrics for each bin
        bin_uncertainties = []
        bin_coverage_68 = []  # ±1σ coverage
        bin_coverage_95 = []  # ±2σ coverage
        bin_counts = []
        bin_expected_coverage_68 = []  # Expected coverage based on normal distribution
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 3:  # Require at least 3 samples per bin
                bin_unc = np.mean(uncertainties[:, i][mask])
                bin_std_errors = standardized_errors[mask]
                
                # Calculate actual coverage
                coverage_68 = np.mean(np.abs(bin_std_errors) <= 1.0)
                coverage_95 = np.mean(np.abs(bin_std_errors) <= 2.0)
                
                # Expected coverage for normal distribution
                expected_68 = 0.68  # This is constant for ±1σ
                
                bin_uncertainties.append(bin_unc)
                bin_coverage_68.append(coverage_68)
                bin_coverage_95.append(coverage_95)
                bin_counts.append(np.sum(mask))
                bin_expected_coverage_68.append(expected_68)
        
        # Plot enhanced reliability diagram
        if len(bin_uncertainties) > 0:
            # Plot actual coverage
            ax2.plot(bin_uncertainties, bin_coverage_68, 'o-', color='steelblue', 
                    label='Actual ±1σ Coverage', linewidth=2.5, markersize=8, markerfacecolor='lightblue', markeredgecolor='steelblue')
            ax2.plot(bin_uncertainties, bin_coverage_95, 's-', color='darkgreen', 
                    label='Actual ±2σ Coverage', linewidth=2.5, markersize=8, markerfacecolor='lightgreen', markeredgecolor='darkgreen')
            
            # Perfect calibration lines
            ax2.axhline(y=0.68, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Perfect ±1σ (68%)')
            ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Perfect ±2σ (95%)')
            
            # Add confidence intervals for binned estimates
            for j, (unc, cov68, count) in enumerate(zip(bin_uncertainties, bin_coverage_68, bin_counts)):
                # Wilson score interval for binomial proportion
                p = cov68
                n = count
                if n > 0:
                    z = 1.96  # 95% confidence
                    wilson_center = (p + z**2/(2*n)) / (1 + z**2/n)
                    wilson_width = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / (1 + z**2/n)
                    ci_lower = max(0, wilson_center - wilson_width)
                    ci_upper = min(1, wilson_center + wilson_width)
                    
                    ax2.errorbar(unc, cov68, yerr=[[cov68-ci_lower], [ci_upper-cov68]], 
                               fmt='none', color='steelblue', alpha=0.5, capsize=3)
            
            # Add sample size annotations
            for j, (unc, cov68, count) in enumerate(zip(bin_uncertainties, bin_coverage_68, bin_counts)):
                if count >= 10:  # Only annotate bins with reasonable sample sizes
                    ax2.annotate(f'n={count}', (unc, cov68), xytext=(0, 10), 
                               textcoords='offset points', ha='center', va='bottom', 
                               fontsize=9, alpha=0.7, weight='bold')
        
        ax2.set_xlabel(f'Predicted Uncertainty {name}', fontsize=12)
        ax2.set_ylabel('Actual Coverage Fraction', fontsize=12)
        ax2.set_title(f'{name} - Uncertainty Calibration (Epoch {epoch})', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Calculate Expected Calibration Error (ECE) and other metrics
        overall_std_errors = standardized_errors
        overall_coverage_68 = np.mean(np.abs(overall_std_errors) <= 1.0)
        overall_coverage_95 = np.mean(np.abs(overall_std_errors) <= 2.0)
        
        # ECE calculation
        ece = 0.0
        if len(bin_uncertainties) > 0:
            total_samples = sum(bin_counts)
            for cov68, count in zip(bin_coverage_68, bin_counts):
                ece += (count / total_samples) * abs(cov68 - 0.68)
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties[:, i])
        
        # Add comprehensive metrics box
        metrics_text = f'Overall ±1σ: {overall_coverage_68:.3f}\nOverall ±2σ: {overall_coverage_95:.3f}\nECE: {ece:.3f}\nSharpness: {sharpness:.4f}'
        ax2.text(0.02, 0.02, metrics_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'), 
                fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        figures[f'{name.replace("[", "_").replace("]", "_").replace(",", "_")}_analysis'] = fig
    
    return figures


def calculate_metrics(predictions: Tuple[torch.Tensor, torch.Tensor], 
                     targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate detailed metrics for evaluation.
    
    Args:
        predictions: Tuple of (values, log_vars) from model
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    values, log_vars = predictions
    values = values.detach().cpu()
    targets = targets.detach().cpu()
    log_vars = log_vars.detach().cpu()
    
    # Overall metrics
    mse = nn.functional.mse_loss(values, targets).item()
    mae = nn.functional.l1_loss(values, targets).item()
    
    # Per-target metrics
    per_target_mse = []
    per_target_mae = []
    target_names = ['Cgd_1_2', 'Cgd_1_3', 'Cgd_0_2']
    
    metrics = {'mse': mse, 'mae': mae}
    
    for i, name in enumerate(target_names):
        target_mse = nn.functional.mse_loss(values[:, i], targets[:, i]).item()
        target_mae = nn.functional.l1_loss(values[:, i], targets[:, i]).item()
        
        metrics[f'{name}_mse'] = target_mse
        metrics[f'{name}_mae'] = target_mae
        
        per_target_mse.append(target_mse)
        per_target_mae.append(target_mae)
    
    # Add percentage errors
    percentage_metrics = calculate_percentage_errors(values, targets)
    metrics.update(percentage_metrics)
    
    # Add uncertainty calibration metrics
    calibration_metrics = calculate_uncertainty_calibration((values, log_vars), targets)
    metrics.update(calibration_metrics)
    
    return metrics


def compute_pcc(logvars: torch.Tensor, errors: torch.Tensor) -> float:
    vars_ = torch.exp(logvars)
    vars_flat = vars_.flatten().detach().cpu().numpy()
    errors_flat = errors.flatten().detach().cpu().numpy()

    if len(vars_flat) > 1 and np.std(vars_flat) > 1e-8 and np.std(errors_flat) > 1e-8:
        try:
            pcc = np.corrcoef(vars_flat, errors_flat)[0, 1]
            if np.isnan(pcc):
                pcc = 0.0
        except Exception as e:
            print(f"Warning: could not compute Pearson coefficient, {e}")
            pcc = 0.0
    else:
        pcc = 0.0

    return pcc


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                loss_fn: nn.Module,
                device: torch.device,
                epoch: int,
                log_wandb: bool = True,
                global_train_step: int = 0) -> Tuple[Dict[str, float], int]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
    

        ###
        # load_dict = True
        # if load_dict:
        #     chkpt = torch.load('./weights/best_model.pth', map_location=torch.device("cuda:0"))
        #     model.load_state_dict(chkpt['model_state_dict'])
        #     print('Loaded state dict successfully')
        # with torch.no_grad():
        #     predictions = model(images)
        #     total_loss_batch, mse_loss, nll_loss, log_vars, errors = loss_fn(predictions, targets)
        #     means, logvars = predictions
        #     means = means.cpu().numpy()
        # print('PREDICTIONS: ', means[:8,:])
        # print('TARGETS: ', targets.cpu().numpy()[:8,:])
        # print('MSE LOSS: ', mse_loss.item())
        # print('MAE: ', torch.sqrt(errors).mean().item())

        # import sys
        # print('--- Train capacitance model debug, auto-exiting')
        # import traceback
        # traceback.print_stack()
        # sys.exit(0)
        ###


        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        total_loss_batch, mse_loss, nll_loss, log_vars, errors = loss_fn(predictions, targets)
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_mse += mse_loss.item()
        total_nll += nll_loss.item()

        # Calculate correlation between prediction errors and uncertainties across the batch
        pcc = compute_pcc(log_vars, errors)
            
        mae = torch.sqrt(errors).mean().item()

        # Log batch metrics to wandb
        if log_wandb and wandb.run is not None:
            log_dict = {
                'train/step': global_train_step,
                'train/loss': total_loss_batch.item(),
                'train/mse': mse_loss.item(),
                'train/mae': mae,
                'train/nll': nll_loss.item(),
                'train/var': np.exp(log_vars.mean().item()),
                'train/pcc': pcc,
                'train/epoch': epoch
            }
            # Log separate NN/NNN losses if using separate heads
            if hasattr(loss_fn, 'get_separate_losses'):
                sep_losses = loss_fn.get_separate_losses()
                log_dict.update({
                    'train/nn_mse': sep_losses['nn_mse'],
                    'train/nn_nll': sep_losses['nn_nll'],
                    'train/nnn_mse': sep_losses['nnn_mse'],
                    'train/nnn_nll': sep_losses['nnn_nll'],
                })
            wandb.log(log_dict)
        
        global_train_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'MSE': f'{mse_loss.item():.4f}',
            'NLL': f'{nll_loss.item():.4f}'
        })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_mse': total_mse / num_batches,
        'train_nll': total_nll / num_batches
    }, global_train_step


def validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   loss_fn: nn.Module,
                   device: torch.device,
                   epoch: int,
                   log_wandb: bool = True,
                   global_val_step: int = 0) -> Tuple[Dict[str, float], int]:
    """Validate for one epoch"""
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            total_loss_batch, mse_loss, nll_loss, log_vars, errors = loss_fn(predictions, targets)

            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_nll += nll_loss.item()

            pcc = compute_pcc(log_vars, errors)

            # Calculate median absolute error
            abs_errors = torch.sqrt(errors)
            median_ae = torch.median(abs_errors).item()

            # Log batch metrics to wandb
            if log_wandb and wandb.run is not None:
                log_dict = {
                    'eval/step': global_val_step,
                    'eval/loss': total_loss_batch.item(),
                    'eval/mse': mse_loss.item(),
                    'eval/mae': abs_errors.mean().item(),
                    'eval/median_ae': median_ae,
                    'eval/nll': nll_loss.item(),
                    'eval/var': np.exp(log_vars.mean().item()),
                    'eval/pcc': pcc,
                    'eval/epoch': epoch,
                }
                # Log separate NN/NNN losses if using separate heads
                if hasattr(loss_fn, 'get_separate_losses'):
                    sep_losses = loss_fn.get_separate_losses()
                    log_dict.update({
                        'eval/nn_mse': sep_losses['nn_mse'],
                        'eval/nn_nll': sep_losses['nn_nll'],
                        'eval/nnn_mse': sep_losses['nnn_mse'],
                        'eval/nnn_nll': sep_losses['nnn_nll'],
                    })
                wandb.log(log_dict)

            global_val_step += 1
    
    num_batches = len(val_loader)
    
    # Base validation metrics
    val_metrics = {
        'val_loss': total_loss / num_batches,
        'val_mse': total_mse / num_batches,
        'val_nll': total_nll / num_batches
    }
    
    return val_metrics, global_val_step


def save_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   epoch: int, 
                   best_loss: float,
                   save_dir: Path,
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = save_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def train_func(config: dict):
    """
    Core training function that can be used for both single-GPU and distributed training.
    
    Args:
        config: Dictionary containing all training parameters
    """
    # Setup device - Ray Train will handle GPU assignment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to weights directory (only on rank 0)
    if train.get_context().get_world_rank() == 0:
        save_config(config, output_dir)
    
    # Initialize Wandb - only on rank 0 for distributed training
    if not config['no_wandb'] and train.get_context().get_world_rank() == 0:
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config
        )
        
        # Define custom metrics with separate step counters for train and validation
        wandb.define_metric("train/step")
        wandb.define_metric("eval/step")
        wandb.define_metric("epoch")
        
        # Training metrics use train step counter
        wandb.define_metric("train/*", step_metric="train/step")
        
        # Validation metrics use eval step counter  
        wandb.define_metric("eval/*", step_metric="eval/step")
        
        # Epoch-level metrics use epoch counter
        wandb.define_metric("val_*", step_metric="epoch")
        wandb.define_metric("train_*", step_metric="epoch") 
        wandb.define_metric("Cgd_*", step_metric="epoch")
        wandb.define_metric("uncertainty_*", step_metric="epoch")
        wandb.define_metric("mean_*", step_metric="epoch")
        wandb.define_metric("median_*", step_metric="epoch")
        wandb.define_metric("std_*", step_metric="epoch")
        wandb.define_metric("plots/*", step_metric="epoch")
        wandb.define_metric("mae", step_metric="epoch")
        wandb.define_metric("mse", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

    data_dirs = [os.path.join(config['root_data_dir'], dir_) for dir_ in config['data_dirs']]

    # Create data loaders
    print("Creating data loaders...")
    transform = get_transforms(normalize=True)
    train_loader, val_loader = create_data_loaders(
        data_dirs=data_dirs,
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        num_workers=config['num_workers'],
        load_to_memory=config['load_to_memory'],
        transform=transform,
        nearest_neighbours=not config['disable_nearest_neighbours']
    )
    
    # Prepare data loaders for distributed training
    train_loader = prepare_data_loader(train_loader)
    val_loader = prepare_data_loader(val_loader)
    
    # Create model and loss function
    separate_heads = config.get('separate_heads', False)
    print(f"Creating model with {config['backbone']} backbone...")
    if separate_heads:
        print("Using SEPARATE NN and NNN heads")
    model = create_model(
        output_size=2 if not config.get('disable_nearest_neighbours', False) else 3,
        backbone=config['backbone'],
        mobilenet=config['mobilenet'],
        separate_heads=separate_heads
    )
    model = prepare_model(model)  # Ray Train preparation

    loss_fn = create_loss_function(
        mse_weight=config['mse_weight'],
        nll_weight=config['nll_weight'],
        separate_heads=separate_heads,
        nnn_weight=config.get('nnn_weight', 1.0)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Print model info (only on rank 0)
    if train.get_context().get_world_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if not config['no_wandb']:
            wandb.watch(model, log_freq=100)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    global_train_step = 0
    global_val_step = 0
    
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        # Train
        train_metrics, global_train_step = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, log_wandb=(not config['no_wandb'] and train.get_context().get_world_rank() == 0), global_train_step=global_train_step)
        
        # Validate
        val_metrics, global_val_step = validate_epoch(model, val_loader, loss_fn, device, epoch, log_wandb=(not config['no_wandb'] and train.get_context().get_world_rank() == 0), global_val_step=global_val_step)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['val_loss'])
        
        epoch_time = time.time() - start_time
        
        
        # Log to wandb (only on rank 0)
        if not config['no_wandb'] and train.get_context().get_world_rank() == 0:
            log_dict = {**train_metrics, **val_metrics}
            log_dict['epoch'] = epoch
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            log_dict['epoch_time'] = epoch_time
            wandb.log(log_dict)
        
        # Report metrics to Ray Train
        metrics_to_report = {**train_metrics, **val_metrics}
        metrics_to_report['epoch_time'] = epoch_time
        train.report(metrics_to_report)
        
        # Save checkpoints (only on rank 0)
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        if train.get_context().get_world_rank() == 0 and (epoch % config['save_freq'] == 0 or is_best):
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, 
                output_dir, is_best
            )
    
    if train.get_context().get_world_rank() == 0:
        if not config['no_wandb']:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Capacitance Prediction Model')
    parser.add_argument('--root_data_dir', type=str, 
                       default='src/swarm/qarray_dataset/',
                       help='Path to dataset directory')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=['dataset'],
                       help='List of data directories')
    parser.add_argument('--output_dir', type=str, default='./weights',
                       help='Directory to save model weights')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split fraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--disable_data_loading', action='store_true',
                       help='Disable loading all data to memory (requires ~15GB RAM)')
    parser.add_argument('--mse_weight', type=float, default=10.0,
                       help='Weight for MSE loss component')
    parser.add_argument('--nll_weight', type=float, default=0.1,
                       help='Weight for NLL loss component (default 0.1 to balance scales)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--gpus', type=str, nargs='+', default=["7"],
                       help='GPU indices to use for training (will automatically use DDP if more than one given)')
    parser.add_argument('--wandb_project', type=str, default='capacitance-prediction',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--mobilenet', type=str, default='small', choices=['small', 'large'],
                       help='MobileNet architecture size (small or large)')
    parser.add_argument('--backbone', type=str, default='mobilenet', choices=['mobilenet', 'impala'],
                       help='CNN backbone: mobilenet (~2.5M params) or impala (~100K params)')
    parser.add_argument('--disable_nearest_neighbours', action='store_true',
                        help='Whether to only predict nearest-neighbour capacitances (disables)')
    parser.add_argument('--separate_heads', action='store_true',
                        help='Use separate NN and NNN heads (recommended for NNN mode)')
    parser.add_argument('--nnn_weight', type=float, default=1.0,
                        help='Extra weight for NNN loss when using separate heads (default: 1.0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()
    args.load_to_memory = not args.disable_data_loading
    
    gpu_list = []
    for gpu_arg in args.gpus:
        if ',' in gpu_arg:
            # Handle comma-separated: "0,1,2" -> ["0", "1", "2"]
            gpu_list.extend(gpu_arg.split(','))
        else:
            # Handle space-separated: already split by argparse
            gpu_list.append(gpu_arg)
    
    # Clean up any empty strings and strip whitespace
    gpu_list = [gpu.strip() for gpu in gpu_list if gpu.strip()]
    args.gpus = gpu_list
    
    # Convert args to config dict for Ray Train compatibility
    config = vars(args)

    num_gpus = len(gpu_list)
    use_distributed = num_gpus > 1

    if use_distributed:
        print("WARNING: Distributed training is not efficient unless your dataset is huge. Consider using a single gpu")
        print(f"Starting distributed training on GPUs {gpu_list}...")
        
        # Set CUDA_VISIBLE_DEVICES to only the specified GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
        print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        # Initialize Ray for distributed training
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,  # Consistent with existing codebase
                _system_config={
                    "verbose": 0,
                    "suppress_warnings": True,
                }
            )
        
        # Create scaling configuration
        scaling_config = ScalingConfig(
            num_workers=num_gpus,
            use_gpu=True,
            resources_per_worker={"GPU": 1}
        )
        
        # Create and configure TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
        )
        
        # Run distributed training
        result = trainer.fit()
        
        print("Distributed training completed!")
        print(f"Best validation loss: {result.metrics.get('val_loss', 'N/A')}")
        
    else:
        # Single-GPU training (original behavior) 
        gpu = args.gpus[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        print(f"Training on single gpu {gpu} ...")
        
        # Setup device
        device = setup_device()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration to weights directory
        save_config(vars(args), output_dir)
        
        # Initialize Wandb
        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=args
            )
            
            # Define custom metrics with separate step counters for train and validation
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("epoch")
            
            # Training metrics use train step counter
            wandb.define_metric("train/*", step_metric="train/step")
            
            # Validation metrics use eval step counter  
            wandb.define_metric("eval/*", step_metric="eval/step")
            
            # Epoch-level metrics use epoch counter
            wandb.define_metric("val_*", step_metric="epoch")
            wandb.define_metric("train_*", step_metric="epoch") 
            wandb.define_metric("Cgd_*", step_metric="epoch")
            wandb.define_metric("uncertainty_*", step_metric="epoch")
            wandb.define_metric("mean_*", step_metric="epoch")
            wandb.define_metric("median_*", step_metric="epoch")
            wandb.define_metric("std_*", step_metric="epoch")
            wandb.define_metric("plots/*", step_metric="epoch")
            wandb.define_metric("mae", step_metric="epoch")
            wandb.define_metric("mse", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")
            wandb.define_metric("epoch_time", step_metric="epoch")

        data_dirs = [os.path.join(args.root_data_dir, dir_) for dir_ in args.data_dirs]

        # Create data loaders
        print("Creating data loaders...")
        transform = get_transforms(normalize=True)
        train_loader, val_loader = create_data_loaders(
            data_dirs=data_dirs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            load_to_memory=args.load_to_memory,
            transform=transform,
            nearest_neighbours=not args.disable_nearest_neighbours
        )
        
        # Create model and loss function
        print(f"Creating model with {args.backbone} backbone...")
        if args.separate_heads:
            print("Using SEPARATE NN and NNN heads")
        model = create_model(
            output_size=2 if not args.disable_nearest_neighbours else 3,
            backbone=args.backbone,
            mobilenet=args.mobilenet,
            separate_heads=args.separate_heads
        )
        model = model.to(device)

        loss_fn = create_loss_function(
            mse_weight=args.mse_weight,
            nll_weight=args.nll_weight,
            separate_heads=args.separate_heads,
            nnn_weight=args.nnn_weight
        )
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Resume from checkpoint if specified
        start_epoch = 1
        best_val_loss = float('inf')
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']}, best_val_loss={best_val_loss:.4f}")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        if not args.no_wandb:
            wandb.watch(model, log_freq=100)

        # Training loop
        print("Starting training...")
        global_train_step = 0
        global_val_step = 0

        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics, global_train_step = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, log_wandb=not args.no_wandb, global_train_step=global_train_step)
            
            # Validate
            val_metrics, global_val_step = validate_epoch(model, val_loader, loss_fn, device, epoch, log_wandb=not args.no_wandb, global_val_step=global_val_step)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['val_loss'])
            
            epoch_time = time.time() - start_time
            
            
            # Log to wandb
            if not args.no_wandb:
                log_dict = {**train_metrics, **val_metrics}
                log_dict['epoch'] = epoch
                log_dict['lr'] = optimizer.param_groups[0]['lr']
                log_dict['epoch_time'] = epoch_time
                wandb.log(log_dict)
            
            # Save checkpoints
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            if epoch % args.save_freq == 0 or is_best:
                save_checkpoint(
                    model, optimizer, epoch, best_val_loss, 
                    output_dir, is_best
                )
        
        if not args.no_wandb:
            # Upload best model to wandb artifacts
            best_model_path = output_dir / 'best_model.pth'
            if best_model_path.exists():
                artifact = wandb.Artifact('capacitance_model', type='model')
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)
                print(f"Uploaded model to wandb artifacts: {best_model_path}")
            else:
                print("Warning: best_model.pth not found, skipping wandb artifact upload.")
            
            wandb.finish()


if __name__ == "__main__":
    main() 