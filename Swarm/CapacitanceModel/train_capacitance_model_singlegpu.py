#!/usr/bin/env python3
"""
Training script for capacitance prediction model.

Trains a MobileNetV3-based model to predict Cgd[1,2], Cgd[1,3], and Cgd[0,2]
from single-channel capacitance images with uncertainty estimation.
"""

import os
import argparse
import time
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


def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                loss_fn: nn.Module,
                device: torch.device,
                epoch: int,
                log_wandb: bool = True) -> Dict[str, float]:
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
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        total_loss_batch, mse_loss, nll_loss = loss_fn(predictions, targets)
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_mse += mse_loss.item()
        total_nll += nll_loss.item()
        
        # Log batch metrics to wandb
        if log_wandb and wandb.run is not None:
            global_step = (epoch - 1) * num_batches + batch_idx
            wandb.log({
                'batch/train_loss': total_loss_batch.item(),
                'batch/train_mse': mse_loss.item(),
                'batch/train_nll': nll_loss.item(),
                'batch/step': global_step,
                'batch/epoch': epoch
            }, step=global_step)
        
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
    }


def validate_epoch(model: nn.Module, 
                   val_loader: DataLoader, 
                   loss_fn: nn.Module,
                   device: torch.device,
                   epoch: int,
                   log_wandb: bool = True) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_nll = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            total_loss_batch, mse_loss, nll_loss = loss_fn(predictions, targets)
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_nll += nll_loss.item()
            
            # Store for detailed metrics
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    num_batches = len(val_loader)
    
    # Calculate detailed metrics on all validation data
    all_pred_values = torch.cat([pred[0] for pred in all_predictions])
    all_pred_logvars = torch.cat([pred[1] for pred in all_predictions])
    all_targ = torch.cat(all_targets)
    
    detailed_metrics = calculate_metrics((all_pred_values, all_pred_logvars), all_targ)
    
    # Base validation metrics
    val_metrics = {
        'val_loss': total_loss / num_batches,
        'val_mse': total_mse / num_batches,
        'val_nll': total_nll / num_batches
    }
    
    # Add detailed metrics
    val_metrics.update(detailed_metrics)
    
    # Create and log scatter plots to wandb
    if log_wandb and wandb.run is not None:
        try:
            scatter_figures = create_scatter_plots((all_pred_values, all_pred_logvars), all_targ, epoch)
            
            # Log plots to wandb with proper step parameter
            for plot_name, fig in scatter_figures.items():
                wandb.log({f"plots/{plot_name}": wandb.Image(fig)}, step=epoch)
                plt.close(fig)  # Free memory
                
        except Exception as e:
            print(f"Warning: Could not create scatter plots: {e}")
    
    return val_metrics


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


def main():
    parser = argparse.ArgumentParser(description='Train Capacitance Prediction Model')
    parser.add_argument('--root_data_dir', type=str, 
                       default='/home/edn/rl-agent-for-qubit-array-tuning/Swarm/CapacitanceModel/',
                       help='Path to dataset directory')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=['dataset'],
                       help='List of data directories')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split fraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--load_to_memory', action='store_true',
                       help='Load all data to memory (requires ~11GB RAM)')
    parser.add_argument('--mse_weight', type=float, default=1.0,
                       help='Weight for MSE loss component')
    parser.add_argument('--nll_weight', type=float, default=0.1,
                       help='Weight for NLL loss component (default 0.1 to balance scales)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--gpu', type=str, default="0",
                        help='GPU index to use for training')
    parser.add_argument('--wandb_project', type=str, default='capacitance-prediction',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use for distributed training')
    parser.add_argument('--use_distributed', action='store_true',
                       help='Enable distributed training with Ray Train')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup
    device = setup_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args
        )
        
        # Define custom metrics to handle different step scales
        wandb.define_metric("batch/step")
        wandb.define_metric("batch/*", step_metric="batch/step")
        wandb.define_metric("epoch")
        wandb.define_metric("train_*", step_metric="epoch")
        wandb.define_metric("val_*", step_metric="epoch") 
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
        transform=transform
    )
    
    # Create model and loss function
    print("Creating model...")
    model = create_model()
    model = model.to(device)
    
    loss_fn = create_loss_function(
        mse_weight=args.mse_weight,
        nll_weight=args.nll_weight
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if not args.no_wandb:
        wandb.watch(model, log_freq=100)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, log_wandb=not args.no_wandb)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, epoch, log_wandb=not args.no_wandb)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['val_loss'])
        
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
              f"MSE: {train_metrics['train_mse']:.4f}, "
              f"NLL: {train_metrics['train_nll']:.4f}")
        print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
              f"MSE: {val_metrics['val_mse']:.4f}, "
              f"MAE: {val_metrics['mae']:.4f}")
        print(f"Val Targets - Cgd_1_2: {val_metrics['Cgd_1_2_mae']:.4f}, "
              f"Cgd_1_3: {val_metrics['Cgd_1_3_mae']:.4f}, "
              f"Cgd_0_2: {val_metrics['Cgd_0_2_mae']:.4f}")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {**train_metrics, **val_metrics}
            log_dict['epoch'] = epoch
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            log_dict['epoch_time'] = epoch_time
            wandb.log(log_dict, step=epoch)
        
        # Save checkpoints
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, 
                output_dir, is_best
            )
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 