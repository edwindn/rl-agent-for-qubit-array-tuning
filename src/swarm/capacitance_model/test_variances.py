
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt

from pathlib import Path
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
sys.path.insert(0, str(src_dir))

from swarm.capacitance_model.dataloader import PercentileNormalize, get_channel_targets
from swarm.capacitance_model.CapacitancePrediction import CapacitancePredictionModel, create_model
from swarm.capacitance_model.dataloader import create_data_loaders, get_transforms


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calibration_plot(all_predictions, all_targets, n_bins=10, save_path='calibration_plot.png'):
    """
    Creates calibration plot comparing predicted vs observed errors
    Perfect calibration = diagonal line
    
    Args:
        all_predictions: List of prediction dicts with 'values', 'log_vars', 'uncertainties'
        all_targets: List of target arrays corresponding to each prediction
        n_bins: Number of bins for calibration plot
        save_path: Path to save the plot
    """
    # Flatten all predictions and targets
    pred_values = []
    pred_uncertainties = []
    actual_errors = []
    
    for pred_dict, target in zip(all_predictions, all_targets):
        values = pred_dict['values']  # Shape: (3,) - three capacitance values
        uncertainties = pred_dict['uncertainties']  # Shape: (3,)
        
        # Calculate actual errors for each capacitance prediction
        errors = np.abs(values - target)
        
        pred_values.extend(values)
        pred_uncertainties.extend(uncertainties)
        actual_errors.extend(errors)
    
    pred_uncertainties = np.array(pred_uncertainties)
    actual_errors = np.array(actual_errors)
    
    # Sort by predicted uncertainty
    sorted_indices = np.argsort(pred_uncertainties)
    sorted_uncertainties = pred_uncertainties[sorted_indices]
    sorted_errors = actual_errors[sorted_indices]
    
    # Bin predictions by uncertainty level
    bin_boundaries = np.linspace(0, len(sorted_uncertainties), n_bins + 1).astype(int)
    
    predicted_uncertainties = []
    observed_errors = []
    bin_counts = []
    
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i+1]
        if start == end:
            continue
            
        bin_predicted = sorted_uncertainties[start:end].mean()
        bin_observed = sorted_errors[start:end].mean()
        bin_count = end - start
        
        predicted_uncertainties.append(bin_predicted)
        observed_errors.append(bin_observed)
        bin_counts.append(bin_count)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Main calibration plot - show all individual points
    plt.subplot(2, 3, 1)
    # Plot all individual data points with some transparency
    plt.scatter(pred_uncertainties, actual_errors, alpha=0.3, s=1, color='lightblue', label='Individual points')
    
    # Overlay the binned averages
    scatter = plt.scatter(predicted_uncertainties, observed_errors, 
                         s=[c/5 for c in bin_counts], alpha=0.8, c=range(len(predicted_uncertainties)), 
                         cmap='viridis', edgecolors='black', linewidths=0.5, label='Binned averages')
    
    max_val = max(max(predicted_uncertainties), max(observed_errors))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration', linewidth=2)
    plt.xlabel('Predicted Standard Deviation')
    plt.ylabel('Observed Mean Absolute Error')
    plt.title('Uncertainty Calibration Plot (All Points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate calibration error (Expected Calibration Error - ECE)
    calib_error = np.mean(np.abs(np.array(predicted_uncertainties) - np.array(observed_errors)))
    plt.text(0.05, 0.95, f'ECE: {calib_error:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Individual residuals plot
    plt.subplot(2, 3, 2)
    # Calculate residuals for all individual points
    individual_residuals = actual_errors - pred_uncertainties
    plt.scatter(pred_uncertainties, individual_residuals, alpha=0.4, s=1, color='lightcoral')
    
    # Also show binned residuals
    binned_residuals = np.array(observed_errors) - np.array(predicted_uncertainties)
    plt.scatter(predicted_uncertainties, binned_residuals, alpha=0.8, s=30, color='darkred', edgecolors='black', linewidths=0.5)
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Predicted Standard Deviation')
    plt.ylabel('Residuals (Observed - Predicted)')
    plt.title('Calibration Residuals (All Points)')
    plt.grid(True, alpha=0.3)
    
    # Histogram of uncertainties
    plt.subplot(2, 3, 3)
    plt.hist(pred_uncertainties, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Distribution of Predicted Uncertainties\n(n={len(pred_uncertainties)})')
    plt.grid(True, alpha=0.3)
    
    # Histogram of errors
    plt.subplot(2, 3, 4)
    plt.hist(actual_errors, bins=50, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
    plt.xlabel('Actual Absolute Errors')
    plt.ylabel('Density')
    plt.title(f'Distribution of Actual Errors\n(n={len(actual_errors)})')
    plt.grid(True, alpha=0.3)
    
    # 2D density plot (hexbin) to show concentration of points
    plt.subplot(2, 3, 5)
    hb = plt.hexbin(pred_uncertainties, actual_errors, gridsize=30, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(hb, label='Point Density')
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration', linewidth=2)
    plt.xlabel('Predicted Standard Deviation')
    plt.ylabel('Observed Absolute Error')
    plt.title('Point Density Heatmap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot for model diagnostics
    plt.subplot(2, 3, 6)
    from scipy.stats import probplot
    probplot(individual_residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals\n(Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved as '{save_path}'")
    
    return calib_error


def coverage_test(all_predictions, all_targets, confidence_levels=[0.5, 0.68, 0.8, 0.9, 0.95, 0.99], save_path='coverage_plot.png'):
    """
    Test if X% confidence intervals actually contain X% of true values
    Returns coverage statistics and z-test p-values
    
    Args:
        all_predictions: List of prediction dicts with 'values', 'log_vars', 'uncertainties'
        all_targets: List of target arrays corresponding to each prediction
        confidence_levels: List of confidence levels to test
        save_path: Path to save the coverage plot
    """
    # Flatten all predictions and targets
    pred_values = []
    pred_uncertainties = []
    actual_errors = []
    
    for pred_dict, target in zip(all_predictions, all_targets):
        values = pred_dict['values']  # Shape: (3,) - three capacitance values
        uncertainties = pred_dict['uncertainties']  # Shape: (3,)
        
        # Calculate actual errors for each capacitance prediction
        errors = np.abs(values - target)
        
        pred_values.extend(values)
        pred_uncertainties.extend(uncertainties)
        actual_errors.extend(errors)
    
    pred_uncertainties = np.array(pred_uncertainties)
    actual_errors = np.array(actual_errors)
    
    results = []
    
    print("\nCoverage Test Results:")
    print("=" * 60)
    
    for conf_level in confidence_levels:
        # Z-score for this confidence level (two-tailed)
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        
        # Count how many errors fall within predicted confidence intervals
        within_interval = actual_errors <= z_score * pred_uncertainties
        observed_coverage = within_interval.mean()
        
        # Statistical test: Is observed coverage significantly different from expected?
        n_samples = len(within_interval)
        expected_coverage = conf_level
        
        # Binomial test
        n_covered = within_interval.sum()
        binom_result = stats.binomtest(n_covered, n_samples, expected_coverage)
        p_value = binom_result.pvalue  # Extract p-value from result object
        
        # Effect size (Cohen's h for proportions)
        cohen_h = 2 * (np.arcsin(np.sqrt(observed_coverage)) - np.arcsin(np.sqrt(expected_coverage)))
        
        results.append({
            'confidence_level': conf_level,
            'expected_coverage': expected_coverage,
            'observed_coverage': observed_coverage,
            'difference': observed_coverage - expected_coverage,
            'p_value': p_value,
            'cohen_h': cohen_h,
            'well_calibrated': p_value > 0.05,  # Not significantly different
            'n_samples': n_samples,
            'n_covered': n_covered
        })
        
        status = "✓" if p_value > 0.05 else "✗"
        print(f"{status} {conf_level*100:4.0f}% interval: {observed_coverage*100:5.1f}% coverage "
              f"(expected {expected_coverage*100:4.0f}%) - p={p_value:.4f} - effect size={cohen_h:.3f}")
    
    # Create coverage plot
    plt.figure(figsize=(12, 8))
    
    # Main coverage plot
    plt.subplot(2, 2, 1)
    expected = [r['expected_coverage'] for r in results]
    observed = [r['observed_coverage'] for r in results]
    
    plt.scatter(expected, observed, s=100, alpha=0.7, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Coverage', linewidth=2)
    
    # Add error bars for binomial confidence intervals
    n_total = len(pred_uncertainties)
    conf_intervals = []
    for r in results:
        # Wilson score interval for binomial proportion
        p = r['observed_coverage']
        n = r['n_samples']
        z = 1.96  # 95% confidence
        
        center = (p + z**2/(2*n)) / (1 + z**2/n)
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
        
        conf_intervals.append((center - margin, center + margin))
    
    lower_bounds = [ci[0] for ci in conf_intervals]
    upper_bounds = [ci[1] for ci in conf_intervals]
    
    plt.errorbar(expected, observed, 
                yerr=[np.array(observed) - np.array(lower_bounds), 
                      np.array(upper_bounds) - np.array(observed)],
                fmt='none', alpha=0.5, color='blue')
    
    plt.xlabel('Expected Coverage')
    plt.ylabel('Observed Coverage')
    plt.title('Coverage Test Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Coverage difference plot
    plt.subplot(2, 2, 2)
    differences = [r['difference'] for r in results]
    plt.bar(range(len(confidence_levels)), differences, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Confidence Level')
    plt.ylabel('Coverage Difference (Observed - Expected)')
    plt.title('Coverage Bias')
    plt.xticks(range(len(confidence_levels)), [f"{cl:.0%}" for cl in confidence_levels])
    plt.grid(True, alpha=0.3)
    
    # P-values plot
    plt.subplot(2, 2, 3)
    p_values = [r['p_value'] for r in results]
    colors = ['green' if p > 0.05 else 'red' for p in p_values]
    plt.bar(range(len(confidence_levels)), p_values, alpha=0.7, color=colors)
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    plt.xlabel('Confidence Level')
    plt.ylabel('p-value')
    plt.title('Statistical Significance Test')
    plt.xticks(range(len(confidence_levels)), [f"{cl:.0%}" for cl in confidence_levels])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, 'Coverage Test Summary:', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    
    # Calculate overall metrics
    well_calibrated_count = sum(1 for r in results if r['well_calibrated'])
    mean_abs_difference = np.mean([abs(r['difference']) for r in results])
    max_abs_difference = max([abs(r['difference']) for r in results])
    
    summary_text = f"""
Total samples: {n_total:,}
Well-calibrated levels: {well_calibrated_count}/{len(results)}
Mean absolute coverage error: {mean_abs_difference:.3f}
Max absolute coverage error: {max_abs_difference:.3f}

Individual Results:
"""
    
    for r in results:
        status = "✓" if r['well_calibrated'] else "✗"
        summary_text += f"{status} {r['confidence_level']:.0%}: {r['observed_coverage']:.3f} ({r['difference']:+.3f})\n"
    
    plt.text(0.1, 0.8, summary_text, fontsize=9, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Coverage test plot saved as '{save_path}'")
    
    return results


def load_trained_model(model_path: str, device: torch.device, disable_nearest_neighbours: bool = False):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create model with correct output size
    output_size = 3 if disable_nearest_neighbours else 2
    model = create_model(output_size)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model


def main():
    # Configuration
    root_data_dir = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/qarray_dataset/'
    data_dirs = ['dataset', '4dot_dataset']
    model_path = '/home/edn/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/weights/best_model.pth'
    batch_size = 64
    val_split = 0.2
    num_workers = 4
    load_to_memory = False
    num_samples = 1000
    disable_nearest_neighbours = False  # Set to True if model was trained without nearest neighbours
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data paths
    data_dirs_full = [os.path.join(root_data_dir, dir_) for dir_ in data_dirs]
    print(f"Loading data from: {data_dirs_full}")
    
    # Create data loaders
    transform = get_transforms(normalize=True)
    train_loader, val_loader = create_data_loaders(
        data_dirs=data_dirs_full,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        load_to_memory=load_to_memory,
        transform=transform
    )
    
    # Load trained model
    model = load_trained_model(model_path, device, disable_nearest_neighbours)
    
    # Collect validation images and run inference
    all_predictions = []
    all_targets = []
    
    loader = iter(val_loader)
    collected_samples = 0
    
    print(f"Collecting {num_samples} validation samples...")
    
    with torch.no_grad():
        while collected_samples < num_samples:
            try:
                images, targets = next(loader)
                batch_size_actual = images.size(0)
                
                # Only process as many samples as we need
                remaining = num_samples - collected_samples
                samples_to_process = min(batch_size_actual, remaining)
                
                # Process this batch
                images_batch = images[:samples_to_process].to(device)
                targets_batch = targets[:samples_to_process]
                values, log_vars = model(images_batch)
                
                # Convert to numpy and add to predictions list
                predicted_values = values.cpu().detach().numpy()
                predicted_log_vars = log_vars.cpu().detach().numpy()
                predicted_uncertainties = np.exp(0.5 * predicted_log_vars)  # Standard deviation from log variance
                target_values = targets_batch.cpu().numpy()
                
                # Add each sample to the list
                for i in range(samples_to_process):
                    prediction = {
                        'values': predicted_values[i],
                        'log_vars': predicted_log_vars[i], 
                        'uncertainties': predicted_uncertainties[i]
                    }
                    all_predictions.append(prediction)
                    all_targets.append(target_values[i])
                
                collected_samples += samples_to_process
                
                if collected_samples % 100 == 0:
                    print(f"Processed {collected_samples}/{num_samples} samples")
                    
            except StopIteration:
                print(f"Reached end of validation set with {collected_samples} samples")
                break
    
    print(f"Completed! Collected {len(all_predictions)} predictions with targets")
    
    # Print some statistics
    all_values = np.array([p['values'] for p in all_predictions])
    all_uncertainties = np.array([p['uncertainties'] for p in all_predictions])
    all_targets_array = np.array(all_targets)
    
    print(f"\nPrediction Statistics:")
    print(f"Values - Mean: {all_values.mean(axis=0)}")
    print(f"Values - Std: {all_values.std(axis=0)}")
    print(f"Targets - Mean: {all_targets_array.mean(axis=0)}")
    print(f"Targets - Std: {all_targets_array.std(axis=0)}")
    print(f"Uncertainties - Mean: {all_uncertainties.mean(axis=0)}")
    print(f"Uncertainties - Std: {all_uncertainties.std(axis=0)}")
    
    # Calculate overall prediction errors
    all_errors = np.abs(all_values - all_targets_array)
    print(f"Prediction Errors - Mean: {all_errors.mean(axis=0)}")
    print(f"Prediction Errors - Std: {all_errors.std(axis=0)}")
    
    # Run calibration analysis
    print(f"\n" + "="*60)
    print("RUNNING CALIBRATION ANALYSIS")
    print("="*60)
    
    calib_error = calibration_plot(all_predictions, all_targets, n_bins=15, save_path='model_calibration_plot.png')
    print(f"Expected Calibration Error: {calib_error:.6f}")
    
    # Run coverage test
    print(f"\n" + "="*60)
    print("RUNNING COVERAGE TEST")
    print("="*60)
    
    coverage_results = coverage_test(all_predictions, all_targets, 
                                   confidence_levels=[0.5, 0.68, 0.8, 0.9, 0.95, 0.99],
                                   save_path='model_coverage_plot.png')
    
    # Print final summary
    print(f"\n" + "="*60)
    print("FINAL UNCERTAINTY QUANTIFICATION SUMMARY")
    print("="*60)
    print(f"Total samples analyzed: {len(all_predictions)}")
    print(f"Expected Calibration Error (ECE): {calib_error:.6f}")
    
    well_calibrated = sum(1 for r in coverage_results if r['well_calibrated'])
    print(f"Well-calibrated confidence levels: {well_calibrated}/{len(coverage_results)}")
    
    mean_coverage_error = np.mean([abs(r['difference']) for r in coverage_results])
    print(f"Mean coverage error: {mean_coverage_error:.4f}")
    
    print(f"\nFiles generated:")
    print(f"  - model_calibration_plot.png")
    print(f"  - model_coverage_plot.png")


if __name__ == "__main__":
    main()