
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
import colorcet as cc
from matplotlib.colors import ListedColormap

def calibration_plot(all_predictions, all_targets, n_bins=10, save_path='calibration_plot.svg'):
    """
    Creates calibration plot comparing predicted vs observed errors.
    Perfect calibration = diagonal line.

    Args:
        all_predictions: List of prediction dicts with 'values', 'log_vars', 'uncertainties'
        all_targets: List of target arrays corresponding to each prediction
        n_bins: Number of bins for calibration plot
        save_path: Path to save the plot
    """
    # Figure styling constants
    cmap = ListedColormap(cc.gouldian)

    label_size = 9
    tick_size = 9
    tick_length = 4
    tick_width = 1

    # Flatten all predictions and targets
    pred_values = []
    pred_uncertainties = []
    actual_errors = []

    for pred_dict, target in zip(all_predictions, all_targets):
        values = pred_dict['values']
        uncertainties = pred_dict['uncertainties']
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

    # Single subplot figure
    fig_width = 3.25
    fig_height = 3.25 #2.4375
    axes_rect = [0.16, 0.15, 0.8, 0.75]

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(axes_rect)

    # Individual data points
    n = len(pred_uncertainties)
    indices = np.arange(n)
    normalized_indices = indices / (n - 1) if n > 1 else np.array([0.5])
    colors = cmap(normalized_indices)
    ax.scatter(pred_uncertainties, actual_errors, alpha=0.3, s=1, color=colors)

    x0, y0, width, height = axes_rect[0], axes_rect[1], axes_rect[2], axes_rect[3]
    cax = fig.add_axes([x0, y0 + height + 0.02, width * 0.5, height * 0.03])  # [x, y, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
    cbar.set_ticks([])
    cbar.set_label('Scan number', fontsize=label_size, labelpad=-15)

    # Binned averages
    ax.scatter(predicted_uncertainties, observed_errors, marker='s', s=10, color='k', ec='black')

    # Perfect calibration line
    max_val = max(max(predicted_uncertainties), max(observed_errors))
    ax.plot([0, max_val], [0, max_val], '--', color='k', linewidth=1)

    ax.set_xlabel(r'$\sigma^2$ (arb.)', fontsize=label_size)
    ax.set_ylabel(r'$\bar{\mu}_{\mathrm{error}}$ (arb.)', fontsize=label_size)

    # Spine and tick styling
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(tick_width)
    ax.tick_params(which='both', direction='in', labelsize=tick_size,
                   top=True, right=True, bottom=True, left=True,
                   length=tick_length, width=tick_width)

    # Calculate calibration error (ECE)
    calib_error = np.mean(np.abs(np.array(predicted_uncertainties) - np.array(observed_errors)))

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, 1)

    #plt.show()
    plt.savefig(save_path, transparent=True, format="svg")
    plt.close()
    print(f"Calibration plot saved as '{save_path}'")

    return calib_error


def coverage_test(all_predictions, all_targets, confidence_levels=[0.5, 0.68, 0.8, 0.9, 0.95, 0.99], save_path='coverage_plot.svg'):
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
    plt.savefig(save_path, transparent=True)
    plt.close()
    print(f"Coverage test plot saved as '{save_path}'")
    
    return results


def load_trained_model(model_path: str, device: torch.device, output_size: int = 3):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Create model
    model = create_model(output_size=output_size)

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


def load_inference_data(
    scan_dir: str,
    capacitance_dir: str,
    model,
    device: torch.device,
    transform=None,
    nearest_neighbours: bool = False,
    max_episodes: int = 0,
    max_steps_per_episode: int = 0,
):
    """
    Load scan data from inference runs and get capacitance predictions.

    Data is loaded in episode/step order so color coding shows progression.
    For each step, we run the model on each scan channel separately.

    Args:
        scan_dir: Path to scan_captures/ folder with episode_XXXX/step_YYYYYY.npy files
        capacitance_dir: Path to collected_data/.../capacitance/ folder with cgd_true_XXXX.npy
        model: Trained capacitance model
        device: torch device
        transform: Image transforms (normalization)
        nearest_neighbours: If True, use 2 targets; if False, use 3 targets (NNN mode)
        max_episodes: Limit number of episodes to process (0 = all)
        max_steps_per_episode: Limit steps per episode (0 = all)

    Returns:
        all_predictions: List of prediction dicts in episode/step order
        all_targets: List of target arrays
        metadata: Dict with episode/step info for each sample
    """
    from swarm.capacitance_model.capacitance_utils import get_targets_with_nnn, get_nearest_targets

    scan_path = Path(scan_dir)
    cap_path = Path(capacitance_dir)

    # Find all episode folders, sorted
    episode_dirs = sorted(scan_path.glob("episode_*"))
    if max_episodes > 0:
        episode_dirs = episode_dirs[:max_episodes]

    print(f"Found {len(episode_dirs)} episodes in {scan_dir}")

    all_predictions = []
    all_targets = []
    metadata = []  # Track episode/step for each sample

    for episode_dir in episode_dirs:
        episode_num = int(episode_dir.name.split("_")[1])

        # Load ground truth CGD for this episode
        cgd_path = cap_path / f"cgd_true_{episode_num:04d}.npy"
        if not cgd_path.exists():
            print(f"Warning: No CGD found for episode {episode_num}, skipping")
            continue

        cgd_matrix = np.load(cgd_path)
        num_dots = cgd_matrix.shape[0]
        num_channels = num_dots - 1  # Number of scan channels

        # Get step files (use .npy if available, otherwise skip)
        step_files = sorted(episode_dir.glob("step_*.npy"))
        if len(step_files) == 0:
            # Try to see if there are PNG files but no NPY
            png_files = sorted(episode_dir.glob("step_*.png"))
            if len(png_files) > 0:
                print(f"Warning: Episode {episode_num} has PNGs but no NPY files. Re-run data collection.")
            continue

        if max_steps_per_episode > 0:
            step_files = step_files[:max_steps_per_episode]

        for step_file in step_files:
            step_num = int(step_file.stem.split("_")[1])

            # Load raw scans: shape (num_channels, H, W)
            scans = np.load(step_file)

            # Process each channel separately
            for channel_idx in range(min(num_channels, scans.shape[0])):
                scan = scans[channel_idx]  # Shape: (H, W)

                # Get target for this channel
                if nearest_neighbours:
                    target = get_nearest_targets(channel_idx, cgd_matrix, num_dots, has_sensor=True)
                else:
                    target = get_targets_with_nnn(channel_idx, cgd_matrix, num_dots, has_sensor=True)

                # Prepare image for model: (1, H, W) tensor
                image = torch.from_numpy(scan.astype(np.float32)).unsqueeze(0)

                # Apply transforms if provided
                if transform:
                    image = transform(image)

                # Run model
                with torch.no_grad():
                    image_batch = image.unsqueeze(0).to(device)  # (1, 1, H, W)
                    values, log_vars = model(image_batch)

                # Store results
                prediction = {
                    'values': values.cpu().numpy()[0],
                    'log_vars': log_vars.cpu().numpy()[0],
                    'uncertainties': np.exp(0.5 * log_vars.cpu().numpy()[0])
                }
                all_predictions.append(prediction)
                all_targets.append(target)
                metadata.append({
                    'episode': episode_num,
                    'step': step_num,
                    'channel': channel_idx
                })

        if episode_num % 10 == 0:
            print(f"Processed episode {episode_num}, total samples: {len(all_predictions)}")

    print(f"Loaded {len(all_predictions)} samples from {len(episode_dirs)} episodes")
    return all_predictions, all_targets, metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test capacitance model variance/uncertainty calibration')
    parser.add_argument('--inference-data', type=str, default=None,
                        help='Path to collected_data/TIMESTAMP_CHECKPOINT/ folder for inference mode')
    parser.add_argument('--scan-dir', type=str, default=None,
                        help='Path to scan_captures/ folder (default: inferred from inference-data parent)')
    parser.add_argument('--model-path', type=str,
                        default='/home/rahul/rl-agent-for-qubit-array-tuning/src/swarm/capacitance_model/mobilenet_final_epoch_8/mobilenet_barrier_weights.pth',
                        help='Path to trained model weights')
    parser.add_argument('--num-samples', type=int, default=5000,
                        help='Number of samples for validation mode')
    parser.add_argument('--max-episodes', type=int, default=0,
                        help='Max episodes to process in inference mode (0 = all)')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Max steps per episode in inference mode (0 = all)')
    parser.add_argument('--output-prefix', type=str, default='model',
                        help='Prefix for output plot files')
    args = parser.parse_args()

    # Configuration
    output_size = 3  # NNN mode
    nearest_neighbours = False  # NNN mode

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    model = load_trained_model(args.model_path, device, output_size=output_size)

    # Get transforms
    transform = get_transforms(normalize=True)

    if args.inference_data:
        # === INFERENCE MODE: Load from collected data ===
        print(f"\n{'='*60}")
        print("INFERENCE MODE: Loading from collected data")
        print(f"{'='*60}\n")

        inference_path = Path(args.inference_data)
        capacitance_dir = inference_path / "capacitance"

        # Infer scan_dir from inference_data path if not provided
        if args.scan_dir:
            scan_dir = args.scan_dir
        else:
            # Assume scan_captures is sibling to collected_data
            scan_dir = inference_path.parent.parent / "scan_captures"

        print(f"Scan directory: {scan_dir}")
        print(f"Capacitance directory: {capacitance_dir}")

        if not capacitance_dir.exists():
            raise FileNotFoundError(f"Capacitance directory not found: {capacitance_dir}")
        if not Path(scan_dir).exists():
            raise FileNotFoundError(f"Scan directory not found: {scan_dir}")

        all_predictions, all_targets, metadata = load_inference_data(
            scan_dir=str(scan_dir),
            capacitance_dir=str(capacitance_dir),
            model=model,
            device=device,
            transform=transform,
            nearest_neighbours=nearest_neighbours,
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps,
        )

        print(f"\nLoaded {len(all_predictions)} samples from inference data")
        if metadata:
            episodes = set(m['episode'] for m in metadata)
            print(f"Episodes: {min(episodes)} to {max(episodes)}")

    else:
        # === VALIDATION MODE: Load from training dataset ===
        print(f"\n{'='*60}")
        print("VALIDATION MODE: Loading from training dataset")
        print(f"{'='*60}\n")

        root_data_dir = '/home/rahul/rl-agent-for-qubit-array-tuning/data/'
        data_dirs = ['symmetric_capacitance_dataset']
        batch_size = 64
        val_split = 0.1
        num_workers = 4
        load_to_memory = False
        num_samples = args.num_samples

        # Create data paths
        data_dirs_full = [os.path.join(root_data_dir, dir_) for dir_ in data_dirs]
        print(f"Loading data from: {data_dirs_full}")

        # Create data loaders
        _, val_loader = create_data_loaders(
            data_dirs=data_dirs_full,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            load_to_memory=load_to_memory,
            transform=transform,
            nearest_neighbours=nearest_neighbours
        )

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
                    predicted_uncertainties = np.exp(0.5 * predicted_log_vars)
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

    calib_path = f'{args.output_prefix}_calibration_plot.svg'
    calib_error = calibration_plot(all_predictions, all_targets, n_bins=15, save_path=calib_path)
    print(f"Expected Calibration Error: {calib_error:.6f}")

    # Run coverage test
    print(f"\n" + "="*60)
    print("RUNNING COVERAGE TEST")
    print("="*60)

    coverage_path = f'{args.output_prefix}_coverage_plot.svg'
    coverage_results = coverage_test(all_predictions, all_targets,
                                   confidence_levels=[0.5, 0.68, 0.8, 0.9, 0.95, 0.99],
                                   save_path=coverage_path)

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
    print(f"  - {calib_path}")
    print(f"  - {coverage_path}")


if __name__ == "__main__":
    main()