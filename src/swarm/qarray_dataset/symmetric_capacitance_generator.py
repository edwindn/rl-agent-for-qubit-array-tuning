#!/usr/bin/env python3
"""
Symmetric Capacitance Dataset Generator

Generates datasets with effective cross-couplings uniformly distributed from -0.7 to +0.7
by manipulating virtual gate matrices (VGMs). The physical Cgd remains positive but the
effective coupling visible in the charge stability diagram spans negative to positive values.

The output format is compatible with the existing capacitance model dataloader:
- Images saved to images/batch_XXX.npy
- Target effective coupling matrix saved to cgd_matrices/batch_XXX.npy (same format)

Usage:
    python symmetric_capacitance_generator.py --total_samples 10000 --workers 8 --num_dots 4 --output_dir ./dataset
    python symmetric_capacitance_generator.py --test --num_dots 4  # Test mode with visualization
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src directory to path for clean imports
from pathlib import Path
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent
src_dir = swarm_package_dir.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.qarray_base_class import QarrayBaseClass


def load_env_config(env_config_path: str = None) -> dict:
    """Load env_config.yaml from environment directory."""
    if env_config_path is None:
        env_config_path = os.path.join(
            os.path.dirname(__file__), '..', 'environment', 'env_config.yaml'
        )
    with open(env_config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    total_samples: int
    workers: int
    num_dots: int
    use_barriers: bool = True  # Must be True for VGM to work correctly
    output_dir: str = "./symmetric_dataset"
    config_path: str = "qarray_config.yaml"
    batch_size: int = 1000
    seed_base: int = 42
    # Symmetric capacitance range for nearest neighbors
    coupling_min: float = -0.7
    coupling_max: float = 0.7
    # Symmetric capacitance range for second-nearest neighbors (next-nearest)
    nnn_coupling_min: float = -0.3
    nnn_coupling_max: float = 0.3
    # These will be loaded from env_config.yaml if not overridden
    env_config: dict = None

    def __post_init__(self):
        """Load env_config.yaml for parameters like radial_noise, voltage offsets, etc."""
        if self.env_config is None:
            self.env_config = load_env_config()

    @property
    def voltage_offset_min(self) -> float:
        return self.env_config['simulator']['constant_voltage_offset']['min']

    @property
    def voltage_offset_max(self) -> float:
        return self.env_config['simulator']['constant_voltage_offset']['max']

    @property
    def min_obs_voltage_size(self) -> float:
        return self.env_config['simulator']['window_delta_range']['min']

    @property
    def max_obs_voltage_size(self) -> float:
        return self.env_config['simulator']['window_delta_range']['max']

    @property
    def obs_image_size(self) -> int:
        return self.env_config['simulator']['resolution']

    @property
    def barrier_offset_range(self) -> float:
        # Use half of the full barrier range
        return self.env_config['simulator']['full_barrier_range_width']['max'] / 2

    def get_radial_noise_config(self) -> dict:
        """Get radial_noise config from env_config.yaml."""
        return self.env_config['simulator'].get('radial_noise')


def generate_sample(sample_id: int, config: GenerationConfig) -> Dict[str, Any]:
    """
    Generate a single sample with symmetric effective capacitances.

    The key difference from standard generation:
    1. Sample target effective couplings from [-0.7, +0.7]
    2. Set VGM to achieve these effective couplings
    3. Generate image (shows the target effective coupling)
    4. Save target matrix as cgd_matrix (for dataloader compatibility)
    """
    try:
        np.random.seed(config.seed_base + sample_id)
        rng = np.random.default_rng(config.seed_base + sample_id)

        obs_voltage_size = np.random.uniform(config.min_obs_voltage_size, config.max_obs_voltage_size)

        # Create QarrayBaseClass (physical Cgd sampled from config as usual)
        qarray = QarrayBaseClass(
            num_dots=config.num_dots,
            use_barriers=config.use_barriers,
            config_path=config.config_path,
            obs_voltage_min=-obs_voltage_size,
            obs_voltage_max=obs_voltage_size,
            obs_image_size=config.obs_image_size,
            radial_noise_config=config.get_radial_noise_config(),
        )

        # --- Sample target effective couplings (symmetric around 0) ---
        # Note: qarray convention has effective_coupling = -target_matrix (off-diagonal)
        # So we negate when building target_matrix to make the saved label match the image
        target_matrix = np.eye(config.num_dots)
        num_dots = config.num_dots

        # Nearest neighbor couplings (distance 1) - SYMMETRIC
        # Cgd[c, c+1] = Cgd[c+1, c] for each channel c
        nn_couplings = {}  # (dot, gate) -> coupling
        for c in range(num_dots - 1):
            coupling = rng.uniform(config.coupling_min, config.coupling_max)
            nn_couplings[(c, c + 1)] = coupling
            nn_couplings[(c + 1, c)] = coupling  # Same value (symmetric)
            target_matrix[c, c + 1] = -coupling
            target_matrix[c + 1, c] = -coupling

        # Second-nearest neighbor couplings (distance 2) - SYMMETRIC with edge case handling
        # For each pair (i, i+2), sample one value and set both positions
        # Edge cases: positions where gate index is out of bounds stay at 0
        nnn_couplings = {}  # (dot, gate) -> coupling
        for i in range(num_dots - 2):
            coupling = rng.uniform(config.nnn_coupling_min, config.nnn_coupling_max)
            # Set both [i, i+2] and [i+2, i] to the same value (symmetric)
            nnn_couplings[(i, i + 2)] = coupling
            nnn_couplings[(i + 2, i)] = coupling
            target_matrix[i, i + 2] = -coupling
            target_matrix[i + 2, i] = -coupling
        # Edge cases handled implicitly: positions like (2,4) and (1,-1) are never set,
        # so they stay at 0 in cgd_matrix_format

        # Set VGM to achieve target effective coupling
        qarray._set_vgm_for_target_effective_coupling(target_matrix)

        # Get ground truth voltages (pass initial zeros for gate/barrier voltages)
        initial_gate_voltages = np.zeros(config.num_dots)
        initial_barrier_voltages = np.zeros(config.num_dots - 1)
        if config.use_barriers:
            gt_voltages, vb_optimal, _ = qarray.calculate_ground_truth(initial_gate_voltages, initial_barrier_voltages)
        else:
            gt_voltages, _, _ = qarray.calculate_ground_truth(initial_gate_voltages, initial_barrier_voltages)

        # Add random offset to ground truth for observation
        # Use ±40V to cover full voltage space and exceed radial_noise full_noise_distance (30-40V)
        # This ensures a mix of clean honeycombs (offset < 30V) and white noise (offset > 30-40V)
        voltage_offset = rng.uniform(-40, 40, size=len(gt_voltages))
        gate_voltages = gt_voltages + voltage_offset

        if config.use_barriers:
            barrier_offset = rng.uniform(
                -config.barrier_offset_range,
                config.barrier_offset_range,
                size=len(vb_optimal)
            )
            barrier_voltages = vb_optimal + barrier_offset
        else:
            barrier_voltages = np.array([0.0] * (config.num_dots - 1))

        # Set ground truth for radial noise (noise increases with distance from GT)
        qarray.gate_ground_truth = gt_voltages

        # Generate observation (image now reflects target effective coupling)
        obs = qarray._get_obs(gate_voltages, barrier_voltages)

        # Build cgd_matrix format for dataloader compatibility
        # Store effective_couplings (what's visible in image) in the positions that get_nearest_targets reads
        # get_nearest_targets extracts: cgd[i, i+1] and cgd[i+1, i] for channel i
        # Shape: (num_dots, num_dots + 1)
        cgd_matrix_format = np.eye(num_dots, num_dots + 1, dtype=np.float32)

        # Store NN couplings (symmetric)
        for (dot, gate), coupling in nn_couplings.items():
            cgd_matrix_format[dot, gate] = coupling

        # Store NNN couplings (edge cases handled - missing entries stay 0)
        for (dot, gate), coupling in nnn_couplings.items():
            cgd_matrix_format[dot, gate] = coupling
        # Sensor column stays as zeros (not used by get_nearest_targets)

        return {
            'sample_id': sample_id,
            'image': obs['image'].astype(np.float32),
            'cgd_matrix': cgd_matrix_format,  # Target effective coupling in cgd format
            'ground_truth_voltages': gt_voltages.astype(np.float32),
            'gate_voltages': gate_voltages.astype(np.float32),
            'success': True
        }

    except Exception as e:
        logging.error(f"Failed to generate sample {sample_id}: {e}")
        return {
            'sample_id': sample_id,
            'success': False,
            'error': str(e)
        }


def save_batch(batch_id: int, batch_samples: list, output_dir: Path) -> bool:
    """Save a batch of samples to disk."""
    try:
        successful_samples = [s for s in batch_samples if s.get('success', False)]

        if not successful_samples:
            logging.warning(f"No successful samples in batch {batch_id}")
            return False

        batch_size = len(successful_samples)
        logging.info(f"Saving batch {batch_id} with {batch_size} samples")

        # Collect batch data
        images = np.stack([s['image'] for s in successful_samples])
        cgd_matrices = np.stack([s['cgd_matrix'] for s in successful_samples])

        ground_truth_data = []
        for s in successful_samples:
            gt_data = {
                'ground_truth_voltages': s['ground_truth_voltages'].tolist(),
                'gate_voltages': s['gate_voltages'].tolist(),
                'sample_id': s['sample_id']
            }
            ground_truth_data.append(gt_data)

        # Save batch files (same format as original generator)
        image_path = output_dir / 'images' / f'batch_{batch_id:03d}.npy'
        np.save(image_path, images)

        cgd_path = output_dir / 'cgd_matrices' / f'batch_{batch_id:03d}.npy'
        np.save(cgd_path, cgd_matrices)

        gt_path = output_dir / 'ground_truth' / f'batch_{batch_id:03d}.json'
        with open(gt_path, 'w') as f:
            json.dump(ground_truth_data, f, indent=2)

        return True

    except Exception as e:
        logging.error(f"Failed to save batch {batch_id}: {e}")
        return False


def create_output_directories(output_dir: Path) -> None:
    """Create necessary output directories."""
    directories = ['images', 'cgd_matrices', 'ground_truth', 'metadata']
    for dir_name in directories:
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)


def run_test_mode(config: GenerationConfig) -> None:
    """
    Run test mode: generate samples with visualization to verify symmetric couplings.
    """
    print("Running test mode: generating 16 random samples with visualization...")
    print(f"NN coupling range: [{config.coupling_min}, {config.coupling_max}]")
    print(f"NNN coupling range: [{config.nnn_coupling_min}, {config.nnn_coupling_max}]")

    import matplotlib
    matplotlib.use('Agg')

    test_samples = []

    for i in range(16):
        try:
            sample = generate_sample(np.random.randint(0, 1000000), config)

            if sample.get('success', False):
                test_samples.append(sample)
            else:
                print(f"✗ Sample {i} (Error: {sample.get('error', 'Unknown')})")

        except Exception as e:
            print(f"✗ Sample {i} (Exception: {e})")

    if not test_samples:
        print("No successful samples generated. Cannot create visualization.")
        return

    print(f"\nSuccessfully generated {len(test_samples)} samples")

    # Extract target couplings for histogram
    nn_couplings = []
    nnn_couplings = []
    for sample in test_samples:
        cgd = sample['cgd_matrix']
        num_dots = cgd.shape[0]
        # Nearest neighbor couplings
        for i in range(num_dots - 1):
            nn_couplings.append(cgd[i, i + 1])
        # Second-nearest neighbor couplings
        for i in range(num_dots - 2):
            nnn_couplings.append(cgd[i, i + 2])

    def plot_images(channel):
        n_samples_to_plot = min(16, len(test_samples))
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for idx in range(n_samples_to_plot):
            sample = test_samples[idx]

            image = sample['image']
            plot_image = image[:, :, channel]
            cgd = sample['cgd_matrix']

            # Get target coupling for this channel
            target_coupling = cgd[channel, channel + 1]

            im = axes[idx].imshow(plot_image, cmap='viridis', aspect='equal')
            axes[idx].set_title(f'Target coupling: {target_coupling:.3f}', fontsize=9)
            axes[idx].set_xlabel('Gate Voltage 1')
            axes[idx].set_ylabel('Gate Voltage 2')

            plt.colorbar(im, ax=axes[idx], shrink=0.8)

        for idx in range(n_samples_to_plot, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Symmetric Capacitance Test: Channel {channel}\n'
                    f'(Coupling range: [{config.coupling_min}, {config.coupling_max}])', fontsize=14)
        plt.tight_layout()

        output_path = f'symmetric_test_channel_{channel}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()

    # Plot images for each channel
    num_channels = min(3, config.num_dots - 1)
    for channel in range(num_channels):
        plot_images(channel)

    # Plot histogram of couplings (NN and NNN side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NN couplings histogram
    axes[0].hist(nn_couplings, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Target Effective Coupling')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Nearest Neighbor Couplings\n(Expected: Uniform [{config.coupling_min}, {config.coupling_max}])')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[0].legend()

    # NNN couplings histogram
    axes[1].hist(nnn_couplings, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Target Effective Coupling')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Second-Nearest Neighbor Couplings\n(Expected: Uniform [{config.nnn_coupling_min}, {config.nnn_coupling_max}])')
    axes[1].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('symmetric_coupling_distribution.png', dpi=150)
    print("Saved coupling distribution to: symmetric_coupling_distribution.png")
    plt.close()

    # Print statistics
    print("\nNearest Neighbor (NN) Statistics:")
    print(f"  Range: {min(nn_couplings):.4f} to {max(nn_couplings):.4f}")
    print(f"  Mean: {np.mean(nn_couplings):.4f} (should be ~0)")
    print(f"  Std: {np.std(nn_couplings):.4f}")

    print("\nSecond-Nearest Neighbor (NNN) Statistics:")
    print(f"  Range: {min(nnn_couplings):.4f} to {max(nnn_couplings):.4f}")
    print(f"  Mean: {np.mean(nnn_couplings):.4f} (should be ~0)")
    print(f"  Std: {np.std(nnn_couplings):.4f}")


def save_metadata(config: GenerationConfig, output_dir: Path,
                 generation_stats: Dict[str, Any]) -> None:
    """Save dataset metadata and generation statistics."""
    num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size

    metadata = {
        'generation_config': {
            'total_samples': config.total_samples,
            'batch_size': config.batch_size,
            'workers': config.workers,
            'num_dots': config.num_dots,
            'nn_coupling_range': [config.coupling_min, config.coupling_max],
            'nnn_coupling_range': [config.nnn_coupling_min, config.nnn_coupling_max],
            'seed_base': config.seed_base
        },
        'generation_stats': generation_stats,
        'data_structure': {
            'total_batches': num_batches,
            'images': f'Batched charge sensor images, shape per batch: (batch_size, {config.obs_image_size}, {config.obs_image_size}, {config.num_dots-1})',
            'cgd_matrices': f'Target effective coupling matrices, shape per batch: (batch_size, {config.num_dots}, {config.num_dots+1})',
            'ground_truth': 'List of ground truth voltages and observation voltages per batch'
        },
        'notes': {
            'nn_couplings': f'Nearest neighbor couplings (positions [i,i+1]) sampled from [{config.coupling_min}, {config.coupling_max}]',
            'nnn_couplings': f'Second-nearest neighbor couplings (positions [i,i+2]) sampled from [{config.nnn_coupling_min}, {config.nnn_coupling_max}]',
            'compatibility': 'Output format is compatible with standard CapacitanceDataset dataloader'
        }
    }

    metadata_path = output_dir / 'metadata' / 'dataset_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_dataset(config: GenerationConfig) -> None:
    """Generate the complete dataset using multiprocessing."""
    output_dir = Path(config.output_dir)

    # Setup logging
    log_dir = output_dir / 'metadata'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'generation.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting symmetric capacitance dataset generation: {config.total_samples} samples")
    logger.info(f"Coupling range: [{config.coupling_min}, {config.coupling_max}]")

    create_output_directories(output_dir)

    start_time = time.time()
    successful_samples = 0
    failed_samples = 0
    saved_batches = 0

    num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size
    logger.info(f"Generating {num_batches} batches of up to {config.batch_size} samples each")

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_to_id = {
            executor.submit(generate_sample, i, config): i
            for i in range(config.total_samples)
        }

        batch_samples = []
        current_batch_id = 0

        pbar = tqdm(total=config.total_samples, desc="Generating samples", unit="samples")

        for future in as_completed(future_to_id):
            sample_id = future_to_id[future]
            try:
                sample = future.result()
                batch_samples.append(sample)

                if sample.get('success', False):
                    successful_samples += 1
                else:
                    failed_samples += 1

                total_processed = successful_samples + failed_samples
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0

                pbar.set_postfix({
                    'Success': successful_samples,
                    'Failed': failed_samples,
                    'Rate': f'{rate:.1f}/s',
                    'Batches': f'{saved_batches}/{num_batches}'
                })
                pbar.update(1)

                if (len(batch_samples) >= config.batch_size or
                    successful_samples + failed_samples == config.total_samples):

                    if save_batch(current_batch_id, batch_samples, output_dir):
                        saved_batches += 1
                        pbar.set_postfix({
                            'Success': successful_samples,
                            'Failed': failed_samples,
                            'Rate': f'{rate:.1f}/s',
                            'Batches': f'{saved_batches}/{num_batches}'
                        })
                        logger.info(f"Saved batch {current_batch_id} ({len(batch_samples)} samples)")

                    batch_samples = []
                    current_batch_id += 1

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                failed_samples += 1
                pbar.update(1)

        pbar.close()

    total_time = time.time() - start_time
    generation_stats = {
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'saved_batches': saved_batches,
        'total_batches': num_batches,
        'total_time_seconds': total_time,
        'samples_per_second': successful_samples / total_time if total_time > 0 else 0
    }

    logger.info(f"Dataset generation completed!")
    logger.info(f"Successful samples: {successful_samples}")
    logger.info(f"Failed samples: {failed_samples}")
    logger.info(f"Saved batches: {saved_batches}/{num_batches}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average rate: {generation_stats['samples_per_second']:.1f} samples/second")

    save_metadata(config, output_dir, generation_stats)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate symmetric capacitance dataset')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes')
    parser.add_argument('--num_dots', type=int, default=4,
                       help='Number of quantum dots')
    parser.add_argument('--use_barriers', action='store_true',
                       help='Whether to use barrier gates in the model')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch file')
    parser.add_argument('--output_dir', type=str, default='./symmetric_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    parser.add_argument('--coupling_min', type=float, default=-0.7,
                       help='Minimum target effective coupling for nearest neighbors')
    parser.add_argument('--coupling_max', type=float, default=0.7,
                       help='Maximum target effective coupling for nearest neighbors')
    parser.add_argument('--nnn_coupling_min', type=float, default=-0.3,
                       help='Minimum target effective coupling for second-nearest neighbors')
    parser.add_argument('--nnn_coupling_max', type=float, default=0.3,
                       help='Maximum target effective coupling for second-nearest neighbors')
    parser.add_argument('--test', action='store_true',
                       help='Run test mode with visualization')

    args = parser.parse_args()

    print(f"Symmetric Capacitance Generator")
    print(f"Using barriers: {args.use_barriers}")
    print(f"NN coupling range: [{args.coupling_min}, {args.coupling_max}]")
    print(f"NNN coupling range: [{args.nnn_coupling_min}, {args.nnn_coupling_max}]")

    config = GenerationConfig(
        total_samples=args.total_samples,
        workers=args.workers,
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed_base=args.seed,
        coupling_min=args.coupling_min,
        coupling_max=args.coupling_max,
        nnn_coupling_min=args.nnn_coupling_min,
        nnn_coupling_max=args.nnn_coupling_max,
    )

    try:
        if args.test:
            run_test_mode(config)
        else:
            generate_dataset(config)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main()
