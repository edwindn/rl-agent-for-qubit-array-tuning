#!/usr/bin/env python3
"""
Simplified Quantum Device Dataset Generator

This script generates datasets of quantum device charge sensor images
with corresponding Cgd matrices and ground truth parameters using QarrayBaseClass.

Usage:
    python dataset_generator.py --total_samples 10000 --workers 8 --num_dots 4 --output_dir ./dataset
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# print("Warning: using CPU for dataset generation, disable these settings in dataset_generator.py to run JAX on GPU")

# Add src directory to path for clean imports
from pathlib import Path
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
sys.path.insert(0, str(src_dir))

from swarm.environment.qarray_base_class import QarrayBaseClass


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    total_samples: int
    workers: int
    num_dots: int
    use_barriers: bool
    output_dir: str
    config_path: str = "qarray_config.yaml"
    batch_size: int = 1000
    voltage_offset_min: float = -20.0
    voltage_offset_max: float = 10.0
    barrier_offset_range: float = 5.0
    seed_base: int = 42
    min_obs_voltage_size: float = 1.5 # allows adjustable random window width, set equal to give fixed values
    max_obs_voltage_size: float = 2.0 #
    obs_image_size: int = 100 # must match size used in RL training


def generate_sample(sample_id: int, config: GenerationConfig) -> Dict[str, Any]:
    """
    Generate a single sample using QarrayBaseClass.
    
    Args:
        sample_id: Unique identifier for this sample
        config: Generation configuration
        
    Returns:
        Dictionary containing image, Cgd matrix, and ground truth data
    """
    try:
        # Create QarrayBaseClass instance with unique seed per sample
        np.random.seed(config.seed_base + sample_id)

        obs_voltage_size = np.random.uniform(config.min_obs_voltage_size, config.max_obs_voltage_size)

        use_barriers = config.use_barriers
        
        qarray = QarrayBaseClass(
            num_dots=config.num_dots,
            use_barriers=use_barriers,
            config_path=config.config_path,
            obs_voltage_min=-obs_voltage_size,
            obs_voltage_max=obs_voltage_size,
            obs_image_size=config.obs_image_size,
        )
        
        # Get ground truth voltages
        if use_barriers:
            gt_voltages, vb_optimal, _ = qarray.calculate_ground_truth()
        else:
            gt_voltages, _, _ = qarray.calculate_ground_truth()
        
        # Add random offset to ground truth for observation
        rng = np.random.default_rng(config.seed_base + sample_id)
        voltage_offset = rng.uniform(
            config.voltage_offset_min,
            config.voltage_offset_max,
            size=len(gt_voltages)
        )
        gate_voltages = gt_voltages + voltage_offset

        if use_barriers:
            barrier_offset = rng.uniform(
                -config.barrier_offset_range,
                config.barrier_offset_range,
                size=len(vb_optimal)
            )

            barrier_voltages = vb_optimal + barrier_offset
        
        else:
            barrier_voltages = np.array([0.0] * (config.num_dots - 1))
        
        # Generate observation
        obs = qarray._get_obs(gate_voltages, barrier_voltages)
        
        # Extract Cgd matrix
        cgd_matrix = qarray.model.Cgd.copy()
        
        return {
            'sample_id': sample_id,
            'image': obs['image'].astype(np.float32),  # Save space with float32
            'cgd_matrix': cgd_matrix.astype(np.float32),
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
    """
    Save a batch of samples to disk.
    
    Args:
        batch_id: Batch identifier
        batch_samples: List of sample dictionaries
        output_dir: Output directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Filter out failed samples
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
        
        # Save batch files
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
    Run test mode: generate 20 random samples with visualization but no file saving.
    
    Args:
        config: Generation configuration (most parameters ignored in test mode)
    """
    print("Running test mode: generating 16 random samples with visualization...")
    
    # Set matplotlib backend for file output
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    
    test_samples = []
    
    for i in range(16):    
        try:
            # Use completely random seeds for each sample
            sample = generate_sample(np.random.randint(0, 1000000), config)
            
            if sample.get('success', False):
                test_samples.append(sample)
            else:
                print(f"✗ (Error: {sample.get('error', 'Unknown')})")
                
        except Exception as e:
            print(f"✗ (Exception: {e})")
    
    if not test_samples:
        print("No successful samples generated. Cannot create visualization.")
        return
    
    print(f"\nSuccessfully generated {len(test_samples)} samples")

    def plot_images(channel):
        # Create visualization grid
        n_samples_to_plot = min(16, len(test_samples))  # Plot up to 16 samples in 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx in range(n_samples_to_plot):
            sample = test_samples[idx]
            
            # Get the middle channel for visualization (assuming multi-channel images)
            image = sample['image']
            plot_image = image[:, :, channel]
            
            # Plot the charge stability diagram
            im = axes[idx].imshow(plot_image, cmap='viridis', aspect='equal')
            cgd_matrix = sample['cgd_matrix']
            #cgd_ground_truth = [cgd_matrix[0,2], cgd_matrix[1,2], cgd_matrix[1,3]]
            #cgd_ground_truth = [np.round(v, 3).item() for v in cgd_ground_truth]
            #axes[idx].set_title(f'Sample {idx+1}\nvalues={cgd_ground_truth}', fontsize=10)
            axes[idx].set_xlabel('Gate Voltage 1')
            axes[idx].set_ylabel('Gate Voltage 2')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], shrink=0.8)
        
        # Hide unused subplots
        for idx in range(n_samples_to_plot, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Test Mode: {n_samples_to_plot} Random Charge Stability Diagrams\n'
                    f'({config.num_dots} dots each)\n'
                    f'Dots swept: {channel+1} and {channel+2}', fontsize=14)
        plt.tight_layout()
        
        # Always save to file
        output_path = f'test_mode_samples_{channel+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()

    for channel in range(3):
        plot_images(channel)
    
    # Print some statistics
    print("\nSample Statistics:")
    cgd_matrices = [s['cgd_matrix'] for s in test_samples]
    mean_cgd = np.mean([np.mean(cgd) for cgd in cgd_matrices])
    std_cgd = np.std([np.mean(cgd) for cgd in cgd_matrices])
    print(f"Average CGD matrix values: {mean_cgd:.4f} ± {std_cgd:.4f}")
    
    voltage_ranges = [np.ptp(s['gate_voltages']) for s in test_samples]
    print(f"Gate voltage ranges: {np.mean(voltage_ranges):.4f} ± {np.std(voltage_ranges):.4f}")


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
            'voltage_offset_range': config.voltage_offset_range,
            'seed_base': config.seed_base
        },
        'generation_stats': generation_stats,
        'data_structure': {
            'total_batches': num_batches,
            'images': f'Batched charge sensor images, shape per batch: (batch_size, 128, 128, {config.num_dots-1})',
            'cgd_matrices': f'Batched gate-to-dot capacitance matrices, shape per batch: (batch_size, {config.num_dots}, {config.num_dots+1})',
            'ground_truth': 'List of ground truth voltages and observation voltages per batch'
        },
        'usage_info': {
            'loading_example': {
                'images': 'images = np.load("images/batch_000.npy")',
                'cgd_matrices': 'cgd = np.load("cgd_matrices/batch_000.npy")',
                'ground_truth': 'with open("ground_truth/batch_000.json") as f: gt = json.load(f)'
            }
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
    logger.info(f"Starting dataset generation: {config.total_samples} samples with {config.workers} workers")
    
    # Create output directories
    create_output_directories(output_dir)
    
    # Generate samples in parallel and collect in batches
    start_time = time.time()
    successful_samples = 0
    failed_samples = 0
    saved_batches = 0
    
    # Calculate number of batches
    num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size
    logger.info(f"Generating {num_batches} batches of up to {config.batch_size} samples each")
    
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        # Submit all jobs
        future_to_id = {
            executor.submit(generate_sample, i, config): i 
            for i in range(config.total_samples)
        }
        
        # Collect samples in batches
        batch_samples = []
        current_batch_id = 0
        
        # Create progress bar
        pbar = tqdm(total=config.total_samples, desc="Generating samples", unit="samples")
        
        # Process completed jobs
        for future in as_completed(future_to_id):
            sample_id = future_to_id[future]
            try:
                sample = future.result()
                batch_samples.append(sample)
                
                if sample.get('success', False):
                    successful_samples += 1
                else:
                    failed_samples += 1
                
                # Update progress bar
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
                
                # Save batch when full or at end
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
                    
                    # Reset for next batch
                    batch_samples = []
                    current_batch_id += 1
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                failed_samples += 1
                pbar.update(1)
        
        pbar.close()
    
    # Final statistics
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
    
    # Save metadata
    save_metadata(config, output_dir, generation_stats)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate quantum device dataset')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes')
    parser.add_argument('--num_dots', type=int, default=8,
                       help='Number of quantum dots')
    parser.add_argument('--use_barriers', action='store_true',
                       help='Whether to use barrier gates in the model')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch file')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    parser.add_argument('--test', action='store_true',
                        help='Whether to run a sample run with visualisation')
    
    args = parser.parse_args()

    if args.test and args.num_dots != 4:
        args.num_dots = 4
        print("Warning: test mode should run with 4 dots, setting num_dots to 4")

    print(f"Using barriers: {args.use_barriers}")
    
    # Create configuration
    config = GenerationConfig(
        total_samples=args.total_samples,
        workers=args.workers,
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed_base=args.seed,
    )
    
    # Run in test mode or normal generation mode
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