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
from typing import Dict, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import ray
from tqdm import tqdm

# Add parent directory to path for imports (works for both main process and Ray workers)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
swarm_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

# Add all necessary paths
for path in [environment_dir, swarm_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Don't import QarrayBaseClass at module level - let Ray workers import it locally
# This prevents Ray serialization issues with the module-level import


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    total_samples: int
    workers: int
    num_dots: int
    output_dir: str
    config_path: str
    batch_size: int = 1000
    voltage_offset_range: float = 0.1
    seed_base: int = 42


def generate_sample(sample_id: int, config: GenerationConfig) -> Dict[str, Any]:
    """
    Generate a single sample using QarrayBaseClass.
    
    Args:
        sample_id: Unique identifier for this sample
        config: Generation configuration
        
    Returns:
        Dictionary containing image, Cgd matrix, and ground truth data
    """
    # Import QarrayBaseClass locally to avoid Ray serialization issues
    try:
        from qarray_base_class import QarrayBaseClass
    except ImportError as e:
        return {
            'sample_id': sample_id,
            'success': False,
            'error': f'Failed to import QarrayBaseClass: {e}'
        }
    
    try:
        # Create QarrayBaseClass instance with unique seed per sample
        np.random.seed(config.seed_base + sample_id)
        
        qarray = QarrayBaseClass(
            num_dots=config.num_dots,
            config_path=config.config_path,
            obs_voltage_min=-1.0,
            obs_voltage_max=1.0,
            obs_image_size=128
        )
        
        # Get ground truth voltages
        gt_voltages = qarray.calculate_ground_truth()
        
        # Add random offset to ground truth for observation
        rng = np.random.default_rng(config.seed_base + sample_id)
        voltage_offset = rng.uniform(
            -config.voltage_offset_range, 
            config.voltage_offset_range, 
            size=len(gt_voltages)
        )
        gate_voltages = gt_voltages + voltage_offset
        
        # Create dummy barrier voltages (not used in current implementation)
        barrier_voltages = [0.0] * (config.num_dots - 1)
        
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


@ray.remote(num_cpus=1, memory=2*1024*1024*1024)  # 2GB per worker
def generate_sample_ray(sample_id: int, config_dict: dict) -> Dict[str, Any]:
    """
    Ray remote function for generating samples with resource limits.
    
    Args:
        sample_id: Unique identifier for this sample
        config_dict: Generation configuration as dictionary (for Ray serialization)
        
    Returns:
        Dictionary containing image, Cgd matrix, and ground truth data
    """
    import os
    import sys
    
    # Force CPU-only mode to prevent CUDA/JAX conflicts in parallel workers
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    # Force JAX to use CPU only
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Set up import paths in Ray worker (must be done in each worker)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
    if environment_dir not in sys.path:
        sys.path.insert(0, environment_dir)
    
    # Import in worker process
    try:
        from qarray_base_class import QarrayBaseClass
    except ImportError as e:
        return {
            'sample_id': sample_id,
            'success': False,
            'error': f'Ray worker import failed: {e}'
        }
    
    # Reconstruct config from dict (Ray requires serializable args)
    config = GenerationConfig(**config_dict)
    return generate_sample(sample_id, config)


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


# Test mode functionality removed - use small sample size instead


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
    """Generate the complete dataset using Ray for safer multiprocessing."""
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
    
    # Initialize Ray with resource limits
    try:
        # Force CPU-only mode to prevent GPU resource conflicts
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        ray.init(
            num_cpus=config.workers,
            num_gpus=0,  # Explicitly disable GPU usage
            object_store_memory=4*1024*1024*1024,  # 4GB object store
            include_dashboard=False,  # Disable dashboard for safety
            ignore_reinit_error=True  # Allow reinit if Ray already running
        )
        logger.info(f"Ray initialized with {config.workers} CPUs and 4GB object store")
        
        # Convert config to dict for Ray serialization
        config_dict = {
            'total_samples': config.total_samples,
            'workers': config.workers,
            'num_dots': config.num_dots,
            'output_dir': config.output_dir,
            'config_path': config.config_path,
            'batch_size': config.batch_size,
            'voltage_offset_range': config.voltage_offset_range,
            'seed_base': config.seed_base
        }
        
        # Process samples in smaller Ray batches to prevent memory issues
        ray_batch_size = min(config.batch_size, 500)  # Smaller Ray batches
        batch_samples = []
        current_batch_id = 0
        
        # Initialize progress bar
        pbar = tqdm(
            total=config.total_samples,
            desc="Generating samples",
            unit="samples"
        )
        
        for batch_start in range(0, config.total_samples, ray_batch_size):
            batch_end = min(batch_start + ray_batch_size, config.total_samples)
            logger.info(f"Submitting Ray batch: samples {batch_start} to {batch_end-1}")
            
            # Submit Ray tasks for this batch
            futures = [
                generate_sample_ray.remote(i, config_dict)
                for i in range(batch_start, batch_end)
            ]
            
            # Process results as they complete with timeout protection
            for future in futures:
                try:
                    sample = ray.get(future, timeout=60)  # 60 second timeout per sample
                    batch_samples.append(sample)
                    
                    if sample.get('success', False):
                        successful_samples += 1
                    else:
                        failed_samples += 1
                        
                except ray.exceptions.GetTimeoutError:
                    logger.warning(f"Sample generation timed out after 60 seconds")
                    failed_samples += 1
                except Exception as e:
                    logger.error(f"Error processing Ray task: {e}")
                    failed_samples += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Success: {successful_samples}, Failed: {failed_samples}, Batches: {saved_batches}/{num_batches}")
                
                # Save batch when full or at end
                if (len(batch_samples) >= config.batch_size or 
                    successful_samples + failed_samples == config.total_samples):
                    
                    if save_batch(current_batch_id, batch_samples, output_dir):
                        saved_batches += 1
                        pbar.set_description(f"Generating samples (Saved batch {current_batch_id})")
                    
                    # Reset for next batch
                    batch_samples = []
                    current_batch_id += 1
            
            # Cleanup Ray objects for this batch
            del futures
            
        # Close progress bar
        pbar.close()
            
    except Exception as e:
        logger.error(f"Ray processing failed: {e}")
        if 'pbar' in locals():
            pbar.close()
        raise
    finally:
        # Always cleanup Ray resources
        try:
            ray.shutdown()
            logger.info("Ray shutdown completed")
        except Exception as e:
            logger.warning(f"Ray shutdown had issues: {e}")
    
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
    parser.add_argument('--num_dots', type=int, default=4,
                       help='Number of quantum dots')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch file')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for dataset')
    parser.add_argument('--config_path', type=str, default='qarray_config.yaml',
                       help='Path to qarray configuration file')
    parser.add_argument('--voltage_offset_range', type=float, default=0.1,
                       help='Range for random voltage offset from ground truth')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create configuration
    config = GenerationConfig(
        total_samples=args.total_samples,
        workers=args.workers,
        num_dots=args.num_dots,
        output_dir=args.output_dir,
        config_path=args.config_path,
        batch_size=args.batch_size,
        voltage_offset_range=args.voltage_offset_range,
        seed_base=args.seed
    )
    
    # Run dataset generation
    try:
        generate_dataset(config)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main()