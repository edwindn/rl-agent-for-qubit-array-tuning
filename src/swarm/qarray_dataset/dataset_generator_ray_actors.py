#!/usr/bin/env python3
"""
Quantum Device Dataset Generator using Ray Actors (One QarrayBaseClass per GPU)

This script generates datasets of quantum device charge sensor images
with corresponding Cgd matrices and ground truth parameters using QarrayBaseClass.
Uses Ray Actors to ensure one QarrayBaseClass instance per GPU for memory safety.

Usage:
    python dataset_generator_ray_actors.py --total_samples 10000 --gpu_ids "6,7" --output_dir ./dataset
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
import ray
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
swarm_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

for path in [environment_dir, swarm_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""
    total_samples: int
    num_dots: int
    use_barriers: bool
    output_dir: str
    gpu_ids: str
    config_path: str = "qarray_config.yaml"
    resolution: int = 100
    batch_size: int = 1000
    voltage_offset_min: float = -40.0
    voltage_offset_max: float = 20.0
    barrier_offset_min: float = -3.0
    barrier_offset_max: float = -1.0
    seed_base: int = 42
    min_obs_voltage_size: float = 1.5 # allows adjustable random window width, set equal to give fixed values
    max_obs_voltage_size: float = 2.0 #


def load_ray_config(config_path: str = None) -> Dict[str, Any]:
    """Load Ray configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "ray_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Ray config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def enforce_cuda_availability(gpu_ids_str: str) -> List[int]:
    """Enforce CUDA is available for specified GPUs, return GPU list"""
    try:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]
        print(f"Requested GPU IDs: {gpu_ids}")
    except ValueError:
        raise RuntimeError(f"Invalid GPU IDs format: '{gpu_ids_str}'. Use comma-separated integers like '1,2' or '0'")
    
    # Set CUDA_VISIBLE_DEVICES to specified GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
        
        if not gpu_devices:
            available_devices = [str(d) for d in devices]
            raise RuntimeError(f"JAX cannot see any CUDA/GPU devices. JAX sees: {available_devices}")
        
        print(f"✓ JAX detected {len(gpu_devices)} GPU device(s): {[str(d) for d in gpu_devices]}")
        return gpu_ids
        
    except ImportError:
        raise RuntimeError("JAX not available. Install JAX with CUDA support.")
    except Exception as e:
        raise RuntimeError(f"JAX CUDA initialization failed: {e}")


@ray.remote(num_cpus=1, num_gpus=1.0, memory=4*1024*1024*1024) # currently one actor per gpu
class QarrayWorkerActor:
    """Ray Actor that holds a single QarrayBaseClass instance for one GPU"""
    
    def __init__(self, gpu_id: int, config_dict: dict, ray_config_dict: dict):
        import os
        import sys
        
        self.worker_pid = os.getpid()
        self.gpu_id = gpu_id
        self.samples_generated = 0
        self.ray_config = ray_config_dict
        
        # Add paths in actor - need to add src directory to find swarm package
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # Set memory settings for this actor from config
        env_config = ray_config_dict['ray']['environment']
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = env_config['XLA_PYTHON_CLIENT_MEM_FRACTION']
        os.environ['XLA_FLAGS'] = env_config['XLA_FLAGS']
        os.environ['TF_GPU_ALLOCATOR'] = env_config['TF_GPU_ALLOCATOR']
                
        # Reconstruct config
        self.config = GenerationConfig(**config_dict)
        
        try:
            from swarm.environment.qarray_base_class import QarrayBaseClass
            
            # Create single QarrayBaseClass instance that will be reused
            self.qarray_class = QarrayBaseClass
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"QarrayWorkerActor {self.worker_pid}: Failed to initialize: {e}")
            raise
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single sample using the persistent QarrayBaseClass instance"""
        try:
            np.random.seed(self.config.seed_base + sample_id)

            obs_voltage_size = np.random.uniform(self.config.min_obs_voltage_size, self.config.max_obs_voltage_size)

            use_barriers = self.config.use_barriers
            
            qarray = self.qarray_class(
                num_dots=self.config.num_dots,
                use_barriers=use_barriers,
                config_path=self.config.config_path,
                obs_voltage_min=-obs_voltage_size,
                obs_voltage_max=obs_voltage_size,
                obs_image_size=self.config.resolution
            )
            
            # Get ground truth voltages
            gt_voltages, vb_optimal, _ = qarray.calculate_ground_truth()
            
            # Add random offset to ground truth for observation
            rng = np.random.default_rng(self.config.seed_base + sample_id)
            voltage_offset = rng.uniform(
                self.config.voltage_offset_min, 
                self.config.voltage_offset_max, 
                size=len(gt_voltages)
            )
            gate_voltages = gt_voltages + voltage_offset

            if use_barriers:
                barrier_offset = rng.uniform(
                    self.config.barrier_offset_min,
                    self.config.barrier_offset_max,
                    size=len(vb_optimal)
                )

                barrier_voltages = vb_optimal + barrier_offset
            
            else:
                barrier_voltages = np.array([0.0] * (self.config.num_dots - 1))
            
            # Generate observation
            obs = qarray._get_obs(gate_voltages, barrier_voltages)
            
            # Extract Cgd matrix
            cgd_matrix = qarray.model.Cgd.copy()
            
            self.samples_generated += 1
            
            return {
                'sample_id': sample_id,
                'image': obs['image'].astype(np.float32),
                'cgd_matrix': cgd_matrix.astype(np.float32),
                'ground_truth_voltages': gt_voltages.astype(np.float32),
                'gate_voltages': gate_voltages.astype(np.float32),
                'success': True
            }
            
        except Exception as e:
            return {
                'sample_id': sample_id,
                'worker_pid': self.worker_pid,
                'gpu_id': self.gpu_id,
                'success': False,
                'error': str(e)
            }
    
    def generate_batch(self, sample_ids: List[int]) -> List[Dict[str, Any]]:
        """Generate multiple samples using the same QarrayBaseClass instance"""
        results = []
        for sample_id in sample_ids:
            result = self.generate_sample(sample_id)
            results.append(result)
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get actor status"""
        return {
            'worker_pid': self.worker_pid,
            'gpu_id': self.gpu_id,
            'initialized': self.initialized,
            'samples_generated': self.samples_generated,
            'init_error': getattr(self, 'init_error', None)
        }

def save_batch(batch_id: int, batch_samples: List[Dict[str, Any]], output_dir: Path) -> bool:
    """Save a batch of samples to disk"""
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
                'sample_id': s['sample_id'],
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
    """Create necessary output directories"""
    directories = ['images', 'cgd_matrices', 'ground_truth', 'metadata']
    for dir_name in directories:
        (output_dir / dir_name).mkdir(parents=True, exist_ok=True)



def run_test_mode(config: GenerationConfig, ray_config_path: str = None) -> None:
    """
    Run test mode: generate 20 random samples with visualization but no file saving.
    
    Args:
        config: Generation configuration (most parameters ignored in test mode)
    """
    print("Running test mode: generating 16 random samples with visualization...")
    
    # Load Ray configuration
    ray_config = load_ray_config(ray_config_path)
    test_config = ray_config['ray']['test_mode']
    
    # Set matplotlib backend for file output
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    
    # Enforce CUDA and get GPU list
    gpu_list = enforce_cuda_availability(config.gpu_ids)
    num_gpus = len(gpu_list)
    print(f"Using {num_gpus} GPU(s) for test mode: {gpu_list}")
    
    test_samples = []
    
    try:
        # Initialize Ray with settings from config
        ray.init(
            num_cpus=num_gpus + 1,
            num_gpus=num_gpus,
            object_store_memory=test_config['object_store_memory_gb']*1024*1024*1024,
            include_dashboard=test_config['include_dashboard'],
            _system_config=test_config['system_config']
        )
        print(f"Ray initialized for test mode")
        
        # Convert config to dict for Ray serialization
        config_dict = {
            'total_samples': config.total_samples,
            'num_dots': config.num_dots,
            'use_barriers': config.use_barriers,
            'output_dir': config.output_dir,
            'config_path': config.config_path,
            'gpu_ids': config.gpu_ids,
            'batch_size': config.batch_size,
            'voltage_offset_min': config.voltage_offset_min,
            'voltage_offset_max': config.voltage_offset_max,
            'barrier_offset_min': config.barrier_offset_min,
            'barrier_offset_max': config.barrier_offset_max,
            'seed_base': config.seed_base,
            'min_obs_voltage_size': config.min_obs_voltage_size,
            'max_obs_voltage_size': config.max_obs_voltage_size
        }
        
        # Create one actor (use first GPU for test mode)
        print(f"Creating test actor for GPU {gpu_list[0]}...")
        actor = QarrayWorkerActor.remote(gpu_list[0], config_dict, ray_config)
        
        # Wait for actor to initialize
        init_timeout = test_config['actor_timeouts']['actor_initialization']
        status = ray.get(actor.get_status.remote(), timeout=init_timeout)
        if not status['initialized']:
            raise RuntimeError(f"Test actor failed to initialize: {status.get('init_error', 'Unknown error')}")
        
        print(f"Test actor initialized: PID {status['worker_pid']}, GPU {status['gpu_id']}")
        
        # Generate 16 test samples
        print("Generating test samples...")
        
        sample_timeout = test_config['actor_timeouts']['sample_generation']
        for i in range(16):    
            try:
                # Use completely random seeds for each sample
                sample_id = np.random.randint(0, 1000000)
                sample = ray.get(actor.generate_sample.remote(sample_id), timeout=sample_timeout)
                
                if sample.get('success', False):
                    test_samples.append(sample)
                else:
                    print(f"✗ Sample {i+1} (Error: {sample.get('error', 'Unknown')})")
                    
            except Exception as e:
                print(f"✗ Sample {i+1} (Exception: {e})")
        
    except Exception as e:
        print(f"Ray Actor test mode failed: {e}")
        raise
    finally:
        # Always cleanup Ray resources
        try:
            ray.shutdown()
            print("Ray shutdown completed")
        except Exception as e:
            print(f"Ray shutdown had issues: {e}")
    
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
            axes[idx].set_title(f'Sample {idx+1}', fontsize=10)
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
    """Save dataset metadata and generation statistics"""
    num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size
    
    metadata = {
        'generation_config': {
            'total_samples': config.total_samples,
            'batch_size': config.batch_size,
            'num_dots': config.num_dots,
            'gpu_ids': config.gpu_ids,
            'voltage_offset_min': config.voltage_offset_min,
            'voltage_offset_max': config.voltage_offset_max,
            'seed_base': config.seed_base
        },
        'generation_stats': generation_stats,
        'data_structure': {
            'total_batches': num_batches,
            'images': f'Batched charge sensor images, shape per batch: (batch_size, 128, 128, {config.num_dots-1})',
            'cgd_matrices': f'Batched gate-to-dot capacitance matrices, shape per batch: (batch_size, {config.num_dots}, {config.num_dots+1})',
            'ground_truth': 'List of ground truth voltages and observation voltages per batch'
        },
        'ray_actor_info': {
            'approach': 'One QarrayBaseClass instance per GPU using Ray Actors',
            'gpus_used': config.gpu_ids,
            'memory_per_actor': '4GB with 80% GPU memory fraction'
        }
    }
    
    metadata_path = output_dir / 'metadata' / 'dataset_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def generate_dataset(config: GenerationConfig, ray_config_path: str = None) -> None:
    """Generate the complete dataset using Ray Actors for reliable GPU usage"""
    output_dir = Path(config.output_dir)
    
    # Load Ray configuration
    ray_config = load_ray_config(ray_config_path)
    prod_config = ray_config['ray']['production']
    
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
    logger.info(f"Starting Ray Actor dataset generation: {config.total_samples} samples")
    
    # Create output directories
    create_output_directories(output_dir)
    
    # Enforce CUDA and get GPU list
    gpu_list = enforce_cuda_availability(config.gpu_ids)
    num_gpus = len(gpu_list)
    logger.info(f"Using {num_gpus} GPU(s): {gpu_list}")
    
    # Initialize Ray with settings from config
    try:
        ray.init(
            num_cpus=num_gpus + 2,  # Extra CPUs for coordination
            num_gpus=num_gpus,      # Exact GPU count
            object_store_memory=prod_config['object_store_memory_gb']*1024*1024*1024,
            include_dashboard=prod_config['include_dashboard'],
            _system_config=prod_config['system_config'],
            #ignore_reinit_error=True
        )
        logger.info(f"Ray initialized with {num_gpus} GPUs and {prod_config['object_store_memory_gb']}GB object store")
        
        # Convert config to dict for Ray serialization
        config_dict = {
            'total_samples': config.total_samples,
            'num_dots': config.num_dots,
            'use_barriers': config.use_barriers,
            'output_dir': config.output_dir,
            'config_path': config.config_path,
            'gpu_ids': config.gpu_ids,
            'batch_size': config.batch_size,
            'voltage_offset_min': config.voltage_offset_min,
            'voltage_offset_max': config.voltage_offset_max,
            'barrier_offset_min': config.barrier_offset_min,
            'barrier_offset_max': config.barrier_offset_max,
            'seed_base': config.seed_base,
            'min_obs_voltage_size': config.min_obs_voltage_size,
            'max_obs_voltage_size': config.max_obs_voltage_size
        }
        
        # Create one QarrayWorkerActor per GPU
        logger.info(f"Creating {num_gpus} QarrayWorkerActors...")
        actors = []
        
        for i, gpu_id in enumerate(gpu_list):
            print(f"  Creating actor {i} for GPU {gpu_id}...")
            actor = QarrayWorkerActor.remote(gpu_id, config_dict, ray_config)
            actors.append((i, actor))
        
        # Wait for all actors to initialize
        logger.info("Waiting for actors to initialize...")
        init_timeout = prod_config['actor_timeouts']['actor_initialization']
        status_futures = [actor.get_status.remote() for _, actor in actors]
        statuses = ray.get(status_futures, timeout=init_timeout)
        
        initialized_actors = []
        for i, status in enumerate(statuses):
            if status['initialized']:
                logger.info(f"Actor {i}: PID {status['worker_pid']}, GPU {status['gpu_id']}")
                initialized_actors.append(actors[i])
            else:
                logger.error(f"Actor {i}: Failed - {status.get('init_error', 'Unknown error')}")
        
        if not initialized_actors:
            raise RuntimeError("No actors initialized successfully")
        
        num_actors = len(initialized_actors)
        logger.info(f"Successfully initialized {num_actors} actors")
        
        # Generate samples by distributing work across actors
        start_time = time.time()
        successful_samples = 0
        failed_samples = 0
        saved_batches = 0
        current_batch_samples = []
        current_batch_id = 0
        
        # Calculate samples per actor for load balancing
        samples_per_actor = config.total_samples // num_actors
        extra_samples = config.total_samples % num_actors
        
        logger.info(f"Distributing {config.total_samples} samples across {num_actors} actors")
        logger.info(f"Base samples per actor: {samples_per_actor}, extra samples: {extra_samples}")
        
        # Create sample assignment for each actor
        actor_assignments = []
        current_sample_id = 0
        
        for i, (actor_id, actor) in enumerate(initialized_actors):
            # Give extra samples to first few actors
            actor_samples = samples_per_actor + (1 if i < extra_samples else 0)
            sample_ids = list(range(current_sample_id, current_sample_id + actor_samples))
            actor_assignments.append((actor_id, actor, sample_ids))
            current_sample_id += actor_samples
            
            logger.info(f"Actor {actor_id}: assigned {len(sample_ids)} samples (IDs {sample_ids[0]}-{sample_ids[-1]})")
        
        # Process samples in chunks to manage memory
        chunk_size = ray_config['ray']['processing']['chunk_size']
        num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size
        
        # Initialize progress bar
        pbar = tqdm(
            total=config.total_samples,
            desc="Generating samples",
            unit="samples"
        )
        
        # Process each actor's assignment in chunks
        remaining_assignments = [(actor_id, actor, sample_ids) for actor_id, actor, sample_ids in actor_assignments]
        
        while remaining_assignments:
            # Submit next chunk for each actor
            chunk_futures = []
            new_remaining = []
            
            for actor_id, actor, sample_ids in remaining_assignments:
                if sample_ids:
                    # Take next chunk
                    chunk_ids = sample_ids[:chunk_size]
                    remaining_ids = sample_ids[chunk_size:]
                    
                    # Submit chunk
                    future = actor.generate_batch.remote(chunk_ids)
                    chunk_futures.append((actor_id, future, len(chunk_ids)))
                    
                    # Keep remaining samples for next iteration
                    if remaining_ids:
                        new_remaining.append((actor_id, actor, remaining_ids))
            
            remaining_assignments = new_remaining
            
            # Process chunk results
            chunk_timeout = prod_config['actor_timeouts']['chunk_processing']
            for actor_id, future, chunk_size_actual in chunk_futures:
                try:
                    chunk_results = ray.get(future, timeout=chunk_timeout)
                    
                    # Add to current batch
                    current_batch_samples.extend(chunk_results)
                    
                    # Count successes/failures
                    chunk_successes = sum(1 for r in chunk_results if r.get('success', False))
                    chunk_failures = len(chunk_results) - chunk_successes
                    
                    successful_samples += chunk_successes
                    failed_samples += chunk_failures
                    
                    logger.info(f"Actor {actor_id}: {chunk_successes}/{len(chunk_results)} samples successful")
                    
                    # Update progress
                    pbar.update(len(chunk_results))
                    pbar.set_postfix_str(f"Success: {successful_samples}, Failed: {failed_samples}, Batches: {saved_batches}/{num_batches}")
                    
                except Exception as e:
                    logger.error(f"Actor {actor_id} chunk failed: {e}")
                    failed_samples += chunk_size_actual
                    pbar.update(chunk_size_actual)
            
            # Save batch when full
            if len(current_batch_samples) >= config.batch_size:
                if save_batch(current_batch_id, current_batch_samples, output_dir):
                    saved_batches += 1
                    pbar.set_description(f"Generating samples (Saved batch {current_batch_id})")
                
                current_batch_samples = []
                current_batch_id += 1
        
        # Save final partial batch
        if current_batch_samples:
            if save_batch(current_batch_id, current_batch_samples, output_dir):
                saved_batches += 1
        
        pbar.close()
        
    except Exception as e:
        logger.error(f"Ray Actor processing failed: {e}")
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
        'samples_per_second': successful_samples / total_time if total_time > 0 else 0,
        'actors_used': num_actors,
        'gpus_used': num_gpus
    }
    
    logger.info(f"Dataset generation completed!")
    logger.info(f"Successful samples: {successful_samples}")
    logger.info(f"Failed samples: {failed_samples}")
    logger.info(f"Success rate: {successful_samples/(successful_samples+failed_samples)*100:.1f}%")
    logger.info(f"Saved batches: {saved_batches}/{num_batches}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average rate: {generation_stats['samples_per_second']:.1f} samples/second")
    
    # Save metadata
    save_metadata(config, output_dir, generation_stats)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate quantum device dataset using Ray Actors')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--num_dots', type=int, default=4,
                       help='Number of quantum dots')
    parser.add_argument('--use_barriers', action='store_true',
                       help='Whether to use barrier gates in the model')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch file')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    parser.add_argument('--gpu_ids', type=str, default="7",
                       help='Comma-separated list of GPU IDs to use (e.g., "6,7" or "0")')
    parser.add_argument('--ray_config', type=str, default="ray_config.yaml",
                       help='Path to Ray configuration YAML file')
    parser.add_argument('--test', action='store_true',
                        help='Whether to run a sample run with visualisation')
    
    args = parser.parse_args()

    if args.test and args.num_dots != 4:
        args.num_dots = 4
        print("Warning: test mode should run with 4 dots, setting num_dots to 4")
    
    print(f"\nUsing barriers: {args.use_barriers}\n")
    
    # Create configuration
    config = GenerationConfig(
        total_samples=args.total_samples,
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
        seed_base=args.seed
    )
    
    # Run dataset generation
    try:
        if args.test:
            run_test_mode(config, args.ray_config)
        else:
            generate_dataset(config, args.ray_config)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise

if __name__ == '__main__':
    main()