#!/usr/bin/env python3
"""
Symmetric Capacitance Dataset Generator using Ray Actors (One QarrayBaseClass per GPU)

Generates datasets with effective cross-couplings uniformly distributed from -0.7 to +0.7
by manipulating virtual gate matrices (VGMs). Uses Ray Actors to ensure one QarrayBaseClass
instance per GPU for memory safety and better GPU utilization.

Usage:
    python symmetric_capacitance_generator_ray.py --total_samples 150000 --gpu_ids "1,2,3,4,5,6,7" --output_dir ./dataset
    python symmetric_capacitance_generator_ray.py --test --gpu_ids "1" --num_dots 4  # Test mode
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import logging
from dataclasses import dataclass, field
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
    num_dots: int
    use_barriers: bool
    output_dir: str
    gpu_ids: str
    config_path: str = "qarray_config.yaml"
    batch_size: int = 1000
    seed_base: int = 42
    # Symmetric capacitance range for nearest neighbors
    coupling_min: float = -0.7
    coupling_max: float = 0.7
    # Symmetric capacitance range for second-nearest neighbors (next-nearest)
    nnn_coupling_min: float = -0.3
    nnn_coupling_max: float = 0.3
    # These will be loaded from env_config.yaml
    env_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Load env_config.yaml for parameters like radial_noise, voltage offsets, etc."""
        if not self.env_config:
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
        return self.env_config['simulator']['full_barrier_range_width']['max'] / 2

    def get_radial_noise_config(self) -> dict:
        """Get radial_noise config from env_config.yaml."""
        return self.env_config['simulator'].get('radial_noise')


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

        print(f"JAX detected {len(gpu_devices)} GPU device(s): {[str(d) for d in gpu_devices]}")
        return gpu_ids

    except ImportError:
        raise RuntimeError("JAX not available. Install JAX with CUDA support.")
    except Exception as e:
        raise RuntimeError(f"JAX CUDA initialization failed: {e}")


@ray.remote(num_cpus=1, num_gpus=1.0, memory=4*1024*1024*1024)
class SymmetricCapacitanceWorkerActor:
    """Ray Actor that generates samples with symmetric effective capacitances via VGM manipulation"""

    def __init__(self, gpu_id: int, config_dict: dict, ray_config_dict: dict):
        import os
        import sys

        self.worker_pid = os.getpid()
        self.gpu_id = gpu_id
        self.samples_generated = 0
        self.ray_config = ray_config_dict

        # Add paths in actor
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Set memory settings for this actor from config
        env_config = ray_config_dict['ray']['environment']
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = env_config['XLA_PYTHON_CLIENT_MEM_FRACTION']
        os.environ['XLA_FLAGS'] = env_config['XLA_FLAGS']
        os.environ['TF_GPU_ALLOCATOR'] = env_config['TF_GPU_ALLOCATOR']

        # Store config dict for reconstruction
        self.config_dict = config_dict

        try:
            from swarm.environment.qarray_base_class import QarrayBaseClass
            self.qarray_class = QarrayBaseClass
            self.initialized = True

        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"SymmetricCapacitanceWorkerActor {self.worker_pid}: Failed to initialize: {e}")
            raise

    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single sample with symmetric effective capacitances via VGM manipulation"""
        try:
            import numpy as np

            config = self.config_dict
            np.random.seed(config['seed_base'] + sample_id)
            rng = np.random.default_rng(config['seed_base'] + sample_id)

            obs_voltage_size = np.random.uniform(
                config['min_obs_voltage_size'],
                config['max_obs_voltage_size']
            )

            # Create QarrayBaseClass (physical Cgd sampled from qarray_config as usual)
            qarray = self.qarray_class(
                num_dots=config['num_dots'],
                use_barriers=config['use_barriers'],
                config_path=config['config_path'],
                obs_voltage_min=-obs_voltage_size,
                obs_voltage_max=obs_voltage_size,
                obs_image_size=config['obs_image_size'],
                radial_noise_config=config.get('radial_noise_config'),
            )

            # --- Sample target effective couplings (symmetric around 0) ---
            # Note: qarray convention has effective_coupling = -target_matrix (off-diagonal)
            target_matrix = np.eye(config['num_dots'])
            num_dots = config['num_dots']

            # Nearest neighbor couplings (distance 1) - SYMMETRIC
            # Cgd[c, c+1] = Cgd[c+1, c] for each channel c
            nn_couplings = {}  # (dot, gate) -> coupling
            for c in range(num_dots - 1):
                coupling = rng.uniform(config['coupling_min'], config['coupling_max'])
                nn_couplings[(c, c + 1)] = coupling
                nn_couplings[(c + 1, c)] = coupling  # Same value (symmetric)
                target_matrix[c, c + 1] = -coupling
                target_matrix[c + 1, c] = -coupling

            # Second-nearest neighbor couplings (distance 2) - SYMMETRIC with edge case handling
            # For each pair (i, i+2), sample one value and set both positions
            # Edge cases: positions where gate index is out of bounds stay at 0
            nnn_couplings = {}  # (dot, gate) -> coupling
            for i in range(num_dots - 2):
                coupling = rng.uniform(config['nnn_coupling_min'], config['nnn_coupling_max'])
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
            initial_gate_voltages = np.zeros(config['num_dots'])
            initial_barrier_voltages = np.zeros(config['num_dots'] - 1)
            if config['use_barriers']:
                gt_voltages, vb_optimal, _ = qarray.calculate_ground_truth(initial_gate_voltages, initial_barrier_voltages)
            else:
                gt_voltages, _, _ = qarray.calculate_ground_truth(initial_gate_voltages, initial_barrier_voltages)

            # Add random offset to ground truth for observation
            # Use ±35V to cover full voltage space and exceed radial_noise full_noise_distance (30-40V)
            # This ensures a mix of clean honeycombs (offset < 30V) and white noise (offset > 30-40V)
            voltage_offset = rng.uniform(-35, 35, size=len(gt_voltages))
            gate_voltages = gt_voltages + voltage_offset

            if config['use_barriers']:
                barrier_offset = rng.uniform(
                    -config['barrier_offset_range'],
                    config['barrier_offset_range'],
                    size=len(vb_optimal)
                )
                barrier_voltages = vb_optimal + barrier_offset
            else:
                barrier_voltages = np.array([0.0] * (config['num_dots'] - 1))

            # Set ground truth for radial noise
            qarray.gate_ground_truth = gt_voltages

            # Generate observation
            obs = qarray._get_obs(gate_voltages, barrier_voltages)

            # Build cgd_matrix format for dataloader compatibility
            cgd_matrix_format = np.eye(num_dots, num_dots + 1, dtype=np.float32)

            # Store NN couplings (asymmetric - each position stored separately)
            for (dot, gate), coupling in nn_couplings.items():
                cgd_matrix_format[dot, gate] = coupling

            # Store NNN couplings (edge cases handled - missing entries stay 0)
            for (dot, gate), coupling in nnn_couplings.items():
                cgd_matrix_format[dot, gate] = coupling

            self.samples_generated += 1

            return {
                'sample_id': sample_id,
                'image': obs['image'].astype(np.float32),
                'cgd_matrix': cgd_matrix_format,
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
        """Generate multiple samples"""
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
        successful_samples = [s for s in batch_samples if s.get('success', False)]

        if not successful_samples:
            logging.warning(f"No successful samples in batch {batch_id}")
            return False

        batch_size = len(successful_samples)
        logging.info(f"Saving batch {batch_id} with {batch_size} samples")

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
    """Run test mode: generate samples with visualization to verify symmetric couplings."""
    print("Running test mode: generating 16 random samples with visualization...")
    print(f"NN coupling range: [{config.coupling_min}, {config.coupling_max}]")
    print(f"NNN coupling range: [{config.nnn_coupling_min}, {config.nnn_coupling_max}]")

    ray_config = load_ray_config(ray_config_path)
    test_config = ray_config['ray']['test_mode']

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gpu_list = enforce_cuda_availability(config.gpu_ids)
    num_gpus = len(gpu_list)
    print(f"Using {num_gpus} GPU(s) for test mode: {gpu_list}")

    test_samples = []

    try:
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
            'seed_base': config.seed_base,
            'coupling_min': config.coupling_min,
            'coupling_max': config.coupling_max,
            'nnn_coupling_min': config.nnn_coupling_min,
            'nnn_coupling_max': config.nnn_coupling_max,
            'voltage_offset_min': config.voltage_offset_min,
            'voltage_offset_max': config.voltage_offset_max,
            'min_obs_voltage_size': config.min_obs_voltage_size,
            'max_obs_voltage_size': config.max_obs_voltage_size,
            'obs_image_size': config.obs_image_size,
            'barrier_offset_range': config.barrier_offset_range,
            'radial_noise_config': config.get_radial_noise_config(),
        }

        print(f"Creating test actor for GPU {gpu_list[0]}...")
        actor = SymmetricCapacitanceWorkerActor.remote(gpu_list[0], config_dict, ray_config)

        init_timeout = test_config['actor_timeouts']['actor_initialization']
        status = ray.get(actor.get_status.remote(), timeout=init_timeout)
        if not status['initialized']:
            raise RuntimeError(f"Test actor failed to initialize: {status.get('init_error', 'Unknown error')}")

        print(f"Test actor initialized: PID {status['worker_pid']}, GPU {status['gpu_id']}")

        print("Generating test samples...")
        sample_timeout = test_config['actor_timeouts']['sample_generation']
        for i in range(16):
            try:
                sample_id = np.random.randint(0, 1000000)
                sample = ray.get(actor.generate_sample.remote(sample_id), timeout=sample_timeout)

                if sample.get('success', False):
                    test_samples.append(sample)
                else:
                    print(f"Sample {i+1} failed: {sample.get('error', 'Unknown')}")

            except Exception as e:
                print(f"Sample {i+1} exception: {e}")

    except Exception as e:
        print(f"Ray Actor test mode failed: {e}")
        raise
    finally:
        try:
            ray.shutdown()
            print("Ray shutdown completed")
        except Exception as e:
            print(f"Ray shutdown had issues: {e}")

    if not test_samples:
        print("No successful samples generated. Cannot create visualization.")
        return

    print(f"\nSuccessfully generated {len(test_samples)} samples")

    # Extract target couplings for histogram
    nn_couplings_all = []
    nnn_couplings_all = []
    for sample in test_samples:
        cgd = sample['cgd_matrix']
        num_dots = cgd.shape[0]
        # Nearest neighbor couplings (both directions)
        for c in range(num_dots - 1):
            nn_couplings_all.append(cgd[c, c + 1])
            nn_couplings_all.append(cgd[c + 1, c])
        # Second-nearest neighbor couplings (all valid positions)
        for c in range(num_dots - 1):
            # NNN for left dot (c) to gate c+2
            if c + 2 < num_dots:
                nnn_couplings_all.append(cgd[c, c + 2])
            # NNN for right dot (c+1) to gate c-1
            if c - 1 >= 0:
                nnn_couplings_all.append(cgd[c + 1, c - 1])

    def plot_images(channel):
        n_samples_to_plot = min(16, len(test_samples))
        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        axes = axes.flatten()

        for idx in range(n_samples_to_plot):
            sample = test_samples[idx]
            image = sample['image']
            plot_image = image[:, :, channel]
            cgd = sample['cgd_matrix']
            num_dots = cgd.shape[0]

            # Get NN coupling for this channel (symmetric: cgd[c,c+1] = cgd[c+1,c])
            nn_coupling = cgd[channel, channel + 1]

            # Get NNN couplings for this channel with edge case handling
            # NNN for left dot (channel) to gate channel+2
            if channel + 2 < num_dots:
                nnn_right = cgd[channel, channel + 2]
                nnn_right_str = f'{nnn_right:.2f}'
            else:
                nnn_right_str = '0 (edge)'

            # NNN for right dot (channel+1) to gate channel-1
            if channel - 1 >= 0:
                nnn_left = cgd[channel + 1, channel - 1]
                nnn_left_str = f'{nnn_left:.2f}'
            else:
                nnn_left_str = '0 (edge)'

            im = axes[idx].imshow(plot_image, cmap='viridis', aspect='equal')
            title = (f'NN({channel},{channel+1}): {nn_coupling:.2f}\n'
                    f'NNN: ({channel},{channel+2})={nnn_right_str}, ({channel+1},{channel-1})={nnn_left_str}')
            axes[idx].set_title(title, fontsize=8)
            axes[idx].set_xlabel('Gate Voltage 1')
            axes[idx].set_ylabel('Gate Voltage 2')
            plt.colorbar(im, ax=axes[idx], shrink=0.8)

        for idx in range(n_samples_to_plot, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Symmetric Capacitance Test (Ray): Channel {channel}\n'
                    f'(NN range: [{config.coupling_min}, {config.coupling_max}], '
                    f'NNN range: [{config.nnn_coupling_min}, {config.nnn_coupling_max}])', fontsize=12)
        plt.tight_layout()

        output_path = f'symmetric_ray_test_channel_{channel}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.close()

    num_channels = min(3, config.num_dots - 1)
    for channel in range(num_channels):
        plot_images(channel)

    # Plot histogram of couplings (NN and NNN side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NN couplings histogram
    axes[0].hist(nn_couplings_all, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Target Effective Coupling')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Nearest Neighbor Couplings (asymmetric)\n(Expected: Uniform [{config.coupling_min}, {config.coupling_max}])')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[0].legend()

    # NNN couplings histogram
    axes[1].hist(nnn_couplings_all, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Target Effective Coupling')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Second-Nearest Neighbor Couplings\n(Expected: Uniform [{config.nnn_coupling_min}, {config.nnn_coupling_max}])')
    axes[1].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('symmetric_ray_coupling_distribution.png', dpi=150)
    print("Saved coupling distribution to: symmetric_ray_coupling_distribution.png")
    plt.close()

    # Print statistics
    print("\nNearest Neighbor (NN) Statistics:")
    print(f"  Range: {min(nn_couplings_all):.4f} to {max(nn_couplings_all):.4f}")
    print(f"  Mean: {np.mean(nn_couplings_all):.4f} (should be ~0)")
    print(f"  Std: {np.std(nn_couplings_all):.4f}")

    print("\nSecond-Nearest Neighbor (NNN) Statistics:")
    print(f"  Range: {min(nnn_couplings_all):.4f} to {max(nnn_couplings_all):.4f}")
    print(f"  Mean: {np.mean(nnn_couplings_all):.4f} (should be ~0)")
    print(f"  Std: {np.std(nnn_couplings_all):.4f}")


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
        'ray_actor_info': {
            'approach': 'One SymmetricCapacitanceWorkerActor per GPU using Ray',
            'gpus_used': config.gpu_ids,
            'memory_per_actor': '4GB with configured GPU memory fraction'
        },
        'notes': {
            'nn_couplings': f'Nearest neighbor couplings (positions [i,i+1]) sampled from [{config.coupling_min}, {config.coupling_max}]',
            'nnn_couplings': f'Second-nearest neighbor couplings (positions [i,i+2]) sampled from [{config.nnn_coupling_min}, {config.nnn_coupling_max}]',
            'vgm_manipulation': 'Uses VGM manipulation to achieve target effective couplings while keeping physical Cgd positive',
            'compatibility': 'Output format is compatible with standard CapacitanceDataset dataloader'
        }
    }

    metadata_path = output_dir / 'metadata' / 'dataset_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_dataset(config: GenerationConfig, ray_config_path: str = None) -> None:
    """Generate the complete dataset using Ray Actors for GPU parallelization"""
    output_dir = Path(config.output_dir)

    ray_config = load_ray_config(ray_config_path)
    prod_config = ray_config['ray']['production']

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
    logger.info(f"Starting Ray Actor symmetric capacitance dataset generation: {config.total_samples} samples")
    logger.info(f"Coupling range: [{config.coupling_min}, {config.coupling_max}]")

    create_output_directories(output_dir)

    gpu_list = enforce_cuda_availability(config.gpu_ids)
    num_gpus = len(gpu_list)
    logger.info(f"Using {num_gpus} GPU(s): {gpu_list}")

    try:
        ray.init(
            num_cpus=num_gpus + 2,
            num_gpus=num_gpus,
            object_store_memory=prod_config['object_store_memory_gb']*1024*1024*1024,
            include_dashboard=prod_config['include_dashboard'],
            _system_config=prod_config['system_config'],
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
            'seed_base': config.seed_base,
            'coupling_min': config.coupling_min,
            'coupling_max': config.coupling_max,
            'nnn_coupling_min': config.nnn_coupling_min,
            'nnn_coupling_max': config.nnn_coupling_max,
            'voltage_offset_min': config.voltage_offset_min,
            'voltage_offset_max': config.voltage_offset_max,
            'min_obs_voltage_size': config.min_obs_voltage_size,
            'max_obs_voltage_size': config.max_obs_voltage_size,
            'obs_image_size': config.obs_image_size,
            'barrier_offset_range': config.barrier_offset_range,
            'radial_noise_config': config.get_radial_noise_config(),
        }

        # Create one actor per GPU
        logger.info(f"Creating {num_gpus} SymmetricCapacitanceWorkerActors...")
        actors = []

        for i, gpu_id in enumerate(gpu_list):
            print(f"  Creating actor {i} for GPU {gpu_id}...")
            actor = SymmetricCapacitanceWorkerActor.remote(gpu_id, config_dict, ray_config)
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

        # Generate samples
        start_time = time.time()
        successful_samples = 0
        failed_samples = 0
        saved_batches = 0
        current_batch_samples = []
        current_batch_id = 0

        samples_per_actor = config.total_samples // num_actors
        extra_samples = config.total_samples % num_actors

        logger.info(f"Distributing {config.total_samples} samples across {num_actors} actors")

        # Create sample assignment for each actor
        actor_assignments = []
        current_sample_id = 0

        for i, (actor_id, actor) in enumerate(initialized_actors):
            actor_samples = samples_per_actor + (1 if i < extra_samples else 0)
            sample_ids = list(range(current_sample_id, current_sample_id + actor_samples))
            actor_assignments.append((actor_id, actor, sample_ids))
            current_sample_id += actor_samples
            logger.info(f"Actor {actor_id}: assigned {len(sample_ids)} samples (IDs {sample_ids[0]}-{sample_ids[-1]})")

        chunk_size = ray_config['ray']['processing']['chunk_size']
        num_batches = (config.total_samples + config.batch_size - 1) // config.batch_size

        pbar = tqdm(
            total=config.total_samples,
            desc="Generating samples",
            unit="samples"
        )

        remaining_assignments = [(actor_id, actor, sample_ids) for actor_id, actor, sample_ids in actor_assignments]

        while remaining_assignments:
            chunk_futures = []
            new_remaining = []

            for actor_id, actor, sample_ids in remaining_assignments:
                if sample_ids:
                    chunk_ids = sample_ids[:chunk_size]
                    remaining_ids = sample_ids[chunk_size:]

                    future = actor.generate_batch.remote(chunk_ids)
                    chunk_futures.append((actor_id, future, len(chunk_ids)))

                    if remaining_ids:
                        new_remaining.append((actor_id, actor, remaining_ids))

            remaining_assignments = new_remaining

            chunk_timeout = prod_config['actor_timeouts']['chunk_processing']
            for actor_id, future, chunk_size_actual in chunk_futures:
                try:
                    chunk_results = ray.get(future, timeout=chunk_timeout)

                    current_batch_samples.extend(chunk_results)

                    chunk_successes = sum(1 for r in chunk_results if r.get('success', False))
                    chunk_failures = len(chunk_results) - chunk_successes

                    successful_samples += chunk_successes
                    failed_samples += chunk_failures

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
        try:
            ray.shutdown()
            logger.info("Ray shutdown completed")
        except Exception as e:
            logger.warning(f"Ray shutdown had issues: {e}")

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

    save_metadata(config, output_dir, generation_stats)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate symmetric capacitance dataset using Ray Actors')
    parser.add_argument('--total_samples', type=int, default=10000,
                       help='Total number of samples to generate')
    parser.add_argument('--num_dots', type=int, default=4,
                       help='Number of quantum dots')
    parser.add_argument('--use_barriers', action='store_true',
                       help='Whether to use barrier gates (required for VGM)')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of samples per batch file')
    parser.add_argument('--output_dir', type=str, default='./symmetric_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed for reproducibility')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5,6,7",
                       help='Comma-separated list of GPU IDs to use (e.g., "1,2,3,4,5,6,7")')
    parser.add_argument('--ray_config', type=str, default=None,
                       help='Path to Ray configuration YAML file')
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

    if args.test and args.num_dots != 4:
        args.num_dots = 4
        print("Warning: test mode should run with 4 dots, setting num_dots to 4")

    # Force use_barriers for VGM to work correctly
    if not args.use_barriers:
        print("Note: Enabling --use_barriers (required for VGM manipulation)")
        args.use_barriers = True

    print(f"\nSymmetric Capacitance Generator (Ray)")
    print(f"Using barriers: {args.use_barriers}")
    print(f"NN coupling range: [{args.coupling_min}, {args.coupling_max}]")
    print(f"NNN coupling range: [{args.nnn_coupling_min}, {args.nnn_coupling_max}]")
    print(f"GPU IDs: {args.gpu_ids}\n")

    config = GenerationConfig(
        total_samples=args.total_samples,
        num_dots=args.num_dots,
        use_barriers=args.use_barriers,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
        seed_base=args.seed,
        coupling_min=args.coupling_min,
        coupling_max=args.coupling_max,
        nnn_coupling_min=args.nnn_coupling_min,
        nnn_coupling_max=args.nnn_coupling_max,
    )

    ray_config_path = args.ray_config
    if ray_config_path is None:
        ray_config_path = os.path.join(os.path.dirname(__file__), "ray_config.yaml")

    try:
        if args.test:
            run_test_mode(config, ray_config_path)
        else:
            generate_dataset(config, ray_config_path)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main()
