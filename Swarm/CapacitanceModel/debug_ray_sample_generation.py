#!/usr/bin/env python3
"""
Comprehensive diagnostic script for Ray-based sample generation failures.

This script runs extensive tests to identify the root cause of sample generation
failures when using Ray with GPU acceleration.
"""

import os
import sys
import time
import psutil
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List
import ray
import numpy as np

# Add parent directory to path for imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
swarm_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

for path in [environment_dir, swarm_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

def get_gpu_memory_usage():
    """Get current GPU memory usage via nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    gpu_info.append({
                        'gpu_id': int(parts[0]),
                        'used_mb': int(parts[1]),
                        'total_mb': int(parts[2]), 
                        'free_mb': int(parts[3])
                    })
            return gpu_info
        else:
            return None
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None

def print_system_info():
    """Print comprehensive system information"""
    print("="*80)
    print("SYSTEM DIAGNOSTICS")
    print("="*80)
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available")
    
    # CPU info
    print(f"CPU cores: {psutil.cpu_count()} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    # GPU memory
    gpu_info = get_gpu_memory_usage()
    if gpu_info:
        print("\nGPU Memory Usage:")
        for gpu in gpu_info:
            used_pct = (gpu['used_mb'] / gpu['total_mb']) * 100
            print(f"  GPU {gpu['gpu_id']}: {gpu['used_mb']} MB / {gpu['total_mb']} MB ({used_pct:.1f}% used)")
    else:
        print("Could not get GPU memory info")
    
    # Environment variables
    print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'Not set')}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
    
    print("="*80)

def test_jax_cuda_single_process():
    """Test JAX CUDA functionality in single process"""
    print("\n" + "="*60)
    print("TEST 1: JAX CUDA Single Process")
    print("="*60)
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"JAX version: {jax.__version__}")
        
        # Get devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
        
        print(f"JAX devices: {[str(d) for d in devices]}")
        print(f"GPU devices: {[str(d) for d in gpu_devices]}")
        
        if not gpu_devices:
            print("❌ No GPU devices found")
            return False
        
        # Test basic operations on each GPU
        for gpu in gpu_devices:
            print(f"\nTesting operations on {gpu}...")
            try:
                with jax.default_device(gpu):
                    # Create test arrays
                    x = jnp.array([1.0, 2.0, 3.0])
                    y = jnp.array([4.0, 5.0, 6.0])
                    
                    # Basic operations
                    z = x + y
                    w = jnp.dot(x, y)
                    
                    print(f"  ✓ Basic ops successful: {x} + {y} = {z}, dot = {w}")
                    
                    # Matrix operations
                    A = jnp.ones((100, 100))
                    B = jnp.ones((100, 100))
                    C = jnp.dot(A, B)
                    
                    print(f"  ✓ Matrix ops successful: (100x100) @ (100x100) = shape {C.shape}")
                    
            except Exception as e:
                print(f"  ❌ GPU {gpu} failed: {e}")
                return False
        
        print("✅ JAX CUDA single process test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ JAX CUDA test failed: {e}")
        return False

def test_qarray_single_process():
    """Test QarrayBaseClass functionality in single process"""
    print("\n" + "="*60)
    print("TEST 2: QarrayBaseClass Single Process")
    print("="*60)
    
    try:
        from qarray_base_class import QarrayBaseClass
        
        print("✓ QarrayBaseClass import successful")
        
        # Test basic instantiation
        qarray = QarrayBaseClass(
            num_dots=4,
            config_path='qarray_config.yaml',
            obs_voltage_min=-1.0,
            obs_voltage_max=1.0,
            obs_image_size=128
        )
        print("✓ QarrayBaseClass instantiation successful")
        
        # Test ground truth calculation
        gt_voltages = qarray.calculate_ground_truth()
        print(f"✓ Ground truth calculation successful: shape {gt_voltages.shape}")
        
        # Test observation generation
        barrier_voltages = [0.0] * (4 - 1)
        obs = qarray._get_obs(gt_voltages, barrier_voltages)
        print(f"✓ Observation generation successful: image shape {obs['image'].shape}")
        
        # Test Cgd matrix access
        cgd_matrix = qarray.model.Cgd.copy()
        print(f"✓ Cgd matrix access successful: shape {cgd_matrix.shape}")
        
        print("✅ QarrayBaseClass single process test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ QarrayBaseClass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@ray.remote(num_cpus=1, num_gpus=0.5, memory=1*1024*1024*1024)  # 1GB, half GPU
def test_ray_worker_basic():
    """Test basic Ray worker functionality"""
    import os
    import sys
    
    worker_pid = os.getpid()
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    return {
        'worker_pid': worker_pid,
        'cuda_visible_devices': cuda_devices,
        'sys_path_count': len(sys.path),
        'success': True
    }

@ray.remote(num_cpus=1, num_gpus=0.5, memory=1*1024*1024*1024)
def test_ray_worker_jax():
    """Test JAX functionality in Ray worker"""
    import os
    
    worker_pid = os.getpid()
    
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]
        
        # Test basic computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        
        return {
            'worker_pid': worker_pid,
            'jax_version': jax.__version__,
            'devices': [str(d) for d in devices],
            'gpu_devices': [str(d) for d in gpu_devices],
            'test_computation': float(y),
            'success': True
        }
    except Exception as e:
        return {
            'worker_pid': worker_pid,
            'success': False,
            'error': str(e)
        }

@ray.remote(num_cpus=1, num_gpus=0.5, memory=1*1024*1024*1024)
def test_ray_worker_qarray():
    """Test QarrayBaseClass functionality in Ray worker"""
    import os
    import sys
    
    worker_pid = os.getpid()
    
    # Add paths (required in Ray worker)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
    if environment_dir not in sys.path:
        sys.path.insert(0, environment_dir)
    
    try:
        from qarray_base_class import QarrayBaseClass
        
        qarray = QarrayBaseClass(
            num_dots=4,
            config_path='qarray_config.yaml',
            obs_voltage_min=-1.0,
            obs_voltage_max=1.0,
            obs_image_size=128
        )
        
        # Generate a single sample
        gt_voltages = qarray.calculate_ground_truth()
        barrier_voltages = [0.0] * 3
        obs = qarray._get_obs(gt_voltages, barrier_voltages)
        cgd_matrix = qarray.model.Cgd.copy()
        
        return {
            'worker_pid': worker_pid,
            'gt_voltages_shape': gt_voltages.shape,
            'obs_image_shape': obs['image'].shape,
            'cgd_matrix_shape': cgd_matrix.shape,
            'success': True
        }
    except Exception as e:
        import traceback
        return {
            'worker_pid': worker_pid,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@ray.remote(num_cpus=1, num_gpus=1.0, memory=4*1024*1024*1024)  # Full GPU, 4GB memory
class QarrayWorkerActor:
    """Ray Actor that holds a single QarrayBaseClass instance for one GPU"""
    
    def __init__(self, gpu_id: int):
        import os
        import sys
        
        self.worker_pid = os.getpid()
        self.gpu_id = gpu_id
        
        # Add paths
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
        if environment_dir not in sys.path:
            sys.path.insert(0, environment_dir)
        
        # Set conservative memory settings
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use 80% of GPU since we have full GPU
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
        
        print(f"🎮 QarrayWorkerActor {self.worker_pid}: Initializing for GPU {gpu_id}")
        
        try:
            from qarray_base_class import QarrayBaseClass
            
            # Create single QarrayBaseClass instance that will be reused
            self.qarray = QarrayBaseClass(
                num_dots=4,
                config_path='qarray_config.yaml',
                obs_voltage_min=-1.0,
                obs_voltage_max=1.0,
                obs_image_size=128
            )
            
            self.initialized = True
            print(f"✅ QarrayWorkerActor {self.worker_pid}: Successfully initialized QarrayBaseClass")
            
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"❌ QarrayWorkerActor {self.worker_pid}: Failed to initialize: {e}")
    
    def generate_sample(self, sample_id: int):
        """Generate a single sample using the persistent QarrayBaseClass instance"""
        if not self.initialized:
            return {
                'sample_id': sample_id,
                'worker_pid': self.worker_pid,
                'gpu_id': self.gpu_id,
                'success': False,
                'error': f'Actor not initialized: {getattr(self, "init_error", "Unknown error")}'
            }
        
        try:
            # Use the persistent instance to generate sample
            gt_voltages = self.qarray.calculate_ground_truth()
            barrier_voltages = [0.0] * 3
            obs = self.qarray._get_obs(gt_voltages, barrier_voltages)
            cgd_matrix = self.qarray.model.Cgd.copy()
            
            return {
                'sample_id': sample_id,
                'worker_pid': self.worker_pid,
                'gpu_id': self.gpu_id,
                'image_shape': obs['image'].shape,
                'cgd_matrix_shape': cgd_matrix.shape,
                'gt_voltages_shape': gt_voltages.shape,
                'success': True
            }
            
        except Exception as e:
            import traceback
            return {
                'sample_id': sample_id,
                'worker_pid': self.worker_pid,
                'gpu_id': self.gpu_id,
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def generate_multiple_samples(self, sample_ids: list):
        """Generate multiple samples using the same QarrayBaseClass instance"""
        results = []
        for sample_id in sample_ids:
            result = self.generate_sample(sample_id)
            results.append(result)
        return results
    
    def get_status(self):
        """Get actor status"""
        return {
            'worker_pid': self.worker_pid,
            'gpu_id': self.gpu_id,
            'initialized': self.initialized,
            'init_error': getattr(self, 'init_error', None)
        }

def test_ray_actor_approach():
    """Test Ray Actor approach with one QarrayBaseClass instance per GPU"""
    print("\n" + "="*60)
    print("TEST 3: Ray Actor Approach (One QarrayBaseClass per GPU)")
    print("="*60)
    
    # GPU IDs already set in main()
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    gpu_list = [int(x.strip()) for x in gpu_ids.split(',') if x.strip()]
    num_gpus = len(gpu_list)
    
    print(f"Testing with {num_gpus} GPUs: {gpu_list}")
    
    try:
        # Initialize Ray with exact GPU count
        ray.init(
            num_cpus=num_gpus * 2,  # 2 CPUs per GPU
            num_gpus=num_gpus,      # Exact GPU count  
            object_store_memory=2*1024*1024*1024,  # 2GB
            include_dashboard=False,
            ignore_reinit_error=True
        )
        
        print(f"✓ Ray initialized with {num_gpus} GPUs")
        
        # Create one QarrayWorkerActor per GPU
        print(f"Creating {num_gpus} QarrayWorkerActors...")
        actors = []
        
        for i, gpu_id in enumerate(gpu_list):
            print(f"  Creating actor {i} for GPU {gpu_id}...")
            actor = QarrayWorkerActor.remote(gpu_id)
            actors.append(actor)
        
        # Wait for all actors to initialize and check status
        print("Checking actor initialization...")
        status_futures = [actor.get_status.remote() for actor in actors]
        statuses = ray.get(status_futures)
        
        initialized_actors = []
        for i, status in enumerate(statuses):
            if status['initialized']:
                print(f"  ✅ Actor {i}: PID {status['worker_pid']}, GPU {status['gpu_id']}")
                initialized_actors.append((i, actors[i]))
            else:
                print(f"  ❌ Actor {i}: Failed - {status.get('init_error', 'Unknown error')}")
        
        if not initialized_actors:
            print("❌ No actors initialized successfully")
            return
        
        print(f"\n🎯 Testing sample generation with {len(initialized_actors)} actors...")
        
        # Test single sample per actor
        print("--- Single sample per actor ---")
        single_futures = []
        for i, actor in initialized_actors:
            future = actor.generate_sample.remote(i * 100)  # Unique sample IDs
            single_futures.append((i, future))
        
        single_results = []
        for actor_id, future in single_futures:
            try:
                result = ray.get(future, timeout=60)
                single_results.append((actor_id, result))
                if result['success']:
                    print(f"  ✅ Actor {actor_id}: Sample {result['sample_id']} - Image {result['image_shape']}")
                else:
                    print(f"  ❌ Actor {actor_id}: Sample {result['sample_id']} - {result['error']}")
            except Exception as e:
                print(f"  ❌ Actor {actor_id}: Exception - {e}")
                single_results.append((actor_id, {'success': False, 'error': str(e)}))
        
        single_success_rate = sum(1 for _, result in single_results if result.get('success', False)) / len(single_results)
        print(f"Single sample success rate: {single_success_rate:.1%}")
        
        if single_success_rate == 1.0:
            # Test multiple samples per actor
            print("\n--- Multiple samples per actor (10 each) ---")
            sample_batches = []
            for i, actor in initialized_actors:
                sample_ids = list(range(i * 100 + 10, i * 100 + 20))  # 10 samples each
                sample_batches.append((i, sample_ids))
            
            multi_futures = []
            for actor_id, sample_ids in sample_batches:
                actor = initialized_actors[actor_id][1]
                future = actor.generate_multiple_samples.remote(sample_ids)
                multi_futures.append((actor_id, future))
            
            multi_results = []
            for actor_id, future in multi_futures:
                try:
                    batch_results = ray.get(future, timeout=120)  # Longer timeout for batch
                    multi_results.extend(batch_results)
                    
                    success_count = sum(1 for r in batch_results if r.get('success', False))
                    print(f"  Actor {actor_id}: {success_count}/{len(batch_results)} samples successful")
                    
                    # Show any failures
                    for result in batch_results:
                        if not result.get('success', False):
                            print(f"    ❌ Sample {result['sample_id']}: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    print(f"  ❌ Actor {actor_id} batch failed: {e}")
                    # Add failed results for this batch
                    for sample_id in sample_batches[actor_id][1]:
                        multi_results.append({
                            'sample_id': sample_id,
                            'success': False,
                            'error': f'Batch exception: {e}'
                        })
            
            multi_success_rate = sum(1 for r in multi_results if r.get('success', False)) / len(multi_results)
            print(f"Multiple samples success rate: {multi_success_rate:.1%}")
            
            if multi_success_rate == 1.0:
                print("🎉 Ray Actor approach with persistent QarrayBaseClass instances WORKS!")
            else:
                print(f"⚠️  Ray Actor approach partially works ({multi_success_rate:.1%} success rate)")
        else:
            print("❌ Single sample generation failed, skipping multiple sample test")
        
    except Exception as e:
        print(f"❌ Ray Actor test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ray.shutdown()
            time.sleep(2)
        except:
            pass

def test_ray_scaling():
    """Test Ray with increasing number of workers to find failure point"""
    print("\n" + "="*60)
    print("TEST 3: Ray Worker Scaling")
    print("="*60)
    
    # GPU IDs already set in main(), just verify
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    print(f"Testing with GPU IDs: {gpu_ids}")
    
    for num_workers in [1, 2, 4, 6, 8]:
        print(f"\n--- Testing with {num_workers} workers ---")
        
        try:
            # Initialize Ray
            ray.init(
                num_cpus=num_workers,
                num_gpus=2,  # 2 GPUs available
                object_store_memory=2*1024*1024*1024,  # 2GB
                include_dashboard=False,
                ignore_reinit_error=True
            )
            
            print(f"✓ Ray initialized: {num_workers} CPUs, 2 GPUs")
            
            # Submit basic worker tests
            print("Testing basic worker functionality...")
            futures = [test_ray_worker_basic.remote() for _ in range(num_workers)]
            
            basic_results = []
            for i, future in enumerate(futures):
                try:
                    result = ray.get(future, timeout=30)
                    basic_results.append(result)
                    if result['success']:
                        print(f"  Worker {i}: ✓ PID {result['worker_pid']}, CUDA: {result['cuda_visible_devices']}")
                    else:
                        print(f"  Worker {i}: ❌ Failed")
                except Exception as e:
                    print(f"  Worker {i}: ❌ Exception: {e}")
                    basic_results.append({'success': False, 'error': str(e)})
            
            basic_success_rate = sum(1 for r in basic_results if r.get('success', False)) / len(basic_results)
            print(f"Basic worker success rate: {basic_success_rate:.1%}")
            
            if basic_success_rate < 1.0:
                print(f"❌ Basic workers failing at {num_workers} workers")
                ray.shutdown()
                continue
            
            # Test JAX functionality
            print("Testing JAX functionality...")
            jax_futures = [test_ray_worker_jax.remote() for _ in range(min(num_workers, 4))]  # Limit JAX tests
            
            jax_results = []
            for i, future in enumerate(jax_futures):
                try:
                    result = ray.get(future, timeout=45)
                    jax_results.append(result)
                    if result['success']:
                        print(f"  JAX Worker {i}: ✓ PID {result['worker_pid']}, GPUs: {result['gpu_devices']}")
                    else:
                        print(f"  JAX Worker {i}: ❌ {result['error']}")
                except Exception as e:
                    print(f"  JAX Worker {i}: ❌ Exception: {e}")
                    jax_results.append({'success': False, 'error': str(e)})
            
            jax_success_rate = sum(1 for r in jax_results if r.get('success', False)) / len(jax_results)
            print(f"JAX worker success rate: {jax_success_rate:.1%}")
            
            if jax_success_rate < 1.0:
                print(f"❌ JAX workers failing at {num_workers} workers")
                ray.shutdown()
                continue
            
            # Test QarrayBaseClass functionality  
            print("Testing QarrayBaseClass functionality...")
            qarray_futures = [test_ray_worker_qarray.remote() for _ in range(min(num_workers, 2))]  # Very limited
            
            qarray_results = []
            for i, future in enumerate(qarray_futures):
                try:
                    result = ray.get(future, timeout=60)
                    qarray_results.append(result)
                    if result['success']:
                        print(f"  QArray Worker {i}: ✓ PID {result['worker_pid']}, Image: {result['obs_image_shape']}")
                    else:
                        print(f"  QArray Worker {i}: ❌ {result['error']}")
                        if 'traceback' in result:
                            print(f"    Traceback: {result['traceback']}")
                except Exception as e:
                    print(f"  QArray Worker {i}: ❌ Exception: {e}")
                    qarray_results.append({'success': False, 'error': str(e)})
            
            qarray_success_rate = sum(1 for r in qarray_results if r.get('success', False)) / len(qarray_results)
            print(f"QArray worker success rate: {qarray_success_rate:.1%}")
            
            # Test multiple samples in single worker
            if qarray_success_rate == 1.0 and num_workers <= 4:
                print("Testing multiple samples in single worker...")
                multi_future = test_ray_worker_multiple_samples.remote()
                try:
                    multi_result = ray.get(multi_future, timeout=120)
                    if multi_result:
                        success_rate = multi_result['total_success'] / multi_result['total_samples']
                        print(f"  Multi-sample test: {multi_result['total_success']}/{multi_result['total_samples']} ({success_rate:.1%})")
                        if success_rate < 1.0:
                            print(f"  ❌ Multiple samples failing in single worker")
                            for result in multi_result['results']:
                                if not result.get('success', False):
                                    print(f"    Sample {result['sample_idx']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"  ❌ Multi-sample test failed: {e}")
            
            print(f"✅ {num_workers} workers: Basic {basic_success_rate:.1%}, JAX {jax_success_rate:.1%}, QArray {qarray_success_rate:.1%}")
            
        except Exception as e:
            print(f"❌ Failed to test {num_workers} workers: {e}")
        finally:
            try:
                ray.shutdown()
                time.sleep(2)  # Give Ray time to cleanup
            except:
                pass

def test_memory_stress():
    """Test memory usage patterns to identify memory leaks"""
    print("\n" + "="*60)
    print("TEST 4: Memory Stress Test")
    print("="*60)
    
    # GPU IDs already set in main()
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    
    try:
        ray.init(
            num_cpus=2,
            num_gpus=2,
            object_store_memory=1*1024*1024*1024,  # 1GB
            include_dashboard=False,
            ignore_reinit_error=True
        )
        
        print("Testing memory usage over time...")
        
        for round_num in range(5):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Get initial memory
            initial_gpu_info = get_gpu_memory_usage()
            if initial_gpu_info:
                for gpu in initial_gpu_info:
                    if gpu['gpu_id'] in [6, 7]:  # Only show our GPUs
                        print(f"  Initial GPU {gpu['gpu_id']}: {gpu['used_mb']} MB used")
            
            # Submit multiple QArray tasks
            futures = [test_ray_worker_qarray.remote() for _ in range(4)]
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = ray.get(future, timeout=90)
                    results.append(result)
                    if result['success']:
                        print(f"    Task {i}: ✓")
                    else:
                        print(f"    Task {i}: ❌ {result['error']}")
                except Exception as e:
                    print(f"    Task {i}: ❌ Exception: {e}")
                    results.append({'success': False, 'error': str(e)})
            
            success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
            print(f"  Round {round_num + 1} success rate: {success_rate:.1%}")
            
            # Get final memory
            final_gpu_info = get_gpu_memory_usage()
            if initial_gpu_info and final_gpu_info:
                for initial, final in zip(initial_gpu_info, final_gpu_info):
                    if initial['gpu_id'] in [6, 7]:
                        memory_increase = final['used_mb'] - initial['used_mb']
                        print(f"  GPU {initial['gpu_id']} memory change: +{memory_increase} MB")
            
            time.sleep(2)  # Brief pause between rounds
        
    except Exception as e:
        print(f"❌ Memory stress test failed: {e}")
    finally:
        try:
            ray.shutdown()
        except:
            pass

def test_environment_variables():
    """Test different JAX/CUDA environment variable combinations"""
    print("\n" + "="*60)
    print("TEST 5: Environment Variable Combinations")
    print("="*60)
    
    original_env = dict(os.environ)
    
    test_configs = [
        {
            'name': 'Conservative Memory',
            'env': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.2',
                'XLA_FLAGS': '--xla_gpu_enable_command_buffer=',
                'TF_GPU_ALLOCATOR': 'cuda_malloc_async'
            }
        },
        {
            'name': 'Very Conservative Memory',
            'env': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.1',
                'XLA_FLAGS': '--xla_gpu_enable_command_buffer= --xla_gpu_strict_conv_algorithm_picker=false',
                'TF_GPU_ALLOCATOR': 'cuda_malloc_async'
            }
        },
        {
            'name': 'Minimal Memory',
            'env': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.05',
                'XLA_FLAGS': '--xla_gpu_enable_command_buffer= --xla_gpu_strict_conv_algorithm_picker=false',
                'JAX_PLATFORM_NAME': 'gpu'
            }
        }
    ]
    
    # GPU IDs already set in main()
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    
    for config in test_configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        # Set environment variables
        for key, value in config['env'].items():
            os.environ[key] = value
            print(f"  {key} = {value}")
        
        try:
            ray.init(
                num_cpus=4,
                num_gpus=2,
                object_store_memory=1*1024*1024*1024,
                include_dashboard=False,
                ignore_reinit_error=True
            )
            
            # Test with 4 workers (same as failing scenario)
            futures = [test_ray_worker_qarray.remote() for _ in range(4)]
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = ray.get(future, timeout=120)
                    results.append(result)
                    if result['success']:
                        print(f"    Worker {i}: ✓")
                    else:
                        print(f"    Worker {i}: ❌ {result['error']}")
                except Exception as e:
                    print(f"    Worker {i}: ❌ Timeout/Exception: {e}")
                    results.append({'success': False, 'error': str(e)})
            
            success_rate = sum(1 for r in results if r.get('success', False)) / len(results)
            print(f"  {config['name']} success rate: {success_rate:.1%}")
            
        except Exception as e:
            print(f"  ❌ Failed to test {config['name']}: {e}")
        finally:
            try:
                ray.shutdown()
                time.sleep(3)  # Longer cleanup time
            except:
                pass
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

def main():
    """Run comprehensive diagnostics"""
    print("🔍 COMPREHENSIVE RAY SAMPLE GENERATION DIAGNOSTICS")
    print("This script will identify the root cause of sample generation failures.")
    
    # FORCE GPU 6,7 usage from the start
    gpu_ids = "6,7"
    print(f"🎮 ENFORCING GPU usage: {gpu_ids}")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    # Print system info
    print_system_info()
    
    # Test 1: Single process tests
    jax_ok = test_jax_cuda_single_process()
    qarray_ok = test_qarray_single_process()
    
    if not jax_ok or not qarray_ok:
        print("\n❌ CRITICAL: Single process tests failed. Fix these issues first.")
        return
    
    print("\n✅ Single process tests passed. Moving to Ray tests...")
    
    # Test 2: Ray Actor approach (main test)
    test_ray_actor_approach()
    
    # Test 3: Ray scaling (for comparison)  
    test_ray_scaling()
    
    # Test 4: Memory stress 
    test_memory_stress()
    
    # Test 5: Environment variables
    test_environment_variables()
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print("Review the output above to identify:")
    print("1. At what worker count failures begin")
    print("2. Whether failures are JAX-related, QArray-related, or memory-related")
    print("3. Which environment variable settings work best")
    print("4. Memory usage patterns and potential leaks")

if __name__ == '__main__':
    main()