#!/usr/bin/env python3
"""
Focused diagnostic for qarray library memory allocation issues.

This script specifically tests the qarray library's GPU memory usage patterns
to understand why it fails in multi-worker Ray environments.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Enforce GPUs 6,7 from start
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# Add parent directory to path for imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
if environment_dir not in sys.path:
    sys.path.insert(0, environment_dir)

def get_gpu_memory_for_our_gpus():
    """Get memory usage specifically for GPUs 6,7"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.used,memory.total,memory.free', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            our_gpus = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    gpu_id = int(parts[0])
                    if gpu_id in [6, 7]:
                        our_gpus[gpu_id] = {
                            'used_mb': int(parts[1]),
                            'total_mb': int(parts[2]), 
                            'free_mb': int(parts[3])
                        }
            return our_gpus
        return {}
    except Exception:
        return {}

def test_jax_memory_settings():
    """Test different JAX memory configuration approaches"""
    print("="*70)
    print("TESTING JAX MEMORY CONFIGURATIONS")
    print("="*70)
    
    configs = [
        {
            'name': 'Default (no limits)',
            'env_vars': {}
        },
        {
            'name': 'Memory fraction 20%',
            'env_vars': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.2'
            }
        },
        {
            'name': 'Memory fraction 10%',
            'env_vars': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.1'
            }
        },
        {
            'name': 'Memory fraction 5%',
            'env_vars': {
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.05'
            }
        },
        {
            'name': 'Preallocate disabled',
            'env_vars': {
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'
            }
        }
    ]
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        # Set environment variables for this test
        for key, value in config['env_vars'].items():
            os.environ[key] = value
            print(f"Set {key} = {value}")
        
        # Get initial GPU memory
        initial_memory = get_gpu_memory_for_our_gpus()
        print(f"Initial memory - GPU 6: {initial_memory.get(6, {}).get('used_mb', 'unknown')} MB, GPU 7: {initial_memory.get(7, {}).get('used_mb', 'unknown')} MB")
        
        try:
            # Import and test JAX
            import jax
            import jax.numpy as jnp
            
            # Force JAX to initialize
            x = jnp.array([1.0])
            y = jnp.sum(x)
            
            print(f"✓ JAX basic operations successful")
            
            # Test larger operations
            A = jnp.ones((1000, 1000))
            B = jnp.ones((1000, 1000))
            C = jnp.dot(A, B)
            print(f"✓ Large matrix operations successful: {C.shape}")
            
            # Get memory after JAX operations
            after_jax_memory = get_gpu_memory_for_our_gpus()
            
            # Now test QarrayBaseClass
            from qarray_base_class import QarrayBaseClass
            
            qarray = QarrayBaseClass(
                num_dots=4,
                config_path='qarray_config.yaml',
                obs_voltage_min=-1.0,
                obs_voltage_max=1.0,
                obs_image_size=128
            )
            print("✓ QarrayBaseClass instantiation successful")
            
            # Test ground truth calculation (this might allocate GPU memory)
            gt_voltages = qarray.calculate_ground_truth()
            print(f"✓ Ground truth calculation: {gt_voltages.shape}")
            
            # Get memory after ground truth
            after_gt_memory = get_gpu_memory_for_our_gpus()
            
            # Test observation generation (this is where failures occur)
            barrier_voltages = [0.0] * 3
            obs = qarray._get_obs(gt_voltages, barrier_voltages)
            print(f"✓ Observation generation successful: {obs['image'].shape}")
            
            # Get final memory
            final_memory = get_gpu_memory_for_our_gpus()
            
            # Calculate memory increases
            for gpu_id in [6, 7]:
                initial = initial_memory.get(gpu_id, {}).get('used_mb', 0)
                after_jax = after_jax_memory.get(gpu_id, {}).get('used_mb', 0)
                after_gt = after_gt_memory.get(gpu_id, {}).get('used_mb', 0)
                final = final_memory.get(gpu_id, {}).get('used_mb', 0)
                
                jax_increase = after_jax - initial
                gt_increase = after_gt - after_jax
                obs_increase = final - after_gt
                
                print(f"  GPU {gpu_id} memory increases: JAX: +{jax_increase}MB, GT: +{gt_increase}MB, Obs: +{obs_increase}MB")
            
            print(f"✅ {config['name']}: All operations successful")
            
        except Exception as e:
            print(f"❌ {config['name']}: Failed with {type(e).__name__}: {e}")
            
            # Show memory even on failure
            error_memory = get_gpu_memory_for_our_gpus()
            for gpu_id in [6, 7]:
                initial = initial_memory.get(gpu_id, {}).get('used_mb', 0)
                current = error_memory.get(gpu_id, {}).get('used_mb', 0)
                increase = current - initial
                print(f"  GPU {gpu_id} memory at failure: +{increase}MB from start")
        
        # Clear environment variables for next test
        for key in config['env_vars'].keys():
            if key in os.environ:
                del os.environ[key]
        
        # Force garbage collection and give time for cleanup
        import gc
        gc.collect()
        time.sleep(3)
        
        print("-" * 50)

def test_qarray_sequential_vs_parallel():
    """Test sequential vs parallel QarrayBaseClass usage"""
    print("\n" + "="*70)
    print("TESTING SEQUENTIAL VS PARALLEL QARRAY USAGE")
    print("="*70)
    
    # Set conservative memory settings
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
    
    print("Testing sequential QarrayBaseClass operations...")
    
    try:
        from qarray_base_class import QarrayBaseClass
        
        # Sequential test - multiple samples one after another
        for i in range(5):
            print(f"  Sample {i}...", end=" ")
            
            initial_memory = get_gpu_memory_for_our_gpus()
            
            qarray = QarrayBaseClass(
                num_dots=4,
                config_path='qarray_config.yaml',
                obs_voltage_min=-1.0,
                obs_voltage_max=1.0,
                obs_image_size=128
            )
            
            gt_voltages = qarray.calculate_ground_truth()
            barrier_voltages = [0.0] * 3
            obs = qarray._get_obs(gt_voltages, barrier_voltages)
            
            final_memory = get_gpu_memory_for_our_gpus()
            
            # Calculate memory increase
            memory_increase = 0
            for gpu_id in [6, 7]:
                initial = initial_memory.get(gpu_id, {}).get('used_mb', 0)
                final = final_memory.get(gpu_id, {}).get('used_mb', 0)
                memory_increase += (final - initial)
            
            print(f"✓ (GPU memory: +{memory_increase}MB)")
            
            # Explicit cleanup
            del qarray, gt_voltages, obs
            import gc
            gc.collect()
            
        print("✅ Sequential operations successful")
        
    except Exception as e:
        print(f"❌ Sequential test failed: {e}")
        import traceback
        traceback.print_exc()

def test_single_qarray_instance_reuse():
    """Test reusing a single QarrayBaseClass instance for multiple samples"""
    print("\n" + "="*70)  
    print("TESTING SINGLE QARRAY INSTANCE REUSE")
    print("="*70)
    
    # Set very conservative memory settings
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.05'
    
    try:
        from qarray_base_class import QarrayBaseClass
        
        print("Creating single QarrayBaseClass instance...")
        qarray = QarrayBaseClass(
            num_dots=4,
            config_path='qarray_config.yaml',
            obs_voltage_min=-1.0,
            obs_voltage_max=1.0,
            obs_image_size=128
        )
        print("✓ Instance created")
        
        # Test reusing the same instance
        for i in range(10):
            print(f"  Sample {i}...", end=" ")
            
            initial_memory = get_gpu_memory_for_our_gpus()
            
            # Generate sample using same instance
            gt_voltages = qarray.calculate_ground_truth()
            barrier_voltages = [0.0] * 3
            obs = qarray._get_obs(gt_voltages, barrier_voltages)
            
            final_memory = get_gpu_memory_for_our_gpus()
            
            memory_increase = 0
            for gpu_id in [6, 7]:
                initial = initial_memory.get(gpu_id, {}).get('used_mb', 0)
                final = final_memory.get(gpu_id, {}).get('used_mb', 0)
                memory_increase += (final - initial)
            
            print(f"✓ (+{memory_increase}MB)")
            
            # Light cleanup
            del gt_voltages, obs
        
        print("✅ Instance reuse successful")
        
    except Exception as e:
        print(f"❌ Instance reuse failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run focused qarray memory diagnostics"""
    print("🔍 QARRAY LIBRARY MEMORY DIAGNOSTICS")
    print("Focused on understanding qarray's GPU memory allocation patterns")
    print(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Show initial GPU state
    initial_memory = get_gpu_memory_for_our_gpus()
    print(f"\nInitial GPU state:")
    for gpu_id, info in initial_memory.items():
        print(f"  GPU {gpu_id}: {info['used_mb']}/{info['total_mb']} MB ({info['used_mb']/info['total_mb']*100:.1f}% used)")
    
    # Test 1: JAX memory settings
    test_jax_memory_settings()
    
    # Test 2: Sequential vs parallel
    test_qarray_sequential_vs_parallel()
    
    # Test 3: Instance reuse
    test_single_qarray_instance_reuse()
    
    print("\n" + "="*80)
    print("MEMORY DIAGNOSTIC COMPLETE")
    print("="*80)
    print("Key findings:")
    print("1. Check which memory fraction settings allow QarrayBaseClass to work")
    print("2. Observe memory allocation patterns (does memory keep growing?)")
    print("3. Compare instance creation vs reuse memory usage")

if __name__ == '__main__':
    main()