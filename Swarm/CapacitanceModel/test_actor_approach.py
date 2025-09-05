#!/usr/bin/env python3
"""
Test Ray Actor approach with one QarrayBaseClass instance per GPU.
"""

import os
import sys
import time
import ray

# Enforce GPUs 6,7 from start  
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# Add parent directory to path for imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
if environment_dir not in sys.path:
    sys.path.insert(0, environment_dir)

@ray.remote(num_cpus=1, num_gpus=1.0, memory=4*1024*1024*1024)  # Full GPU, 4GB memory
class QarrayWorkerActor:
    """Ray Actor that holds a single QarrayBaseClass instance for one GPU"""
    
    def __init__(self, gpu_id: int):
        import os
        import sys
        
        self.worker_pid = os.getpid()
        self.gpu_id = gpu_id
        
        # Add paths in actor
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        environment_dir = os.path.abspath(os.path.join(current_file_dir, '..', 'Environment'))
        if environment_dir not in sys.path:
            sys.path.insert(0, environment_dir)
        
        # Set conservative memory settings for this actor
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use 80% since we have full GPU
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
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
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
            print(f"  🔧 Actor {self.worker_pid}: Generating sample {sample_id}")
            
            # Use the persistent instance to generate sample
            gt_voltages = self.qarray.calculate_ground_truth()
            barrier_voltages = [0.0] * 3
            obs = self.qarray._get_obs(gt_voltages, barrier_voltages)
            cgd_matrix = self.qarray.model.Cgd.copy()
            
            print(f"  ✅ Actor {self.worker_pid}: Sample {sample_id} completed")
            
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
            print(f"  ❌ Actor {self.worker_pid}: Sample {sample_id} failed: {e}")
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
        print(f"  🔄 Actor {self.worker_pid}: Generating {len(sample_ids)} samples")
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

def main():
    """Test Ray Actor approach with one QarrayBaseClass per GPU"""
    print("🔍 TESTING RAY ACTOR APPROACH")
    print("One QarrayBaseClass instance per GPU (GPUs 6,7)")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    gpu_list = [int(x.strip()) for x in gpu_ids.split(',') if x.strip()]
    num_gpus = len(gpu_list)
    
    print(f"Using {num_gpus} GPUs: {gpu_list}")
    
    try:
        # Initialize Ray with exact GPU count - one actor per GPU
        ray.init(
            num_cpus=num_gpus + 2,  # Extra CPUs for coordination
            num_gpus=num_gpus,      # Exact GPU count  
            object_store_memory=2*1024*1024*1024,  # 2GB
            include_dashboard=False,
            ignore_reinit_error=True
        )
        
        print(f"✅ Ray initialized with {num_gpus} GPUs")
        
        # Create one QarrayWorkerActor per GPU
        print(f"\n🏗️  Creating {num_gpus} QarrayWorkerActors...")
        actors = []
        
        for i, gpu_id in enumerate(gpu_list):
            print(f"  Creating actor {i} for GPU {gpu_id}...")
            actor = QarrayWorkerActor.remote(gpu_id)
            actors.append((i, actor))
        
        # Wait for all actors to initialize and check status
        print(f"\n🔍 Checking actor initialization...")
        status_futures = [actor.get_status.remote() for _, actor in actors]
        statuses = ray.get(status_futures, timeout=120)  # Give plenty of time
        
        initialized_actors = []
        for i, status in enumerate(statuses):
            if status['initialized']:
                print(f"  ✅ Actor {i}: PID {status['worker_pid']}, GPU {status['gpu_id']}")
                initialized_actors.append(actors[i])
            else:
                print(f"  ❌ Actor {i}: Failed - {status.get('init_error', 'Unknown error')}")
        
        if not initialized_actors:
            print("❌ No actors initialized successfully")
            return
        
        print(f"\n🎯 Testing sample generation with {len(initialized_actors)} actors...")
        
        # Test 1: Single sample per actor
        print("\n--- Test 1: Single sample per actor ---")
        single_futures = []
        for actor_id, actor in initialized_actors:
            future = actor.generate_sample.remote(actor_id * 100)  # Unique sample IDs
            single_futures.append((actor_id, future))
        
        single_results = []
        for actor_id, future in single_futures:
            try:
                result = ray.get(future, timeout=90)
                single_results.append((actor_id, result))
                if result['success']:
                    print(f"  ✅ Actor {actor_id}: Sample {result['sample_id']} - Image {result['image_shape']}")
                else:
                    print(f"  ❌ Actor {actor_id}: Sample {result['sample_id']} - {result['error']}")
            except Exception as e:
                print(f"  ❌ Actor {actor_id}: Exception - {e}")
                single_results.append((actor_id, {'success': False, 'error': str(e)}))
        
        single_success_rate = sum(1 for _, result in single_results if result.get('success', False)) / len(single_results)
        print(f"\n📊 Single sample success rate: {single_success_rate:.1%}")
        
        if single_success_rate == 1.0:
            # Test 2: Multiple samples per actor  
            print("\n--- Test 2: Multiple samples per actor (10 each) ---")
            sample_batches = []
            for actor_id, actor in initialized_actors:
                sample_ids = list(range(actor_id * 100 + 10, actor_id * 100 + 20))  # 10 samples each
                sample_batches.append((actor_id, sample_ids, actor))
            
            multi_futures = []
            for actor_id, sample_ids, actor in sample_batches:
                future = actor.generate_multiple_samples.remote(sample_ids)
                multi_futures.append((actor_id, future))
            
            all_multi_results = []
            for actor_id, future in multi_futures:
                try:
                    batch_results = ray.get(future, timeout=180)  # Longer timeout for batch
                    all_multi_results.extend(batch_results)
                    
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
                        all_multi_results.append({
                            'sample_id': sample_id,
                            'success': False,
                            'error': f'Batch exception: {e}'
                        })
            
            multi_success_rate = sum(1 for r in all_multi_results if r.get('success', False)) / len(all_multi_results)
            print(f"\n📊 Multiple samples success rate: {multi_success_rate:.1%}")
            
            if multi_success_rate == 1.0:
                print("\n🎉 SUCCESS: Ray Actor approach with persistent QarrayBaseClass instances WORKS!")
                print("✅ Each GPU can reliably run one QarrayBaseClass instance")
                print("✅ Multiple samples can be generated from the same instance")
                
                # Test 3: Stress test with more samples
                print("\n--- Test 3: Stress test (50 samples per actor) ---")
                stress_futures = []
                for actor_id, actor in initialized_actors:
                    sample_ids = list(range(actor_id * 1000, actor_id * 1000 + 50))  # 50 samples each
                    future = actor.generate_multiple_samples.remote(sample_ids)
                    stress_futures.append((actor_id, future))
                
                stress_results = []
                for actor_id, future in stress_futures:
                    try:
                        batch_results = ray.get(future, timeout=300)  # 5 minute timeout
                        stress_results.extend(batch_results)
                        
                        success_count = sum(1 for r in batch_results if r.get('success', False))
                        print(f"  Actor {actor_id}: {success_count}/{len(batch_results)} stress samples successful")
                        
                    except Exception as e:
                        print(f"  ❌ Actor {actor_id} stress test failed: {e}")
                
                stress_success_rate = sum(1 for r in stress_results if r.get('success', False)) / len(stress_results)
                print(f"\n📊 Stress test success rate: {stress_success_rate:.1%}")
                
                if stress_success_rate >= 0.95:  # Allow for minor failures
                    print("🚀 EXCELLENT: Ray Actor approach scales well!")
                else:
                    print(f"⚠️  Ray Actor approach has issues at scale ({stress_success_rate:.1%})")
                
            else:
                print(f"\n⚠️  Ray Actor approach partially works ({multi_success_rate:.1%} success rate)")
        else:
            print("❌ Single sample generation failed, skipping further tests")
        
    except Exception as e:
        print(f"❌ Ray Actor test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            print("\n🔧 Shutting down Ray...")
            ray.shutdown()
            time.sleep(2)
            print("✅ Ray shutdown complete")
        except Exception as e:
            print(f"⚠️  Ray shutdown had issues: {e}")

if __name__ == '__main__':
    main()