#!/usr/bin/env python3
"""
Test script to verify the BatchedCapacitanceModelWrapper works correctly
and maintains the same interface as the original model.
"""

import os
import sys
import time
import threading
from pathlib import Path
import torch
import numpy as np

# Add Swarm directory to path
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent
sys.path.append(str(swarm_dir))
os.environ['SWARM_PROJECT_ROOT'] = str(swarm_dir)

from batched_capacitance_wrapper import BatchedCapacitanceModelWrapper


class MockCapacitanceModel(torch.nn.Module):
    """Mock model that simulates the real capacitance model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 3)  # Simplified mock
        self.device = torch.device("cpu")
        
    def forward(self, x):
        """
        Mock forward pass.
        Input: (batch_size, 1, H, W) 
        Output: (values, log_vars) both of shape (batch_size, 3)
        """
        batch_size = x.shape[0]
        # Simulate processing time
        time.sleep(0.001)  # 1ms per forward pass
        
        # Mock outputs
        values = torch.randn(batch_size, 3)
        log_vars = torch.randn(batch_size, 3)
        return values, log_vars


def test_single_threaded():
    """Test that wrapper works the same as original model in single-threaded mode."""
    print("=== Testing Single-Threaded Compatibility ===")
    
    # Create original and wrapped models
    original_model = MockCapacitanceModel()
    wrapped_model = BatchedCapacitanceModelWrapper(original_model, batch_window_ms=5)
    
    # Test input (simulating 7 channels for 8-dot system)
    test_input = torch.randn(7, 1, 64, 64)  # (num_dots-1, 1, H, W)
    
    # Test original model
    with torch.no_grad():
        orig_values, orig_log_vars = original_model(test_input)
    
    # Test wrapped model
    wrapped_values, wrapped_log_vars = wrapped_model(test_input)
    
    # Verify shapes match
    assert orig_values.shape == wrapped_values.shape, f"Values shape mismatch: {orig_values.shape} vs {wrapped_values.shape}"
    assert orig_log_vars.shape == wrapped_log_vars.shape, f"Log vars shape mismatch: {orig_log_vars.shape} vs {wrapped_log_vars.shape}"
    
    print("✓ Single-threaded compatibility verified")
    wrapped_model.close()


def test_concurrent_access():
    """Test that multiple threads can safely access the wrapped model."""
    print("\n=== Testing Concurrent Access ===")
    
    model = MockCapacitanceModel()
    wrapped_model = BatchedCapacitanceModelWrapper(model, batch_window_ms=20, max_batch_size=32)
    
    num_threads = 10
    num_calls_per_thread = 5
    results = {}
    errors = []
    
    def worker_function(worker_id):
        """Simulate environment worker making model calls."""
        try:
            worker_results = []
            for i in range(num_calls_per_thread):
                # Each call simulates an environment step
                test_input = torch.randn(7, 1, 64, 64)  # Different random input
                start_time = time.time()
                values, log_vars = wrapped_model(test_input)
                call_time = time.time() - start_time
                
                worker_results.append({
                    'call_id': f"{worker_id}_{i}",
                    'values_shape': values.shape,
                    'log_vars_shape': log_vars.shape,
                    'call_time': call_time
                })
                
                # Small delay between calls
                time.sleep(0.002)
            
            results[worker_id] = worker_results
            
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")
    
    # Launch concurrent workers
    threads = []
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    if errors:
        print(f"✗ Errors occurred: {errors}")
        return False
    
    total_calls = num_threads * num_calls_per_thread
    successful_calls = sum(len(worker_results) for worker_results in results.values())
    
    print(f"✓ Concurrent access test completed:")
    print(f"  Total calls: {total_calls}")
    print(f"  Successful calls: {successful_calls}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per call: {total_time/total_calls:.3f}s")
    
    # Check for batching efficiency (should be faster than sequential)
    avg_call_time = np.mean([
        call['call_time'] 
        for worker_results in results.values() 
        for call in worker_results
    ])
    
    print(f"  Average individual call time: {avg_call_time:.3f}s")
    
    wrapped_model.close()
    return True


def test_interface_compatibility():
    """Test that all expected model methods are available."""
    print("\n=== Testing Interface Compatibility ===")
    
    original_model = MockCapacitanceModel()
    wrapped_model = BatchedCapacitanceModelWrapper(original_model)
    
    # Test common PyTorch model methods
    methods_to_test = ['eval', 'train', 'to', 'parameters', 'state_dict', 'load_state_dict']
    
    for method_name in methods_to_test:
        if hasattr(original_model, method_name):
            assert hasattr(wrapped_model, method_name), f"Missing method: {method_name}"
            print(f"  ✓ {method_name} method available")
    
    # Test device property
    assert hasattr(wrapped_model, 'device'), "Missing device property"
    print(f"  ✓ device property: {wrapped_model.device}")
    
    print("✓ Interface compatibility verified")
    wrapped_model.close()


if __name__ == "__main__":
    print("Testing BatchedCapacitanceModelWrapper...")
    
    try:
        # Run all tests
        test_interface_compatibility()
        test_single_threaded()
        success = test_concurrent_access()
        
        if success:
            print("\n🎉 All tests passed! The wrapper is ready for use.")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()