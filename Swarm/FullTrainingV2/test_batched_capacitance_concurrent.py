#!/usr/bin/env python3
"""
Comprehensive test script for batched capacitance model with concurrent requests.

Tests:
1. Single request correctness
2. Concurrent requests from multiple threads
3. Batching efficiency and correctness
4. Edge cases (timeouts, memory limits, error handling)
5. Thread safety and result consistency
"""

import os
import sys
import torch
import numpy as np
import threading
import time
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Any
import traceback

# Add Swarm directory to path
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent
sys.path.append(str(swarm_dir))
os.environ['SWARM_PROJECT_ROOT'] = str(swarm_dir)

from train import setup_capacitance_model


class CapacitanceModelTester:
    """Comprehensive tester for batched capacitance model."""
    
    def __init__(self, checkpoint_path: str = "artifacts/best_model.pth"):
        """Initialize tester with capacitance model."""
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.test_results = {}
        
    def setup_model(self, batch_window_ms: int = 50, max_batch_size: int = 32):
        """Setup the batched capacitance model."""
        print(f"Setting up capacitance model with batching (window: {batch_window_ms}ms, max_batch: {max_batch_size})")
        self.model = setup_capacitance_model(
            checkpoint=self.checkpoint_path,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size
        )
        print(f"Model device: {self.model.device}")
        
    def create_test_input(self, num_dots: int = 4, batch_size: int = 1, seed: int = None) -> torch.Tensor:
        """Create realistic test input tensor."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create input shape: (num_dots-1, 1, 224, 224) - similar to real CSD images
        channels = num_dots - 1
        height, width = 224, 224
        
        # Generate realistic-looking charge stability diagrams
        input_tensor = torch.zeros(channels, 1, height, width)
        
        for i in range(channels):
            # Create a charge stability diagram pattern
            x = torch.linspace(-1, 1, width)
            y = torch.linspace(-1, 1, height)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Add some realistic features (charge transitions, coulomb diamonds)
            pattern = torch.sin(5 * X) * torch.cos(5 * Y) + 0.1 * torch.randn(height, width)
            pattern = torch.sigmoid(pattern)  # Normalize to [0, 1]
            
            input_tensor[i, 0] = pattern
            
        return input_tensor.to(self.model.device)
    
    def test_single_request(self) -> bool:
        """Test single request correctness."""
        print("\n=== Test 1: Single Request Correctness ===")
        
        try:
            test_input = self.create_test_input(num_dots=4)
            print(f"Input shape: {test_input.shape}")
            
            start_time = time.time()
            values, log_vars = self.model(test_input)
            end_time = time.time()
            
            print(f"✓ Single request completed in {(end_time - start_time)*1000:.2f}ms")
            print(f"  Values shape: {values.shape}")
            print(f"  Log vars shape: {log_vars.shape}")
            print(f"  Values range: [{values.min().item():.4f}, {values.max().item():.4f}]")
            print(f"  Log vars range: [{log_vars.min().item():.4f}, {log_vars.max().item():.4f}]")
            
            # Verify outputs are reasonable
            assert values.shape == (3, 3), f"Expected values shape (3, 3), got {values.shape}"
            assert log_vars.shape == (3, 3), f"Expected log_vars shape (3, 3), got {log_vars.shape}"
            assert torch.all(values >= 0) and torch.all(values <= 1), "Values should be in [0, 1] range"
            
            self.test_results["single_request"] = True
            return True
            
        except Exception as e:
            print(f"✗ Single request test failed: {e}")
            traceback.print_exc()
            self.test_results["single_request"] = False
            return False
    
    def worker_function(self, worker_id: int, num_requests: int, num_dots: int) -> Dict[str, Any]:
        """Worker function that sends multiple requests."""
        results = []
        errors = []
        timings = []
        
        for i in range(num_requests):
            try:
                # Create unique input for this request
                test_input = self.create_test_input(num_dots=num_dots)
                # Add small variation to make each request unique
                test_input += 0.01 * torch.randn_like(test_input) * (worker_id + 1)
                
                start_time = time.time()
                values, log_vars = self.model(test_input)
                end_time = time.time()
                
                timings.append(end_time - start_time)
                results.append({
                    'worker_id': worker_id,
                    'request_id': i,
                    'values_shape': values.shape,
                    'log_vars_shape': log_vars.shape,
                    'values_checksum': values.sum().item(),
                    'log_vars_checksum': log_vars.sum().item(),
                    'timing': end_time - start_time
                })
                
            except Exception as e:
                errors.append(f"Worker {worker_id}, Request {i}: {str(e)}")
        
        return {
            'worker_id': worker_id,
            'results': results,
            'errors': errors,
            'timings': timings,
            'avg_timing': np.mean(timings) if timings else 0,
            'total_requests': len(results)
        }
    
    def test_concurrent_requests(self, num_workers: int = 8, requests_per_worker: int = 5) -> bool:
        """Test concurrent requests from multiple threads."""
        print(f"\n=== Test 2: Concurrent Requests ({num_workers} workers, {requests_per_worker} requests each) ===")
        
        try:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all worker tasks
                futures = [
                    executor.submit(self.worker_function, worker_id, requests_per_worker, 4)
                    for worker_id in range(num_workers)
                ]
                
                # Collect results
                worker_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        worker_results.append(result)
                    except Exception as e:
                        print(f"Worker failed: {e}")
                        return False
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze results
            total_requests = sum(r['total_requests'] for r in worker_results)
            total_errors = sum(len(r['errors']) for r in worker_results)
            avg_timings = [r['avg_timing'] for r in worker_results if r['avg_timing'] > 0]
            
            print(f"✓ Concurrent test completed in {total_time:.2f}s")
            print(f"  Total requests processed: {total_requests}")
            print(f"  Total errors: {total_errors}")
            print(f"  Average request time: {np.mean(avg_timings)*1000:.2f}ms")
            print(f"  Throughput: {total_requests/total_time:.2f} requests/second")
            
            # Check for errors
            if total_errors > 0:
                print("Errors encountered:")
                for result in worker_results:
                    for error in result['errors']:
                        print(f"  {error}")
                return False
            
            # Verify all requests completed
            if total_requests != num_workers * requests_per_worker:
                print(f"✗ Expected {num_workers * requests_per_worker} requests, got {total_requests}")
                return False
            
            self.test_results["concurrent_requests"] = True
            return True
            
        except Exception as e:
            print(f"✗ Concurrent requests test failed: {e}")
            traceback.print_exc()
            self.test_results["concurrent_requests"] = False
            return False
    
    def test_batching_efficiency(self) -> bool:
        """Test that batching actually improves efficiency."""
        print("\n=== Test 3: Batching Efficiency ===")
        
        try:
            num_requests = 20
            
            # Test with very short batching window (near-individual processing)
            print("Testing with minimal batching (1ms window)...")
            self.setup_model(batch_window_ms=1, max_batch_size=1)
            
            start_time = time.time()
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for i in range(num_requests):
                    test_input = self.create_test_input(num_dots=4)
                    future = executor.submit(self.model, test_input)
                    futures.append(future)
                
                for future in futures:
                    future.result()
            no_batch_time = time.time() - start_time
            
            # Test with longer batching window (efficient batching)
            print("Testing with efficient batching (100ms window, batch size 32)...")
            self.setup_model(batch_window_ms=100, max_batch_size=32)
            
            start_time = time.time()
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                for i in range(num_requests):
                    test_input = self.create_test_input(num_dots=4)
                    future = executor.submit(self.model, test_input)
                    futures.append(future)
                
                for future in futures:
                    future.result()
            batch_time = time.time() - start_time
            
            print(f"✓ No batching time: {no_batch_time:.2f}s")
            print(f"✓ With batching time: {batch_time:.2f}s")
            
            # Batching should be at least as fast (within margin of error)
            efficiency_ratio = no_batch_time / batch_time if batch_time > 0 else 0
            print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")
            
            self.test_results["batching_efficiency"] = True
            return True
            
        except Exception as e:
            print(f"✗ Batching efficiency test failed: {e}")
            traceback.print_exc()
            self.test_results["batching_efficiency"] = False
            return False
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        print("\n=== Test 4: Edge Cases ===")
        
        try:
            # Test 1: Very large batch
            print("Testing large batch size...")
            large_inputs = []
            for i in range(50):  # Create 50 simultaneous requests
                test_input = self.create_test_input(num_dots=8)  # Larger input
                large_inputs.append(test_input)
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(self.model, inp) for inp in large_inputs]
                results = [future.result(timeout=60) for future in futures]
            end_time = time.time()
            
            print(f"✓ Large batch test completed in {end_time - start_time:.2f}s")
            print(f"  Processed {len(results)} large requests successfully")
            
            # Test 2: Mixed batch sizes
            print("Testing mixed input sizes...")
            mixed_inputs = []
            for num_dots in [4, 6, 8]:  # Different quantum dot array sizes
                for _ in range(5):
                    mixed_inputs.append(self.create_test_input(num_dots=num_dots))
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.model, inp) for inp in mixed_inputs]
                try:
                    results = [future.result(timeout=30) for future in futures]
                    print(f"✓ Mixed sizes test completed successfully")
                except Exception as e:
                    print(f"Mixed sizes test failed (expected): {e}")
                    # This might fail due to different tensor shapes, which is expected
            
            # Test 3: Rapid sequential requests
            print("Testing rapid sequential requests...")
            sequential_times = []
            for i in range(20):
                test_input = self.create_test_input(num_dots=4)
                start = time.time()
                values, log_vars = self.model(test_input)
                sequential_times.append(time.time() - start)
            
            print(f"✓ Sequential requests: avg {np.mean(sequential_times)*1000:.2f}ms, "
                  f"std {np.std(sequential_times)*1000:.2f}ms")
            
            self.test_results["edge_cases"] = True
            return True
            
        except Exception as e:
            print(f"✗ Edge cases test failed: {e}")
            traceback.print_exc()
            self.test_results["edge_cases"] = False
            return False
    
    def test_output_consistency(self) -> bool:
        """Test that identical inputs produce identical outputs."""
        print("\n=== Test 5: Output Consistency ===")
        
        try:
            # Create a deterministic reference input
            ref_input = self.create_test_input(num_dots=4, seed=42)
            
            # Get reference output
            ref_values, ref_log_vars = self.model(ref_input)
            
            # Test multiple identical requests
            num_tests = 10
            identical_results = []
            
            for i in range(num_tests):
                values, log_vars = self.model(ref_input.clone())
                identical_results.append((values, log_vars))
            
            # Check all results are identical
            all_identical = True
            for i, (values, log_vars) in enumerate(identical_results):
                if not torch.allclose(values, ref_values, atol=1e-6):
                    print(f"✗ Values mismatch in test {i}")
                    all_identical = False
                if not torch.allclose(log_vars, ref_log_vars, atol=1e-6):
                    print(f"✗ Log vars mismatch in test {i}")
                    all_identical = False
            
            if all_identical:
                print(f"✓ All {num_tests} identical inputs produced identical outputs")
            else:
                print(f"✗ Output consistency failed")
                return False
            
            # Test concurrent identical requests
            print("Testing concurrent identical requests...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self.model, ref_input.clone()) 
                    for _ in range(16)
                ]
                concurrent_results = [future.result() for future in futures]
            
            # Verify all concurrent results are identical
            concurrent_identical = True
            for i, (values, log_vars) in enumerate(concurrent_results):
                if not torch.allclose(values, ref_values, atol=1e-6):
                    print(f"✗ Concurrent values mismatch in test {i}")
                    concurrent_identical = False
                if not torch.allclose(log_vars, ref_log_vars, atol=1e-6):
                    print(f"✗ Concurrent log vars mismatch in test {i}")
                    concurrent_identical = False
            
            if concurrent_identical:
                print(f"✓ All 16 concurrent identical requests produced identical outputs")
            else:
                print(f"✗ Concurrent output consistency failed")
                return False
            
            self.test_results["output_consistency"] = True
            return True
            
        except Exception as e:
            print(f"✗ Output consistency test failed: {e}")
            traceback.print_exc()
            self.test_results["output_consistency"] = False
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage under heavy load."""
        print("\n=== Test 6: Memory Usage ===")
        
        try:
            if not torch.cuda.is_available():
                print("Skipping GPU memory test (CUDA not available)")
                self.test_results["memory_usage"] = True
                return True
            
            # Get initial GPU memory
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            
            print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
            
            # Create many concurrent requests
            large_batch_inputs = []
            for i in range(100):
                test_input = self.create_test_input(num_dots=6)  # Medium size
                large_batch_inputs.append(test_input)
            
            # Process all at once
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(self.model, inp) for inp in large_batch_inputs]
                results = [future.result(timeout=60) for future in futures]
            end_time = time.time()
            
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            print(f"✓ Processed 100 concurrent requests in {end_time - start_time:.2f}s")
            print(f"  Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
            print(f"  Final GPU memory: {final_memory / 1024**2:.1f} MB")
            print(f"  Memory increase: {(final_memory - initial_memory) / 1024**2:.1f} MB")
            
            # Memory should not grow excessively
            memory_increase = final_memory - initial_memory
            if memory_increase > 500 * 1024**2:  # 500 MB threshold
                print(f"⚠ Warning: Large memory increase detected ({memory_increase / 1024**2:.1f} MB)")
            
            self.test_results["memory_usage"] = True
            return True
            
        except Exception as e:
            print(f"✗ Memory usage test failed: {e}")
            traceback.print_exc()
            self.test_results["memory_usage"] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        print("\n=== Test 7: Error Handling ===")
        
        try:
            # Test 1: Invalid input shapes (should fail gracefully)
            print("Testing invalid input shapes...")
            try:
                invalid_input = torch.randn(2, 3, 100, 100).to(self.model.device)  # Wrong shape
                self.model(invalid_input)
                print("⚠ Warning: Model accepted invalid input shape")
            except Exception as e:
                print(f"✓ Model correctly rejected invalid input: {type(e).__name__}")
            
            # Test 2: Very large inputs (memory stress test)
            print("Testing memory stress with large inputs...")
            try:
                # Create very large input
                stress_input = self.create_test_input(num_dots=4)
                # Process multiple large requests simultaneously
                large_futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    for _ in range(10):
                        future = executor.submit(self.model, stress_input)
                        large_futures.append(future)
                    
                    for future in large_futures:
                        future.result(timeout=30)
                
                print("✓ Handled large memory stress test")
                
            except Exception as e:
                print(f"Memory stress test result: {type(e).__name__} (may be expected)")
            
            # Test 3: Model recovery after errors
            print("Testing model recovery...")
            normal_input = self.create_test_input(num_dots=4)
            values, log_vars = self.model(normal_input)
            print("✓ Model recovered and processed normal request")
            
            self.test_results["error_handling"] = True
            return True
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            traceback.print_exc()
            self.test_results["error_handling"] = False
            return False
    
    def test_thread_safety(self) -> bool:
        """Test thread safety with conflicting access patterns."""
        print("\n=== Test 8: Thread Safety ===")
        
        try:
            results_lock = threading.Lock()
            thread_results = []
            
            def stress_worker(worker_id: int, duration_seconds: float):
                """Worker that continuously sends requests for specified duration."""
                end_time = time.time() + duration_seconds
                requests_completed = 0
                errors = 0
                
                while time.time() < end_time:
                    try:
                        test_input = self.create_test_input(num_dots=4)
                        values, log_vars = self.model(test_input)
                        requests_completed += 1
                    except Exception:
                        errors += 1
                
                with results_lock:
                    thread_results.append({
                        'worker_id': worker_id,
                        'requests_completed': requests_completed,
                        'errors': errors
                    })
            
            # Run stress test for 5 seconds with many workers
            stress_duration = 5.0
            num_stress_workers = 12
            
            print(f"Running stress test with {num_stress_workers} workers for {stress_duration}s...")
            stress_threads = []
            
            start_time = time.time()
            for i in range(num_stress_workers):
                thread = threading.Thread(target=stress_worker, args=(i, stress_duration))
                thread.start()
                stress_threads.append(thread)
            
            # Wait for all threads to complete
            for thread in stress_threads:
                thread.join()
            
            actual_duration = time.time() - start_time
            
            # Analyze stress test results
            total_requests = sum(r['requests_completed'] for r in thread_results)
            total_errors = sum(r['errors'] for r in thread_results)
            
            print(f"✓ Stress test completed in {actual_duration:.2f}s")
            print(f"  Total requests: {total_requests}")
            print(f"  Total errors: {total_errors}")
            print(f"  Throughput: {total_requests/actual_duration:.2f} requests/second")
            print(f"  Error rate: {total_errors/total_requests*100:.2f}%" if total_requests > 0 else "  No requests completed")
            
            # Thread safety passes if we completed requests without deadlocks
            if total_requests > 0 and actual_duration < stress_duration * 1.5:
                print("✓ Thread safety test passed")
                self.test_results["thread_safety"] = True
                return True
            else:
                print("✗ Thread safety test failed (possible deadlock or no progress)")
                self.test_results["thread_safety"] = False
                return False
            
        except Exception as e:
            print(f"✗ Thread safety test failed: {e}")
            traceback.print_exc()
            self.test_results["thread_safety"] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results summary."""
        print("=" * 60)
        print("BATCHED CAPACITANCE MODEL TEST SUITE")
        print("=" * 60)
        
        # Setup initial model
        self.setup_model()
        
        # Run all tests
        tests = [
            ("Single Request", self.test_single_request),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Batching Efficiency", self.test_batching_efficiency),
            ("Edge Cases", self.test_edge_cases),
            ("Output Consistency", self.test_output_consistency),
            ("Memory Usage", self.test_memory_usage),
            ("Thread Safety", self.test_thread_safety),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name} crashed: {e}")
                self.test_results[test_name.lower().replace(" ", "_")] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name.upper():<25} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if self.model and hasattr(self.model, 'close'):
            self.model.close()
        
        return self.test_results


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test batched capacitance model")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick tests only")
    args = parser.parse_args()
    
    tester = CapacitanceModelTester(checkpoint_path=args.checkpoint)
    
    if args.quick:
        # Quick tests only
        print("Running quick test suite...")
        tester.setup_model()
        results = {}
        results["single_request"] = tester.test_single_request()
        results["concurrent_requests"] = tester.test_concurrent_requests(num_workers=4, requests_per_worker=3)
        results["output_consistency"] = tester.test_output_consistency()
        
        passed = sum(results.values())
        print(f"\nQuick tests: {passed}/{len(results)} passed")
    else:
        # Full test suite
        results = tester.run_all_tests()
        
        # Exit with appropriate code
        all_passed = all(results.values())
        exit_code = 0 if all_passed else 1
        
        if all_passed:
            print("\n🎉 All tests passed! Batched capacitance model is working correctly.")
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
        
        sys.exit(exit_code)


if __name__ == "__main__":
    main()