#!/usr/bin/env python3
"""
Minimal test to isolate the concurrency issue in batched capacitance model.
"""

import os
import sys
import torch
import numpy as np
import threading
import time
import concurrent.futures
from pathlib import Path

# Add Swarm directory to path
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent
sys.path.append(str(swarm_dir))
os.environ['SWARM_PROJECT_ROOT'] = str(swarm_dir)

from train import setup_capacitance_model


### cudnn race condition debugging
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
###

def ptrs(t):
    return (t.data_ptr(), t.storage().data_ptr() if t.storage() is not None else None)


def create_fixed_input(device) -> torch.Tensor:
    """Create a completely fixed input tensor."""
    # Fixed pattern - no randomness
    input_tensor = torch.ones(3, 1, 224, 224) * 0.5  # All 0.5
    return input_tensor.to(device)


def test_concurrent_identical_calls():
    """Test concurrent identical calls to isolate the race condition."""
    print("Setting up batched model...")
    model = setup_capacitance_model("artifacts/best_model.pth", batch_window_ms=100, max_batch_size=16)
    
    # Create fixed input
    fixed_input = create_fixed_input(model.device)
    print(f"Fixed input checksum: {fixed_input.sum().item()}")

    # model.to('cpu')
    # fixed_input = fixed_input.cpu()
    model.eval()
    
    # Get reference result (single threaded)
    print("Getting reference result...")
    ref_values, ref_log_vars = model(fixed_input)
    print(f"Reference values checksum: {ref_values.sum().item()}")
    print(f"Reference log_vars checksum: {ref_log_vars.sum().item()}")
    
    # Test concurrent identical calls
    print("\nTesting concurrent identical calls...")
    
    def concurrent_call(call_id):
        try:
            # Use the same fixed input tensor
            with torch.no_grad():
                values, log_vars = model(fixed_input.clone())
                torch.cuda.synchronize()
                print("values ptrs:", ptrs(values), "log_vars ptrs:", ptrs(log_vars))
            return {
                'call_id': call_id,
                'values_checksum': values.sum().item(),
                'log_vars_checksum': log_vars.sum().item(),
                'values_shape': values.shape,
                'log_vars_shape': log_vars.shape,
                'success': True,
                'values': values.detach().cpu(),
                'log_vars': log_vars.detach().cpu()
            }
        except Exception as e:
            return {
                'call_id': call_id,
                'success': False,
                'error': str(e)
            }
    
    # Run concurrent calls
    num_concurrent = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(concurrent_call, i) for i in range(num_concurrent)]
        results = [future.result() for future in futures]
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\nResults: {len(successful_results)} successful, {len(failed_results)} failed")
    
    if failed_results:
        print("Failed calls:")
        for result in failed_results:
            print(f"  Call {result['call_id']}: {result['error']}")
    
    if successful_results:
        # Check consistency
        ref_values_checksum = ref_values.sum().item()
        ref_log_vars_checksum = ref_log_vars.sum().item()
        
        print(f"\nConsistency check (reference checksums: values={ref_values_checksum:.6f}, log_vars={ref_log_vars_checksum:.6f}):")
        
        all_consistent = True
        for result in successful_results:
            values_match = abs(result['values_checksum'] - ref_values_checksum) < 1e-4
            log_vars_match = abs(result['log_vars_checksum'] - ref_log_vars_checksum) < 1e-4
            
            status_v = "✓" if values_match else "✗"
            status_l = "✓" if log_vars_match else "✗"
            
            print(f"  Call {result['call_id']}: {status_v} values ({result['values_checksum']:.6f}) "
                  f"{status_l} log_vars ({result['log_vars_checksum']:.6f})")
            
            if not (values_match and log_vars_match):
                all_consistent = False
                
                # Detailed comparison
                detailed_values_diff = torch.abs(result['values'] - ref_values.cpu()).max().item()
                detailed_log_vars_diff = torch.abs(result['log_vars'] - ref_log_vars.cpu()).max().item()
                print(f"    Max values diff: {detailed_values_diff:.8f}")
                print(f"    Max log_vars diff: {detailed_log_vars_diff:.8f}")
        
        if all_consistent:
            print("\n✓ All concurrent calls produced consistent results!")
        else:
            print("\n✗ Inconsistent results detected - there's a race condition!")
    
    # Cleanup
    if hasattr(model, 'close'):
        model.close()
    
    return all_consistent


def test_batching_timing():
    """Test to see how batching timing affects results."""
    print("\n" + "="*50)
    print("TESTING BATCHING TIMING EFFECTS")
    print("="*50)
    
    # Test with different batch window settings
    window_configs = [
        (1, 4),      # Very fast batching
        (50, 16),    # Medium batching  
        (200, 32),   # Slow batching
    ]
    
    for batch_window_ms, max_batch_size in window_configs:
        print(f"\nTesting with batch_window_ms={batch_window_ms}, max_batch_size={max_batch_size}")
        
        model = setup_capacitance_model(
            "artifacts/best_model.pth", 
            batch_window_ms=batch_window_ms, 
            max_batch_size=max_batch_size
        )
        
        fixed_input = create_fixed_input(model.device)
        
        # Test concurrent calls with this configuration
        def quick_call(call_id):
            values, log_vars = model(fixed_input.clone())
            return values.sum().item(), log_vars.sum().item()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(quick_call, i) for i in range(12)]
            checksums = [future.result() for future in futures]
        
        # Check if all checksums are identical
        values_checksums = [c[0] for c in checksums]
        log_vars_checksums = [c[1] for c in checksums]
        
        values_consistent = all(abs(c - values_checksums[0]) < 1e-4 for c in values_checksums)
        log_vars_consistent = all(abs(c - log_vars_checksums[0]) < 1e-4 for c in log_vars_checksums)
        
        status = "✓" if (values_consistent and log_vars_consistent) else "✗"
        print(f"  {status} Consistency: values={values_consistent}, log_vars={log_vars_consistent}")
        
        if not values_consistent:
            print(f"    Values range: {min(values_checksums):.6f} to {max(values_checksums):.6f}")
        if not log_vars_consistent:
            print(f"    Log vars range: {min(log_vars_checksums):.6f} to {max(log_vars_checksums):.6f}")
        
        # Cleanup
        if hasattr(model, 'close'):
            model.close()


if __name__ == "__main__":
    print("MINIMAL CONCURRENCY TEST FOR BATCHED CAPACITANCE MODEL")
    print("="*60)
    
    try:
        # Test basic concurrency
        consistent = test_concurrent_identical_calls()
        
        # Test timing effects
        test_batching_timing()
        
        if consistent:
            print("\n🎉 Concurrency test PASSED!")
        else:
            print("\n❌ Concurrency test FAILED - race condition detected!")
            
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()