#!/usr/bin/env python3
"""
Debug script to identify the thread safety issue in batched capacitance model.
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


def create_deterministic_input(seed: int = 42) -> torch.Tensor:
    """Create deterministic test input for consistency testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create 3-channel input (for 4-dot system)
    input_tensor = torch.randn(3, 1, 224, 224)
    return input_tensor


def test_direct_model_consistency():
    """Test if the underlying model itself has consistency issues."""
    print("=== Testing Direct Model Consistency ===")
    
    # Setup model without batching wrapper
    try:
        from CapacitanceModel.CapacitancePrediction import CapacitancePredictionModel
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.insert(0, parent_dir)
        from CapacitanceModel.CapacitancePrediction import CapacitancePredictionModel
    
    # Load model directly
    swarm_dir = os.environ['SWARM_PROJECT_ROOT']
    weights_path = os.path.join(swarm_dir, 'CapacitanceModel', 'artifacts', 'best_model.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CapacitancePredictionModel()
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Test direct model consistency
    test_input = create_deterministic_input().to(device)
    
    with torch.no_grad():
        ref_values, ref_log_vars = model(test_input)
        
        # Test multiple calls
        for i in range(5):
            values, log_vars = model(test_input.clone())
            
            if torch.allclose(values, ref_values, atol=1e-6):
                print(f"✓ Direct model call {i}: values consistent")
            else:
                print(f"✗ Direct model call {i}: values inconsistent!")
                print(f"  Max diff: {(values - ref_values).abs().max().item()}")
                
            if torch.allclose(log_vars, ref_log_vars, atol=1e-6):
                print(f"✓ Direct model call {i}: log_vars consistent")
            else:
                print(f"✗ Direct model call {i}: log_vars inconsistent!")
                print(f"  Max diff: {(log_vars - ref_log_vars).abs().max().item()}")
    
    return model, test_input, ref_values, ref_log_vars


def test_batching_wrapper_issue():
    """Test the batching wrapper specifically."""
    print("\n=== Testing Batching Wrapper Issue ===")
    
    # Get reference from direct model
    direct_model, test_input, ref_values, ref_log_vars = test_direct_model_consistency()
    
    # Setup batched model
    batched_model = setup_capacitance_model("artifacts/best_model.pth", batch_window_ms=50, max_batch_size=8)
    
    print(f"\nDirect model device: {next(direct_model.parameters()).device}")
    print(f"Batched model device: {batched_model.device}")
    
    # Test single call through batched wrapper
    print("\nTesting single call through batched wrapper...")
    batched_values, batched_log_vars = batched_model(test_input)
    
    if torch.allclose(batched_values, ref_values, atol=1e-5):
        print("✓ Batched wrapper single call: values match direct model")
    else:
        print("✗ Batched wrapper single call: values don't match!")
        print(f"  Max diff: {(batched_values - ref_values).abs().max().item()}")
    
    if torch.allclose(batched_log_vars, ref_log_vars, atol=1e-5):
        print("✓ Batched wrapper single call: log_vars match direct model")
    else:
        print("✗ Batched wrapper single call: log_vars don't match!")
        print(f"  Max diff: {(batched_log_vars - ref_log_vars).abs().max().item()}")
    
    # Test multiple sequential calls through batched wrapper
    print("\nTesting sequential calls through batched wrapper...")
    for i in range(3):
        seq_values, seq_log_vars = batched_model(test_input.clone())
        
        if torch.allclose(seq_values, ref_values, atol=1e-5):
            print(f"✓ Sequential call {i}: values consistent")
        else:
            print(f"✗ Sequential call {i}: values inconsistent!")
            print(f"  Max values diff: {(seq_values - ref_values).abs().max().item()}")
            
        if torch.allclose(seq_log_vars, ref_log_vars, atol=1e-5):
            print(f"✓ Sequential call {i}: log_vars consistent")
        else:
            print(f"✗ Sequential call {i}: log_vars inconsistent!")
            print(f"  Max log_vars diff: {(seq_log_vars - ref_log_vars).abs().max().item()}")
    
    # Test concurrent calls
    print("\nTesting concurrent calls through batched wrapper...")
    def concurrent_call(call_id):
        return call_id, batched_model(test_input.clone())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(concurrent_call, i) for i in range(8)]
        concurrent_results = [future.result() for future in futures]
    
    for call_id, (conc_values, conc_log_vars) in concurrent_results:
        if torch.allclose(conc_values, ref_values, atol=1e-5):
            print(f"✓ Concurrent call {call_id}: values consistent")
        else:
            print(f"✗ Concurrent call {call_id}: values inconsistent!")
            print(f"  Max values diff: {(conc_values - ref_values).abs().max().item()}")
            
        if torch.allclose(conc_log_vars, ref_log_vars, atol=1e-5):
            print(f"✓ Concurrent call {call_id}: log_vars consistent")
        else:
            print(f"✗ Concurrent call {call_id}: log_vars inconsistent!")
            print(f"  Max log_vars diff: {(conc_log_vars - ref_log_vars).abs().max().item()}")
    
    # Cleanup
    if hasattr(batched_model, 'close'):
        batched_model.close()


def test_tensor_cloning_issue():
    """Test if tensor cloning is causing issues."""
    print("\n=== Testing Tensor Cloning Issues ===")
    
    test_input = create_deterministic_input()
    
    # Test different cloning strategies
    clone1 = test_input.clone()
    clone2 = test_input.clone().detach()
    
    print(f"Original: {test_input.sum().item():.6f}")
    print(f"Clone: {clone1.sum().item():.6f}")
    print(f"Clone+detach: {clone2.sum().item():.6f}")
    
    print(f"Are clones equal? {torch.equal(test_input, clone1)}")
    print(f"Are clone+detach equal? {torch.equal(test_input, clone2)}")


def main():
    """Main debug function."""
    print("DEBUGGING BATCHED CAPACITANCE MODEL THREAD SAFETY")
    print("=" * 60)
    
    try:
        # Test tensor cloning first
        test_tensor_cloning_issue()
        
        # Test direct model consistency
        test_direct_model_consistency()
        
        # Test batching wrapper
        test_batching_wrapper_issue()
        
    except Exception as e:
        print(f"Debug test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()