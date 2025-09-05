#!/usr/bin/env python3
"""
Comprehensive test script for SingleAgentRecurrentPPOModel.
Tests all forward methods, state handling, and input/output shapes.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from gymnasium import spaces
from ray.rllib.policy.sample_batch import SampleBatch

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ppo_trainer_recurrent import SingleAgentRecurrentPPOModel


def create_test_spaces():
    """Create test observation and action spaces."""
    # Gate agent: 2-channel image
    gate_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(64, 64, 2),  # 2 channels for gate agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    # Barrier agent: 1-channel image
    barrier_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(64, 64, 1),  # 1 channel for barrier agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    action_space = spaces.Box(
        low=-1.0, high=1.0,
        shape=(1,),  # Single voltage output
        dtype=np.float32
    )
    
    return gate_obs_space, barrier_obs_space, action_space


def create_test_batch(obs_space, batch_size=4, include_states=True, include_prev=True):
    """Create a test batch for forward pass."""
    # Create observations
    if 'image' in obs_space.spaces:
        image_shape = obs_space.spaces['image'].shape  # (H, W, C)
        image = torch.randn(batch_size, *image_shape)
        voltage = torch.randn(batch_size, 1)
        
        obs = {
            'image': image,
            'voltage': voltage
        }
    else:
        obs = obs_space.sample()
        obs = {k: torch.tensor(v).unsqueeze(0).repeat(batch_size, *([1] * len(v.shape))) 
               for k, v in obs.items()}
    
    batch = {
        SampleBatch.OBS: obs
    }
    
    # Add LSTM states if requested
    if include_states:
        batch["state_in"] = {
            "h_state": torch.zeros(1, batch_size, 64),  # lstm_cell_size = 64
            "c_state": torch.zeros(1, batch_size, 64)
        }
    
    # Add previous actions and rewards if requested
    if include_prev:
        batch[SampleBatch.PREV_ACTIONS] = torch.randn(batch_size, 1)
        batch[SampleBatch.PREV_REWARDS] = torch.randn(batch_size)
    
    return batch


def test_model_creation():
    """Test model creation with different configurations."""
    print("=== Testing Model Creation ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    try:
        # Test gate agent model (2-channel)
        gate_model = SingleAgentRecurrentPPOModel(
            observation_space=gate_obs_space,
            action_space=action_space,
            model_config=model_config
        )
        print("✓ Gate agent model created successfully")
        print(f"  Image channels: {gate_model.image_channels}")
        print(f"  Image input shape: {gate_model.image_input_shape}")
        
        # Test barrier agent model (1-channel)
        barrier_model = SingleAgentRecurrentPPOModel(
            observation_space=barrier_obs_space,
            action_space=action_space,
            model_config=model_config
        )
        print("✓ Barrier agent model created successfully")
        print(f"  Image channels: {barrier_model.image_channels}")
        print(f"  Image input shape: {barrier_model.image_input_shape}")
        
        return gate_model, barrier_model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_input_shapes():
    """Test various input tensor shapes."""
    print("\n=== Testing Input Shapes ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    gate_model = SingleAgentRecurrentPPOModel(
        observation_space=gate_obs_space,
        action_space=action_space,
        model_config=model_config
    )
    
    # Test different batch sizes and tensor shapes
    test_cases = [
        ("Small batch", 1),
        ("Medium batch", 4),
        ("Large batch", 16),
    ]
    
    for test_name, batch_size in test_cases:
        try:
            print(f"\nTesting {test_name} (batch_size={batch_size})")
            
            # Create batch
            batch = create_test_batch(gate_obs_space, batch_size)
            
            # Check input shapes
            image = batch[SampleBatch.OBS]['image']
            voltage = batch[SampleBatch.OBS]['voltage']
            
            print(f"  Input image shape: {image.shape}")
            print(f"  Input voltage shape: {voltage.shape}")
            
            # Test forward pass
            output = gate_model._forward_train(batch)
            
            print(f"  Output action logits shape: {output[SampleBatch.ACTION_DIST_INPUTS].shape}")
            print(f"  Output value shape: {output[SampleBatch.VF_PREDS].shape}")
            
            # Check state output
            if 'state_out' in output:
                h_state = output['state_out']['h_state']
                c_state = output['state_out']['c_state']
                print(f"  Output h_state shape: {h_state.shape}")
                print(f"  Output c_state shape: {c_state.shape}")
            
            print("  ✓ Forward pass successful")
            
        except Exception as e:
            print(f"  ✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()


def test_edge_cases():
    """Test edge cases and potential failure modes."""
    print("\n=== Testing Edge Cases ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    gate_model = SingleAgentRecurrentPPOModel(
        observation_space=gate_obs_space,
        action_space=action_space,
        model_config=model_config
    )
    
    test_cases = [
        ("No previous states", False, True),
        ("No previous actions/rewards", True, False),
        ("Minimal batch", True, True),
        ("Different image shapes", True, True),
    ]
    
    for test_name, include_states, include_prev in test_cases:
        try:
            print(f"\nTesting {test_name}")
            
            if test_name == "Different image shapes":
                # Test with potentially problematic shapes
                batch_size = 2
                
                # Create batch with potentially 5D tensor (common RLlib issue)
                obs = {
                    'image': torch.randn(batch_size, 1, 64, 64, 2),  # 5D tensor
                    'voltage': torch.randn(batch_size, 1)
                }
                
                batch = {SampleBatch.OBS: obs}
                
                # Add states
                if include_states:
                    batch["state_in"] = {
                        "h_state": torch.zeros(1, batch_size, 64),
                        "c_state": torch.zeros(1, batch_size, 64)
                    }
                
                print(f"  Input image shape: {obs['image'].shape}")
                
                # This should fail with current implementation
                try:
                    output = gate_model._forward_train(batch)
                    print("  ✓ 5D tensor handled successfully")
                except Exception as e:
                    print(f"  ⚠ 5D tensor failed as expected: {e}")
                    
                    # Test fix: squeeze the tensor
                    obs['image'] = obs['image'].squeeze(1)  # Remove extra dimension
                    print(f"  Fixed image shape: {obs['image'].shape}")
                    output = gate_model._forward_train(batch)
                    print("  ✓ Fixed tensor works")
                
            else:
                batch = create_test_batch(gate_obs_space, 2, include_states, include_prev)
                output = gate_model._forward_train(batch)
                print("  ✓ Forward pass successful")
            
        except Exception as e:
            print(f"  ✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()


def test_forward_methods():
    """Test all forward methods."""
    print("\n=== Testing Forward Methods ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    gate_model = SingleAgentRecurrentPPOModel(
        observation_space=gate_obs_space,
        action_space=action_space,
        model_config=model_config
    )
    
    batch = create_test_batch(gate_obs_space, 4)
    
    methods = [
        ("_forward_train", gate_model._forward_train),
        ("_forward_inference", gate_model._forward_inference),
        ("_forward_exploration", gate_model._forward_exploration),
    ]
    
    for method_name, method in methods:
        try:
            print(f"\nTesting {method_name}")
            
            if method_name == "_forward_exploration":
                # Test with t parameter
                output = method(batch, t=1000)
            else:
                output = method(batch)
            
            print(f"  Output keys: {list(output.keys())}")
            print(f"  Action logits shape: {output[SampleBatch.ACTION_DIST_INPUTS].shape}")
            print(f"  Value shape: {output[SampleBatch.VF_PREDS].shape}")
            print(f"  ✓ {method_name} successful")
            
        except Exception as e:
            print(f"  ✗ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()


def test_state_management():
    """Test LSTM state management."""
    print("\n=== Testing State Management ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    gate_model = SingleAgentRecurrentPPOModel(
        observation_space=gate_obs_space,
        action_space=action_space,
        model_config=model_config
    )
    
    # Test initial state
    try:
        initial_state = gate_model.get_initial_state()
        print(f"Initial state keys: {list(initial_state.keys())}")
        print(f"Initial h_state shape: {initial_state['h_state'].shape}")
        print(f"Initial c_state shape: {initial_state['c_state'].shape}")
        print("✓ Initial state generation successful")
    except Exception as e:
        print(f"✗ Initial state failed: {e}")
    
    # Test state continuity
    try:
        batch_size = 4
        batch1 = create_test_batch(gate_obs_space, batch_size)
        
        # First forward pass
        output1 = gate_model._forward_train(batch1)
        state_out_1 = output1['state_out']
        
        # Second forward pass with state from first
        batch2 = create_test_batch(gate_obs_space, batch_size)
        batch2['state_in'] = state_out_1
        output2 = gate_model._forward_train(batch2)
        
        print("✓ State continuity test successful")
        
        # Check if states changed
        h1 = state_out_1['h_state']
        h2 = output2['state_out']['h_state']
        
        if not torch.equal(h1, h2):
            print("✓ States properly updated between steps")
        else:
            print("⚠ States did not change (might be expected)")
            
    except Exception as e:
        print(f"✗ State continuity test failed: {e}")
        import traceback
        traceback.print_exc()


def test_both_agent_types():
    """Test both gate and barrier agent models."""
    print("\n=== Testing Both Agent Types ===")
    
    gate_obs_space, barrier_obs_space, action_space = create_test_spaces()
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    try:
        # Gate model (2-channel)
        gate_model = SingleAgentRecurrentPPOModel(
            observation_space=gate_obs_space,
            action_space=action_space,
            model_config=model_config
        )
        
        gate_batch = create_test_batch(gate_obs_space, 4)
        gate_output = gate_model._forward_train(gate_batch)
        
        print("✓ Gate model forward pass successful")
        print(f"  Gate image channels: {gate_model.image_channels}")
        
        # Barrier model (1-channel)
        barrier_model = SingleAgentRecurrentPPOModel(
            observation_space=barrier_obs_space,
            action_space=action_space,
            model_config=model_config
        )
        
        barrier_batch = create_test_batch(barrier_obs_space, 4)
        barrier_output = barrier_model._forward_train(barrier_batch)
        
        print("✓ Barrier model forward pass successful")
        print(f"  Barrier image channels: {barrier_model.image_channels}")
        
        # Compare outputs
        print(f"  Gate action logits shape: {gate_output[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Barrier action logits shape: {barrier_output[SampleBatch.ACTION_DIST_INPUTS].shape}")
        
    except Exception as e:
        print(f"✗ Both agent types test failed: {e}")
        import traceback
        traceback.print_exc()


def run_diagnostic_tests():
    """Run diagnostic tests to identify common issues."""
    print("\n=== Running Diagnostic Tests ===")
    
    # Test tensor shapes that commonly cause issues
    print("\nTesting problematic tensor shapes:")
    
    shapes_to_test = [
        ("4D Normal", (4, 64, 64, 2)),
        ("5D Problematic", (4, 1, 64, 64, 2)),
        ("3D Missing batch", (64, 64, 2)),
        ("6D Extra dims", (1, 4, 1, 64, 64, 2)),
    ]
    
    for shape_name, shape in shapes_to_test:
        try:
            print(f"  {shape_name}: {shape}")
            tensor = torch.randn(*shape)
            
            # Test permutation
            if len(shape) == 4:
                # Should work
                permuted = tensor.permute(0, 3, 1, 2)
                print(f"    ✓ Permute successful: {permuted.shape}")
            elif len(shape) == 5:
                # Common issue - need to squeeze first
                print(f"    ⚠ 5D tensor detected")
                squeezed = tensor.squeeze(1)  # Remove extra dimension
                permuted = squeezed.permute(0, 3, 1, 2)
                print(f"    ✓ After squeeze and permute: {permuted.shape}")
            else:
                print(f"    ✗ Unsupported shape: {len(shape)} dimensions")
                
        except Exception as e:
            print(f"    ✗ Failed: {e}")


def main():
    """Run all tests."""
    print("Starting comprehensive test of SingleAgentRecurrentPPOModel")
    print("=" * 60)
    
    # Basic model creation
    gate_model, barrier_model = test_model_creation()
    if gate_model is None:
        print("Model creation failed, skipping other tests")
        return
    
    # Test input shapes
    test_input_shapes()
    
    # Test edge cases
    test_edge_cases()
    
    # Test forward methods
    test_forward_methods()
    
    # Test state management
    test_state_management()
    
    # Test both agent types
    test_both_agent_types()
    
    # Run diagnostics
    run_diagnostic_tests()
    
    print("\n" + "=" * 60)
    print("Test completed. Check output above for any failures.")


if __name__ == "__main__":
    main()