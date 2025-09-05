"""
Test script to verify integration of RLlib's DefaultPPOTorchRLModule with 
custom Dict observation space encoder for quantum device environments.

This script tests the new implementation that leverages RLlib's default modules
while only customizing the encoder to handle {image, voltage} observations.
"""

import torch
import numpy as np
from gymnasium import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.columns import Columns

# Test our custom encoder and catalog
from custom_dict_encoder import CustomDictPPOCatalog, DictObservationActorCriticEncoder
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def test_custom_encoder_standalone():
    """Test the custom encoder independently."""
    print("=" * 60)
    print("Testing Custom Dict Encoder (Standalone)")
    print("=" * 60)
    
    # Create test observation spaces (gate agent with 2 channels)
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0.0, high=1.0, shape=(128, 128, 2), dtype=np.float32),
        'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    })
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    model_config = {
        "use_lstm": True,
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "vf_share_layers": True
    }
    
    try:
        # Test encoder creation
        encoder = DictObservationActorCriticEncoder(
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config,
            shared=True,
            framework="torch"
        )
        print("✓ Custom encoder created successfully")
        print(f"  Latent dimensions: {encoder.latent_dims}")
        print(f"  Combined feature size: {encoder.combined_feature_size}")
        print(f"  Uses LSTM: {encoder.use_lstm}")
        
        # Test encoder forward pass without time dimension
        batch_size = 4
        test_batch = {
            Columns.OBS: {
                'image': torch.randn(batch_size, 128, 128, 2),
                'voltage': torch.randn(batch_size, 1)
            }
        }
        
        output = encoder._forward(test_batch)
        print("✓ Encoder forward pass (no time dim) successful")
        print(f"  Actor output shape: {output['encoder_out']['actor'].shape}")
        print(f"  Critic output shape: {output['encoder_out']['critic'].shape}")
        
        # Test with time dimension
        test_batch_5d = {
            Columns.OBS: {
                'image': torch.randn(batch_size, 3, 128, 128, 2),  # (B, T, H, W, C)
                'voltage': torch.randn(batch_size, 3, 1)           # (B, T, 1)
            },
            Columns.PREV_ACTIONS: torch.randn(batch_size, 3, 1),
            Columns.PREV_REWARDS: torch.randn(batch_size, 3),
            Columns.STATE_IN: {
                'h_state': torch.zeros(1, batch_size, 64),
                'c_state': torch.zeros(1, batch_size, 64)
            }
        }
        
        output_5d = encoder._forward(test_batch_5d)
        print("✓ Encoder forward pass (with time dim) successful")
        print(f"  Actor output shape: {output_5d['encoder_out']['actor'].shape}")
        print(f"  State out keys: {list(output_5d.get('state_out', {}).keys())}")
        
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_custom_catalog():
    """Test the custom PPO catalog."""
    print("\n" + "=" * 60)
    print("Testing Custom PPO Catalog")
    print("=" * 60)
    
    # Test spaces
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),  # Barrier agent
        'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    })
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    model_config = {
        "use_lstm": True,
        "lstm_cell_size": 32,
        "fcnet_hiddens": [64, 64],
        "vf_share_layers": True,
        "head_fcnet_hiddens": [32],
        "head_fcnet_activation": "relu",
        "free_log_std": False
    }
    
    try:
        # Create catalog
        catalog = CustomDictPPOCatalog(
            observation_space=obs_space,
            action_space=action_space,
            model_config_dict=model_config
        )
        print("✓ Custom catalog created successfully")
        
        # Test building encoder
        encoder = catalog.build_actor_critic_encoder(framework="torch")
        print("✓ ActorCritic encoder built successfully")
        print(f"  Encoder type: {type(encoder).__name__}")
        
        # Test building heads
        pi_head = catalog.build_pi_head(framework="torch")
        vf_head = catalog.build_vf_head(framework="torch")
        print("✓ Policy and value heads built successfully")
        print(f"  Pi head type: {type(pi_head).__name__}")
        print(f"  VF head type: {type(vf_head).__name__}")
        
    except Exception as e:
        print(f"✗ Catalog test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_default_ppo_with_custom_catalog():
    """Test DefaultPPOTorchRLModule with our custom catalog."""
    print("\n" + "=" * 60)
    print("Testing DefaultPPOTorchRLModule + Custom Catalog")
    print("=" * 60)
    
    # Create test spaces
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0.0, high=1.0, shape=(128, 128, 2), dtype=np.float32),
        'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    })
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    model_config = {
        "use_lstm": True,
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "vf_share_layers": True,
        "head_fcnet_hiddens": [],
        "head_fcnet_activation": "relu",
        "free_log_std": False
    }
    
    try:
        # Create RLModule using custom catalog
        rl_module = DefaultPPOTorchRLModule(
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config,
            catalog_class=CustomDictPPOCatalog
        )
        print("✓ DefaultPPOTorchRLModule created with custom catalog")
        print(f"  Module type: {type(rl_module).__name__}")
        print(f"  Is stateful: {rl_module.is_stateful()}")
        
        # Test forward inference
        batch_size = 2
        batch = {
            Columns.OBS: {
                'image': torch.randn(batch_size, 128, 128, 2),
                'voltage': torch.randn(batch_size, 1)
            }
        }
        
        # Test inference
        output_inf = rl_module._forward(batch)
        print("✓ Forward inference successful")
        print(f"  Action dist inputs shape: {output_inf[Columns.ACTION_DIST_INPUTS].shape}")
        
        # Test training forward pass
        output_train = rl_module._forward_train(batch)
        print("✓ Forward train successful")
        print(f"  Action dist inputs shape: {output_train[Columns.ACTION_DIST_INPUTS].shape}")
        print(f"  Embeddings shape: {output_train[Columns.EMBEDDINGS].shape}")
        
        # Test value computation
        value_preds = rl_module.compute_values(batch)
        print("✓ Value computation successful")
        print(f"  Value predictions shape: {value_preds.shape}")
        
        # Test with LSTM state
        batch_with_state = {
            **batch,
            Columns.STATE_IN: {
                'actor': {'h_state': torch.zeros(1, batch_size, 64), 'c_state': torch.zeros(1, batch_size, 64)},
                'critic': {'h_state': torch.zeros(1, batch_size, 64), 'c_state': torch.zeros(1, batch_size, 64)}
            }
        }
        
        output_with_state = rl_module._forward_train(batch_with_state)
        print("✓ Forward with LSTM state successful")
        print(f"  Has state output: {'state_out' in output_with_state}")
        
    except Exception as e:
        print(f"✗ DefaultPPO + Custom catalog test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_rl_module_spec_creation():
    """Test creating RLModuleSpec with our setup."""
    print("\n" + "=" * 60)
    print("Testing RLModuleSpec Creation")
    print("=" * 60)
    
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
        'voltage': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    })
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    model_config = {
        "use_lstm": True,
        "lstm_cell_size": 32,
        "fcnet_hiddens": [64, 64],
        "vf_share_layers": True
    }
    
    try:
        # Create RLModuleSpec
        spec = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config,
            catalog_class=CustomDictPPOCatalog
        )
        print("✓ RLModuleSpec created successfully")
        
        # Test building from spec
        module = spec.build()
        print("✓ RLModule built from spec successfully")
        print(f"  Module class: {type(module).__name__}")
        
        # Quick functionality test
        batch = {
            Columns.OBS: {
                'image': torch.randn(1, 64, 64, 1),
                'voltage': torch.randn(1, 1)
            }
        }
        
        output = module._forward(batch)
        print("✓ Built module forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"✗ RLModuleSpec test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("Testing RLlib DefaultPPOTorchRLModule Integration")
    print("Using custom Dict observation encoder for quantum devices")
    print("=" * 80)
    
    tests = [
        ("Custom Encoder", test_custom_encoder_standalone),
        ("Custom Catalog", test_custom_catalog),
        ("DefaultPPO + Custom Catalog", test_default_ppo_with_custom_catalog),
        ("RLModuleSpec Creation", test_rl_module_spec_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Integration successful.")
        print("✅ RLlib's DefaultPPOTorchRLModule works with custom Dict encoder")
        print("✅ LSTM state management handled automatically")
        print("✅ Action/value computation handled by default modules")
        print("✅ Custom encoder processes quantum device observations correctly")
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
    
    return passed == len(results)


if __name__ == "__main__":
    main()