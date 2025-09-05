#!/usr/bin/env python3
"""
Comprehensive test script for Ray/RLlib compatibility with multi-agent quantum environment.
Tests the specific configuration issues that cause EnvRunner failures.
"""

import os
import sys
import argparse
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import SingleAgentRLModuleSpec, MultiAgentRLModuleSpec
import time

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import components from train.py
from train import (
    setup_environment, 
    import_environment, 
    log_memory_usage, 
    force_garbage_collect
)

def test_environment_compatibility():
    """Test environment compatibility with RLlib expectations."""
    print("="*60)
    print("TESTING: Environment-RLlib Compatibility")
    print("="*60)
    
    try:
        log_memory_usage("before_env_compat_test")
        
        # Setup environment
        setup_environment()
        env_factory = import_environment()
        
        # Create environment with minimal dots
        num_dots = 2
        env = env_factory(num_quantum_dots=num_dots)
        log_memory_usage("after_env_creation")
        
        print(f"✓ Environment created with {num_dots} quantum dots")
        
        # Test Gymnasium/RLlib compatibility
        print("Testing Gymnasium compatibility...")
        
        # Check if environment has required attributes
        required_attrs = ['observation_space', 'action_space', 'reset', 'step', 'close']
        missing_attrs = []
        for attr in required_attrs:
            if hasattr(env, attr):
                print(f"  ✓ Has {attr}")
            else:
                print(f"  ✗ Missing {attr}")
                missing_attrs.append(attr)
        
        # Check multi-agent specific attributes
        ma_attrs = ['get_agent_ids', 'observation_spaces', 'action_spaces']
        missing_ma_attrs = []
        for attr in ma_attrs:
            if hasattr(env, attr):
                print(f"  ✓ Has multi-agent {attr}")
            else:
                print(f"  ✗ Missing multi-agent {attr}")
                missing_ma_attrs.append(attr)
        
        if missing_attrs or missing_ma_attrs:
            print(f"SUMMARY: Missing {len(missing_attrs)} required attrs, {len(missing_ma_attrs)} multi-agent attrs")
            return False
        
        # Test agent IDs
        agent_ids = env.get_agent_ids()
        print(f"  ✓ Agent IDs: {agent_ids}")
        print(f"  ✓ Number of agents: {len(agent_ids)}")
        
        # Test observation spaces
        for agent_id in agent_ids:
            obs_space = env.observation_space[agent_id]
            action_space = env.action_space[agent_id]
            print(f"  ✓ {agent_id}:")
            print(f"    Obs space: {obs_space}")
            print(f"    Action space: {action_space}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"  ✓ Reset successful, got {len(obs)} observations")
        
        # Test step with random actions
        actions = {}
        for agent_id in agent_ids:
            actions[agent_id] = env.action_space[agent_id].sample()
        
        obs_next, rewards, terminated, truncated, info = env.step(actions)
        print(f"  ✓ Step successful, got {len(rewards)} rewards")
        
        env.close()
        log_memory_usage("after_env_compat_test")
        return True
        
    except Exception as e:
        print(f"✗ Environment compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_specs_creation():
    """Test policy specs creation for multi-agent setup."""
    print("="*60)
    print("TESTING: Policy Specs Creation")
    print("="*60)
    
    try:
        from utils.policy_mapping import create_rl_module_spec, get_policy_mapping_fn, get_policies_to_train
        
        log_memory_usage("before_policy_specs")
        
        # Create environment
        setup_environment()
        env_factory = import_environment()
        env = env_factory(num_quantum_dots=2)
        
        # Test policy specs creation
        print("Creating policy specifications...")
        rl_module_spec = create_rl_module_spec(env)
        print(f"✓ Created RLModule spec with {len(rl_module_spec.rl_module_specs)} modules")
        
        # Validate RLModule specs structure
        for policy_id, single_spec in rl_module_spec.rl_module_specs.items():
            print(f"  RLModule '{policy_id}':")
            print(f"    Type: {type(single_spec)}")
            print(f"    Is SingleAgentRLModuleSpec: {isinstance(single_spec, SingleAgentRLModuleSpec)}")
            
            if hasattr(single_spec, 'observation_space'):
                print(f"    Obs space: {single_spec.observation_space}")
            if hasattr(single_spec, 'action_space'):
                print(f"    Action space: {single_spec.action_space}")
            if hasattr(single_spec, 'model_config_dict'):
                print(f"    Model config keys: {list(single_spec.model_config_dict.keys()) if single_spec.model_config_dict else 'None'}")
        
        # Test policy mapping function
        print("\nTesting policy mapping function...")
        policy_mapping_fn = get_policy_mapping_fn(2)
        
        mapping_errors = []
        for agent_id in env.get_agent_ids():
            mapped_policy = policy_mapping_fn(agent_id, None, None)
            print(f"  Agent '{agent_id}' -> Policy '{mapped_policy}'")
            
            if mapped_policy not in rl_module_spec.rl_module_specs:
                print(f"  ✗ ERROR: Policy '{mapped_policy}' not found in RLModule specs")
                mapping_errors.append((agent_id, mapped_policy))
            else:
                print(f"  ✓ Policy mapping valid")
        
        if mapping_errors:
            print(f"SUMMARY: {len(mapping_errors)} policy mapping errors")
            return False
        
        # Test policies to train
        policies_to_train = get_policies_to_train()
        print(f"\nPolicies to train: {policies_to_train}")
        
        env.close()
        log_memory_usage("after_policy_specs")
        return True
        
    except Exception as e:
        print(f"✗ Policy specs creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_config_creation():
    """Test PPO configuration creation with multi-agent setup."""
    print("="*60)
    print("TESTING: PPO Configuration Creation")
    print("="*60)
    
    try:
        from utils.policy_mapping import create_rl_module_spec, get_policy_mapping_fn, get_policies_to_train
        from utils.config_loader import load_config_from_file
        
        log_memory_usage("before_ppo_config")
        
        # Load config
        config_path = current_dir / "configs" / "config.yaml"
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        config = load_config_from_file(str(config_path))
        
        # Create environment and policies
        setup_environment()
        env_factory = import_environment()
        env = env_factory(num_quantum_dots=2)
        
        rl_module_spec = create_rl_module_spec(env)
        policy_mapping_fn = get_policy_mapping_fn(2)
        policies_to_train = get_policies_to_train()
        
        print("Creating PPO configuration...")
        
        # Create PPO config (similar to trainer setup)
        ppo_config = PPOConfig()
        ppo_config = ppo_config.environment(env=env_factory, env_config={"num_quantum_dots": 2})
        ppo_config = ppo_config.multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        ppo_config = ppo_config.framework("torch")
        ppo_config = ppo_config.resources(
            num_gpus=0,  # No GPU for testing
        )
        ppo_config = ppo_config.env_runners(
            num_env_runners=1,  # Minimal workers for testing
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        
        print("✓ PPO configuration created successfully")
        
        # Test configuration validation
        print("Validating PPO configuration...")
        
        # Check multi-agent config
        ma_config = ppo_config.multi_agent_config
        print(f"  Multi-agent policies: {list(ma_config['policies'].keys())}")
        print(f"  Policies to train: {ma_config['policies_to_train']}")
        
        # Validate policy mapping
        test_agent_id = env.get_agent_ids()[0]
        mapped_policy = ma_config['policy_mapping_fn'](test_agent_id, None, None)
        print(f"  Policy mapping test: {test_agent_id} -> {mapped_policy}")
        
        if mapped_policy not in ma_config['policies']:
            print(f"  ✗ ERROR: Mapped policy '{mapped_policy}' not in policies")
        else:
            print(f"  ✓ Policy mapping validation passed")
        
        print("✓ PPO configuration validation completed")
        
        env.close()
        log_memory_usage("after_ppo_config")
        return True
        
    except Exception as e:
        print(f"✗ PPO configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ray_envrunner_creation():
    """Test Ray EnvRunner creation with actual configuration."""
    print("="*60)
    print("TESTING: Ray EnvRunner Creation")
    print("="*60)
    
    try:
        from utils.policy_mapping import create_rl_module_spec, get_policy_mapping_fn, get_policies_to_train
        from utils.config_loader import load_config_from_file
        
        log_memory_usage("before_envrunner_test")
        
        # Initialize Ray
        ray_config = {
            "num_gpus": 0,
            "num_cpus": 2,
            "object_store_memory": 1024*1024*512,  # 512MB
            "include_dashboard": False,
            "_node_ip_address": "127.0.0.1",
            "dashboard_port": None,
        }
        
        ray.init(**ray_config)
        print("✓ Ray initialized")
        
        # Load config
        config_path = current_dir / "configs" / "config.yaml"
        config = load_config_from_file(str(config_path))
        
        # Setup environment and policies
        setup_environment()
        env_factory = import_environment()
        env = env_factory(num_quantum_dots=2)
        
        rl_module_spec = create_rl_module_spec(env)
        policy_mapping_fn = get_policy_mapping_fn(2)
        policies_to_train = get_policies_to_train()
        
        print("Creating minimal PPO algorithm...")
        
        # Create PPO algorithm with minimal configuration
        ppo_config = PPOConfig()
        ppo_config = ppo_config.environment(env=env_factory, env_config={"num_quantum_dots": 2})
        ppo_config = ppo_config.multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        ppo_config = ppo_config.framework("torch")
        ppo_config = ppo_config.resources(num_gpus=0)
        ppo_config = ppo_config.env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=10,  # Very short for testing
        )
        ppo_config = ppo_config.training(
            train_batch_size=10,  # Very small for testing
            sgd_minibatch_size=5,
        )
        
        log_memory_usage("before_algorithm_build")
        
        # This is where the EnvRunner error typically occurs
        print("Building PPO algorithm (this tests EnvRunner creation)...")
        algorithm = ppo_config.build()
        
        log_memory_usage("after_algorithm_build")
        print("✓ PPO algorithm built successfully (EnvRunner creation succeeded)")
        
        # Test a single training step
        print("Testing single training iteration...")
        result = algorithm.train()
        print(f"✓ Training iteration completed")
        print(f"  Episodes: {result.get('episodes_total', 'N/A')}")
        print(f"  Timesteps: {result.get('timesteps_total', 'N/A')}")
        
        # Cleanup
        algorithm.stop()
        env.close()
        ray.shutdown()
        
        log_memory_usage("after_envrunner_test")
        print("✓ EnvRunner test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ EnvRunner creation test failed: {e}")
        import traceback
        traceback.print_exc()
        
        if ray.is_initialized():
            ray.shutdown()
        return False


def test_rl_module_spec_issue():
    """Test for RLModuleSpec configuration issues."""
    print("="*60)
    print("TESTING: RLModuleSpec Configuration")
    print("="*60)
    
    try:
        # RLModuleSpec already imported at top
        from utils.policy_mapping import create_rl_module_spec
        
        # Create environment
        setup_environment()
        env_factory = import_environment()
        env = env_factory(num_quantum_dots=2)
        
        # Get RLModule spec
        rl_module_spec = create_rl_module_spec(env)
        
        print("Checking RLModule spec structure for compatibility...")
        
        spec_errors = []
        missing_attrs = []
        
        print(f"\nAnalyzing MultiAgentRLModuleSpec:")
        print(f"  Type: {type(rl_module_spec)}")
        print(f"  Is MultiAgentRLModuleSpec: {isinstance(rl_module_spec, MultiAgentRLModuleSpec)}")
        
        if not isinstance(rl_module_spec, MultiAgentRLModuleSpec):
            print(f"  ✗ ERROR: Expected MultiAgentRLModuleSpec, got {type(rl_module_spec)}")
            spec_errors.append(("root", f"Wrong type: {type(rl_module_spec)}"))
        else:
            print(f"  ✓ MultiAgentRLModuleSpec is valid")
            
            # Check individual RLModule specs
            if hasattr(rl_module_spec, 'rl_module_specs'):
                for policy_id, single_spec in rl_module_spec.rl_module_specs.items():
                    print(f"\n  Analyzing SingleAgentRLModuleSpec '{policy_id}':")
                    print(f"    Type: {type(single_spec)}")
                    print(f"    Is SingleAgentRLModuleSpec: {isinstance(single_spec, SingleAgentRLModuleSpec)}")
                    
                    if not isinstance(single_spec, SingleAgentRLModuleSpec):
                        print(f"    ✗ ERROR: Expected SingleAgentRLModuleSpec, got {type(single_spec)}")
                        spec_errors.append((policy_id, f"Wrong type: {type(single_spec)}"))
                        continue
                    
                    # Check required attributes
                    attrs_to_check = [
                        ('module_class', 'module_class'),
                        ('observation_space', 'observation_space'),
                        ('action_space', 'action_space'),
                        ('model_config_dict', 'model_config_dict')
                    ]
                    
                    for attr_name, display_name in attrs_to_check:
                        if hasattr(single_spec, attr_name):
                            value = getattr(single_spec, attr_name)
                            print(f"    ✓ {display_name}: {type(value)}")
                        else:
                            print(f"    ✗ Missing {display_name}")
                            missing_attrs.append((policy_id, attr_name))
            else:
                print("  ✗ ERROR: Missing rl_module_specs attribute")
                missing_attrs.append(("root", "rl_module_specs"))
        
        # Report all errors found
        total_errors = len(spec_errors) + len(missing_attrs)
        if total_errors > 0:
            print(f"\nERROR SUMMARY:")
            print(f"  RLModule spec format errors: {len(spec_errors)}")
            print(f"  Missing attributes: {len(missing_attrs)}")
            
            if spec_errors:
                print(f"  Format errors: {spec_errors}")
            if missing_attrs:
                print(f"  Missing attrs: {missing_attrs}")
                
            return False
        
        env.close()
        print("\n✓ All policy specs are properly formatted")
        return True
        
    except Exception as e:
        print(f"✗ RLModuleSpec test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_compatibility_tests():
    """Run all compatibility tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE RAY/RLLIB COMPATIBILITY TESTS")
    print("="*80)
    
    tests = [
        ("Environment Compatibility", test_environment_compatibility),
        ("Policy Specs Creation", test_policy_specs_creation),
        ("RLModuleSpec Configuration", test_rl_module_spec_issue),
        ("PPO Configuration", test_ppo_config_creation),
        ("Ray EnvRunner Creation", test_ray_envrunner_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        # Cleanup between tests
        force_garbage_collect(f"after_{test_name.lower().replace(' ', '_')}")
        if ray.is_initialized():
            ray.shutdown()
        time.sleep(2)  # Longer pause for Ray cleanup
    
    # Summary
    print("\n" + "="*80)
    print("COMPATIBILITY TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n" + "!"*80)
        print("FAILURES DETECTED - Check the failed tests above for RLlib configuration issues")
        print("!"*80)
    
    return passed == total


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Ray/RLlib compatibility")
    parser.add_argument(
        "--test", 
        type=str, 
        default="all",
        choices=["all", "env", "policy", "rlmodule", "ppo", "envrunner"],
        help="Specific test to run"
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_compatibility_tests()
    elif args.test == "env":
        success = test_environment_compatibility()
    elif args.test == "policy":
        success = test_policy_specs_creation()
    elif args.test == "rlmodule":
        success = test_rl_module_spec_issue()
    elif args.test == "ppo":
        success = test_ppo_config_creation()
    elif args.test == "envrunner":
        success = test_ray_envrunner_creation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()