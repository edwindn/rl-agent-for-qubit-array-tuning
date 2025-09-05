#!/usr/bin/env python3
"""
Test script to isolate and test individual components of train.py.
This helps debug memory issues and component failures without full training runs.
"""

import os
import sys
import argparse
from pathlib import Path
import ray
import time

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import components from train.py
from train import (
    setup_environment, 
    import_environment, 
    log_memory_usage, 
    force_garbage_collect,
    setup_memory_logger
)

def test_memory_logging():
    """Test memory logging functionality."""
    print("="*60)
    print("TESTING: Memory Logging System")
    print("="*60)
    
    try:
        # Test logger setup
        logger = setup_memory_logger()
        print(f"✓ Memory logger created successfully")
        
        # Test memory usage logging
        log_memory_usage("test_start")
        print("✓ Memory usage logged successfully")
        
        # Test garbage collection
        force_garbage_collect("test_gc")
        print("✓ Garbage collection logged successfully")
        
        return True
    except Exception as e:
        print(f"✗ Memory logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_setup():
    """Test environment variable setup."""
    print("="*60)
    print("TESTING: Environment Setup")
    print("="*60)
    
    try:
        log_memory_usage("before_env_setup")
        
        # Test environment setup
        setup_environment()
        print("✓ Environment variables set successfully")
        
        # Verify critical environment variables
        critical_vars = ['MPLBACKEND', 'JAX_PLATFORMS', 'RAY_DISABLE_IMPORT_WARNING']
        for var in critical_vars:
            if var in os.environ:
                print(f"  ✓ {var} = {os.environ[var]}")
            else:
                print(f"  ✗ {var} not set")
                return False
        
        log_memory_usage("after_env_setup")
        return True
        
    except Exception as e:
        print(f"✗ Environment setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_import():
    """Test environment import and factory creation."""
    print("="*60)
    print("TESTING: Environment Import")
    print("="*60)
    
    try:
        log_memory_usage("before_env_import")
        
        # Test environment import
        env_factory = import_environment()
        print("✓ Environment factory created successfully")
        
        log_memory_usage("after_env_import")
        
        # Test factory function (without creating full environment)
        print(f"✓ Factory function: {type(env_factory)}")
        print(f"✓ Factory callable: {callable(env_factory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation(num_quantum_dots=2):
    """Test actual environment creation with minimal quantum dots."""
    print("="*60)
    print(f"TESTING: Environment Creation ({num_quantum_dots} quantum dots)")
    print("="*60)
    
    try:
        log_memory_usage("before_env_creation")
        
        # Get factory
        env_factory = import_environment()
        
        # Create minimal environment
        print(f"Creating environment with {num_quantum_dots} quantum dots...")
        env = env_factory(num_quantum_dots=num_quantum_dots)
        
        log_memory_usage("after_env_creation")
        print("✓ Environment created successfully")
        
        # Test basic properties
        print(f"  Agent IDs: {env.get_agent_ids()}")
        print(f"  Number of agents: {len(env.get_agent_ids())}")
        
        # Test reset (this is often where memory issues occur)
        print("Testing environment reset...")
        log_memory_usage("before_reset")
        obs, info = env.reset()
        log_memory_usage("after_reset")
        print("✓ Environment reset successful")
        print(f"  Got observations for {len(obs)} agents")
        
        # Check first observation structure
        if obs:
            first_agent = list(obs.keys())[0]
            agent_obs = obs[first_agent]
            print(f"  Sample observation ({first_agent}):")
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value.shape}")
                    else:
                        print(f"    {key}: {type(value)}")
        
        # Clean up
        env.close()
        log_memory_usage("after_env_close")
        force_garbage_collect("after_env_test")
        print("✓ Environment closed and cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("="*60)
    print("TESTING: Configuration Loading")
    print("="*60)
    
    try:
        from utils.config_loader import load_config_from_file
        
        log_memory_usage("before_config_load")
        
        # Try to load default config
        default_config_path = current_dir / "configs" / "config.yaml"
        if default_config_path.exists():
            config = load_config_from_file(str(default_config_path))
            print(f"✓ Configuration loaded from: {default_config_path}")
            print(f"  Experiment name: {config.get('experiment', {}).get('name', 'N/A')}")
            print(f"  Ray workers: {config.get('ray', {}).get('num_workers', 'N/A')}")
            print(f"  Ray GPUs: {config.get('ray', {}).get('num_gpus', 'N/A')}")
        else:
            print(f"✗ Config file not found: {default_config_path}")
            return False
        
        log_memory_usage("after_config_load")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ray_initialization():
    """Test Ray initialization with minimal resources."""
    print("="*60)
    print("TESTING: Ray Initialization")
    print("="*60)
    
    try:
        log_memory_usage("before_ray_init")
        
        # Minimal Ray config for testing
        ray_config = {
            "num_gpus": 0,  # No GPUs for testing
            "num_cpus": 2,  # Minimal CPUs
            "object_store_memory": 1024*1024*512,  # 512MB object store
            "include_dashboard": False,
            "_node_ip_address": "127.0.0.1",
            "dashboard_port": None,
            "_temp_dir": "/tmp/ray_temp_test",
        }
        
        # Initialize Ray
        ray.init(**ray_config)
        print("✓ Ray initialized successfully")
        
        log_memory_usage("after_ray_init")
        
        # Test Ray is working
        @ray.remote
        def test_ray_task():
            return "Ray is working"
        
        result = ray.get(test_ray_task.remote())
        print(f"✓ Ray task executed: {result}")
        
        # Cleanup
        ray.shutdown()
        log_memory_usage("after_ray_shutdown")
        print("✓ Ray shut down successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Ray initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        if ray.is_initialized():
            ray.shutdown()
        return False


def test_trainer_creation():
    """Test trainer class creation (without full setup)."""
    print("="*60)
    print("TESTING: Trainer Creation")
    print("="*60)
    
    try:
        # Add VoltageAgent to path
        sys.path.append(str(current_dir.parent))
        from VoltageAgent import get_trainer_class
        
        log_memory_usage("before_trainer_class")
        
        # Test getting trainer class
        trainer_type = "recurrent_ppo"
        trainer_class = get_trainer_class(trainer_type)
        print(f"✓ Got trainer class: {trainer_class}")
        
        log_memory_usage("after_trainer_class")
        print("✓ Trainer class creation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_individual_test(test_name):
    """Run a single test by name."""
    test_functions = {
        "memory": test_memory_logging,
        "env_setup": test_environment_setup,
        "env_import": test_environment_import,
        "env_create": test_environment_creation,
        "config": test_config_loading,
        "ray": test_ray_initialization,
        "trainer": test_trainer_creation,
    }
    
    if test_name not in test_functions:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(test_functions.keys())}")
        return False
    
    print(f"\nRunning individual test: {test_name}")
    return test_functions[test_name]()


def run_all_tests():
    """Run all component tests in sequence."""
    print("\n" + "="*80)
    print("RUNNING ALL COMPONENT TESTS")
    print("="*80)
    
    tests = [
        ("Memory Logging", test_memory_logging),
        ("Environment Setup", test_environment_setup),
        ("Environment Import", test_environment_import),
        ("Environment Creation", lambda: test_environment_creation(num_quantum_dots=2)),
        ("Config Loading", test_config_loading),
        ("Ray Initialization", test_ray_initialization),
        ("Trainer Creation", test_trainer_creation),
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
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    return passed == total


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test individual components of train.py")
    parser.add_argument(
        "--test", 
        type=str, 
        default="all",
        help="Test to run (memory, env_setup, env_import, env_create, config, ray, trainer, all)"
    )
    parser.add_argument(
        "--num-dots", 
        type=int, 
        default=2,
        help="Number of quantum dots for environment creation test"
    )
    
    args = parser.parse_args()
    
    # Setup environment first
    setup_environment()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "env_create":
        success = test_environment_creation(args.num_dots)
    else:
        success = run_individual_test(args.test)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()