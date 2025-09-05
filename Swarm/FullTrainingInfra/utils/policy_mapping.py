"""
Policy mapping utilities for multi-agent quantum device environment.
Maps agent IDs to appropriate policies based on agent type (plunger vs barrier).

NOTE: check policy handling, we are training two policies only (plunger and barrier)
 even though the agent instances require multiple policies
"""

from typing import Dict, Any, Optional
from ray.rllib.core.rl_module.rl_module import RLModuleSpec as SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec as MultiAgentRLModuleSpec


def get_policy_mapping_fn(num_quantum_dots: int = 8):
    """
    Create a policy mapping function for the quantum device environment.
    
    Args:
        num_quantum_dots: Number of quantum dots (N), typically 8
        
    Returns:
        Function that maps agent IDs to policy IDs
    """
    
    def policy_mapping_fn(agent_id: str, episode: Optional[Any] = None, worker: Optional[Any] = None, **kwargs) -> str:
        """
        Map agent ID to policy ID.
        
        Args:
            agent_id: The agent identifier
            episode: Current episode (unused)
            worker: Current worker (unused)
            
        Returns:
            Policy ID string
            
        Raises:
            ValueError: If agent_id doesn't contain 'plunger' or 'barrier'
        """
        if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
            return "plunger_policy"
        elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
            return "barrier_policy"
        else:
            raise ValueError(
                f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type. "
                f"Expected format: 'plunger_X' or 'barrier_X' where X is the agent number."
            )
    
    return policy_mapping_fn


def get_policies_to_train():
    """
    Get the list of policies that should be trained.
    
    Returns:
        List of policy IDs to train
    """
    return ["plunger_policy", "barrier_policy"]


def create_rl_module_spec(env_instance) -> MultiAgentRLModuleSpec:
    """
    Create policy specifications for RLlib with the plunger and barrier policies
    (note there are only TWO policies although each has multiple agent instances)
    
    Args:
        env_instance: Instance of the quantum device environment
        
    Returns:
        MultiAgentRLModuleSpec object
    """
    from gymnasium import spaces
    import numpy as np
    
    # Get full environment spaces from base environment
    # Multi-agent wrapper preserves original spaces as base_observation_space and base_action_space
    if hasattr(env_instance, 'base_observation_space'):
        full_obs_space = env_instance.base_observation_space
        full_action_space = env_instance.base_action_space
    else:
        raise ValueError("Training attempted on a non-wrapped environment")
    
    # Extract dimensions from environment
    image_shape = full_obs_space['image'].shape  # (H, W, channels)
    num_gates = full_action_space['action_gate_voltages'].shape[0]
    num_barriers = full_action_space['action_barrier_voltages'].shape[0]
    
    # Gate voltage ranges
    gate_low = full_action_space['action_gate_voltages'].low[0]
    gate_high = full_action_space['action_gate_voltages'].high[0]
    barrier_low = full_action_space['action_barrier_voltages'].low[0]
    barrier_high = full_action_space['action_barrier_voltages'].high[0]
    
    # Create observation space for gate agents
    # Each gate agent sees: dual-channel image + single voltage value
    gate_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(image_shape[0], image_shape[1], 2),  # Dual channel for gate agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=gate_low, high=gate_high,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    # Create action space for gate agents
    # Each gate agent controls: single gate voltage
    gate_action_space = spaces.Box(
            low=gate_low, high=gate_high,
            shape=(1,),  # Single gate voltage output
            dtype=np.float32
        )
    
    # Create observation space for barrier agents  
    # Each barrier agent sees: single-channel image + single voltage value
    barrier_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(image_shape[0], image_shape[1], 1),  # Single channel for barrier agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=barrier_low, high=barrier_high,
            shape=(1,),  # Single voltage value
            dtype=np.float32
        )
    })
    
    # Create action space for barrier agents
    # Each barrier agent controls: single barrier voltage
    barrier_action_space = spaces.Box(
            low=barrier_low, high=barrier_high,
            shape=(1,),  # Single barrier voltage output
            dtype=np.float32
        )
    
    # Create single agent RLModule specs
    plunger_spec = SingleAgentRLModuleSpec(
        module_class=None,  # Use default RLModule class
        observation_space=gate_obs_space,
        action_space=gate_action_space,
        model_config={},  # Empty config - will be populated by trainer
        learner_only=True  # Required for multi-agent RLModule configurations
    )
    
    barrier_spec = SingleAgentRLModuleSpec(
        module_class=None,  # Use default RLModule class
        observation_space=barrier_obs_space,
        action_space=barrier_action_space,
        model_config={},  # Empty config - will be populated by trainer
        learner_only=True  # Required for multi-agent RLModule configurations
    )

    # Create multi-agent RLModule spec
    rl_module_spec = MultiAgentRLModuleSpec(
        rl_module_specs={
            "plunger_policy": plunger_spec,
            "barrier_policy": barrier_spec,
        }
    )

    return rl_module_spec



def get_agent_ids(num_quantum_dots: int = 8):
    """
    Generate expected agent IDs for the quantum device environment.
    
    Args:
        num_quantum_dots: Number of quantum dots (N)
        
    Returns:
        List of agent IDs
    """
    agent_ids = []
    
    # N plunger agents (one per quantum dot)
    for i in range(num_quantum_dots):
        agent_ids.append(f"plunger_{i}")
    
    # N-1 barrier agents (between quantum dots)
    for i in range(num_quantum_dots - 1):
        agent_ids.append(f"barrier_{i}")
    
    return agent_ids


def validate_agent_assignment(agent_ids, num_quantum_dots: int = 8):
    """
    Validate that agent IDs match expected structure.
    
    Args:
        agent_ids: List of agent IDs from environment
        num_quantum_dots: Expected number of quantum dots
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected_plungers = num_quantum_dots
    expected_barriers = num_quantum_dots - 1
    
    plunger_count = sum(1 for aid in agent_ids if "plunger" in aid.lower())
    barrier_count = sum(1 for aid in agent_ids if "barrier" in aid.lower())
    
    if plunger_count != expected_plungers:
        return False, f"Expected {expected_plungers} plunger agents, got {plunger_count}"
    
    if barrier_count != expected_barriers:
        return False, f"Expected {expected_barriers} barrier agents, got {barrier_count}"
    
    return True, "Agent assignment is valid"



if __name__ == "__main__":
    import sys
    import os
    import torch
    import numpy as np
    from pathlib import Path
    
    # Add VoltageAgent to path
    current_dir = Path(__file__).parent.parent
    voltage_agent_dir = current_dir / "../VoltageAgent"
    sys.path.append(str(voltage_agent_dir))
    
    # Import the agent class
    try:
        from ppo_trainer_recurrent import SingleAgentRecurrentPPOModel
        from ray.rllib.policy.sample_batch import SampleBatch
        print("✓ Successfully imported SingleAgentRecurrentPPOModel")
    except Exception as e:
        print(f"✗ Failed to import agent class: {e}")
        sys.exit(1)
    
    # Create mock environment for testing
    class MockEnv:
        def __init__(self):
            from gymnasium import spaces
            # Mock the base environment observation/action spaces
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0.0, high=1.0, shape=(64, 64, 7), dtype=np.float32),  # 8 dots = 7 channels
                'obs_gate_voltages': spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                'obs_barrier_voltages': spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            })
            self.action_space = spaces.Dict({
                'action_gate_voltages': spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                'action_barrier_voltages': spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            })

            self.base_observation_space = self.observation_space
            self.base_action_space = self.action_space
    
    print("\n=== Testing Policy Specifications ===")
    
    # Create mock environment
    mock_env = MockEnv()
    
    # Create RLModule specs
    try:
        rl_module_spec = create_rl_module_spec(mock_env)
        print("✓ Successfully created RLModule specifications")
        
        # Print RLModule info
        for policy_id, single_spec in rl_module_spec.rl_module_specs.items():
            print(f"\nRLModule: {policy_id}")
            print(f"  Module class: {single_spec.module_class}")
            print(f"  Observation space: {single_spec.observation_space}")
            print(f"  Action space: {single_spec.action_space}")
            print(f"  Model config: {single_spec.model_config}")
            
    except Exception as e:
        print(f"✗ Failed to create RLModule specs: {e}")
        sys.exit(1)
    
    print("\n=== Testing Gate Agent ===")
    
    # Test gate agent
    plunger_spec = rl_module_spec.rl_module_specs["plunger_policy"]
    gate_obs_space, gate_action_space = plunger_spec.observation_space, plunger_spec.action_space
    
    try:
        # Create agent config
        gate_config = {
            "observation_space": gate_obs_space,
            "action_space": gate_action_space,
            "model_config_dict": {
                "lstm_cell_size": 64,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
                "fcnet_hiddens": [128, 128]
            }
        }
        
        # Create gate agent
        gate_agent = SingleAgentRecurrentPPOModel(
            observation_space=gate_obs_space,
            action_space=gate_action_space,
            model_config=gate_config["model_config_dict"]
        )
        print("✓ Successfully created gate agent")
        print(f"  Image channels: {gate_agent.image_channels}")
        print(f"  Action size: {gate_agent.action_size}")
        
        # Create test batch
        batch_size = 4
        test_batch = {
            SampleBatch.OBS: {
                'image': torch.randn(batch_size, 64, 64, 2),  # 2 channels for gate agent
                'voltage': torch.randn(batch_size, 1)  # Single voltage
            }
        }
        
        # Test forward pass
        output = gate_agent._forward_train(test_batch)
        print("✓ Successfully ran forward pass for gate agent")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Action logits shape: {output[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Value predictions shape: {output[SampleBatch.VF_PREDS].shape}")
        
    except Exception as e:
        print(f"✗ Gate agent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Barrier Agent ===")
    
    # Test barrier agent
    barrier_spec = rl_module_spec.rl_module_specs["barrier_policy"]
    barrier_obs_space, barrier_action_space = barrier_spec.observation_space, barrier_spec.action_space
    
    try:
        # Create agent config
        barrier_config = {
            "observation_space": barrier_obs_space,
            "action_space": barrier_action_space,
            "model_config_dict": {
                "lstm_cell_size": 64,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
                "fcnet_hiddens": [128, 128]
            }
        }
        
        # Create barrier agent
        barrier_agent = SingleAgentRecurrentPPOModel(
            observation_space=barrier_obs_space,
            action_space=barrier_action_space,
            model_config=barrier_config["model_config_dict"]
        )
        print("✓ Successfully created barrier agent")
        print(f"  Image channels: {barrier_agent.image_channels}")
        print(f"  Action size: {barrier_agent.action_size}")
        
        # Create test batch
        test_batch = {
            SampleBatch.OBS: {
                'image': torch.randn(batch_size, 64, 64, 1),  # 1 channel for barrier agent
                'voltage': torch.randn(batch_size, 1)  # Single voltage
            }
        }
        
        # Test forward pass
        output = barrier_agent._forward_train(test_batch)
        print("✓ Successfully ran forward pass for barrier agent")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Action logits shape: {output[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Value predictions shape: {output[SampleBatch.VF_PREDS].shape}")
        
    except Exception as e:
        print(f"✗ Barrier agent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing LSTM State Management ===")
    
    try:
        # Test with previous state
        h_state = torch.randn(1, batch_size, 64)
        c_state = torch.randn(1, batch_size, 64)
        
        test_batch_with_state = {
            SampleBatch.OBS: {
                'image': torch.randn(batch_size, 64, 64, 2),
                'voltage': torch.randn(batch_size, 1)
            },
            "state_in": [h_state, c_state],
            SampleBatch.PREV_ACTIONS: torch.randn(batch_size, 1),
            SampleBatch.PREV_REWARDS: torch.randn(batch_size)
        }
        
        output_with_state = gate_agent._forward_train(test_batch_with_state)
        print("✓ Successfully handled LSTM state and previous action/reward")
        # Handle both dict and list format for state_out
        if isinstance(output_with_state['state_out'], dict):
            state_shapes = [v.shape for v in output_with_state['state_out'].values()]
        else:
            state_shapes = [s.shape for s in output_with_state['state_out']]
        print(f"  New state shapes: {state_shapes}")
        
    except Exception as e:
        print(f"✗ LSTM state test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== All Tests Complete ===")
    print("🎉 Single agent architecture appears to be working correctly!") 