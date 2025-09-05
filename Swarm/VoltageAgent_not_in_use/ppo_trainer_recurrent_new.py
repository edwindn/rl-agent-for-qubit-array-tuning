"""
PPO trainer setup for multi-agent quantum device environment.
Handles distributed training, policy configuration, and model setup.
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork, LSTMWrapper
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Type
import numpy as np
import sys
from pathlib import Path

torch, nn = try_import_torch()

"""
todo:

update obs space to use actual keys

update nets to use custom models

add stopping model
"""

class ObservationProcessor(nn.Module):
    """Processes quantum device observations (images + voltages) into features."""
    
    def __init__(self, observation_space: spaces.Dict, model_config: Dict[str, Any]):
        super().__init__()
        
        # Parse observation space
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("Observation space should be a Dict")
            
        self.image_space = observation_space['image']
        self.voltage_space = observation_space['voltage']
        
        # Handle image observations
        image_shape = self.image_space.shape  # (H, W, C)
        if len(image_shape) == 3 and image_shape[2] in [1, 2]:
            self.image_channels = image_shape[2]
            self.image_input_shape = (self.image_channels, image_shape[0], image_shape[1])
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}")
        
        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.image_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.image_input_shape)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.shape[1]
        
        # Voltage processing
        voltage_dim = self.voltage_space.shape[0]
        self.voltage_fc = nn.Linear(voltage_dim, 32)
        
        # Combined features processing
        combined_size = conv_output_size + 32
        hidden_dims = model_config.get("fcnet_hiddens", [128, 128])
        
        layers = []
        prev_size = combined_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_dim),
                nn.ReLU()
            ])
            prev_size = hidden_dim
            
        self.shared_layers = nn.Sequential(*layers)
        self.output_size = prev_size
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observations into feature vector."""
        image = obs_dict["image"]
        voltage = obs_dict["voltage"]
        
        # Handle batch dimension edge case
        if image.ndim == 3:
            image = image.unsqueeze(0)
            
        # Flatten time dimension if present
        original_shape = None
        if image.ndim == 5:  # (B, T, H, W, C)
            original_shape = image.shape[:2]  # (B, T)
            image = image.flatten(0, 1)  # (B*T, H, W, C)
            voltage = voltage.flatten(0, 1)  # (B*T, voltage_dim)
        
        # Convert to channels-first format
        if image.shape[-1] in [1, 2]:
            image = image.permute(0, 3, 1, 2)  # (batch, C, H, W)
        
        # Process through CNN
        image_features = self.conv_layers(image)
        
        # Process voltage
        voltage_features = torch.relu(self.voltage_fc(voltage))
        
        # Combine and process
        combined = torch.cat([image_features, voltage_features], dim=1)
        features = self.shared_layers(combined)
        
        # Restore time dimension if needed
        if original_shape is not None:
            B, T = original_shape
            features = features.view(B, T, -1)
        
        return features


class SingleAgentRecurrentPPOModel(TorchRLModule):
    """
    RLModule using Ray's proven LSTMWrapper for LSTM memory handling.
    This class processes quantum device observations and delegates LSTM logic to Ray.
    """
    
    def __init__(self, observation_space=None, action_space=None, inference_only=False, 
                 learner_only=False, model_config=None, catalog_class=None, config=None, **kwargs):
        # Handle both new and old API formats
        if config is not None:
            obs_space = config.get("observation_space", observation_space)
            action_space = config.get("action_space", action_space)
            model_config = config.get("model_config_dict", config.get("model_config", model_config or {}))
            inference_only = config.get("inference_only", inference_only)
            learner_only = config.get("learner_only", learner_only)
            catalog_class = config.get("catalog_class", catalog_class)
        else:
            obs_space = observation_space
            model_config = model_config or {}
        
        # Filter out None values for parent
        parent_kwargs = {}
        for key, value in [("observation_space", obs_space), ("action_space", action_space), 
                          ("inference_only", inference_only), ("learner_only", learner_only),
                          ("model_config", model_config), ("catalog_class", catalog_class)]:
            if value is not None:
                parent_kwargs[key] = value
        
        super().__init__(**parent_kwargs)
        
        # Store configuration
        self.lstm_cell_size = model_config["lstm_cell_size"]
        self.action_size = int(np.prod(action_space.shape)) if action_space.shape else 1
        
        # Create observation processor
        self.obs_processor = ObservationProcessor(obs_space, model_config)
        
        # Create a flattened observation space for Ray's LSTMWrapper
        # LSTMWrapper expects a flat observation space
        flat_obs_size = self.obs_processor.output_size
        flat_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(flat_obs_size,), 
            dtype=np.float32
        )
        
        # Create Ray's LSTMWrapper to handle all LSTM logic
        self.lstm_wrapper = LSTMWrapper(
            obs_space=flat_obs_space,
            action_space=action_space,
            num_outputs=self.action_size,  # Policy outputs
            model_config=model_config,
            name="quantum_lstm"
        )
        
        # The LSTM wrapper already has its own value branch, 
        # so we don't need to create separate heads
    
    def _forward_inference(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """Forward pass for inference."""
        return self._forward_impl(batch)
    
    def _forward_exploration(self, batch: Dict[str, TensorType], t: int = None, **kwargs) -> Dict[str, TensorType]:
        """Forward pass for exploration."""
        return self._forward_impl(batch)
    
    def _forward_train(self, batch: Dict[str, TensorType], t: int | None = None) -> Dict[str, TensorType]:
        """Forward pass for training."""
        return self._forward_impl(batch)
    
    def _forward_impl(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """Implementation using Ray's LSTMWrapper for all LSTM logic."""
        
        # Step 1: Process quantum observations through our custom CNN
        obs_features = self.obs_processor(batch[SampleBatch.OBS])
        
        # Step 2: Create input_dict compatible with Ray's LSTMWrapper
        input_dict = {
            "obs_flat": obs_features,
            **{k: v for k, v in batch.items() if k.startswith("prev_") or k in [SampleBatch.PREV_ACTIONS, SampleBatch.PREV_REWARDS]}
        }
        
        # Step 3: Determine seq_lens for time dimension handling
        seq_lens = None
        if obs_features.ndim == 3:  # (B, T, features)
            seq_lens = torch.full((obs_features.shape[0],), obs_features.shape[1], dtype=torch.long, device=obs_features.device)
        elif obs_features.ndim == 2:  # (B, features)
            seq_lens = torch.ones(obs_features.shape[0], dtype=torch.long, device=obs_features.device)
        
        # Step 4: Extract and convert LSTM state
        lstm_state = None
        if "state_in" in batch and batch["state_in"]:
            state_dict = batch["state_in"]
            if isinstance(state_dict, dict) and "h_state" in state_dict and "c_state" in state_dict:
                h_state = state_dict["h_state"]
                c_state = state_dict["c_state"]
                # Convert RLModule state format to Ray's format
                if h_state.ndim == 3:  # (1, B, hidden) -> (B, hidden)
                    h_state = h_state.squeeze(0)
                if c_state.ndim == 3:  # (1, B, hidden) -> (B, hidden)  
                    c_state = c_state.squeeze(0)
                lstm_state = [h_state, c_state]
        
        # Step 5: Use Ray's proven LSTM implementation
        action_logits, new_state = self.lstm_wrapper.forward(input_dict, lstm_state, seq_lens)
        value_preds = self.lstm_wrapper.value_function()
        
        # Step 6: Convert state back to RLModule format
        state_out = {}
        if new_state and len(new_state) == 2:
            h, c = new_state
            # Ensure proper dimensions for RLModule API
            if h.ndim == 2:  # (B, hidden) -> (1, B, hidden)
                h = h.unsqueeze(0)
            if c.ndim == 2:  # (B, hidden) -> (1, B, hidden)
                c = c.unsqueeze(0)
            state_out = {"h_state": h, "c_state": c}
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: value_preds,
            "state_out": state_out,
        }
    
    
    def get_initial_state(self) -> Dict[str, TensorType]:
        """Return initial LSTM state using Ray's implementation."""
        ray_state = self.lstm_wrapper.get_initial_state()
        
        # Convert Ray's state format to RLModule format
        if len(ray_state) == 2:
            h, c = ray_state
            # Convert to proper tensor format and dimensions
            if isinstance(h, np.ndarray):
                h = torch.from_numpy(h)
            if isinstance(c, np.ndarray):
                c = torch.from_numpy(c)
            
            # Ensure proper dimensions (1, 1, hidden)
            if h.ndim == 1:
                h = h.unsqueeze(0).unsqueeze(0)
            elif h.ndim == 2:
                h = h.unsqueeze(0)
            if c.ndim == 1:
                c = c.unsqueeze(0).unsqueeze(0)
            elif c.ndim == 2:
                c = c.unsqueeze(0)
            
            return {"h_state": h, "c_state": c}
        
        # Fallback
        return {
            "h_state": torch.zeros(1, 1, self.lstm_cell_size),
            "c_state": torch.zeros(1, 1, self.lstm_cell_size)
        }



def create_recurrent_ppo_config(config: Dict[str, Any], env_factory, rl_module_spec, 
                     policy_mapping_fn, callback_class, 
                     num_quantum_dots: int = 8, train_agent_types: list = None) -> PPOConfig:
    """
    Create PPO configuration for multi-agent training using new RLModule API.
    
    Args:
        config: Training configuration dictionary
        env_factory: Environment factory function
        rl_module_spec: MultiRLModuleSpec from FullTrainingInfra
        policy_mapping_fn: Policy mapping function from FullTrainingInfra
        callback_class: Callback class from FullTrainingInfra
        num_quantum_dots: Number of quantum dots (N)
        train_agent_types: List of agent types to train ['gate', 'barrier'] or ['gate'] (default: ['gate', 'barrier'])
        
    Returns:
        Configured PPOConfig instance
    """
    from .ppo_config import get_ppo_config
    
    # Default to training both agent types if not specified
    if train_agent_types is None:
        train_agent_types = ['gate', 'barrier']
    
    # With RLModule API, we don't use ModelCatalog anymore
    
    # Get PPO configuration with any overrides from training config
    ppo_overrides = config.get("ppo_overrides", {})
    ppo_config_dict = get_ppo_config(ppo_overrides)
    
    # Configure model for RLModule
    base_model_config = ppo_config_dict["model"].copy()
    
    # Create model configuration for the LSTM model
    model_config_dict = base_model_config.copy()
    model_config_dict.update({
        "use_lstm": config["agent_models"]["use_lstm"],
        "lstm_cell_size": config["agent_models"]["lstm_cell_size"],
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128],  # Feed-forward hidden layers
    })

    # Update the RLModule specs with model configuration and custom RLModule class
    updated_rl_module_specs = {}
    for policy_id, single_spec in rl_module_spec.rl_module_specs.items():
        # Determine agent type for filtering
        if "gate" in policy_id.lower() or "plunger" in policy_id.lower():
            agent_type = "gate"
        elif "barrier" in policy_id.lower():
            agent_type = "barrier"
        else:
            agent_type = "gate"  # Default fallback
        
        # Skip this policy if its agent type is not in train_agent_types
        if agent_type not in train_agent_types:
            continue
        
        # Create new SingleAgentRLModuleSpec with updated model config
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec as SingleAgentRLModuleSpec
        updated_spec = SingleAgentRLModuleSpec(
            module_class=SingleAgentRecurrentPPOModel,  # Use our custom RLModule
            observation_space=single_spec.observation_space,
            action_space=single_spec.action_space,
            model_config=model_config_dict,
            learner_only=True  # Required for multi-agent RLModule configurations
        )
        updated_rl_module_specs[policy_id] = updated_spec

    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    
    updated_rl_module_spec = MultiRLModuleSpec(
        rl_module_specs=updated_rl_module_specs
    )
    
    # With RLModules, all modules in the rl_module_spec will be trained
    # train_agent_types parameter is maintained for potential future filtering
    
    # Create environment function for Ray RLlib
    def env_creator(env_config):
        return env_factory(num_quantum_dots=num_quantum_dots)
    
    # Register environment with Ray
    from ray.tune.registry import register_env
    env_name = f"quantum_device_env_{num_quantum_dots}"
    register_env(env_name, env_creator)
    
    # Create PPO configuration with new RLModule API
    ppo_config = (
        PPOConfig()
        .environment(env=env_name)
        .rl_module(
            rl_module_spec=updated_rl_module_spec,
        )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(
            rollout_fragment_length=config["env"]["rollout_fragment_length"],
            batch_mode=config["ray"]["batch_mode"],
            remote_worker_envs=config["ray"]["remote_worker_envs"],
            num_env_runners=config["ray"]["num_workers"],
            num_envs_per_env_runner=config["ray"]["num_envs_per_env_runner"],
            num_cpus_per_env_runner=config["ray"]["num_cpus_per_worker"],
            num_gpus_per_env_runner=config["ray"]["num_gpus_per_worker"],
        )
        .training(
            train_batch_size=config["env"]["train_batch_size"],
            minibatch_size=config["env"].get("mini_batch_size", 128),
            num_sgd_iter=ppo_config_dict["num_sgd_iter"],
            lr=ppo_config_dict["lr"],
            lr_schedule=ppo_config_dict["lr_schedule"],
            clip_param=ppo_config_dict["clip_param"],
            vf_clip_param=ppo_config_dict["vf_clip_param"],
            entropy_coeff=ppo_config_dict["entropy_coeff"],
            vf_loss_coeff=ppo_config_dict["vf_loss_coeff"],
            kl_coeff=ppo_config_dict["kl_coeff"],
            kl_target=ppo_config_dict["kl_target"],
            gamma=ppo_config_dict["gamma"],
            lambda_=ppo_config_dict["lambda"],
        )
        .resources(
            num_gpus=config["ray"]["num_gpus"],
        )
        .callbacks(callback_class)
        .debugging(seed=config["experiment"]["seed"])
    )
    
    # Add evaluation configuration if specified
    eval_config = config.get("evaluation", {})
    if eval_config:
        ppo_config = ppo_config.evaluation(
            evaluation_interval=eval_config.get("evaluation_interval"),
            evaluation_duration=eval_config.get("evaluation_duration"),
            evaluation_parallel_to_training=eval_config.get("evaluation_parallel_to_training", True),
        )
    
    return ppo_config


def create_tune_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Ray Tune configuration for hyperparameter optimization.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Ray Tune configuration dictionary
    """
    tune_config = {
        "stop": config["stopping_criteria"],
        "checkpoint_config": {
            "checkpoint_frequency": config["checkpointing"]["frequency"],
            "num_to_keep": config["checkpointing"]["keep_checkpoints_num"],
            "checkpoint_score_attribute": config["checkpointing"]["checkpoint_score_attr"],
        },
        "verbose": 1,
    }
    
    return tune_config


class RecurrentPPOTrainer:
    """Manages PPO training for quantum device environment."""
    
    def __init__(self, config: Dict[str, Any], env_factory):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration dictionary
            env_factory: Environment factory function
        """
        self.config = config
        self.env_factory = env_factory
        self.algorithm = None
        self.ppo_config = None
        
        # Setup Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_gpus=config["ray"]["num_gpus"],
                object_store_memory=config["ray"]["object_store_memory"]
            )
    
    def setup_training(self, rl_module_spec, policy_mapping_fn, 
                      callback_class, num_quantum_dots: int = 8,
                      train_agent_types: list = None):
        """
        Setup training configuration.
        
        Args:
            rl_module_spec: MultiRLModuleSpec from FullTrainingInfra
            policy_mapping_fn: Policy mapping function from FullTrainingInfra
            callback_class: Callback class from FullTrainingInfra
            num_quantum_dots: Number of quantum dots
            train_agent_types: List of agent types to train ['gate', 'barrier'] or ['gate'] (default: ['gate', 'barrier'])
        """
        self.ppo_config = create_recurrent_ppo_config(
            self.config, self.env_factory, rl_module_spec, policy_mapping_fn, 
            callback_class, num_quantum_dots, 
            train_agent_types
        )
        return self.ppo_config
    
    def train(self, num_iterations: int = None):
        """
        Run training.
        
        Args:
            num_iterations: Number of training iterations (overrides config)
        """
        if self.ppo_config is None:
            raise RuntimeError("Must call setup_training() before train()")
        
        # Use config iterations if not specified
        if num_iterations is None:
            num_iterations = self.config["stopping_criteria"]["training_iteration"]
        
        # Build algorithm using new API
        self.algorithm = self.ppo_config.build()
        
        # Training loop
        for i in range(num_iterations):
            result = self.algorithm.train()
            
            # Log progress
            if i % self.config["logging"]["log_frequency"] == 0:
                print(f"Iteration {i}: "
                      f"reward_mean={result.get('episode_reward_mean', 'N/A')}, "
                      f"len_mean={result.get('episode_len_mean', 'N/A')}")
            
            # Save checkpoint
            if i % self.config["checkpointing"]["frequency"] == 0:
                checkpoint_path = self.algorithm.save()
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Check stopping criteria
            if self._should_stop(result):
                print(f"Stopping criteria met at iteration {i}")
                break
        
        return self.algorithm
    
    def _should_stop(self, result: Dict[str, Any]) -> bool:
        """Check if training should stop based on criteria."""
        criteria = self.config["stopping_criteria"]
        
        # Check reward threshold
        if "episode_reward_mean" in criteria:
            if result.get("episode_reward_mean", -float('inf')) >= criteria["episode_reward_mean"]:
                return True
        
        # Check timesteps
        if "timesteps_total" in criteria:
            if result.get("timesteps_total", 0) >= criteria["timesteps_total"]:
                return True
        
        return False
    
    def cleanup(self):
        """Cleanup resources."""
        if self.algorithm:
            self.algorithm.stop()
        ray.shutdown() 


if __name__ == "__main__":
    """Simple test for SingleAgentRecurrentPPOModel with time dimension."""
    import torch
    from gymnasium import spaces
    import numpy as np
    from ray.rllib.policy.sample_batch import SampleBatch
    
    print("Testing SingleAgentRecurrentPPOModel with time dimension")
    print("=" * 60)
    
    # Create test spaces (gate agent with 2 channels)
    obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(128, 128, 2),  # 2 channels for gate agents
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
    
    model_config = {
        "lstm_cell_size": 64,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        "fcnet_hiddens": [128, 128]
    }
    
    # Create model
    try:
        model = SingleAgentRecurrentPPOModel(
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        exit(1)
    
    # Test with 4D tensors (no time dimension)
    print("\n--- Testing 4D input (no time dimension) ---")
    batch_size = 4
    
    batch_4d = {
        SampleBatch.OBS: {
            'image': torch.randn(batch_size, 128, 128, 2),  # (B, H, W, C)
            'voltage': torch.randn(batch_size, 1)           # (B, 1)
        },
        'state_in': {
            'h_state': torch.zeros(1, batch_size, 64),
            'c_state': torch.zeros(1, batch_size, 64)
        },
        SampleBatch.PREV_ACTIONS: torch.randn(batch_size, 1),
        SampleBatch.PREV_REWARDS: torch.randn(batch_size)
    }
    
    try:
        output_4d = model._forward_train(batch_4d)
        print(f"✓ 4D forward pass successful")
        print(f"  Action logits shape: {output_4d[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Value predictions shape: {output_4d[SampleBatch.VF_PREDS].shape}")
    except Exception as e:
        print(f"✗ 4D forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with 5D tensors (with time dimension)
    print("\n--- Testing 5D input (with time dimension) ---")
    batch_size = 4
    seq_len = 3
    
    batch_5d = {
        SampleBatch.OBS: {
            'image': torch.randn(batch_size, seq_len, 128, 128, 2),  # (B, T, H, W, C)
            'voltage': torch.randn(batch_size, seq_len, 1)           # (B, T, 1)
        },
        'state_in': {
            'h_state': torch.zeros(1, batch_size, 64),
            'c_state': torch.zeros(1, batch_size, 64)
        },
        SampleBatch.PREV_ACTIONS: torch.randn(batch_size, seq_len, 1),  # (B, T, 1) - Fixed!
        SampleBatch.PREV_REWARDS: torch.randn(batch_size, seq_len)      # (B, T) - Fixed!
    }
    
    try:
        output_5d = model._forward_train(batch_5d)
        print(f"✓ 5D forward pass successful")
        print(f"  Action logits shape: {output_5d[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Value predictions shape: {output_5d[SampleBatch.VF_PREDS].shape}")
    except Exception as e:
        print(f"✗ 5D forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test barrier agent (1 channel)
    print("\n--- Testing Barrier Agent (1 channel) ---")
    barrier_obs_space = spaces.Dict({
        'image': spaces.Box(
            low=0.0, high=1.0,
            shape=(128, 128, 1),  # 1 channel for barrier agents
            dtype=np.float32
        ),
        'voltage': spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    })
    
    try:
        barrier_model = SingleAgentRecurrentPPOModel(
            observation_space=barrier_obs_space,
            action_space=action_space,
            model_config=model_config
        )
        
        batch_barrier = {
            SampleBatch.OBS: {
                'image': torch.randn(batch_size, 128, 128, 1),  # (B, H, W, 1)
                'voltage': torch.randn(batch_size, 1)           # (B, 1)
            },
            'state_in': {
                'h_state': torch.zeros(1, batch_size, 64),
                'c_state': torch.zeros(1, batch_size, 64)
            }
        }
        
        output_barrier = barrier_model._forward_train(batch_barrier)
        print(f"✓ Barrier agent forward pass successful")
        print(f"  Action logits shape: {output_barrier[SampleBatch.ACTION_DIST_INPUTS].shape}")
        print(f"  Value predictions shape: {output_barrier[SampleBatch.VF_PREDS].shape}")
        
    except Exception as e:
        print(f"✗ Barrier agent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test forward_exploration with t parameter
    print("\n--- Testing forward_exploration with t parameter ---")
    try:
        output_explore = model._forward_exploration(batch_4d, t=1000)
        print(f"✓ forward_exploration with t=1000 successful")
        print(f"  Action logits shape: {output_explore[SampleBatch.ACTION_DIST_INPUTS].shape}")
    except Exception as e:
        print(f"✗ forward_exploration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")