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
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Type
import numpy as np

torch, nn = try_import_torch()

"""
todo:

update obs space to use actual keys

update nets to use custom models

add stopping model
"""

class SingleAgentRecurrentPPOModel(TorchRLModule):
    """
    Single-agent RLModule with LSTM memory for individual quantum device agents.
    Each agent sees a single voltage and a one-channel (two-channel) image if it is a gate (barrier)
    and outputs a single voltage.
    """
    
    def __init__(self, observation_space=None, action_space=None, inference_only=False, 
                 learner_only=False, model_config=None, catalog_class=None, config=None, **kwargs):
        # Handle both new and old API formats
        if config is not None:
            # Old format: config dictionary
            obs_space = config.get("observation_space", observation_space)
            action_space = config.get("action_space", action_space)
            model_config = config.get("model_config_dict", config.get("model_config", model_config or {}))
            inference_only = config.get("inference_only", inference_only)
            learner_only = config.get("learner_only", learner_only)
            catalog_class = config.get("catalog_class", catalog_class)
        else:
            # New format: individual parameters
            obs_space = observation_space
            model_config = model_config or {}
        
        # Filter out None values to avoid passing them to parent
        parent_kwargs = {}
        if obs_space is not None:
            parent_kwargs["observation_space"] = obs_space
        if action_space is not None:
            parent_kwargs["action_space"] = action_space
        if inference_only is not None:
            parent_kwargs["inference_only"] = inference_only
        if learner_only is not None:
            parent_kwargs["learner_only"] = learner_only
        if model_config is not None:
            parent_kwargs["model_config"] = model_config
        if catalog_class is not None:
            parent_kwargs["catalog_class"] = catalog_class
        
        # Call parent with new API format
        super().__init__(**parent_kwargs)
        
        # LSTM configuration
        self.lstm_cell_size = model_config["lstm_cell_size"]
        self.use_prev_action = model_config.get("lstm_use_prev_action", False)
        self.use_prev_reward = model_config.get("lstm_use_prev_reward", False)
        
        self.action_size = 1
        self.num_voltages = 1
        
        # Parse observation space
        if isinstance(obs_space, spaces.Dict):
            self.is_multimodal = True
            self.image_space = obs_space['image']
            self.voltage_space = obs_space['voltage']  # Single voltage for this agent
        else:
            raise ValueError("Observation space should be a Dict")


        # Handle single-channel image observations
        image_shape = self.image_space.shape  # Should be (H, W, C)
        if len(image_shape) == 3:
            # Convert (height, width, channels) -> (channels, height, width)
            if image_shape[2] in [1, 2]:  # Single channel
                self.image_channels = image_shape[2]
                self.image_input_shape = (self.image_channels, image_shape[0], image_shape[1])
            else:
                raise ValueError(f"Unexpected image shape: {image_shape}")
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}")
        
        # Convolutional layers for single-channel image processing
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
        
        self.voltage_fc = nn.Linear(self.num_voltages, 32)
        
        # Combined feature size
        combined_size = conv_output_size + 32
        
        # Shared layers before LSTM
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
        
        # LSTM layer
        lstm_input_size = prev_size
        if self.use_prev_action:
            lstm_input_size += self.action_size
        if self.use_prev_reward:
            lstm_input_size += 1
            
        self.lstm = nn.LSTM(lstm_input_size, self.lstm_cell_size, batch_first=True)
        
        # Value function head
        self.value_head = nn.Linear(self.lstm_cell_size, 1)
        
        # Policy head - outputs single voltage value
        self.policy_head = nn.Linear(self.lstm_cell_size, self.action_size)

    
    def _forward_inference(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """Forward pass for inference."""
        return self.forward(batch, None, None)
    
    def _forward_exploration(self, batch: Dict[str, TensorType], t: int = None, **kwargs) -> Dict[str, TensorType]:
        """Forward pass for exploration."""
        return self.forward(batch, None, None)
    
    def _forward_train(self, batch: Dict[str, TensorType], t: int | None = None) -> Dict[str, TensorType]:
        """Forward pass for training."""
        return self.forward(batch, None, None)
    
    def forward(self, batch: Dict[str, TensorType], state=None, seq_lens=None) -> Dict[str, TensorType]:
        """Forward pass for training."""
        # use time parameter t for modulating exploration / behaviour over time
        obs = batch[SampleBatch.OBS]

        print("[DEBUG]: agent forward pass")
        
        # Process single/dual-channel image
        image = obs["image"]  # Shape: (batch_size, H, W, C) where C is 1 or 2, or (batch_size, seq_len, H, W, C)
        voltage = obs["voltage"]  # Shape (batch_size, N) or (batch_size, seq_len, N)

        print(image.shape)
        print(voltage.shape)

        if image.ndim == 3:
            image = image.unsqueeze(0)

        has_time_dim = False
        if image.ndim == 5:
            has_time_dim = True
            B, T = image.size(0), image.size(1)
            assert voltage.ndim == 3, "Image has time dimension but voltage does not"

            image = image.flatten(0, 1)
            voltage = voltage.flatten(0, 1)
        else:
            assert image.ndim == 4, f"Unexpected image size: {image.shape}"
            assert voltage.ndim == 2, f"Unexpected voltage size: {voltage.shape}"

        # Reshape image: (batch_size, H, W, C) -> (batch_size, C, H, W)
        if image.shape[-1] in [1, 2]:  # channels last -> channels first
            # image = image.swapaxes(-1, -3).swapaxes(-1, -2) # permutes ignoring the first few dims
            image = image.permute(0, 3, 1, 2)
        
        print('forward passes')
        print(image.shape)
        print(voltage.shape)
        image_features = self.conv_layers(image)
        
        # Process single voltage value
        voltage_features = torch.relu(self.voltage_fc(voltage))
        
        # Combine features
        combined = torch.cat([image_features, voltage_features], dim=1)
        
        # Shared processing
        shared_out = self.shared_layers(combined)

        # Prepare LSTM input
        lstm_input = shared_out
        
        # Add previous action if configured
        if self.use_prev_action:
            if SampleBatch.PREV_ACTIONS in batch:
                prev_actions = batch[SampleBatch.PREV_ACTIONS].float()
                
                # Handle time dimension properly
                if has_time_dim:
                    if prev_actions.ndim == 3:  # (B, T, action_size)
                        prev_actions = prev_actions.flatten(0, 1)  # (B*T, action_size)
                    elif prev_actions.ndim == 2:  # (B, action_size)
                        raise ValueError("prev_actions must have 3 dimensions when has_time_dim is True")
                elif len(prev_actions.shape) > 2:
                    prev_actions = prev_actions.view(-1, prev_actions.shape[-1])
            else:
                # Pad with zeros if previous actions are expected but not provided
                batch_size = lstm_input.shape[0]
                prev_actions = torch.zeros(batch_size, self.action_size, device=lstm_input.device)
            
            lstm_input = torch.cat([lstm_input, prev_actions], dim=-1)
        
        # Add previous reward if configured
        if self.use_prev_reward:
            if SampleBatch.PREV_REWARDS in batch:
                prev_rewards = batch[SampleBatch.PREV_REWARDS].float()
                
                # Handle time dimension properly
                if has_time_dim:
                    # prev_rewards should be (B, T), need (B*T, 1)
                    if prev_rewards.ndim == 2:  # (B, T)
                        prev_rewards = prev_rewards.flatten(0, 1).unsqueeze(-1)  # (B*T, 1)
                    elif prev_rewards.ndim == 1:  # (B,) - need to expand
                        prev_rewards = prev_rewards.unsqueeze(1).expand(-1, T)  # (B, T)
                        prev_rewards = prev_rewards.flatten(0, 1).unsqueeze(-1)  # (B*T, 1)
                else:
                    prev_rewards = prev_rewards.unsqueeze(-1) if prev_rewards.ndim == 1 else prev_rewards
            else:
                # Pad with zeros if previous rewards are expected but not provided
                batch_size = lstm_input.shape[0]
                prev_rewards = torch.zeros(batch_size, 1, device=lstm_input.device)
            lstm_input = torch.cat([lstm_input, prev_rewards], dim=-1)
        
        print(lstm_input.shape)
        # Reshape for LSTM (batch_size, seq_len, features)
        if len(lstm_input.shape) == 2:
            if has_time_dim:
                lstm_input = lstm_input.view(B, T, -1)
            else:
                lstm_input = lstm_input.unsqueeze(1)
        print(lstm_input.shape)
        
        # Handle LSTM state for RLModule API
        if "state_in" in batch and batch["state_in"]:
            # RLModule passes state as dict with h_state and c_state keys
            if isinstance(batch["state_in"], dict):
                h_state = batch["state_in"].get("h_state")
                c_state = batch["state_in"].get("c_state")
            else:
                # Fallback for list format
                state = batch["state_in"]
                h_state, c_state = state[0], state[1]
        else:
            # Initialize hidden state
            batch_size = lstm_input.shape[0]
            h_state = torch.zeros(1, batch_size, self.lstm_cell_size, device=lstm_input.device)
            c_state = torch.zeros(1, batch_size, self.lstm_cell_size, device=lstm_input.device)
        
        lstm_out, (new_h, new_c) = self.lstm(lstm_input, (h_state, c_state))
        lstm_out = lstm_out.squeeze(1)  # Remove sequence dimension
        
        # Policy output - single voltage value
        action_logits = self.policy_head(lstm_out)
        
        # Value output
        vf_out = self.value_head(lstm_out).squeeze(-1)
        
        return {
            SampleBatch.ACTION_DIST_INPUTS: action_logits,
            SampleBatch.VF_PREDS: vf_out,
            "state_out": {
                "h_state": new_h,
                "c_state": new_c
            },
        }
    
    def get_initial_state(self) -> Dict[str, TensorType]:
        """Return initial LSTM state for RLModule."""
        return {
            "h_state": torch.zeros(1, 1, self.lstm_cell_size),
            "c_state": torch.zeros(1, 1, self.lstm_cell_size)
        }



def create_recurrent_ppo_config(config: Dict[str, Any], env_factory, rl_module_spec, 
                     policy_mapping_fn, callback_class, 
                     num_quantum_dots: int = 8, train_agent_types: list = None) -> PPOConfig:
    """
    Create PPO configuration for multi-agent training using RLlib's default PPO module
    with custom Dict observation space encoder.
    
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
    from .custom_dict_encoder import CustomDictPPOCatalog
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
    
    # Default to training both agent types if not specified
    if train_agent_types is None:
        train_agent_types = ['gate', 'barrier']
    
    # Get PPO configuration with any overrides from training config
    ppo_overrides = config.get("ppo_overrides", {})
    ppo_config_dict = get_ppo_config(ppo_overrides)
    
    # Configure model for RLModule - use DefaultModelConfig compatible settings
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    
    # Create model configuration that works with our custom encoder
    model_config_dict = {
        # LSTM configuration
        "use_lstm": config["agent_models"]["use_lstm"],
        "lstm_cell_size": config["agent_models"]["lstm_cell_size"],
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": True,
        
        # Feed-forward layers (used by our custom encoder)
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        
        # Shared layers between actor and critic
        "vf_share_layers": True,
        
        # Standard PPO head configuration
        "head_fcnet_hiddens": ppo_config_dict["model"].get("head_fcnet_hiddens", []),
        "head_fcnet_activation": ppo_config_dict["model"].get("head_fcnet_activation", "relu"),
        
        # Other PPO model settings
        "free_log_std": ppo_config_dict["model"].get("free_log_std", False),
        "log_std_clip_param": ppo_config_dict["model"].get("log_std_clip_param", 20.0),
    }

    # Update the RLModule specs to use DefaultPPOTorchRLModule with custom catalog
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
        
        # Create new SingleAgentRLModuleSpec using RLlib's default PPO module with custom catalog
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec as SingleAgentRLModuleSpec
        updated_spec = SingleAgentRLModuleSpec(
            module_class=DefaultPPOTorchRLModule,  # Use RLlib's default PPO module
            observation_space=single_spec.observation_space,
            action_space=single_spec.action_space,
            model_config=model_config_dict,
            catalog_class=CustomDictPPOCatalog,  # Use our custom catalog for Dict observations
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