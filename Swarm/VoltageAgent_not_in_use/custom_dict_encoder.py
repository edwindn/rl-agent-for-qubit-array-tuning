"""
Custom encoder and catalog for handling Dict observation spaces in RLlib PPO.
Handles quantum device observations with {image, voltage} keys while leveraging
RLlib's default LSTM and action/value head implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import gymnasium as gym
from gymnasium import spaces

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import ActorCriticEncoder, ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.models.configs import ActorCriticEncoderConfig
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class DictObservationActorCriticEncoder(ActorCriticEncoder):
    """
    Custom ActorCriticEncoder that handles Dict observation spaces with 'image' and 'voltage' keys.
    
    This encoder processes:
    - 'image': Multi-channel images (1 or 2 channels) via CNN
    - 'voltage': Single voltage value via MLP
    
    Combines features and feeds them to shared/separate actor-critic networks with LSTM support.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config: Dict[str, Any],
        shared: bool = True,
        framework: str = "torch",
    ):
        """
        Initialize the custom Dict encoder.
        
        Args:
            observation_space: Dict space with 'image' and 'voltage' keys
            action_space: Action space for the agent
            model_config: Model configuration dictionary
            shared: Whether to share encoder between actor and critic
            framework: Framework to use ("torch")
        """
        super().__init__(framework=framework)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_config = model_config
        self.shared = shared
        
        # Validate observation space
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("observation_space must be a Dict space")
        if 'image' not in observation_space.spaces or 'voltage' not in observation_space.spaces:
            raise ValueError("observation_space must contain 'image' and 'voltage' keys")
        
        self.image_space = observation_space['image']
        self.voltage_space = observation_space['voltage']
        
        # Parse image dimensions
        image_shape = self.image_space.shape  # Should be (H, W, C)
        if len(image_shape) == 3:
            if image_shape[2] in [1, 2]:  # 1 or 2 channels
                self.image_channels = image_shape[2]
                self.image_height, self.image_width = image_shape[0], image_shape[1]
            else:
                raise ValueError(f"Unexpected image channels: {image_shape[2]}. Expected 1 or 2.")
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}. Expected (H, W, C).")
        
        # Parse voltage dimensions
        if len(self.voltage_space.shape) != 1 or self.voltage_space.shape[0] != 1:
            raise ValueError(f"Expected voltage space shape (1,), got {self.voltage_space.shape}")
        
        # Build feature extractors
        self._build_feature_extractors()
        
        # Build encoder networks
        self._build_encoders()
    
    def _build_feature_extractors(self):
        """Build CNN for image and MLP for voltage processing."""
        # CNN for image processing
        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.image_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_image = torch.zeros(1, self.image_channels, self.image_height, self.image_width)
            conv_output = self.image_encoder(dummy_image)
            self.conv_output_size = conv_output.shape[1]
        
        # MLP for voltage processing
        self.voltage_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined feature size
        self.combined_feature_size = self.conv_output_size + 32
    
    def _build_encoders(self):
        """Build actor and critic encoder networks."""
        # Get configuration
        hidden_dims = self.model_config.get("fcnet_hiddens", [128, 128])
        activation = self.model_config.get("fcnet_activation", "relu")
        use_lstm = self.model_config.get("use_lstm", False)
        lstm_cell_size = self.model_config.get("lstm_cell_size", 256)
        
        # Build shared trunk
        layers = []
        prev_size = self.combined_feature_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_dim),
                self._get_activation(activation)
            ])
            prev_size = hidden_dim
        
        self.shared_trunk = nn.Sequential(*layers)
        
        if self.shared:
            # Shared encoder for both actor and critic
            if use_lstm:
                # Add LSTM layer
                lstm_input_size = prev_size
                if self.model_config.get("lstm_use_prev_action", False):
                    # Add action size (will be handled in forward pass)
                    pass  # Dynamic sizing in forward
                if self.model_config.get("lstm_use_prev_reward", False):
                    # Add reward size (will be handled in forward pass)
                    pass  # Dynamic sizing in forward
                
                self.lstm_cell_size = lstm_cell_size
                self.use_lstm = True
                self.use_prev_action = self.model_config.get("lstm_use_prev_action", False)
                self.use_prev_reward = self.model_config.get("lstm_use_prev_reward", False)
                
                # LSTM will be created dynamically in forward pass based on actual input size
            else:
                self.use_lstm = False
            
            self.latent_dims = lstm_cell_size if use_lstm else prev_size
        else:
            # Separate encoders (not commonly used, but supported)
            self.use_lstm = False
            self.latent_dims = prev_size
            
            # Create separate actor and critic heads
            self.actor_encoder = nn.Linear(prev_size, prev_size)
            self.critic_encoder = nn.Linear(prev_size, prev_size)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    @override(ActorCriticEncoder)
    def _forward(self, inputs: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """
        Forward pass for the Dict observation encoder.
        
        Args:
            inputs: Dictionary containing observations under Columns.OBS key
            
        Returns:
            Dictionary with ENCODER_OUT containing {ACTOR: features, CRITIC: features}
            and optional STATE_OUT for LSTM states
        """
        # Extract observations
        obs = inputs[Columns.OBS]
        
        # Process image: (batch_size, H, W, C) -> (batch_size, C, H, W)
        image = obs["image"]
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension if missing
        
        # Handle time dimension if present (RNN case)
        has_time_dim = False
        if image.ndim == 5:  # (B, T, H, W, C)
            has_time_dim = True
            B, T = image.shape[0], image.shape[1]
            # Flatten time dimension: (B*T, H, W, C)
            image = image.view(B * T, *image.shape[2:])
        
        # Convert to channels-first: (batch, C, H, W)
        if image.shape[-1] in [1, 2]:  # channels last -> channels first
            image = image.permute(0, 3, 1, 2)
        
        # Process voltage
        voltage = obs["voltage"]
        if has_time_dim:
            voltage = voltage.view(B * T, -1)
        
        # Extract features
        image_features = self.image_encoder(image)
        voltage_features = self.voltage_encoder(voltage)
        
        # Combine features
        combined_features = torch.cat([image_features, voltage_features], dim=-1)
        
        # Apply shared trunk
        trunk_output = self.shared_trunk(combined_features)
        
        # Handle LSTM if enabled
        if self.use_lstm:
            # Prepare LSTM input
            lstm_input = trunk_output
            
            # Add previous action if configured
            if self.use_prev_action and Columns.PREV_ACTIONS in inputs:
                prev_actions = inputs[Columns.PREV_ACTIONS].float()
                if has_time_dim and prev_actions.ndim == 3:
                    prev_actions = prev_actions.view(B * T, -1)
                elif not has_time_dim and prev_actions.ndim > 2:
                    prev_actions = prev_actions.view(-1, prev_actions.shape[-1])
                lstm_input = torch.cat([lstm_input, prev_actions], dim=-1)
            
            # Add previous reward if configured
            if self.use_prev_reward and Columns.PREV_REWARDS in inputs:
                prev_rewards = inputs[Columns.PREV_REWARDS].float()
                if has_time_dim and prev_rewards.ndim == 2:
                    prev_rewards = prev_rewards.view(B * T, -1)
                elif not has_time_dim and prev_rewards.ndim == 1:
                    prev_rewards = prev_rewards.unsqueeze(-1)
                lstm_input = torch.cat([lstm_input, prev_rewards], dim=-1)
            
            # Create LSTM if not already created (dynamic sizing)
            if not hasattr(self, 'lstm'):
                self.lstm = nn.LSTM(lstm_input.shape[-1], self.lstm_cell_size, batch_first=True)
            
            # Reshape for LSTM: (batch_size, seq_len, features)
            if has_time_dim:
                lstm_input = lstm_input.view(B, T, -1)
            else:
                lstm_input = lstm_input.unsqueeze(1)  # Add seq_len dimension
            
            # Handle LSTM state
            if Columns.STATE_IN in inputs and inputs[Columns.STATE_IN] is not None:
                if isinstance(inputs[Columns.STATE_IN], dict):
                    h_state = inputs[Columns.STATE_IN].get("h_state")
                    c_state = inputs[Columns.STATE_IN].get("c_state")
                    lstm_state = (h_state, c_state)
                else:
                    lstm_state = inputs[Columns.STATE_IN]
            else:
                # Initialize hidden state
                batch_size = lstm_input.shape[0]
                h_state = torch.zeros(1, batch_size, self.lstm_cell_size, device=lstm_input.device)
                c_state = torch.zeros(1, batch_size, self.lstm_cell_size, device=lstm_input.device)
                lstm_state = (h_state, c_state)
            
            # LSTM forward pass
            lstm_output, (new_h, new_c) = self.lstm(lstm_input, lstm_state)
            
            # Remove sequence dimension for non-recurrent case
            if not has_time_dim:
                lstm_output = lstm_output.squeeze(1)
            
            final_features = lstm_output
            state_out = {"h_state": new_h, "c_state": new_c}
        else:
            final_features = trunk_output
            state_out = None
        
        # Create output dictionary
        output = {
            ENCODER_OUT: {
                ACTOR: final_features,
                CRITIC: final_features if self.shared else self.critic_encoder(final_features)
            }
        }
        
        if state_out is not None:
            output[Columns.STATE_OUT] = state_out
        
        return output


class CustomDictPPOCatalog(PPOCatalog):
    """
    Custom PPO catalog that builds DictObservationActorCriticEncoder for handling
    Dict observation spaces with 'image' and 'voltage' keys.
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: dict):
        """Initialize catalog with custom encoder configuration."""
        # Validate that we have a Dict observation space
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("CustomDictPPOCatalog requires a Dict observation space")
        
        if 'image' not in observation_space.spaces or 'voltage' not in observation_space.spaces:
            raise ValueError("Dict observation space must contain 'image' and 'voltage' keys")
        
        super().__init__(observation_space, action_space, model_config_dict)
    
    @override(PPOCatalog)
    def build_actor_critic_encoder(self, framework: str) -> ActorCriticEncoder:
        """Build custom Dict observation encoder."""
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got: {framework}")
        
        return DictObservationActorCriticEncoder(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=self._model_config_dict,
            shared=self._model_config_dict.get("vf_share_layers", True),
            framework=framework
        )