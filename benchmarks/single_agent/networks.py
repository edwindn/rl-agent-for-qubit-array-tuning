from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path
import numpy as np
import yaml

from ray.rllib.core.models.base import ENCODER_OUT, Encoder, tokenize
from ray.rllib.core.models.configs import CNNEncoderConfig, MLPHeadConfig, ModelConfig
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.columns import Columns
from ray.rllib.models.utils import get_initializer_fn
from ray.rllib.utils.framework import try_import_torch
import tree

torch, nn = try_import_torch()
import torch.nn.functional as F

current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

from swarm.voltage_model.custom_neural_nets import (
    SimpleCNN,
    SimpleCNNConfig,
)


# Load configuration from config.yaml
def load_config():
    """Load network configuration from config.yaml."""
    config_path = current_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please ensure config.yaml exists in the benchmarks/single_agent directory."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'neural_networks' not in config:
        raise ValueError("Configuration file must contain 'neural_networks' section")

    if 'single_agent_policy' not in config['neural_networks']:
        raise ValueError("Configuration must contain 'single_agent_policy' under 'neural_networks'")

    return config['neural_networks']['single_agent_policy']


# Global config loaded once at module import
_CONFIG = load_config()


class InputEncoderConfig(CNNEncoderConfig):

    num_input_scans: Optional[int] = None
    feature_size: Optional[int] = None
    cnn_activation: Optional[str] = None

    def __post_init__(self):
        """Load values from config.yaml - no defaults allowed."""
        encoder_config = _CONFIG.get('encoder', {})

        if self.num_input_scans is None:
            if 'num_input_scans' not in encoder_config:
                raise ValueError("num_input_scans must be specified in config.yaml under encoder")
            self.num_input_scans = encoder_config['num_input_scans']

        if self.feature_size is None:
            if 'feature_size' not in encoder_config:
                raise ValueError("feature_size must be specified in config.yaml under encoder")
            self.feature_size = encoder_config['feature_size']

        if self.cnn_activation is None:
            if 'cnn_activation' not in encoder_config:
                raise ValueError("cnn_activation must be specified in config.yaml under encoder")
            self.cnn_activation = encoder_config['cnn_activation']

    @property
    def output_dims(self):
        if self.feature_size is None:
            raise ValueError("feature_size not set - __post_init__ was not called")
        return (self.feature_size,)

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return InputEncoder(self)


class InputEncoder(TorchModel, Encoder):
    """Single-agent encoder for processing all scans at once."""

    def __init__(self, config: InputEncoderConfig):

        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Create SimpleCNN config and build the CNN
        cnn_config = SimpleCNNConfig(
            input_dims=config.input_dims,
            feature_size=config.feature_size // config.num_input_scans,  # Each scan contributes a portion
            cnn_activation=config.cnn_activation
        )
        self.cnn = cnn_config.build()

        # Calculate CNN output size
        self._cnn_output_size = cnn_config.feature_size

        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size * self.config.num_input_scans, config.feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )

        self._output_dims = (config.feature_size,)

    @property
    def output_dims(self):
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        """
        Process observation from QuantumDeviceEnv into single-agent features.

        Expected input format from QuantumDeviceEnv:
        {
            "image": (H, W, N-1),  # All N-1 charge stability diagrams
            "obs_gate_voltages": (N,),
            "obs_barrier_voltages": (N-1,)
        }

        We extract each channel from the image array and process separately,
        ignoring voltage observations.
        """
        # Handle nested dict structure from RLlib
        if isinstance(inputs, dict) and "obs" in inputs:
            inputs = inputs["obs"]

        if not isinstance(inputs, dict):
            raise ValueError(f"Expected dict input from QuantumDeviceEnv, got {type(inputs)}")

        # Extract the multi-channel image
        if "image" not in inputs:
            raise ValueError(f"Expected 'image' key in observation, got keys: {list(inputs.keys())}")

        image = inputs["image"]

        # Convert numpy to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # image shape: (H, W, N-1) or (B, H, W, N-1) if batched
        if image.dim() == 3:
            # Add batch dimension: (H, W, N-1) -> (1, H, W, N-1)
            image = image.unsqueeze(0)

        batch_size, height, width, num_channels = image.shape

        if num_channels != self.config.num_input_scans:
            raise ValueError(
                f"Expected {self.config.num_input_scans} channels in image, "
                f"got {num_channels} (shape: {image.shape})"
            )

        # Process each channel through CNN separately
        features = []
        for channel_idx in range(num_channels):
            # Extract single channel: (B, H, W, 1)
            channel_image = image[:, :, :, channel_idx:channel_idx+1]

            # Process through CNN
            out = self.cnn._forward(channel_image)
            features.append(out[ENCODER_OUT])

        # Concatenate all features and pass through final MLP
        features = torch.cat(features, dim=-1)  # (B, num_scans * cnn_feature_size)
        latent = self.final_mlp(features)  # (B, feature_size)
        return {ENCODER_OUT: latent}


@dataclass
class ValueHeadConfig(MLPHeadConfig):
    """Value head configuration for single-agent quantum device RL."""

    hidden_layers: Optional[List[int]] = None
    activation: Optional[str] = None
    use_attention: Optional[bool] = None
    num_outputs: Optional[int] = None

    def __post_init__(self):
        """Load values from config.yaml - no defaults allowed."""
        value_head_config = _CONFIG.get('value_head', {})

        if self.hidden_layers is None:
            if 'hidden_layers' not in value_head_config:
                raise ValueError("hidden_layers must be specified in config.yaml under value_head")
            self.hidden_layers = value_head_config['hidden_layers']

        if self.activation is None:
            if 'activation' not in value_head_config:
                raise ValueError("activation must be specified in config.yaml under value_head")
            self.activation = value_head_config['activation']

        if self.use_attention is None:
            if 'use_attention' not in value_head_config:
                raise ValueError("use_attention must be specified in config.yaml under value_head")
            self.use_attention = value_head_config['use_attention']

        if self.num_outputs is None:
            if 'num_outputs' not in value_head_config:
                raise ValueError("num_outputs must be specified in config.yaml under value_head")
            self.num_outputs = value_head_config['num_outputs']

        self.hidden_layer_dims = self.hidden_layers
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = self.num_outputs

    def build(self, framework: str = "torch") -> "ValueHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return ValueHead(self)


class ValueHead(TorchModel):
    """Value head for single-agent quantum device RL with optional attention mechanism.

    Outputs num_outputs values - one for each scan/agent in the single-agent setup.
    """

    def __init__(self, config: ValueHeadConfig):
        super().__init__(config)

        self.config = config

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.num_outputs))

        self.mlp = nn.Sequential(*layers)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )

        self._output_dims = (config.num_outputs,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)

        return self.mlp(inputs)


@dataclass
class PolicyHeadConfig(MLPHeadConfig):
    """Policy head configuration for single-agent quantum device control."""

    hidden_layers: Optional[List[int]] = None
    activation: Optional[str] = None
    use_attention: Optional[bool] = None
    num_outputs: Optional[int] = None

    def __post_init__(self):
        """Load values from config.yaml - no defaults allowed."""
        policy_head_config = _CONFIG.get('policy_head', {})

        if self.hidden_layers is None:
            if 'hidden_layers' not in policy_head_config:
                raise ValueError("hidden_layers must be specified in config.yaml under policy_head")
            self.hidden_layers = policy_head_config['hidden_layers']

        if self.activation is None:
            if 'activation' not in policy_head_config:
                raise ValueError("activation must be specified in config.yaml under policy_head")
            self.activation = policy_head_config['activation']

        if self.use_attention is None:
            if 'use_attention' not in policy_head_config:
                raise ValueError("use_attention must be specified in config.yaml under policy_head")
            self.use_attention = policy_head_config['use_attention']

        if self.num_outputs is None:
            if 'num_outputs' not in policy_head_config:
                raise ValueError("num_outputs must be specified in config.yaml under policy_head")
            self.num_outputs = policy_head_config['num_outputs']

        self.hidden_layer_dims = self.hidden_layers
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = self.num_outputs

    def build(self, framework: str = "torch") -> "PolicyHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return PolicyHead(self)


class PolicyHead(TorchModel):
    """Policy head for single-agent quantum device control with optional attention.

    Outputs num_outputs action values - one for each scan/agent in the single-agent setup.
    """

    def __init__(self, config: PolicyHeadConfig):
        super().__init__(config)

        self.config = config

        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims

        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.num_outputs))

        self.mlp = nn.Sequential(*layers)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )

        self._output_dims = (config.num_outputs,)

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)

        x = self.mlp(inputs)
        return x