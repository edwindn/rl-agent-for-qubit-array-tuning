"""Configuration classes for policy and value heads."""

from dataclasses import dataclass
from typing import List, Optional

from ray.rllib.core.models.configs import MLPHeadConfig


@dataclass
class PolicyHeadConfig(MLPHeadConfig):
    """Policy head configuration for quantum device control."""

    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    log_std_bounds: Optional[List[float]] = None  # [min, max] for log_std clamping
    voltage_dim: int = 1  # Number of voltage inputs (1 for multi-agent, N for single-agent)

    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 128]

        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.heads import PolicyHead
        return PolicyHead(self)


@dataclass
class ValueHeadConfig(MLPHeadConfig):
    """Value head configuration for quantum device RL."""

    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    voltage_dim: int = 1  # Number of voltage inputs (1 for multi-agent, N for single-agent)

    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 64]

        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = 1

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.heads import ValueHead
        return ValueHead(self)


@dataclass
class QValueHeadConfig(MLPHeadConfig):
    """Q-function head configuration for SAC.

    Takes dict input with image_features, voltage, and action.
    Processes voltage the same way as ValueHead for consistency.
    """

    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    action_dim: int = 1  # Dimension of action space

    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [256, 256]

        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = 1

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.heads import QValueHead
        return QValueHead(self)


@dataclass
class DeterministicPolicyHeadConfig(MLPHeadConfig):
    """Policy head configuration for TD3's deterministic policy.

    Unlike PolicyHeadConfig which outputs mean+log_std (2*action_dim),
    this outputs action directly (action_dim) with tanh activation.
    """

    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False

    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 128]

        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "tanh"  # Bound actions to [-1, 1]

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        from swarm.voltage_model.models.heads import DeterministicPolicyHead
        return DeterministicPolicyHead(self)
