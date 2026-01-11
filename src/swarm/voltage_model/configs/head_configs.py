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
