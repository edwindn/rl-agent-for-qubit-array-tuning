import torch as th
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, MlpExtractor

"""
some not great layer size handling, to be fixed
"""

class CustomFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int,
        cnn_output_dim: int = 256,
        voltage_dim: int = 1,
        normalized_image: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.cnn = NatureCNN(
            observation_space=observation_space['image'],
            normalized_image=normalized_image
        )
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + voltage_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )


    def forward(self, obs: TensorDict) -> th.Tensor:
        image = obs['image']
        voltages = obs['obs_voltages']

        image_out = self.cnn(image)
        image_out = image_out.reshape(image_out.shape[0], -1)
        latents = th.cat([image_out, voltages], dim=1)
        latents = self.mlp(latents)
        return latents

