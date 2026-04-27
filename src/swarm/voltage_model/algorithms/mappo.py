"""MAPPO (Multi-Agent PPO with centralized critic) integration for RLlib.

The actor side is identical to IPPO: each policy reads its own per-agent
``{image, voltage}``. The critic side is centralized: each policy's value head
consumes ``{global_image, global_voltages}`` shared across all agents.

The encoder-level change is a custom routing encoder that splits the Dict
observation by key and feeds each sub-encoder. The PPO learner's normal path
(``_forward_train`` writes ``EMBEDDINGS = encoder_outs[CRITIC]``, learner passes
those into ``compute_values``) works automatically. We additionally override
``compute_values`` so the *fallback* path (``embeddings=None``, used by GAE
bootstrap and any learner code that calls ``compute_values`` without first
running ``_forward_train``) also routes through the full encoder rather than
calling ``self.encoder.critic_encoder`` directly with a per-agent obs slice.

Expected ``observation_space`` (registered in ``factory.create_rl_module_spec``
when ``algo == 'mappo'``)::

    Dict({
        'image':            Box(H, W, agent_channels),   # 2 (plunger) or 1 (barrier)
        'voltage':          Box(1,),
        'global_image':     Box(H, W, num_dots - 1),
        'global_voltages':  Box(num_agents,),            # gates + barriers
    })

Expected ``model_config_dict`` shape (per policy)::

    {
        'backbone':           {...},   # actor encoder spec (same as IPPO)
        'policy_head':        {...},
        'value_head':         {...},   # unused; centralized_critic.value_head wins
        'centralized_critic': {
            'backbone':       {...},   # critic encoder spec
            'value_head':     {...},
        },
        ...
    }
"""

from dataclasses import dataclass

import gymnasium as gym
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import (
    build_encoder_config,
    get_head_input_dim,
)
from swarm.voltage_model.algorithms.ppo import CustomPPOCatalog
from swarm.voltage_model.configs import ValueHeadConfig


class MAPPORoutingEncoder(TorchModel, Encoder):
    """ActorCritic-style encoder that routes Dict obs by key.

    Mirrors RLlib's ``ActorCriticEncoder`` (``ray/rllib/core/models/base.py``)
    so that ``DefaultPPOTorchRLModule.compute_values`` automatically picks up
    ``self.critic_encoder``. The only behavioural difference is that the actor
    sub-encoder receives the per-agent ``{image, voltage}`` slice while the
    critic sub-encoder receives ``{image: global_image, voltage: global_voltages}``.
    """

    framework = "torch"

    def __init__(self, config: "MAPPOEncoderConfig") -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.actor_encoder = config.actor_encoder_config.build(framework="torch")
        self.critic_encoder = None
        if not config.inference_only:
            self.critic_encoder = config.critic_encoder_config.build(framework="torch")

    @override(Encoder)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        obs = inputs[Columns.OBS]
        actor_inputs = {
            **inputs,
            Columns.OBS: {"image": obs["image"], "voltage": obs["voltage"]},
        }
        actor_out = self.actor_encoder(actor_inputs, **kwargs)

        result = {ACTOR: actor_out[ENCODER_OUT]}
        if self.critic_encoder is not None:
            critic_inputs = {
                **inputs,
                Columns.OBS: {
                    "image": obs["global_image"],
                    "voltage": obs["global_voltages"],
                },
            }
            critic_out = self.critic_encoder(critic_inputs, **kwargs)
            result[CRITIC] = critic_out[ENCODER_OUT]

        return {ENCODER_OUT: result}


@dataclass
class MAPPOEncoderConfig(ModelConfig):
    """Encoder config for ``MAPPORoutingEncoder``.

    Holds two pre-built sub-encoder configs. ``output_dims`` reports the actor's
    output dims so ``Catalog.latent_dims`` is consistent with the policy head's
    expected input.
    """

    actor_encoder_config: ModelConfig = None
    critic_encoder_config: ModelConfig = None
    inference_only: bool = False

    @property
    def output_dims(self):
        return self.actor_encoder_config.output_dims

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return MAPPORoutingEncoder(self)


class CustomMAPPOCatalog(CustomPPOCatalog):
    """PPO catalog with a centralized critic.

    Reads centralized critic spec from ``model_config_dict['centralized_critic']``
    (keys: ``backbone``, ``value_head`` — same shape as the per-policy block).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(
                f"CustomMAPPOCatalog requires a Dict observation_space, got "
                f"{type(observation_space).__name__}"
            )
        for key in ("image", "voltage", "global_image", "global_voltages"):
            if key not in observation_space.spaces:
                raise KeyError(
                    f"observation_space missing required key {key!r}; got "
                    f"{list(observation_space.spaces.keys())}"
                )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        # Build the centralized critic encoder config. Reuses the standard
        # backbone helper; just sized against the global image Box and driven
        # by the centralized_critic config block.
        self._critic_encoder_config = build_encoder_config(
            observation_space=observation_space["global_image"],
            model_config_dict=model_config_dict["centralized_critic"],
        )

    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        # Actor encoder is built against the per-agent image Box only.
        # voltage is plumbed through the encoder unchanged at runtime.
        return build_encoder_config(observation_space["image"], model_config_dict)

    @override(PPOCatalog)
    def build_actor_critic_encoder(self, framework: str):
        # ``DefaultPPORLModule.setup`` mutates
        # ``self.actor_critic_encoder_config.inference_only`` for inference-only
        # rollout modules. We piggyback on that flag to skip building the
        # centralized critic encoder when it won't be used.
        inference_only = self.actor_critic_encoder_config.inference_only
        cfg = MAPPOEncoderConfig(
            actor_encoder_config=self._encoder_config,
            critic_encoder_config=self._critic_encoder_config,
            inference_only=inference_only,
        )
        return cfg.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        cc = self._model_config_dict["centralized_critic"]
        value_config = cc["value_head"]
        input_dim = get_head_input_dim(cc)
        # Centralized critic concatenates all agent voltages, so the value
        # head's voltage embedding is sized by the global_voltages Box.
        voltage_dim = self.observation_space["global_voltages"].shape[0]
        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            use_attention=value_config["use_attention"],
            voltage_dim=voltage_dim,
        )
        return config.build(framework=framework)


class CustomMAPPOTorchRLModule(DefaultPPOTorchRLModule):
    """PPO RLModule with a centralized critic.

    Two differences vs ``DefaultPPOTorchRLModule``:

    1. The ``embeddings=None`` fallback (GAE bootstrap) is routed through the
       full ``MAPPORoutingEncoder._forward`` so the Dict obs gets split into
       ``{image, voltage}`` (actor) vs ``{global_image, global_voltages}``
       (critic). The parent calls ``self.encoder.critic_encoder(batch)``
       directly, which would feed a per-agent slice into the centralized
       encoder.
    2. The fallback runs under ``torch.no_grad()``. The GAE connector
       immediately converts the output to numpy
       (``general_advantage_estimation.py:112``), so the autograd graph for
       the centralized critic forward (IMPALA on the full ``(H, W, N-1)``
       image, batch of thousands) is wasted memory and triggered CUDA OOM
       on a 20 GiB card. The training-time path (``embeddings`` non-None,
       called from ``ppo_torch_learner.py:96``) still runs with grads since
       only ``self.vf`` runs there — the encoder grads were already produced
       in ``_forward_train``.
    """

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            with torch.no_grad():
                embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]
                return self.vf(embeddings).squeeze(-1)
        return self.vf(embeddings).squeeze(-1)
