"""
Factory for creating RLlib module specifications for the single-agent benchmark.

Note: We use MultiRLModuleSpec even for single-agent training because RLlib's
single-agent PPO code path has different behavior than the multi-agent code path.
The multi-agent path works correctly for our continuous action space setup.
"""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.utils.annotations import override

from swarm.voltage_model.algorithms.common import build_encoder_config, get_head_input_dim
from swarm.voltage_model.algorithms.sac import CustomSACTorchRLModule
from swarm.voltage_model.configs import PolicyHeadConfig, QValueHeadConfig, ValueHeadConfig


class CustomSingleAgentCatalog(PPOCatalog):
    """Single-agent catalog that uses the full voltage vector."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        if isinstance(observation_space, gym.spaces.Dict):
            self.voltage_dim = observation_space["voltage"].shape[0]
        else:
            self.voltage_dim = 1

    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if isinstance(observation_space, gym.spaces.Dict):
            encoder_obs_space = observation_space["image"]
        else:
            encoder_obs_space = observation_space
        return build_encoder_config(encoder_obs_space, model_config_dict)

    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,
            log_std_bounds=self._model_config_dict.get("log_std_bounds", [-10, 2]),
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)

    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        value_config = self._model_config_dict["value_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            use_attention=value_config["use_attention"],
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)


class SingleAgentSACCatalog(SACCatalog):
    """SAC catalog that supports Dict observations with image + voltage."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )
        if isinstance(observation_space, gym.spaces.Dict):
            self.voltage_dim = observation_space["voltage"].shape[0]
        else:
            self.voltage_dim = 1

    @override(SACCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        if isinstance(observation_space, gym.spaces.Dict):
            encoder_obs_space = observation_space["image"]
        else:
            encoder_obs_space = observation_space
        return build_encoder_config(encoder_obs_space, model_config_dict)

    @override(SACCatalog)
    def build_pi_head(self, framework: str = "torch"):
        policy_config = self._model_config_dict["policy_head"]
        input_dim = get_head_input_dim(self._model_config_dict)

        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config["hidden_layers"],
            activation=policy_config["activation"],
            use_attention=policy_config["use_attention"],
            output_layer_dim=self.action_space.shape[0] * 2,
            log_std_bounds=self._model_config_dict.get("log_std_bounds", [-5, 2]),
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        value_config = self._model_config_dict["value_head"]
        encoder_output_dim = get_head_input_dim(self._model_config_dict)
        action_dim = self.action_space.shape[0]

        config = QValueHeadConfig(
            input_dims=(encoder_output_dim,),
            hidden_layers=value_config["hidden_layers"],
            activation=value_config["activation"],
            action_dim=action_dim,
            voltage_dim=self.voltage_dim,
        )

        return config.build(framework=framework)

    @override(SACCatalog)
    def build_qf_encoder(self, framework: str = "torch"):
        return self.build_encoder(framework=framework)


def create_rl_module_spec(env_config: dict, algo: str = "ppo", config: dict = None) -> MultiRLModuleSpec:
    """Create a MultiRLModuleSpec with single agent based on env_config.

    Note: We use MultiRLModuleSpec instead of RLModuleSpec because RLlib's
    single-agent PPO code path has issues with continuous action spaces.
    Using multi-agent config with a single agent uses the correct code path.
    """
    import numpy as np
    from gymnasium import spaces

    algo = algo.lower()
    if algo not in {"ppo", "sac"}:
        raise NotImplementedError("Single-agent benchmark currently supports PPO and SAC only.")

    resolution = env_config["simulator"]["resolution"]
    num_dots = env_config["simulator"]["num_dots"]
    use_barriers = env_config["simulator"]["use_barriers"]
    # TEMPORARY: bypass_barriers - agent only controls gates
    bypass_barriers = env_config["simulator"].get("bypass_barriers", False)
    # TEMPORARY: single_gate_mode - agent only controls ONE gate
    single_gate_mode = env_config["simulator"].get("single_gate_mode", False)
    num_channels = num_dots - 1

    # Determine number of actions based on mode
    if single_gate_mode:
        # Agent only controls ONE gate
        num_actions = 1
    elif bypass_barriers:
        # Agent controls all gates (barriers auto-set)
        num_actions = num_dots
    else:
        num_barriers = num_dots - 1 if use_barriers else 0
        num_actions = num_dots + num_barriers

    image_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(resolution, resolution, num_channels),
        dtype=np.float32,
    )
    voltage_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(num_actions,),
        dtype=np.float32,
    )

    observation_space = spaces.Dict({
        "image": image_space,
        "voltage": voltage_space,
    })

    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(num_actions,),
        dtype=np.float32,
    )

    model_config = config if isinstance(config, dict) else {}
    if "backbone" in model_config:
        backbone = model_config["backbone"]
        memory_layer = backbone.get("memory_layer")
        if memory_layer == "lstm":
            model_config["max_seq_len"] = backbone["lstm"]["max_seq_len"]
        elif memory_layer == "transformer":
            model_config["max_seq_len"] = backbone["transformer"]["max_seq_len"]

    # Select algorithm-specific module + catalog
    if algo == "ppo":
        module_class = DefaultPPOTorchRLModule
        catalog_class = CustomSingleAgentCatalog
    else:
        module_class = CustomSACTorchRLModule
        catalog_class = SingleAgentSACCatalog

    # Create single-agent spec
    agent_spec = RLModuleSpec(
        module_class=module_class,
        observation_space=observation_space,
        action_space=action_space,
        model_config=model_config,
        catalog_class=catalog_class,
        inference_only=False,
    )

    # Wrap in MultiRLModuleSpec with single agent
    return MultiRLModuleSpec(
        rl_module_specs={"agent_0": agent_spec}
    )
