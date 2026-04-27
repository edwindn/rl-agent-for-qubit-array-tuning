"""
Factory for creating RLlib module specifications.

This is the single entry point for creating RL modules for different algorithms.
To add a new algorithm (e.g., TD3), add a new branch in create_rl_module_spec().
"""

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec

from swarm.voltage_model.algorithms import (
    CustomMAPPOCatalog,
    CustomMAPPOTorchRLModule,
    CustomPPOCatalog,
    CustomSACCatalog,
    CustomSACTorchRLModule,
    CustomTD3Catalog,
    CustomTD3TorchRLModule,
)


def create_rl_module_spec(env_config: dict, algo: str = "ppo", config: dict = None) -> MultiRLModuleSpec:
    """
    Create policy specifications for RLlib with the plunger and barrier policies.

    This is the single entry point for all algorithms (PPO, SAC, TD3, etc.)

    Args:
        env_config: Environment configuration dict (from env_config.yaml)
        algo: Algorithm type ("ppo", "sac", "td3")
        config: Neural network config dict

    Returns:
        MultiRLModuleSpec object
    """
    import numpy as np
    from gymnasium import spaces

    # Extract dimensions from env config (no GPU initialization needed)
    resolution = env_config['simulator']['resolution']
    num_dots = env_config['simulator']['num_dots']
    num_channels = num_dots - 1  # N-1 charge stability diagrams

    image_shape = (resolution, resolution, num_channels)

    # Voltage ranges are always [-1, 1] (normalized in env)
    gate_low, gate_high = -1.0, 1.0
    barrier_low, barrier_high = -1.0, 1.0

    # Create observation space for gate agents
    gate_obs_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(image_shape[0], image_shape[1], 2),  # Dual channel for gate agents
        dtype=np.float32,
    )

    # Create action space for gate agents
    gate_action_space = spaces.Box(
        low=gate_low,
        high=gate_high,
        shape=(1,),  # Single gate voltage output
        dtype=np.float32,
    )

    # Create observation space for barrier agents
    barrier_obs_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(image_shape[0], image_shape[1], 1),  # Single channel for barrier agents
        dtype=np.float32,
    )

    # Create action space for barrier agents
    barrier_action_space = spaces.Box(
        low=barrier_low,
        high=barrier_high,
        shape=(1,),  # Single barrier voltage output
        dtype=np.float32,
    )

    # Load neural network configuration
    if config is not None and isinstance(config, dict):
        neural_networks_config = config
    else:
        neural_networks_config = {}

    # Create model configs for each policy
    plunger_config = neural_networks_config['plunger_policy']
    barrier_config = neural_networks_config['barrier_policy']

    # Add max_seq_len to plunger model_config if using LSTM or transformer
    plunger_backbone = plunger_config['backbone']
    plunger_memory_layer = plunger_backbone.get('memory_layer')

    if plunger_memory_layer == 'lstm':
        lstm_config = plunger_backbone['lstm']
        plunger_config['max_seq_len'] = lstm_config['max_seq_len']
    elif plunger_memory_layer == 'transformer':
        transformer_config = plunger_backbone['transformer']
        plunger_config['max_seq_len'] = transformer_config['max_seq_len']

    # Add max_seq_len to barrier model_config if using LSTM or transformer
    barrier_backbone = barrier_config['backbone']
    barrier_memory_layer = barrier_backbone.get('memory_layer')

    if barrier_memory_layer == 'lstm':
        lstm_config = barrier_backbone['lstm']
        barrier_config['max_seq_len'] = lstm_config['max_seq_len']
    elif barrier_memory_layer == 'transformer':
        transformer_config = barrier_backbone['transformer']
        barrier_config['max_seq_len'] = transformer_config['max_seq_len']

    # Select algorithm-specific classes
    if algo == "ppo":
        module_class = DefaultPPOTorchRLModule
        catalog_class = CustomPPOCatalog
    elif algo == "mappo":
        # MAPPO is PPO with a centralized critic. The custom RLModule overrides
        # compute_values to route the embeddings=None fallback through the
        # MAPPORoutingEncoder (the default impl calls critic_encoder directly,
        # which would feed per-agent obs into the centralized encoder).
        module_class = CustomMAPPOTorchRLModule
        catalog_class = CustomMAPPOCatalog
    elif algo == "sac":
        module_class = CustomSACTorchRLModule
        catalog_class = CustomSACCatalog
    elif algo == "td3":
        module_class = CustomTD3TorchRLModule
        catalog_class = CustomTD3Catalog
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Supported: 'ppo', 'mappo', 'sac', 'td3'")

    # MAPPO needs each agent's obs_space to also expose the global state so the
    # catalog can size the centralized critic encoder. The actor still reads
    # only its per-agent {image, voltage}; the critic reads {global_image,
    # global_voltages}. The env wrapper populates these keys when constructed
    # with return_global_state=True.
    if algo == "mappo":
        global_image_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(image_shape[0], image_shape[1], num_channels),
            dtype=np.float32,
        )
        num_agents = num_dots + (num_dots - 1)  # plungers + barriers
        global_voltage_space = spaces.Box(
            low=min(gate_low, barrier_low),
            high=max(gate_high, barrier_high),
            shape=(num_agents,),
            dtype=np.float32,
        )
        gate_voltage_space = spaces.Box(
            low=gate_low, high=gate_high, shape=(1,), dtype=np.float32,
        )
        barrier_voltage_space = spaces.Box(
            low=barrier_low, high=barrier_high, shape=(1,), dtype=np.float32,
        )
        gate_obs_space = spaces.Dict({
            "image": gate_obs_space,
            "voltage": gate_voltage_space,
            "global_image": global_image_space,
            "global_voltages": global_voltage_space,
        })
        barrier_obs_space = spaces.Dict({
            "image": barrier_obs_space,
            "voltage": barrier_voltage_space,
            "global_image": global_image_space,
            "global_voltages": global_voltage_space,
        })

    # Create single agent RLModule specs
    plunger_spec = RLModuleSpec(
        module_class=module_class,
        observation_space=gate_obs_space,
        action_space=gate_action_space,
        model_config=plunger_config,
        catalog_class=catalog_class,
        inference_only=False,
    )

    barrier_spec = RLModuleSpec(
        module_class=module_class,
        observation_space=barrier_obs_space,
        action_space=barrier_action_space,
        model_config=barrier_config,
        catalog_class=catalog_class,
        inference_only=False,
    )

    # Create multi-agent RLModule spec
    rl_module_spec = MultiRLModuleSpec(
        rl_module_specs={"plunger_policy": plunger_spec, "barrier_policy": barrier_spec}
    )

    return rl_module_spec
