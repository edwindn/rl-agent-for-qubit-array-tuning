from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from swarm.voltage_model.custom_sac_rl_module import CustomSACTorchRLModule

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec

from swarm.voltage_model.custom_catalog import CustomPPOCatalog


def create_rl_module_spec(env_config: dict, algo: str="ppo", config: dict=None) -> MultiRLModuleSpec:
    """
    Create policy specifications for RLlib.

    Routes by env_config["env_type"]:
      - default (dot tuning):  two policies — plunger_policy + barrier_policy.
      - "supersims" (All-XY):  one policy — qubit_policy (shared across all qubits).

    Args:
        env_config: Environment configuration dict (from env_config.yaml)
        algo: Algorithm type ("ppo" or "sac")
        config: Neural network config dict

    Returns:
        MultiRLModuleSpec object
    """
    import numpy as np
    from gymnasium import spaces

    env_type = env_config.get("env_type", "dot")
    if env_type == "supersims":
        return _create_supersims_rl_module_spec(env_config, algo=algo, config=config)

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
    # Each gate agent controls: single gate voltage
    gate_action_space = spaces.Box(
        low=gate_low,
        high=gate_high,
        shape=(1,),  # Single gate voltage output
        dtype=np.float32,
    )

    # Create observation space for barrier agents
    # Each barrier agent sees: single-channel image + single voltage value
    barrier_obs_space = spaces.Dict(
        {
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    image_shape[0],
                    image_shape[1],
                    1,
                ),  # Single channel for barrier agents
                dtype=np.float32,
            ),
            "voltage": spaces.Box(
                low=barrier_low,
                high=barrier_high,
                shape=(1,),  # Single voltage value
                dtype=np.float32,
            ),
        }
    )
    # IMAGE ONLY SPACE
    barrier_obs_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(image_shape[0], image_shape[1], 1),  # Single channel for barrier agents
        dtype=np.float32,
    )

    # Create action space for barrier agents
    # Each barrier agent controls: single barrier voltage
    barrier_action_space = spaces.Box(
        low=barrier_low,
        high=barrier_high,
        shape=(1,),  # Single barrier voltage output
        dtype=np.float32,
    )

    # Load neural network configuration from YAML file
    if config is not None and isinstance(config, dict):
        neural_networks_config = config
    else:
        neural_networks_config = {}
    
    # Create model configs for each policy
    plunger_config = neural_networks_config['plunger_policy']
    barrier_config = neural_networks_config['barrier_policy']

    # Add max_seq_len to plunger model_config if using LSTM or transformer
    backbone = plunger_config['backbone']
    memory_layer = backbone.get('memory_layer')

    if memory_layer == 'lstm':
        lstm_config = backbone['lstm']
        plunger_config['max_seq_len'] = lstm_config['max_seq_len']
    elif memory_layer == 'transformer':
        transformer_config = backbone['transformer']
        plunger_config['max_seq_len'] = transformer_config['max_seq_len']

    if algo=="ppo":
        module_class = DefaultPPOTorchRLModule
        catalog_class = CustomPPOCatalog
    elif algo=="sac":
        module_class = CustomSACTorchRLModule
        catalog_class = CustomSACCatalog
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Create single agent RLModule specs using new API
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


_SUPERSIMS_PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]
# Grouped mode: 2 policies, action dims set by group size (must match
# PARAM_GROUPS in supersims_multi_agent_wrapper).
_SUPERSIMS_GROUP_NAMES = ["freq", "env"]
_SUPERSIMS_GROUP_ACTION_DIMS = {"freq": 3, "env": 2}


def _create_supersims_rl_module_spec(env_config: dict, algo: str = "ppo", config: dict = None) -> MultiRLModuleSpec:
    """SuperSims All-XY env. Three layouts (selected by env_config['policy_split']):

      - "per_qubit": single shared `qubit_policy`. Action shape (5,).
      - "per_param": five shared policies (omega01_policy, omegad_policy, phi_policy,
        drive_policy, beta_policy), each with action shape (1,).
      - "grouped":   two shared policies — `freq_policy` (action_dim=3 covering
        omega_01, omega_d, phi) and `env_policy` (action_dim=2 covering Omega, beta).

    Per-agent obs is Dict({staircase: (21,), params: (5,)}) in all modes.
    """
    import copy
    import numpy as np
    from gymnasium import spaces

    # Pull shapes from the SuperSims module so this stays in sync with the env.
    # Use the symlinked sibling under src/swarm/_supersims (rides with Ray's working_dir).
    import sys
    from pathlib import Path
    supersims_dir = Path(__file__).resolve().parent.parent / "_supersims"
    if supersims_dir.exists() and str(supersims_dir) not in sys.path:
        sys.path.insert(0, str(supersims_dir))
    from all_xy_sequence import N_ALLXY  # noqa: E402

    n_params = 5  # [omega_01, omega_d, phi, Omega, beta]

    qubit_obs_space = spaces.Dict({
        "staircase": spaces.Box(low=0.0, high=1.0, shape=(N_ALLXY,), dtype=np.float32),
        "params":    spaces.Box(low=-np.inf, high=np.inf, shape=(n_params,), dtype=np.float32),
    })

    if config is None or not isinstance(config, dict):
        raise ValueError("SuperSims expects a non-empty neural-network config.")
    if algo != "ppo":
        raise ValueError(f"SuperSims env currently only supports PPO, got algo={algo!r}")

    policy_split = env_config.get("policy_split", "per_qubit")

    if policy_split == "per_qubit":
        if "qubit_policy" not in config:
            raise ValueError("per_qubit mode expects neural_networks.qubit_policy in the config.")
        qubit_action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_params,), dtype=np.float32)
        spec = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=qubit_obs_space,
            action_space=qubit_action_space,
            model_config=config["qubit_policy"],
            catalog_class=CustomPPOCatalog,
            inference_only=False,
        )
        return MultiRLModuleSpec(rl_module_specs={"qubit_policy": spec})

    if policy_split == "per_param":
        per_param_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        specs = {}
        for pname in _SUPERSIMS_PARAM_NAMES:
            policy_id = f"{pname}_policy"
            if policy_id not in config:
                raise ValueError(f"per_param mode expects neural_networks.{policy_id} in the config.")
            specs[policy_id] = RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=qubit_obs_space,
                action_space=per_param_action_space,
                model_config=copy.deepcopy(config[policy_id]),
                catalog_class=CustomPPOCatalog,
                inference_only=False,
            )
        return MultiRLModuleSpec(rl_module_specs=specs)

    if policy_split == "grouped":
        specs = {}
        for gname in _SUPERSIMS_GROUP_NAMES:
            policy_id = f"{gname}_policy"
            if policy_id not in config:
                raise ValueError(f"grouped mode expects neural_networks.{policy_id} in the config.")
            action_dim = _SUPERSIMS_GROUP_ACTION_DIMS[gname]
            group_action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
            specs[policy_id] = RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=qubit_obs_space,
                action_space=group_action_space,
                model_config=copy.deepcopy(config[policy_id]),
                catalog_class=CustomPPOCatalog,
                inference_only=False,
            )
        return MultiRLModuleSpec(rl_module_specs=specs)

    raise ValueError(f"Unknown policy_split={policy_split!r}; expected 'per_qubit', 'per_param', or 'grouped'.")