"""Factory for SuperSims All-XY env RLModule specs.

This is the single-policy / per-param / grouped factory for the SuperSims
calibration env. The default dot-tuning factory lives in `factory.py`; the
caller dispatches based on `env_config["env_type"]`.
"""
from __future__ import annotations

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec

from swarm.voltage_model.supersims_catalog import CustomPPOCatalog


_SUPERSIMS_PARAM_NAMES = ["omega01", "omegad", "phi", "drive", "beta"]
# Grouped mode: 2 policies, action dims set by group size (must match
# PARAM_GROUPS in supersims_multi_agent_wrapper).
_SUPERSIMS_GROUP_NAMES = ["freq", "env"]
_SUPERSIMS_GROUP_ACTION_DIMS = {"freq": 3, "env": 2}


def create_rl_module_spec_supersims(
    env_config: dict,
    algo: str = "ppo",
    config: dict = None,
) -> MultiRLModuleSpec:
    """SuperSims All-XY env RLModule spec.

    Three layouts (selected by env_config['policy_split']):
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

    raise ValueError(
        f"Unknown policy_split={policy_split!r}; expected 'per_qubit', 'per_param', or 'grouped'."
    )
