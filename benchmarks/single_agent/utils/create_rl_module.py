"""Create single-agent RL module specification for RLlib."""

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def create_single_agent_rl_module_spec(env_instance, algo: str = "ppo", config: dict = None) -> RLModuleSpec:
    """
    Create single-agent RL module specification for quantum device tuning.

    Args:
        env_instance: Instance of the quantum device environment (SingleAgentEnvWrapper)
        algo: Algorithm type ("ppo" or "sac")
        config: Optional config dict containing network architecture settings

    Returns:
        RLModuleSpec object for single-agent training
    """
    # Import here to avoid circular dependencies
    from utils.custom_catalog import CustomSingleAgentCatalog

    if algo == "ppo":
        module_class = DefaultPPOTorchRLModule
        catalog_class = CustomSingleAgentCatalog
    elif algo == "sac":
        # TODO: Implement SAC support if needed
        raise NotImplementedError("SAC not yet implemented for single-agent benchmark")
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Get observation and action spaces from environment
    obs_space = env_instance.observation_space
    action_space = env_instance.action_space

    # Load neural network configuration
    if config is not None and isinstance(config, dict):
        model_config = config
    else:
        model_config = {}

    # Add max_seq_len to model_config if using LSTM or transformer
    if 'backbone' in model_config:
        backbone = model_config['backbone']
        memory_layer = backbone.get('memory_layer')

        if memory_layer == 'lstm':
            lstm_config = backbone['lstm']
            model_config['max_seq_len'] = lstm_config['max_seq_len']
        elif memory_layer == 'transformer':
            transformer_config = backbone['transformer']
            model_config['max_seq_len'] = transformer_config['max_seq_len']

    # Create single-agent RLModule spec
    rl_module_spec = RLModuleSpec(
        module_class=module_class,
        observation_space=obs_space,
        action_space=action_space,
        model_config=model_config,
        catalog_class=catalog_class,
        inference_only=False,
    )

    return rl_module_spec
