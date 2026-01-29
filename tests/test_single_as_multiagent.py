#!/usr/bin/env python3
"""
Test: Wrap single-agent env as a multi-agent env and use multi-agent RLlib config.

If this works but the regular single-agent doesn't, the issue is in how
RLlib handles single vs multi-agent mode.
"""
import sys
from pathlib import Path
import numpy as np
from gymnasium import spaces

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SingleAsMultiAgentWrapper(MultiAgentEnv):
    """Wrap single-agent env to look like multi-agent with one agent."""

    def __init__(self, config=None):
        super().__init__()

        # Create the single-agent env
        from swarm.single_agent_ablations.utils.env_wrapper import SingleAgentEnvWrapper
        config_path = str(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml")
        self.base_env = SingleAgentEnvWrapper(training=True, config_path=config_path)

        # Define single "agent"
        self._agent_ids = {"agent_0"}
        self.agents = self._agent_ids.copy()
        self.possible_agents = list(self._agent_ids)

        # Copy spaces
        self._obs_space = spaces.Dict({
            "agent_0": self.base_env.observation_space
        })
        self._action_space = spaces.Dict({
            "agent_0": self.base_env.action_space
        })

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return {"agent_0": obs}, {"agent_0": info}

    def step(self, action_dict):
        action = action_dict["agent_0"]
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        obs_dict = {"agent_0": obs}
        reward_dict = {"agent_0": reward}
        terminated_dict = {"agent_0": terminated, "__all__": terminated}
        truncated_dict = {"agent_0": truncated, "__all__": truncated}
        info_dict = {"agent_0": info}

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def get_agent_ids(self):
        return self._agent_ids

    def close(self):
        return self.base_env.close()


def test_wrapper():
    """Test the wrapper works correctly."""
    print("Testing SingleAsMultiAgentWrapper...")

    env = SingleAsMultiAgentWrapper()
    obs, info = env.reset()

    print(f"Agent IDs: {env.get_agent_ids()}")
    print(f"Observation keys: {obs.keys()}")
    print(f"agent_0 obs type: {type(obs['agent_0'])}")
    if isinstance(obs['agent_0'], dict):
        for k, v in obs['agent_0'].items():
            print(f"  {k}: shape={v.shape}")

    # Step
    action = {"agent_0": env.action_space["agent_0"].sample()}
    obs, reward, term, trunc, info = env.step(action)

    print(f"\nAfter step:")
    print(f"  reward: {reward}")
    print(f"  terminated: {term}")
    print(f"  truncated: {trunc}")

    env.close()
    print("✓ Wrapper test passed")


def test_with_rllib():
    """Test with RLlib multi-agent config."""
    print("\n" + "=" * 60)
    print("Testing with RLlib multi-agent config...")
    print("=" * 60)

    import warnings
    warnings.filterwarnings("ignore")

    import ray
    import yaml
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec, RLModuleSpec
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

    # Load single-agent config
    with open(src_dir / "swarm/single_agent_ablations/training_config.yaml") as f:
        config = yaml.safe_load(f)
    with open(src_dir / "swarm/single_agent_ablations/single_agent_env_config.yaml") as f:
        env_config = yaml.safe_load(f)

    ray.init(include_dashboard=False, logging_level=40)

    try:
        # Register env
        register_env("test_multi_env", lambda c: SingleAsMultiAgentWrapper(c))

        # Create module spec (use single-agent catalog but wrap in MultiRLModuleSpec)
        from swarm.single_agent_ablations.utils.factory import (
            create_rl_module_spec,
            CustomSingleAgentCatalog
        )

        resolution = env_config['simulator']['resolution']
        num_channels = env_config['simulator']['num_dots'] - 1

        obs_space = spaces.Dict({
            'image': spaces.Box(0.0, 1.0, (resolution, resolution, num_channels), np.float32),
            'voltage': spaces.Box(-1.0, 1.0, (1,), np.float32),
        })
        action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        model_config = {
            **config['neural_networks']['single_agent_policy'],
            "free_log_std": config['rl_config']['single_agent']['free_log_std'],
            "log_std_bounds": config['rl_config']['single_agent']['log_std_bounds'],
        }

        agent_spec = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config,
            catalog_class=CustomSingleAgentCatalog,
            inference_only=False,
        )

        multi_spec = MultiRLModuleSpec(
            rl_module_specs={"agent_0": agent_spec}
        )

        def policy_mapping_fn(agent_id, episode, **kwargs):
            return "agent_0"

        # Build algorithm
        algo_config = (
            PPOConfig()
            .environment(env="test_multi_env")
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies={"agent_0"},
                policies_to_train=["agent_0"],
            )
            .rl_module(rl_module_spec=multi_spec)
            .env_runners(
                num_env_runners=4,
                num_gpus_per_env_runner=1.0,
                rollout_fragment_length=50,
                sample_timeout_s=300,
            )
            .learners(
                num_learners=1,
                num_gpus_per_learner=1,
            )
            .training(
                train_batch_size_per_learner=200,  # 4 runners * 50 fragment = 200
                minibatch_size=50,
                num_epochs=4,
                lr=0.0005,
                gamma=0.0,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
            )
        )

        print("\nBuilding algorithm...")
        algo = algo_config.build()

        print("\nRunning 50 training iterations...")
        for i in range(50):
            result = algo.train()
            reward = result.get("env_runners", {}).get("episode_return_mean", "N/A")
            print(f"  Iteration {i+1}: reward_mean = {reward}")

        algo.stop()
        print("\n✓ Multi-agent wrapper test passed")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_wrapper()
    test_with_rllib()
