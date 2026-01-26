"""TD3 algorithm configuration.

Extends SAC configuration with TD3-specific parameters.
TD3 (Twin Delayed DDPG) key differences from SAC:
- Deterministic policy (no stochastic sampling)
- Exploration via Gaussian noise injection
- Target policy smoothing (clipped noise added to target actions)
- Delayed policy updates (update actor less frequently than critic)
- No entropy/alpha optimization
"""

from typing import Optional

from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.annotations import override


class TD3Config(SACConfig):
    """Configuration for TD3 algorithm.

    TD3-specific parameters:
    - exploration_noise: Gaussian noise stddev for exploration (default: 0.1)
    - policy_noise: Target policy smoothing noise stddev (default: 0.2)
    - noise_clip: Clipping range for target policy noise (default: 0.5)
    - policy_frequency: Update actor every N critic updates (default: 2)
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class)

        # TD3-specific hyperparameters
        self.exploration_noise = 0.1
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_frequency = 2

        # Override SAC defaults - TD3 doesn't use alpha/entropy
        # Keep these for compatibility but they won't be used
        self.initial_alpha = 1.0
        self.target_entropy = None
        self.alpha_lr = None

    def training(
        self,
        *,
        exploration_noise: Optional[float] = None,
        policy_noise: Optional[float] = None,
        noise_clip: Optional[float] = None,
        policy_frequency: Optional[int] = None,
        **kwargs,
    ) -> "TD3Config":
        """Set TD3 training parameters.

        Args:
            exploration_noise: Gaussian noise stddev added during exploration.
            policy_noise: Noise stddev for target policy smoothing.
            noise_clip: Clipping range for target policy noise.
            policy_frequency: Update actor every N critic updates.
            **kwargs: Additional training parameters passed to SACConfig.

        Returns:
            Self for chaining.
        """
        super().training(**kwargs)

        if exploration_noise is not None:
            self.exploration_noise = exploration_noise
        if policy_noise is not None:
            self.policy_noise = policy_noise
        if noise_clip is not None:
            self.noise_clip = noise_clip
        if policy_frequency is not None:
            self.policy_frequency = policy_frequency

        return self

    @override(SACConfig)
    def get_default_rl_module_spec(self):
        """Return the default RLModule spec for TD3."""
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from swarm.voltage_model.algorithms.td3 import CustomTD3TorchRLModule

        return RLModuleSpec(module_class=CustomTD3TorchRLModule)

    @override(SACConfig)
    def get_default_learner_class(self):
        """Return the default Learner class for TD3."""
        from swarm.training.utils.td3_learner import TD3TorchLearner

        return TD3TorchLearner
