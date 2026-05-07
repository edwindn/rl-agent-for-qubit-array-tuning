"""
Monkey-patch for Ray RLlib 2.49.0 to fix complex observation comparison in replay buffers.
Applies the fix from https://github.com/ray-project/ray/pull/57017

This patch fixes the error:
    ValueError: The truth value of an array with more than one element is ambiguous.

Which occurs in SingleAgentEpisode.concat_episode() when using replay buffers (DQN/SAC)
with complex observation structures.

Remove this file after upgrading to Ray >= 2.51.0
"""

import numpy as np


def apply_patch():
    """Call this before any RLlib imports that use replay buffers."""
    from ray.rllib.env import single_agent_episode

    def _patched_concat_episode(self, other):
        """Patched concat_episode that handles complex/nested observations.

        This is a near-exact copy of the original method, with only the
        observation comparison assertion replaced with a structure-aware check.
        """
        # Original assertions (unchanged)
        assert other.id_ == self.id_
        assert not self.is_done
        assert self.t == other.t_started
        other.validate()

        # PATCHED: Replace np.all(other.observations[0] == self.observations[-1])
        # with structure-aware comparison for complex observations
        if not _safe_obs_equal(self.observations[-1], other.observations[0]):
            raise AssertionError(
                "Observations don't match at episode boundary. "
                f"Last obs: {_get_obs_info(self.observations[-1])}, "
                f"First obs: {_get_obs_info(other.observations[0])}"
            )

        # Pop the duplicate observation/info (original behavior)
        self.observations.pop()
        self.infos.pop()

        # Extend all fields (original behavior)
        self.observations.extend(other.get_observations())
        self.actions.extend(other.get_actions())
        self.rewards.extend(other.get_rewards())
        self.infos.extend(other.get_infos())
        self.t = other.t

        if other.is_terminated:
            self.is_terminated = True
        elif other.is_truncated:
            self.is_truncated = True

        for key in other.extra_model_outputs.keys():
            assert key in self.extra_model_outputs
            self.extra_model_outputs[key].extend(
                other.get_extra_model_outputs(key)
            )

        self.custom_data.update(other.custom_data)
        self.validate()

    single_agent_episode.SingleAgentEpisode.concat_episode = _patched_concat_episode
    print("[ray_episode_patch] Applied concat_episode fix for complex observations")


def _safe_obs_equal(obs1, obs2):
    """Compare observations that may be nested structures (dicts, tuples, arrays)."""
    try:
        import tree

        flat1 = tree.flatten(obs1)
        flat2 = tree.flatten(obs2)
        if len(flat1) != len(flat2):
            return False
        return all(np.array_equal(a, b) for a, b in zip(flat1, flat2))
    except ImportError:
        # Fallback without dm-tree
        return np.array_equal(obs1, obs2)


def _get_obs_info(obs):
    """Get shape/type info for error messages."""
    if isinstance(obs, np.ndarray):
        return f"ndarray{obs.shape}"
    if isinstance(obs, dict):
        return {k: _get_obs_info(v) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return f"{type(obs).__name__}[{len(obs)}]"
    return type(obs).__name__
