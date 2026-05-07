"""
PlungerCNNAgentInitBias — variant of PlungerCNNAgent for rescue-campaign M2.

Hypothesis: vanilla MADDPG actor-collapse-to-no-op stems from the final layer's
default Linear init (kaiming-uniform on weights, ZERO bias). Combined with the
tanh squash, the actor outputs near-zero actions across all input states. The
critic sees only (s, a≈0) → learns Q ≈ const → no actor gradient → loop.

Fix: initialise the final-layer bias uniformly in [-0.1, 0.1] so the actor
emits non-zero actions from step 0, breaks the symmetry, and gives the critic
some Q-curvature to learn from.

Everything else matches PlungerCNNAgent exactly.
"""

from __future__ import annotations

from typing import Any

import torch as th

from .plunger_cnn import PlungerCNNAgent


class PlungerCNNAgentInitBias(PlungerCNNAgent):

    def __init__(self, obs_shape: tuple[int, int, int], args: Any) -> None:
        super().__init__(obs_shape, args)
        with th.no_grad():
            self.fc2.bias.uniform_(-0.1, 0.1)
