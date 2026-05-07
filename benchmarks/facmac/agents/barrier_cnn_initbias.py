"""
BarrierCNNAgentInitBias — variant of BarrierCNNAgent for rescue-campaign M2.
See plunger_cnn_initbias.py for the hypothesis + rationale.
"""

from __future__ import annotations

from typing import Any

import torch as th

from .barrier_cnn import BarrierCNNAgent


class BarrierCNNAgentInitBias(BarrierCNNAgent):

    def __init__(self, obs_shape: tuple[int, int, int], args: Any) -> None:
        super().__init__(obs_shape, args)
        with th.no_grad():
            self.fc2.bias.uniform_(-0.1, 0.1)
