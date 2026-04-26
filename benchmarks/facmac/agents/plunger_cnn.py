"""
PlungerCNNAgent — Nature-CNN (Mnih et al. 2015) encoder + small MLP head that
outputs a single continuous voltage in [-1, 1].

The agent expects already-shaped input: (B, 2, H, W). GroupedMAC handles the
flat->shaped reshape and pad-stripping before dispatch, so the agent itself is
independent of the buffer's flat-storage convention.

Forward signature mirrors vendor/modules/agents/mlp_agent.MLPAgent so it slots
into CQMixMAC's "mlp-like" code path unchanged:
    forward(inputs, hidden_state, actions=None) -> {"actions", "hidden_state"}
"""

from __future__ import annotations

from typing import Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PlungerCNNAgent(nn.Module):

    native_channels = 2

    def __init__(self, obs_shape: tuple[int, int, int], args: Any) -> None:
        super().__init__()
        self.args = args
        C, H, W = obs_shape
        assert C == self.native_channels, f"PlungerCNNAgent expects {self.native_channels}-channel input, got {C}"
        self.C, self.H, self.W = C, H, W

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            flat_size = self.cnn(th.zeros(1, C, H, W)).shape[1]

        self.fc1 = nn.Linear(flat_size, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self) -> th.Tensor:
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(
        self,
        inputs: th.Tensor,
        hidden_state: th.Tensor,
        actions: th.Tensor | None = None,
    ) -> dict[str, th.Tensor]:
        x = self.cnn(inputs)
        x = F.relu(self.fc1(x))
        x = th.tanh(self.fc2(x))
        return {"actions": x, "hidden_state": hidden_state}
