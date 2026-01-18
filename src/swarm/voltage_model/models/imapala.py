from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn


# -----------------------------
# Canonical IMPALA-CNN building blocks
# -----------------------------

class ImpalaResidualBlock(nn.Module):
    """
    Canonical IMPALA residual block:
      y = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
    This matches the common reproduction of the IMPALA-CNN residual unit.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        return x + y


class ImpalaConvSequence(nn.Module):
    """
    One IMPALA conv sequence:
      Conv3x3 -> MaxPool(3x3, stride=2, padding=1) -> ResidualBlock -> ResidualBlock
    Output spatial dims are downsampled by ~2x via the maxpool.
    """
    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ImpalaResidualBlock(out_ch) for _ in range(num_res_blocks)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.blocks(x)
        x = self.relu(x)  # many reference implementations include a ReLU after the sequence
        return x


# -----------------------------
# IMPALA-CNN encoder (non-GAP)
# -----------------------------

@dataclass(frozen=True)
class ImpalaCnnConfig:
    input_hw: Tuple[int, int] = (100, 100)
    input_channels: int = 3

    # Canonical channel progression from the IMPALA paper figure for the conv trunk.
    # (Many later works scale width, but this is the classic baseline.)
    channels: Tuple[int, int, int] = (16, 32, 32)

    num_res_blocks_per_sequence: int = 2

    # Projection head: Flatten -> Linear(256) -> ReLU, as shown in IMPALA diagram.
    feature_dim: int = 256

    # If your env already normalizes to [0,1], set False.
    normalize_255: bool = True


class ImpalaCnnEncoder(nn.Module):
    """
    Faithful IMPALA-CNN image encoder (non-GAP).
    - Accepts NCHW float tensors (preferred).
    - Will also accept NHWC and auto-permute if last dim matches channels.

    Output: (B, feature_dim) where feature_dim defaults to 256.
    """
    def __init__(self, cfg: ImpalaCnnConfig):
        super().__init__()
        self.cfg = cfg

        c0, c1, c2 = cfg.channels
        self.trunk = nn.Sequential(
            ImpalaConvSequence(cfg.input_channels, c0, cfg.num_res_blocks_per_sequence),
            ImpalaConvSequence(c0, c1, cfg.num_res_blocks_per_sequence),
            ImpalaConvSequence(c1, c2, cfg.num_res_blocks_per_sequence),
        )

        # For 100x100, with 3x (MaxPool 3,s2,p1): 100 -> 50 -> 25 -> 13
        # so final map is (B, c2, 13, 13). We compute this to avoid mistakes.
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.input_hw[0], cfg.input_hw[1])
            out = self.trunk(dummy)
            flat_dim = int(out.numel() // out.shape[0])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
    ) -> torch.Tensor:
        # Handle common dict obs patterns
        if isinstance(obs, dict):
            if "image" in obs:
                x = obs["image"]
            elif "obs" in obs:
                x = obs["obs"]
            else:
                raise KeyError(f"IMPALA encoder expected key 'image' or 'obs', got: {list(obs.keys())}")
        else:
            x = obs

        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Accept NHWC or NCHW
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W) or (B,H,W,C), got shape {tuple(x.shape)}")

        # If looks like NHWC, permute to NCHW
        if x.shape[-1] == self.cfg.input_channels and x.shape[1] != self.cfg.input_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        # Basic sanity
        if x.shape[1] != self.cfg.input_channels:
            raise ValueError(
                f"Expected {self.cfg.input_channels} channels, got {x.shape[1]} (shape={tuple(x.shape)})"
            )

        # Normalize like the IMPALA diagram shows dividing by 255 (if using uint8 frames).
        x = x.float()
        if self.cfg.normalize_255:
            x = x / 255.0

        x = self.trunk(x)
        x = self.head(x)
        return x
