from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if use_conv else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)