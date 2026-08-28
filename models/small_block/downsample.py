from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: bool = True,
        padding: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.use_conv = use_conv
        self.padding = padding
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding)
        else:
            self.conv = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            if self.padding == 0:
                hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states
