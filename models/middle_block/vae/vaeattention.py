from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...small_block.downsample import Downsample2D
from ...small_block.resnet import ResnetBlock2D
from ...small_block.upsample import Upsample2D

class VAEAttention(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.heads = 1
        self.group_norm = nn.GroupNorm(num_groups, channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True), nn.Dropout(0.0)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        b, c, h, w = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        q = self.to_q(hidden_states).unsqueeze(1)
        k = self.to_k(hidden_states).unsqueeze(1)
        v = self.to_v(hidden_states).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1)

        out = self.to_out[1](self.to_out[0](out))
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return out + residual