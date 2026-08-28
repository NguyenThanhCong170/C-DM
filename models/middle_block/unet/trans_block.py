from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_trans_block import BasicTransformerBlock

class Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        norm_num_groups: int = 32,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.inner_dim = inner_dim

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(dim=inner_dim, n_heads=n_heads, d_head=d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if self.use_linear_projection:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, self.inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, channel).permute(0, 3, 1, 2).contiguous()
        else:
            hidden_states = hidden_states.reshape(batch, height, width, self.inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual