from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vaeattention import VAEAttention
from ...small_block.downsample import Downsample2D
from ...small_block.resnet import ResnetBlock2D
from ...small_block.upsample import Upsample2D


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, num_layers=1, resnet_eps=1e-6, resnet_groups=32):
        super().__init__()
        resnets = [ResnetBlock2D(in_channels, in_channels, None, resnet_groups, resnet_eps)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(VAEAttention(in_channels, resnet_groups, resnet_eps))
            resnets.append(ResnetBlock2D(in_channels, in_channels, None, resnet_groups, resnet_eps))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x):
        x = self.resnets[0](x, None)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, None)
        return x