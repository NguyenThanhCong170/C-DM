from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...small_block.downsample import Downsample2D
from ...small_block.resnet import ResnetBlock2D
from ...small_block.upsample import Upsample2D

class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, resnet_eps=1e-6,
                 resnet_groups=32, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                          temb_channels=None, groups=resnet_groups, eps=resnet_eps)
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList(
            [Upsample2D(out_channels, out_channels, use_conv=True)]
        ) if add_upsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x, None)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        return x