from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...small_block.downsample import Downsample2D
from ...small_block.resnet import ResnetBlock2D
from ...small_block.upsample import Upsample2D


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, resnet_eps=1e-6,
                 resnet_groups=32, add_downsample=True, downsample_padding=0):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels,
                          temb_channels=None, groups=resnet_groups, eps=resnet_eps)
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList(
            [Downsample2D(out_channels, out_channels, use_conv=True, padding=downsample_padding)]
        ) if add_downsample else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x, None)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x