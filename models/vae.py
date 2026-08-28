from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_config import _Config

from .middle_block.vae.downencoderblock import DownEncoderBlock2D
from .middle_block.vae.upencoderblock import UpDecoderBlock2D
from .middle_block.vae.midblock import UNetMidBlock2D



class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, block_out_channels=(64,),
                 layers_per_block=2, norm_num_groups=32, double_z=True):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(DownEncoderBlock2D(
                input_channel, output_channel, num_layers=layers_per_block,
                resnet_groups=norm_num_groups, add_downsample=not is_final,
                downsample_padding=0))
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, block_out_channels=(64,),
                 layers_per_block=2, norm_num_groups=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], 3, padding=1)
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.up_blocks = nn.ModuleList()
        reversed_block_out = list(reversed(block_out_channels))
        output_channel = reversed_block_out[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out[i]
            is_final = i == len(block_out_channels) - 1
            self.up_blocks.append(UpDecoderBlock2D(
                prev_output_channel, output_channel, num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups, add_upsample=not is_final))
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        z = self.conv_in(z)
        z = self.mid_block(z)
        for block in self.up_blocks:
            z = block(z)
        return self.conv_out(self.conv_act(self.conv_norm_out(z)))


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator,
                            device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])


class _EncoderOutput:
    __slots__ = ("latent_dist",)

    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class _DecoderOutput:
    __slots__ = ("sample",)

    def __init__(self, sample):
        self.sample = sample


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.18215,
        **ignored: Any,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels,
                               layers_per_block, norm_num_groups, double_z=True)
        self.decoder = Decoder(latent_channels, out_channels, block_out_channels,
                               layers_per_block, norm_num_groups)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.config = _Config(latent_channels=latent_channels, scaling_factor=scaling_factor,
                              sample_size=sample_size, block_out_channels=tuple(block_out_channels))

    def encode(self, x: torch.Tensor) -> _EncoderOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return _EncoderOutput(DiagonalGaussianDistribution(moments))

    def decode(self, z: torch.Tensor) -> _DecoderOutput:
        z = self.post_quant_conv(z)
        return _DecoderOutput(self.decoder(z))

    def forward(self, sample: torch.Tensor, sample_posterior: bool = False) -> _DecoderOutput:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z)