from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Downsample2D, ResnetBlock2D, Upsample2D, _Config


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