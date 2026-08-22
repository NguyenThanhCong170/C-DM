from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Containers & Helpers
# --------------------------------------------------------------------------


class _Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class UNet2DConditionOutput:
    """Wrapper đầu ra tương thích Diffusers API."""
    __slots__ = ("sample",)

    def __init__(self, sample: torch.Tensor):
        self.sample = sample

    def __getitem__(self, idx):
        return self.sample[idx]

    @property
    def shape(self):
        return self.sample.shape

    @property
    def dtype(self):
        return self.sample.dtype

    @property
    def device(self):
        return self.sample.device


# --------------------------------------------------------------------------
# Timestep Embedding (Chuẩn toán học 100% của Diffusers)
# --------------------------------------------------------------------------


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "timesteps phải là tensor 1 chiều"
    half_dim = embedding_dim // 2
    
    # Tính toán chuẩn bit HF diffusers
    exponent = math.log(max_period) / (half_dim - downscale_freq_shift)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    cos_emb = torch.cos(emb)
    sin_emb = torch.sin(emb)

    if flip_sin_to_cos:
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
    else:
        emb = torch.cat([sin_emb, cos_emb], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# --------------------------------------------------------------------------
# Attention & Spatial Transformer
# --------------------------------------------------------------------------


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        batch_size, seq_len, _ = q.shape
        ctx_len = k.shape[1]

        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, ctx_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, ctx_len, self.heads, self.dim_head).transpose(1, 2)

        # PyTorch FlashAttention / SDPA (O(N) memory complexity)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.heads * self.dim_head)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[1](hidden_states)
        hidden_states = self.net[2](hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states=encoder_hidden_states) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


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


# --------------------------------------------------------------------------
# ResNet & Up/Down Sampling
# --------------------------------------------------------------------------


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = 1280,
        groups: int = 32,
        eps: float = 1e-5,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None and temb_channels > 0:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None and temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return self.conv_shortcut(input_tensor) + hidden_states


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


# --------------------------------------------------------------------------
# Down, Mid, Up Blocks
# --------------------------------------------------------------------------


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int],
        num_layers: int = 2,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        resnets: List[nn.Module] = []
        attentions: List[nn.Module] = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_c, out_channels=out_channels, temb_channels=temb_channels))
            attentions.append(
                Transformer2DModel(
                    in_channels=out_channels,
                    n_heads=num_attention_heads,
                    d_head=out_channels // num_attention_heads,
                    depth=transformer_layers_per_block,
                    context_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.downsamplers = nn.ModuleList([
            Downsample2D(out_channels, out_channels, use_conv=True, padding=downsample_padding)
        ]) if add_downsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int],
        num_layers: int = 2,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets: List[nn.Module] = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_c, out_channels=out_channels, temb_channels=temb_channels))
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList([
            Downsample2D(out_channels, out_channels, use_conv=True, padding=downsample_padding)
        ]) if add_downsample else None

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: Optional[int],
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels),
            ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels),
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(
                in_channels=in_channels,
                n_heads=num_attention_heads,
                d_head=in_channels // num_attention_heads,
                depth=transformer_layers_per_block,
                context_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
            )
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: Optional[int],
        num_layers: int = 2,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_upsample: bool = True,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        resnets: List[nn.Module] = []
        attentions: List[nn.Module] = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )
            attentions.append(
                Transformer2DModel(
                    in_channels=out_channels,
                    n_heads=num_attention_heads,
                    d_head=out_channels // num_attention_heads,
                    depth=transformer_layers_per_block,
                    context_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.upsamplers = nn.ModuleList([
            Upsample2D(out_channels, out_channels, use_conv=True)
        ]) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: Optional[int],
        num_layers: int = 2,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets: List[nn.Module] = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([
            Upsample2D(out_channels, out_channels, use_conv=True)
        ]) if add_upsample else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------
# UNet2DConditionModel
# --------------------------------------------------------------------------


class UNet2DConditionModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        layers_per_block: int = 2,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim: int = 768,
        attention_head_dim: Union[int, Tuple[int, ...], List[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        downsample_padding: int = 1,
        transformer_layers_per_block: int = 1,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.config = _Config(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            cross_attention_dim=cross_attention_dim,
            use_linear_projection=use_linear_projection,
        )

        time_embed_dim = block_out_channels[0] * 4

        # XỬ LÝ LỖI NUM_HEADS ĐẢM BẢO CHUẨN XÁC VỚI DIFFUSERS
        if num_attention_heads is not None:
            heads = num_attention_heads
        else:
            heads = attention_head_dim
            
        if isinstance(heads, (int, float)):
            heads_list = [int(heads)] * len(block_out_channels)
        elif isinstance(heads, (list, tuple)):
            heads_list = list(heads)
        else:
            heads_list = [8] * len(block_out_channels)

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.time_proj = nn.Identity()
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Down Blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            n_heads = heads_list[i]

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = CrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block,
                    num_attention_heads=n_heads,
                    cross_attention_dim=cross_attention_dim,
                    add_downsample=not is_final_block,
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                )
            elif down_block_type == "DownBlock2D":
                down_block = DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    downsample_padding=downsample_padding,
                )
            else:
                raise ValueError(f"down_block_type không hợp lệ: {down_block_type}")
            self.down_blocks.append(down_block)

        # Mid Block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            num_attention_heads=heads_list[-1],
            cross_attention_dim=cross_attention_dim,
            use_linear_projection=use_linear_projection,
        )

        # Up Blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_heads_list = list(reversed(heads_list))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1
            n_heads = reversed_heads_list[i]

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=transformer_layers_per_block,
                    num_attention_heads=n_heads,
                    cross_attention_dim=cross_attention_dim,
                    add_upsample=not is_final_block,
                    use_linear_projection=use_linear_projection,
                )
            elif up_block_type == "UpBlock2D":
                up_block = UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final_block,
                )
            else:
                raise ValueError(f"up_block_type không hợp lệ: {up_block_type}")
            self.up_blocks.append(up_block)

        # Out
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, torch.Tensor]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(sample.shape[0])

        t_emb = get_timestep_embedding(
            timesteps,
            self.down_blocks[0].resnets[0].in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=emb,
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if return_dict:
            return UNet2DConditionOutput(sample=sample)
        return sample
 