from __future__ import annotations

"""
Vòng lặp sinh ảnh có điều kiện bằng vector multi-hot (không dùng prompt).

Khác duy nhất so với `pipeline.inference.sample`: encoder_hidden_states đến từ
MultiHotLabelEncoder thay vì CLIP, và nhánh uncond của CFG là `null_tokens` học
được thay vì negative prompt. Sampler / scheduler / VAE dùng lại y nguyên.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

from .inference import (
    NoiseScheduler,
    decode_latents,
    dpm_solver_2m_step,
    randn_tensor,
)


@dataclass
class LabelSDComponents:
    unet: "torch.nn.Module"
    vae: "torch.nn.Module"
    label_encoder: "torch.nn.Module"


def _prepare_labels(labels, num_images: int, num_labels: int,
                    device, dtype) -> torch.Tensor:
    """Chấp nhận tensor (K,) / (B,K) / list[float] / list[list[float]]."""
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    labels = labels.to(dtype=torch.float32)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if labels.shape[-1] != num_labels:
        raise ValueError(f"labels phải có {num_labels} chiều, nhận {labels.shape[-1]}")
    if labels.shape[0] == 1 and num_images > 1:
        labels = labels.expand(num_images, -1)
    elif labels.shape[0] != num_images:
        raise ValueError(f"num_images={num_images} nhưng labels có {labels.shape[0]} hàng")
    return labels.contiguous().to(device=device)


@torch.no_grad()
def sample_from_labels(
    components: LabelSDComponents,
    labels,
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 4.0,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[torch.dtype] = None,
    scheduler: Optional[NoiseScheduler] = None,
    latents: Optional[torch.Tensor] = None,
) -> List[Image.Image]:
    """
    labels: vector multi-hot 5 chiều (hoặc batch vector). Có thể dùng giá trị mềm
    trong [0,1] để nội suy mức độ biểu hiện bệnh.

    guidance_scale: với điều kiện nhãn (thông tin ít hơn text) 3–5 thường đủ;
    đẩy lên 7.5 như prompt hay tạo ảnh quá tương phản, mất chi tiết nhu mô.
    """
    unet, vae, label_encoder = components.unet, components.vae, components.label_encoder
    if dtype is None:
        dtype = next(unet.parameters()).dtype

    scheduler = scheduler or NoiseScheduler()
    do_cfg = guidance_scale > 1.0

    y = _prepare_labels(labels, num_images, label_encoder.num_labels, device, dtype)
    context = label_encoder.encode_for_cfg(y, do_cfg=do_cfg).to(dtype=dtype)

    if latents is None:
        latents = randn_tensor(
            (num_images, unet.config.in_channels, height // 8, width // 8),
            generator, device, dtype,
        )
    else:
        latents = latents.to(device=device, dtype=dtype)

    timesteps = scheduler.set_timesteps(num_inference_steps, device=device)
    x0_pred_prev, r_t = None, None

    for i, t in enumerate(timesteps):
        latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        t_batch = t.expand(latent_input.shape[0])

        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=context)

        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        prev_t = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
        latents, x0_pred_prev = dpm_solver_2m_step(
            scheduler, noise_pred, int(t), prev_t, r_t, latents, x0_pred_prev
        )
        r_t = int(t)

    return decode_latents(vae, latents)
