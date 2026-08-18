"""
From-scratch diffusion math + a standalone DDIM sampling loop with
classifier-free guidance (CFG), decoupled from any Diffusers `pipeline`
object so it works with a raw (unet, vae, text_encoder, tokenizer) tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VAE_SCALING_FACTOR = 0.18215  # SD 1.x latent scale factor (AutoencoderKL config)


# --------------------------------------------------------------------------
# Forward process / schedule
# --------------------------------------------------------------------------


class NoiseScheduler:
    """Re-implementation of the SD 1.x "scaled_linear" DDPM beta schedule."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        self.num_train_timesteps = num_train_timesteps
        betas = (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()  # (T,)

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """z_t = sqrt(alpha_bar_t) z0 + sqrt(1 - alpha_bar_t) eps, batched over `timesteps`."""
        ac = self.alphas_cumprod.to(device=x0.device, dtype=torch.float32)[timesteps]
        while ac.dim() < x0.dim():
            ac = ac.unsqueeze(-1)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise

    def compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """SNR(t) = alpha_bar_t / (1 - alpha_bar_t), used for Min-SNR loss weighting."""
        ac = self.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
        return ac / (1.0 - ac).clamp(min=1e-8)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Evenly-spaced descending subset of the training timesteps, for DDIM sampling."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps must be <= num_train_timesteps")
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64).copy()
        return torch.from_numpy(timesteps).to(device=device, dtype=torch.long)


def min_snr_weights(snr: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR-gamma per-sample loss weight: weight = min(SNR(t), gamma) / SNR(t)."""
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-8)


@torch.no_grad()
def ddim_step(
    scheduler: NoiseScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    prev_timestep: torch.Tensor,
    sample: torch.Tensor,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One deterministic (eta=0) DDIM reverse step, epsilon-prediction parameterization."""
    device = sample.device
    dtype = sample.dtype
    ac_t = scheduler.alphas_cumprod.to(device=device, dtype=dtype)[timestep]
    ac_prev = (
        scheduler.alphas_cumprod.to(device=device, dtype=dtype)[prev_timestep]
        if prev_timestep >= 0
        else torch.tensor(1.0, device=device, dtype=dtype)
    )

    pred_x0 = (sample - (1.0 - ac_t).sqrt() * model_output) / ac_t.sqrt()

    sigma_t = 0.0
    if eta > 0:
        sigma_t = eta * (((1.0 - ac_prev) / (1.0 - ac_t)) * (1.0 - ac_t / ac_prev)).clamp(min=0.0).sqrt()

    dir_coeff = (1.0 - ac_prev - sigma_t**2).clamp(min=0.0).sqrt()
    x_prev = ac_prev.sqrt() * pred_x0 + dir_coeff * model_output

    if eta > 0:
        noise = torch.randn(sample.shape, generator=generator, device=device, dtype=dtype)
        x_prev = x_prev + sigma_t * noise

    return x_prev


# --------------------------------------------------------------------------
# Text conditioning / CFG helpers
# --------------------------------------------------------------------------


@torch.no_grad()
def encode_prompt(
    tokenizer,
    text_encoder,
    prompts: List[str],
    negative_prompts: List[str],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Tokenize + encode negative and positive prompts as [uncond; cond]."""
    def _encode(texts: List[str]) -> torch.Tensor:
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(tokens)[0]

    uncond_embeds = _encode(negative_prompts)
    cond_embeds = _encode(prompts)
    return torch.cat([uncond_embeds, cond_embeds], dim=0)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> List[Image.Image]:
    """Undo 0.18215 scaling, run vae.decode in float32 to avoid NaN, export PIL."""
    # Ép sang float32 để chống tràn số / ảnh đen trên VAE
    latents = (latents / VAE_SCALING_FACTOR).to(dtype=torch.float32)
    pixel_values = vae.decode(latents).sample  # (B, 3, H, W)
    pixel_values = (pixel_values / 2.0 + 0.5).clamp(0.0, 1.0)
    pixel_values = pixel_values.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (pixel_values * 255.0).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


# --------------------------------------------------------------------------
# Full sampling loop
# --------------------------------------------------------------------------


@dataclass
class SDComponents:
    unet: "torch.nn.Module"
    vae: "torch.nn.Module"
    text_encoder: "torch.nn.Module"
    tokenizer: object


@torch.no_grad()
def sample(
    components: SDComponents,
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]] = "",
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[torch.dtype] = None,
    scheduler: Optional[NoiseScheduler] = None,
) -> List[Image.Image]:
    """Standalone DDIM + classifier-free-guidance sampling loop."""
    unet, vae, text_encoder, tokenizer = (
        components.unet,
        components.vae,
        components.text_encoder,
        components.tokenizer,
    )
    # Tự động đồng bộ dtype theo trọng số của UNet nếu không truyền vào
    if dtype is None:
        dtype = next(unet.parameters()).dtype

    scheduler = scheduler or NoiseScheduler()
    do_cfg = guidance_scale > 1.0

    prompts = [prompt] * num_images if isinstance(prompt, str) else list(prompt)
    negatives = [negative_prompt] * len(prompts) if isinstance(negative_prompt, str) else list(negative_prompt)
    bsz = len(prompts)

    if do_cfg:
        text_embeddings = encode_prompt(tokenizer, text_encoder, prompts, negatives, device).to(dtype=dtype)
    else:
        tokens = tokenizer(
            prompts, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        text_embeddings = text_encoder(tokens)[0].to(dtype=dtype)

    in_channels = unet.config.in_channels
    latent_h, latent_w = height // 8, width // 8
    latents = torch.randn(
        (bsz, in_channels, latent_h, latent_w), generator=generator, device=device, dtype=dtype
    )

    timesteps = scheduler.set_timesteps(num_inference_steps, device=device)

    for i, t in enumerate(timesteps):
        latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        t_batch = t.expand(latent_input.shape[0])

        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=text_embeddings).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device, dtype=torch.long)
        latents = ddim_step(scheduler, noise_pred, t, prev_t, latents, eta=eta, generator=generator)

    return decode_latents(vae, latents)