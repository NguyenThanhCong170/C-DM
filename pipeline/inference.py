
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

VAE_SCALING_FACTOR = 0.18215  # hệ số scale latent của SD 1.x (AutoencoderKL config)


# --------------------------------------------------------------------------
# Forward process / schedule
# --------------------------------------------------------------------------


class NoiseScheduler:
    """Cài đặt lại lịch beta "scaled_linear" (DDPM) của SD 1.x."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule

        if beta_schedule == "scaled_linear":
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
            )
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        else:
            raise NotImplementedError(f"beta_schedule '{beta_schedule}' chưa được hỗ trợ")

        alphas = 1.0 - betas
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).float()  # (T,)

    @classmethod
    def from_diffusers_config(cls, config) -> "NoiseScheduler":
        """
        Dựng scheduler từ config của checkpoint (DDPMScheduler.config) để lịch nhiễu
        lúc train/sinh ảnh trùng đúng với lịch mà checkpoint nền được huấn luyện.
        """
        get = (lambda k, d=None: config.get(k, d)) if isinstance(config, dict) else (lambda k, d=None: getattr(config, k, d))

        prediction_type = get("prediction_type", "epsilon")
        if prediction_type != "epsilon":
            raise NotImplementedError(
                f"Checkpoint dùng prediction_type='{prediction_type}', code này chỉ hỗ trợ 'epsilon'."
            )
        return cls(
            num_train_timesteps=int(get("num_train_timesteps", 1000)),
            beta_start=float(get("beta_start", 0.00085)),
            beta_end=float(get("beta_end", 0.012)),
            beta_schedule=str(get("beta_schedule", "scaled_linear")),
        )

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """z_t = sqrt(alpha_bar_t) z0 + sqrt(1 - alpha_bar_t) eps, batch theo `timesteps`."""
        ac = self.alphas_cumprod.to(device=x0.device, dtype=torch.float32)[timesteps]
        while ac.dim() < x0.dim():
            ac = ac.unsqueeze(-1)
        return (ac.sqrt() * x0.float() + (1 - ac).sqrt() * noise.float()).to(x0.dtype)

    def compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """SNR(t) = alpha_bar_t / (1 - alpha_bar_t), dùng cho Min-SNR weighting."""
        ac = self.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
        return ac / (1.0 - ac).clamp(min=1e-8)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Tập con giảm dần, cách đều, của các timestep lúc train (cho DDIM)."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps phải <= num_train_timesteps")
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64).copy()
        return torch.from_numpy(timesteps).to(device=device, dtype=torch.long)


def min_snr_weights(snr: torch.Tensor, gamma: float) -> torch.Tensor:
    """Trọng số Min-SNR-gamma cho epsilon-prediction: w = min(SNR(t), gamma) / SNR(t)."""
    return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-8)


def randn_tensor(
    shape,
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    """randn an toàn khi generator ở CPU còn tensor cần nằm trên CUDA."""
    device = torch.device(device)
    if generator is not None and generator.device.type != device.type:
        return torch.randn(shape, generator=generator, device=generator.device, dtype=dtype).to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@torch.no_grad()
def ddim_step(
    scheduler: NoiseScheduler,
    model_output: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    prev_timestep: Union[int, torch.Tensor],
    sample: torch.Tensor,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Một bước ngược DDIM (eta=0 là tất định), tham số hóa epsilon-prediction."""
    device, dtype = sample.device, sample.dtype
    t = int(timestep)
    t_prev = int(prev_timestep)

    ac_t = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[t]
    ac_prev = (
        scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[t_prev]
        if t_prev >= 0
        else torch.tensor(1.0, device=device, dtype=torch.float32)
    )

    # Tính ở fp32 rồi mới trả về dtype gốc, tránh mất chính xác khi chạy fp16.
    x = sample.float()
    eps = model_output.float()

    pred_x0 = (x - (1.0 - ac_t).sqrt() * eps) / ac_t.sqrt()

    sigma_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    if eta > 0:
        sigma_t = eta * (((1.0 - ac_prev) / (1.0 - ac_t)) * (1.0 - ac_t / ac_prev)).clamp(min=0.0).sqrt()

    dir_coeff = (1.0 - ac_prev - sigma_t**2).clamp(min=0.0).sqrt()
    x_prev = ac_prev.sqrt() * pred_x0 + dir_coeff * eps

    if eta > 0:
        noise = randn_tensor(sample.shape, generator, device, torch.float32)
        x_prev = x_prev + sigma_t * noise

    return x_prev.to(dtype)


# --------------------------------------------------------------------------
# Text conditioning / CFG
# --------------------------------------------------------------------------


@torch.no_grad()
def encode_prompt(
    tokenizer,
    text_encoder,
    prompts: List[str],
    negative_prompts: Optional[List[str]],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    Token hóa + encode prompt. Nếu có negative prompt thì trả về [uncond; cond].

    Cố ý KHÔNG truyền attention_mask: CLIP của SD được huấn luyện trên toàn bộ chuỗi
    đã pad (chỉ dùng causal mask), truyền thêm padding mask sẽ ra embedding khác với
    lúc train.
    """
    def _encode(texts: List[str]) -> torch.Tensor:
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(tokens)[0]

    cond_embeds = _encode(prompts)
    if negative_prompts is None:
        return cond_embeds
    uncond_embeds = _encode(negative_prompts)
    return torch.cat([uncond_embeds, cond_embeds], dim=0)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> List[Image.Image]:
    """Gỡ scale 0.18215, decode theo đúng dtype của VAE, xuất PIL."""
    vae_dtype = next(vae.parameters()).dtype
    latents = (latents / VAE_SCALING_FACTOR).to(dtype=vae_dtype)
    pixel_values = vae.decode(latents).sample  # (B, 3, H, W)
    pixel_values = (pixel_values.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    pixel_values = pixel_values.cpu().permute(0, 2, 3, 1).numpy()
    images = (pixel_values * 255.0).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


# --------------------------------------------------------------------------
# Vòng lặp sinh ảnh
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
    """Vòng lặp DDIM + classifier-free guidance độc lập với Diffusers pipeline."""
    unet, vae, text_encoder, tokenizer = (
        components.unet, components.vae, components.text_encoder, components.tokenizer,
    )
    if dtype is None:
        dtype = next(unet.parameters()).dtype

    scheduler = scheduler or NoiseScheduler()
    do_cfg = guidance_scale > 1.0

    prompts = [prompt] * num_images if isinstance(prompt, str) else list(prompt)
    if isinstance(negative_prompt, str):
        negatives = [negative_prompt] * len(prompts)
    else:
        negatives = list(negative_prompt)
        if len(negatives) != len(prompts):
            raise ValueError("negative_prompt phải cùng số lượng với prompt")
    bsz = len(prompts)

    text_embeddings = encode_prompt(
        tokenizer, text_encoder, prompts, negatives if do_cfg else None, device
    ).to(dtype=dtype)

    latents = randn_tensor(
        (bsz, unet.config.in_channels, height // 8, width // 8), generator, device, dtype
    )
    timesteps = scheduler.set_timesteps(num_inference_steps, device=device)

    for i, t in enumerate(timesteps):
        latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        t_batch = t.expand(latent_input.shape[0])

        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=text_embeddings).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        prev_t = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
        latents = ddim_step(scheduler, noise_pred, int(t), prev_t, latents, eta=eta, generator=generator)

    return decode_latents(vae, latents)