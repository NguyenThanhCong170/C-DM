from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

VAE_SCALING_FACTOR = 0.18215  # Hệ số scale latent của SD 1.x (AutoencoderKL config)


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
        Dựng scheduler từ config của checkpoint (DDPMScheduler.config).
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

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """Tập con giảm dần, cách đều, của các timestep lúc train."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps phải <= num_train_timesteps")
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64).copy()
        return torch.from_numpy(timesteps).to(device=device, dtype=torch.long)


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
def dpm_solver_2m_step(
    scheduler: NoiseScheduler,
    model_output: torch.Tensor,
    t: int,
    s: int,
    r_t: Optional[int],
    sample: torch.Tensor,
    x0_pred_prev: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Một bước ngược DPM-Solver++ 2M, tham số hóa data-prediction."""
    device, dtype = sample.device, sample.dtype

    ac_t = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[t]
    ac_s = (
        scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[s]
        if s >= 0
        else torch.tensor(1.0, device=device, dtype=torch.float32)
    )

    alpha_t, sigma_t = ac_t.sqrt(), (1.0 - ac_t).sqrt()
    alpha_s, sigma_s = ac_s.sqrt(), (1.0 - ac_s).sqrt()

    lambda_t = torch.log(alpha_t / sigma_t)
    lambda_s = torch.log(alpha_s / sigma_s)
    h = lambda_s - lambda_t

    # Chuyển đổi epsilon-prediction thành data-prediction (x_0)
    x = sample.float()
    eps = model_output.float()
    x0_pred = (x - sigma_t * eps) / alpha_t

    # Bậc 1 (Euler) khi chưa có lịch sử, và ở bước cuối (s < 0: sigma_s = 0,
    # lambda_s = +inf nên h = inf -> ngoại suy bậc 2 cho ra NaN).
    if x0_pred_prev is None or r_t is None or s < 0:
        x_prev = alpha_s * x0_pred + sigma_s * eps
        return x_prev.to(dtype), x0_pred

    # Xấp xỉ bậc 2 (DPM-Solver++ 2M)
    ac_r = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)[r_t]
    alpha_r, sigma_r = ac_r.sqrt(), (1.0 - ac_r).sqrt()
    lambda_r = torch.log(alpha_r / sigma_r)

    h_old = lambda_t - lambda_r
    r = h / h_old

    # Ngoại suy x_0 (D trong bài báo DPM-Solver++)
    x0_pred_hat = x0_pred + 0.5 * r * (x0_pred - x0_pred_prev)

    # Cập nhật nghiệm theo đúng công thức bậc 2:
    #     x_s = (sigma_s/sigma_t)·x_t − alpha_s·(e^(−h) − 1)·D
    # KHÔNG dùng dạng DDIM `alpha_s·D + sigma_s·eps`: hai vế chỉ bằng nhau khi
    # D == x0_pred (tức bậc 1). Ở bậc 2 nó thiếu hạng (alpha_t·sigma_s/sigma_t)·(x0_pred − D),
    # sai số tích lũy qua từng bước làm ảnh bị gắt và cháy tương phản.
    x_prev = (sigma_s / sigma_t) * x - alpha_s * (torch.exp(-h) - 1.0) * x0_pred_hat

    return x_prev.to(dtype), x0_pred


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
    vae_dtype = next(vae.parameters()).dtype
    latents = (latents / VAE_SCALING_FACTOR).to(dtype=vae_dtype)
    pixel_values = vae.decode(latents).sample
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
    num_inference_steps: int = 25,   
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cuda",
    dtype: Optional[torch.dtype] = None,
    scheduler: Optional[NoiseScheduler] = None,
) -> List[Image.Image]:
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

    x0_pred_prev = None
    r_t = None

    for i, t in enumerate(timesteps):
        latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        t_batch = t.expand(latent_input.shape[0])

        noise_pred = unet(latent_input, t_batch, encoder_hidden_states=text_embeddings).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        prev_t = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
        
        latents, x0_pred_prev = dpm_solver_2m_step(
            scheduler, noise_pred, int(t), prev_t, r_t, latents, x0_pred_prev
        )
        r_t = int(t)

    return decode_latents(vae, latents)