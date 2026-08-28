from __future__ import annotations

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
    x_prev = (sigma_s / sigma_t) * x - alpha_s * (torch.exp(-h) - 1.0) * x0_pred_hat

    return x_prev.to(dtype), x0_pred


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> List[Image.Image]:
    vae_dtype = next(vae.parameters()).dtype
    latents = (latents / VAE_SCALING_FACTOR).to(dtype=vae_dtype)
    pixel_values = vae.decode(latents).sample
    pixel_values = (pixel_values.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    pixel_values = pixel_values.cpu().permute(0, 2, 3, 1).numpy()
    images = (pixel_values * 255.0).round().astype("uint8")
    return [Image.fromarray(img) for img in images]
