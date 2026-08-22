from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.xray_dataset import DreamBoothXrayDataset, collate_fn
from models.lora import (
    DEFAULT_TARGET_MODULES,
    LoRAConfig,
    inject_lora,
    load_lora_weights_into,
    lora_parameters,
    num_trainable_parameters,
    save_lora_config,
    save_lora_weights,
)
from pipeline.inference import (
    VAE_SCALING_FACTOR,
    NoiseScheduler,
    SDComponents,
    min_snr_weights,
    sample as sd_sample,
)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Backbone
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="THƯ MỤC local chứa checkpoint theo bố cục diffusers "
                        "(unet/, vae/, text_encoder/, tokenizer/, scheduler/).")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None,
                   help="Hậu tố file trọng số, ví dụ 'fp16'.")

    # Dữ liệu
    p.add_argument("--concepts_list", type=str, default=None,
                   help="File JSON chứa danh sách multi-concept.")
    p.add_argument("--instance_data_dir", type=str, default=None)
    p.add_argument("--instance_prompt", type=str, default=None)
    p.add_argument("--class_data_dir", type=str, default=None)
    p.add_argument("--class_prompt", type=str, default="a chest x-ray")
    p.add_argument("--with_prior_preservation", action="store_true")
    p.add_argument("--prior_loss_weight", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--random_flip", action="store_true",
                   help="KHÔNG khuyến nghị với X-quang ngực (lật ngang làm sai vị trí tim).")
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    # LoRA
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=64.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--target_modules", type=str, nargs="+", default=list(DEFAULT_TARGET_MODULES))

    # Diffusion / loss
    p.add_argument("--snr_gamma", type=float, default=5.0,
                   help="Min-SNR-gamma weighting (Hang et al.). <=0 để tắt.")

    # Tối ưu
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=2500)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant",
                   choices=["constant", "linear", "cosine"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Precision / device
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Output / logging
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_negative_prompt", type=str, default="")
    p.add_argument("--num_validation_images", type=int, default=2)
    p.add_argument("--validation_steps", type=int, default=250)
    p.add_argument("--validation_inference_steps", type=int, default=30)

    args = p.parse_args(argv)
    if args.concepts_list is None and (args.instance_data_dir is None or args.instance_prompt is None):
        p.error("Bắt buộc cung cấp --concepts_list hoặc cả (--instance_data_dir, --instance_prompt).")
    if args.with_prior_preservation and not args.class_data_dir:
        p.error("--with_prior_preservation cần --class_data_dir.")
    if args.mixed_precision == "bf16" and args.device == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("GPU hiện tại không hỗ trợ bf16 (ví dụ T4) — dùng --mixed_precision fp16.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------


def load_frozen_backbone(pretrained_model_name_or_path: str, revision: Optional[str] = None,
                         variant: Optional[str] = None):
    """
    Nạp tokenizer / text encoder / VAE / U-Net + lịch nhiễu bằng code tự viết
    (models/unet.py, models/vae.py, models/text_encoder.py, models/tokenizer.py).
    Không dùng `diffusers` hay `transformers` — chỉ đọc file bằng `safetensors`.

    `pretrained_model_name_or_path` phải là THƯ MỤC local theo bố cục của diffusers.
    Tải trước bằng:  huggingface-cli download <repo_id> --local-dir ./sd15
    """
    from models.loading import load_sd_components

    tokenizer, text_encoder, vae, unet, scheduler_config = load_sd_components(
        pretrained_model_name_or_path, variant=variant)
    noise_scheduler = NoiseScheduler.from_diffusers_config(scheduler_config)
    print(f"[schedule] {noise_scheduler.beta_schedule}, T={noise_scheduler.num_train_timesteps} "
          f"(đọc từ scheduler_config.json của checkpoint)")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def freeze_backbone(vae, text_encoder, unet) -> None:
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()


# --------------------------------------------------------------------------
# Optimizer / LR schedule
# --------------------------------------------------------------------------


def build_optimizer(params, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                params, lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
        except ImportError:
            print("[warn] --use_8bit_adam nhưng chưa cài bitsandbytes; dùng torch.optim.AdamW.")
    return torch.optim.AdamW(
        params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)


def build_lr_lambda(name: str, num_warmup_steps: int, num_training_steps: int):
    def fn(step: int) -> float:
        if num_warmup_steps > 0 and step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        if name == "constant":
            return 1.0
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(name)
    return fn


# --------------------------------------------------------------------------
# DreamBooth + Min-SNR loss
# --------------------------------------------------------------------------


def compute_dreambooth_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: NoiseScheduler,
    with_prior_preservation: bool,
    prior_loss_weight: float,
    snr_gamma: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Loss được tính ở fp32. Với prior preservation, batch có dạng
    [instance..., class...] nên tách nửa đầu / nửa sau (khớp collate_fn).

    Trả về logs là tensor detach — chỉ gọi .item() lúc in để không ép đồng bộ GPU
    mỗi micro-step.
    """
    def _weighted_mse(pred: torch.Tensor, tgt: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        per_sample = F.mse_loss(pred.float(), tgt.float(), reduction="none")
        per_sample = per_sample.mean(dim=list(range(1, pred.dim())))
        if snr_gamma is not None and snr_gamma > 0:
            weights = min_snr_weights(noise_scheduler.compute_snr(ts), snr_gamma)
            per_sample = per_sample * weights
        return per_sample.mean()

    logs: Dict[str, torch.Tensor] = {}

    if with_prior_preservation:
        bsz = model_pred.shape[0]
        if bsz % 2 != 0:
            raise ValueError(f"with_prior_preservation cần batch chẵn, nhận {bsz}")
        half = bsz // 2
        loss_instance = _weighted_mse(model_pred[:half], target[:half], timesteps[:half])
        loss_prior = _weighted_mse(model_pred[half:], target[half:], timesteps[half:])
        loss = loss_instance + prior_loss_weight * loss_prior
        logs["loss_instance"] = loss_instance.detach()
        logs["loss_prior"] = loss_prior.detach()
    else:
        loss = _weighted_mse(model_pred, target, timesteps)
        logs["loss_instance"] = loss.detach()
        logs["loss_prior"] = torch.zeros((), device=loss.device)

    logs["loss"] = loss.detach()
    return loss, logs


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def cycle(dataloader: DataLoader) -> Iterator[dict]:
    """Lặp vô hạn qua dataloader — cửa sổ gradient accumulation không bị reset theo epoch."""
    while True:
        for batch in dataloader:
            yield batch


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_frozen_backbone(
        args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant)
    run_training(args, tokenizer, text_encoder, vae, unet, noise_scheduler)


def run_training(args, tokenizer, text_encoder, vae, unet, noise_scheduler: NoiseScheduler) -> None:
    set_seed(args.seed)
    device = args.device
    amp_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    use_amp = args.mixed_precision in ("fp16", "bf16") and device == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16  # GradScaler chỉ cần cho fp16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_backbone(vae, text_encoder, unet)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # 1. Tiêm LoRA (adapter tự tạo trên device của layer gốc)
    print("[lora] injecting adapters ...")
    injected = inject_lora(
        unet,
        target_modules=args.target_modules,
        rank=args.rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    n_trainable = num_trainable_parameters(injected)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"[lora] {len(injected)} adapters | trainable {n_trainable:,} "
          f"({100 * n_trainable / n_total:.3f}% của U-Net {n_total:,})")

    lora_config = LoRAConfig(
        rank=args.rank, alpha=args.lora_alpha,
        target_modules=tuple(args.target_modules), dropout=args.lora_dropout)
    save_lora_config(lora_config, str(output_dir / "lora_config.json"))

    if args.resume_from_checkpoint:
        load_lora_weights_into(injected, args.resume_from_checkpoint)
        print(f"[resume] đã nạp LoRA từ {args.resume_from_checkpoint}")

    # 2. Dataset
    concepts_list = None
    if args.concepts_list is not None:
        with open(args.concepts_list, "r", encoding="utf-8") as f:
            concepts_list = json.load(f)

    train_dataset = DreamBoothXrayDataset(
        tokenizer=tokenizer,
        concepts_list=concepts_list,
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        use_percentile_norm=True,
        random_flip=args.random_flip,
        seed=args.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, with_prior_preservation=args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    if len(train_dataloader) == 0:
        raise RuntimeError("DataLoader rỗng — giảm --train_batch_size hoặc bỏ drop_last.")

    # 3. Optimizer / scaler
    params = lora_parameters(injected)
    optimizer = build_optimizer(params, args)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.lr_scheduler, args.lr_warmup_steps, args.max_train_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    components = SDComponents(unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)
    batches = cycle(train_dataloader)

    eff_batch = args.train_batch_size * args.gradient_accumulation_steps
    print(f"[train] {len(train_dataset)} mẫu | batch hiệu dụng {eff_batch} "
          f"| {args.max_train_steps} optimizer steps | amp={args.mixed_precision if use_amp else 'off'}")

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    grad_checked = False

    for global_step in range(1, args.max_train_steps + 1):
        logs: Dict[str, torch.Tensor] = {}

        for _ in range(args.gradient_accumulation_steps):
            batch = next(batches)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * VAE_SCALING_FACTOR
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states=encoder_hidden_states).sample
                loss, logs = compute_dreambooth_loss(
                    model_pred, noise, timesteps, noise_scheduler,
                    with_prior_preservation=args.with_prior_preservation,
                    prior_loss_weight=args.prior_loss_weight,
                    snr_gamma=args.snr_gamma,
                )

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

        if args.max_grad_norm and args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

        # Kiểm tra một lần: gradient có thật sự chảy về LoRA không.
        # (gradient checkpointing kiểu reentrant + base đóng băng có thể làm grad = None
        #  mà loss vẫn in ra bình thường.)
        if not grad_checked:
            with_grad = sum(1 for p in params if p.grad is not None and torch.isfinite(p.grad).any())
            if with_grad == 0:
                raise RuntimeError(
                    "Không có gradient nào tới tham số LoRA. Thử bỏ --gradient_checkpointing "
                    "hoặc kiểm tra lại inject_lora / target_modules.")
            print(f"[sanity] {with_grad}/{len(params)} tensor LoRA có gradient ✓")
            grad_checked = True

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if global_step % args.logging_steps == 0 or global_step == 1:
            print(f"[step {global_step}/{args.max_train_steps}] "
                  f"loss={logs['loss'].item():.4f} "
                  f"instance={logs['loss_instance'].item():.4f} "
                  f"prior={logs['loss_prior'].item():.4f} "
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e}")

        if global_step % args.checkpointing_steps == 0:
            ckpt_path = output_dir / f"checkpoint-{global_step}.safetensors"
            save_lora_weights(injected, str(ckpt_path), alpha=args.lora_alpha, rank=args.rank,
                              extra_metadata={"step": global_step})
            print(f"[checkpoint] {ckpt_path}")

        if args.validation_prompt and global_step % args.validation_steps == 0:
            _run_validation(components, args, global_step, output_dir, device,
                            noise_scheduler, use_amp, amp_dtype)

    final_path = output_dir / "pytorch_lora_weights.safetensors"
    save_lora_weights(injected, str(final_path), alpha=args.lora_alpha, rank=args.rank,
                      extra_metadata={"step": args.max_train_steps})
    print(f"[done] LoRA cuối cùng: {final_path}")


@torch.no_grad()
def _run_validation(components: SDComponents, args, step: int, output_dir: Path, device: str,
                    noise_scheduler: NoiseScheduler, use_amp: bool, amp_dtype: torch.dtype) -> None:
    """Sinh vài ảnh kiểm tra. Chạy dưới autocast để không ngốn VRAM fp32 giữa lúc train."""
    was_training = components.unet.training
    components.unet.eval()
    val_dir = output_dir / "validation" / f"step_{step}"
    val_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            images = sd_sample(
                components,
                prompt=args.validation_prompt,
                negative_prompt=args.validation_negative_prompt,
                num_images=args.num_validation_images,
                height=args.resolution, width=args.resolution,
                num_inference_steps=args.validation_inference_steps,
                guidance_scale=7.5,
                generator=generator, device=device,
                scheduler=noise_scheduler,
            )
        for i, img in enumerate(images):
            img.save(val_dir / f"sample_{i:02d}.png")
        print(f"[validation] step {step}: lưu {len(images)} ảnh tại {val_dir}")
    except torch.cuda.OutOfMemoryError:
        print("[validation] ⚠ OOM — bỏ qua lượt này (giảm --num_validation_images).")
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            components.unet.train()


if __name__ == "__main__":
    main()