#!/usr/bin/env python
"""
train_custom_dreambooth.py — pure-PyTorch DreamBooth + custom LoRA fine-tuning
of a Stable Diffusion 1.x U-Net on chest X-ray data.

Orchestrates:
  1. Loading the frozen backbone (VAE, CLIP text encoder, U-Net) from a
     Diffusers-format checkpoint (SD 1.5, or a chest-X-ray-pretrained
     checkpoint such as danyalmalik/stable-diffusion-chest-xray).
  2. Injecting `LoRALinear` adapters into the U-Net's attention projections
     (models/lora.py) and freezing everything else.
  3. Building the DreamBooth dual-sampling dataset/dataloader
     (dataset/xray_dataset.py).
  4. Running the training loop: forward diffusion, fp16 autocast + GradScaler,
     DreamBooth loss (instance + prior-preservation) with Min-SNR-gamma
     weighting, AdamW (or 8-bit AdamW) over the LoRA parameters only.
  5. Periodic checkpointing (models/lora.py safetensors) and optional
     validation image sampling (pipeline/inference.py).

Example:
    python train_custom_dreambooth.py \\
        --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \\
        --instance_data_dir ./data/pneumonia \\
        --instance_prompt "a chest x-ray of sks pneumonia" \\
        --class_data_dir ./data/normal \\
        --class_prompt "a chest x-ray" \\
        --with_prior_preservation \\
        --output_dir ./lora_dreambooth_pneumonia \\
        --rank 24 --lora_alpha 24 --snr_gamma 5.0 \\
        --train_batch_size 2 --gradient_accumulation_steps 2 \\
        --max_train_steps 1500 --learning_rate 1e-4 \\
        --mixed_precision fp16 --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.xray_dataset import DreamBoothXrayDataset, collate_fn
from models.lora import (
    DEFAULT_TARGET_MODULES,
    LoRAConfig,
    inject_lora,
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
    p.add_argument("--pretrained_model_name_or_path", type=str,
                    default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--revision", type=str, default=None)

    # Data
    p.add_argument("--instance_data_dir", type=str, required=True)
    p.add_argument("--instance_prompt", type=str, required=True)
    p.add_argument("--class_data_dir", type=str, default=None)
    p.add_argument("--class_prompt", type=str, default=None)
    p.add_argument("--with_prior_preservation", action="store_true")
    p.add_argument("--prior_loss_weight", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=2)

    # LoRA
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--lora_alpha", type=float, default=24.0)
    p.add_argument("--target_modules", type=str, nargs="+", default=list(DEFAULT_TARGET_MODULES))

    # Diffusion / loss
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--snr_gamma", type=float, default=5.0,
                    help="Min-SNR-gamma weighting (Hang et al.). Set <=0 to disable.")

    # Optimization
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=1500)
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
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--validation_steps", type=int, default=250)

    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Backbone loading / freezing
# --------------------------------------------------------------------------


def load_frozen_backbone(pretrained_model_name_or_path: str, revision: Optional[str] = None):
    """Load tokenizer + frozen VAE / CLIP text encoder / U-Net from a
    Diffusers-format checkpoint (SD 1.5 layout: subfolders
    tokenizer/text_encoder/vae/unet)."""
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )
    return tokenizer, text_encoder, vae, unet


def freeze_backbone(vae, text_encoder, unet) -> None:
    """Freeze every parameter of all three sub-modules. LoRA injection
    (models.lora.inject_lora) freezes the base_layer of each wrapped Linear
    too, but calling this first guarantees *everything* starts frozen —
    including conv layers, GroupNorms, and the timestep/text embeddings that
    LoRA never touches."""
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
                weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
            )
        except ImportError:
            print("[warn] --use_8bit_adam requested but bitsandbytes is not installed; "
                  "falling back to torch.optim.AdamW.")
    return torch.optim.AdamW(
        params, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )


def build_lr_lambda(name: str, num_warmup_steps: int, num_training_steps: int):
    """Hand-rolled warmup + {constant, linear, cosine} decay — avoids an
    extra dependency on `diffusers.optimization.get_scheduler`."""
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
# Core DreamBooth + Min-SNR loss
# --------------------------------------------------------------------------


def compute_dreambooth_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: NoiseScheduler,
    with_prior_preservation: bool,
    prior_loss_weight: float,
    snr_gamma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DreamBooth loss with optional prior preservation and optional
    Min-SNR-gamma per-sample weighting, epsilon-prediction parameterization.

    If `with_prior_preservation`, `model_pred`/`target`/`timesteps` are
    assumed to be laid out as [instance_batch ; class_batch] along dim 0 in
    equal halves (this is exactly what `dataset.collate_fn` produces) —
    the two halves get their own (optionally Min-SNR-weighted) MSE loss, and
    are combined as `loss = loss_instance + prior_loss_weight * loss_prior`.
    """

    def _weighted_mse(pred: torch.Tensor, tgt: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        per_sample = F.mse_loss(pred.float(), tgt.float(), reduction="none").mean(dim=list(range(1, pred.dim())))
        if snr_gamma is not None and snr_gamma > 0:
            snr = noise_scheduler.compute_snr(ts)
            weights = min_snr_weights(snr, snr_gamma)
            per_sample = per_sample * weights
        return per_sample.mean()

    logs: Dict[str, float] = {}

    if with_prior_preservation:
        bsz = model_pred.shape[0]
        if bsz % 2 != 0:
            raise ValueError(
                f"with_prior_preservation expects an even batch (instance+class halves), got batch size {bsz}"
            )
        half = bsz // 2
        instance_pred, class_pred = model_pred[:half], model_pred[half:]
        instance_tgt, class_tgt = target[:half], target[half:]
        instance_ts, class_ts = timesteps[:half], timesteps[half:]

        loss_instance = _weighted_mse(instance_pred, instance_tgt, instance_ts)
        loss_prior = _weighted_mse(class_pred, class_tgt, class_ts)
        loss = loss_instance + prior_loss_weight * loss_prior
        logs.update(loss_instance=loss_instance.item(), loss_prior=loss_prior.item())
    else:
        loss = _weighted_mse(model_pred, target, timesteps)
        logs.update(loss_instance=loss.item(), loss_prior=0.0)

    logs["loss"] = loss.item()
    return loss, logs


# --------------------------------------------------------------------------
# Optional: fill missing prior/class images by sampling the (LoRA-free) base model
# --------------------------------------------------------------------------


@torch.no_grad()
def generate_class_images(
    components: SDComponents,
    class_prompt: str,
    class_dir: Path,
    num_class_images: int,
    resolution: int,
    device: str,
    sample_batch_size: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 40,
    seed: int = 0,
) -> None:
    """Top up `class_dir` up to `num_class_images` PNGs by sampling the base
    model at `class_prompt` (PRIOR_MODE="generated" in the original
    notebook). Skipped entirely if you'd rather point --class_data_dir at
    real images (PRIOR_MODE="real"), which is generally the better choice
    when the base checkpoint is *not* already chest-X-ray-pretrained.
    Sampling dtype follows `components.unet`'s own parameter dtype (see
    `pipeline.inference.sample`) — cast the U-Net first if you want fp16."""
    class_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(class_dir.glob("*.png")))
    remaining = max(0, num_class_images - existing)
    if remaining == 0:
        return

    print(f"[class-images] generating {remaining} images for prior preservation -> {class_dir}")
    generator = torch.Generator(device=device).manual_seed(seed)
    done = existing
    while done < num_class_images:
        n = min(sample_batch_size, num_class_images - done)
        images = sd_sample(
            components, prompt=class_prompt, num_images=n,
            height=resolution, width=resolution,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            generator=generator, device=device,
        )
        for img in images:
            img.save(class_dir / f"class_{done:05d}.png")
            done += 1


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    # ---- 1. Backbone -----------------------------------------------------
    tokenizer, text_encoder, vae, unet = load_frozen_backbone(
        args.pretrained_model_name_or_path, revision=args.revision
    )
    run_training(args, tokenizer, text_encoder, vae, unet)


def run_training(args: argparse.Namespace, tokenizer, text_encoder, vae, unet) -> None:
    """Everything after the backbone is loaded: freeze, inject LoRA, build the
    dataset/optimizer, and run the training loop. Split out from `main()` so
    it can be exercised directly (e.g. in tests) with any (tokenizer,
    text_encoder, vae, unet) tuple — real or a small offline stand-in —
    without going through `load_frozen_backbone`'s `from_pretrained` calls.
    """
    torch.manual_seed(args.seed)
    device = args.device
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    use_amp = args.mixed_precision == "fp16" and device == "cuda"
    if args.mixed_precision == "fp16" and device != "cuda":
        print(f"[warn] --mixed_precision=fp16 requested but device={device!r} (not 'cuda'); "
              "training will run in fp32 instead. fp16 autocast is only enabled on CUDA here.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_backbone(vae, text_encoder, unet)

    # VAE and text encoder are only ever run under `torch.no_grad()` here,
    # so keep them in fp32 for numerically stable encode/decode (fp16 VAE
    # decode is a known source of all-black images on Turing GPUs). Both
    # are already fp32 right out of `from_pretrained` (no torch_dtype was
    # requested), so a plain `.to(device)` is enough -- passing an explicit
    # `dtype=` here just to re-assert fp32 trips a noisy Diffusers warning.
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)  # LoRA math done in fp32; autocast casts matmuls as needed

    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # ---- 2. Inject LoRA ----------------------------------------------------
    injected = inject_lora(unet, target_modules=args.target_modules, rank=args.rank, alpha=args.lora_alpha)
    n_trainable = num_trainable_parameters(injected)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"[lora] injected {len(injected)} adapters | trainable params: {n_trainable:,} "
          f"({100 * n_trainable / n_total:.3f}% of U-Net's {n_total:,})")
    save_lora_config(LoRAConfig(rank=args.rank, alpha=args.lora_alpha, target_modules=tuple(args.target_modules)),
                      str(output_dir / "lora_config.json"))

    if args.resume_from_checkpoint:
        from models.lora import load_lora_weights_into
        load_lora_weights_into(injected, args.resume_from_checkpoint)
        print(f"[resume] loaded LoRA weights from {args.resume_from_checkpoint}")

    # ---- 3. Optional prior-preservation class images ----------------------
    class_data_dir = Path(args.class_data_dir) if args.class_data_dir else None
    if args.with_prior_preservation and class_data_dir is not None:
        existing = len(list(class_data_dir.glob("*.png"))) if class_data_dir.is_dir() else 0
        print(f"[class-images] {existing} images already present in {class_data_dir}")
        # If you want on-the-fly generation instead of pointing at real
        # images, call generate_class_images(...) here before building the
        # dataset. Left as an explicit opt-in (see module docstring) since
        # for a non-chest-X-ray-pretrained base model, generated priors can
        # anchor the "class" concept to the wrong visual distribution.

    # ---- 4. Dataset / DataLoader -------------------------------------------
    train_dataset = DreamBoothXrayDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        class_data_root=str(class_data_dir) if (args.with_prior_preservation and class_data_dir) else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        random_flip=args.random_flip,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation=args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # ---- 5. Optimizer / LR schedule / AMP ---------------------------------
    optimizer = build_optimizer(lora_parameters(injected), args)
    lr_lambda = build_lr_lambda(args.lr_scheduler, args.lr_warmup_steps, args.max_train_steps)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # torch>=2.0 non-deprecated API
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # fallback for older torch

    noise_scheduler = NoiseScheduler(num_train_timesteps=args.num_train_timesteps)
    components = SDComponents(unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer)

    steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    num_epochs = math.ceil(args.max_train_steps / steps_per_epoch)
    print(f"[train] {len(train_dataset)} examples | {steps_per_epoch} opt steps/epoch | "
          f"{num_epochs} epochs -> {args.max_train_steps} total steps")

    global_step = 0
    unet.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for micro_step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * VAE_SCALING_FACTOR
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=weight_dtype, enabled=use_amp):
                model_pred = unet(
                    noisy_latents.to(weight_dtype if use_amp else torch.float32),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(weight_dtype if use_amp else torch.float32),
                ).sample
                loss, logs = compute_dreambooth_loss(
                    model_pred, noise, timesteps, noise_scheduler,
                    with_prior_preservation=args.with_prior_preservation,
                    prior_loss_weight=args.prior_loss_weight,
                    snr_gamma=args.snr_gamma,
                )
                loss_to_backprop = loss / args.gradient_accumulation_steps

            scaler.scale(loss_to_backprop).backward()

            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_parameters(injected), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    print(f"[step {global_step}/{args.max_train_steps}] "
                          f"loss={logs['loss']:.4f} instance={logs['loss_instance']:.4f} "
                          f"prior={logs['loss_prior']:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step % args.checkpointing_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}.safetensors"
                    save_lora_weights(injected, str(ckpt_path), alpha=args.lora_alpha, rank=args.rank)
                    print(f"[checkpoint] saved {ckpt_path}")

                if args.validation_prompt and global_step % args.validation_steps == 0:
                    _run_validation(components, args, global_step, output_dir, device)

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    final_path = output_dir / "pytorch_lora_weights.safetensors"
    save_lora_weights(injected, str(final_path), alpha=args.lora_alpha, rank=args.rank)
    print(f"[done] final LoRA weights saved to {final_path}")


@torch.no_grad()
def _run_validation(components: SDComponents, args, step: int, output_dir: Path, device: str) -> None:
    was_training = components.unet.training
    components.unet.eval()
    val_dir = output_dir / "validation" / f"step_{step}"
    val_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = sd_sample(
        components, prompt=args.validation_prompt, num_images=args.num_validation_images,
        height=args.resolution, width=args.resolution,
        num_inference_steps=30, guidance_scale=7.5,
        generator=generator, device=device,
    )
    for i, img in enumerate(images):
        img.save(val_dir / f"sample_{i:02d}.png")
    print(f"[validation] step {step}: saved {len(images)} images to {val_dir}")
    if was_training:
        components.unet.train()


if __name__ == "__main__":
    main()