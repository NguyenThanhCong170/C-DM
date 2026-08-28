from __future__ import annotations

"""
Fine-tune Stable Diffusion trên NIH ChestX-ray14 với điều kiện là vector multi-hot
5 chiều (No Finding / Infiltration / Effusion / Atelectasis / Others).

Kiến trúc:
    labels (B,5) --MultiHotLabelEncoder--> (B,L,768) --cross-attn--> U-Net(+LoRA)

Train: LoRA của U-Net + toàn bộ LabelEncoder. VAE và phần còn lại của U-Net đóng băng.
(Decoder VAE được tinh chỉnh riêng ở `train_vae_decoder.py`)

    python train_multilabel.py --config config/multilabel.yaml
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from logging_utils import WandbLogger, gpu_memory_gb
from dataset.nih_multilabel import (
    LABEL_NAMES,
    NIHMultiLabelDataset,
    collate_multilabel,
    patient_level_split,
)
from models.label_encoder import (
    LabelEncoderConfig,
    MultiHotLabelEncoder,
    batch_multihot,
    load_label_encoder_into,
    save_label_encoder,
)
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
from models.loading import load_scheduler_config, load_unet, load_vae
from pipeline.inference import VAE_SCALING_FACTOR, NoiseScheduler, min_snr_weights
from pipeline.label_inference import LabelSDComponents, sample_from_labels

# --------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------

DEFAULTS: Dict[str, object] = {
    # Model
    "variant": None,
    "vae_decoder_checkpoint": None,     # .safetensors từ train_vae_decoder.py
    # LoRA
    "rank": 64,
    "lora_alpha": 64.0,
    "lora_dropout": 0.0,
    "target_modules": list(DEFAULT_TARGET_MODULES),
    "cross_attention_only": False,      # True -> chỉ tiêm vào attn2 (khối nhận nhãn)
    "resume_lora": None,
    "resume_label_encoder": None,
    # Label encoder
    "tokens_per_label": 2,
    "label_encoder_layers": 2,
    "label_encoder_heads": 8,
    "label_encoder_dropout": 0.0,
    "label_encoder_lr": None,           # None -> dùng learning_rate
    "cond_dropout_prob": 0.1,           # tỉ lệ thay nhãn bằng null -> học CFG
    # Dataset
    "data_root": "./data/nih",
    "csv_path": None,
    "resolution": 512,
    "view_position": None,              # "PA" | "AP" | None
    "max_per_label": None,              # {"No Finding": 15000}
    "max_images": None,
    "cache_dir": None,
    "val_ratio": 0.0,                   # >0 -> tách theo Patient ID
    "balance_beta": 0.5,                # 0 = giữ phân phối gốc, 1 = cân bằng hẳn
    "dataloader_num_workers": 4,
    # Optimizer
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    # Training
    "seed": 42,
    "max_train_steps": 20000,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "snr_gamma": 5.0,
    # Logging / checkpoint / validation
    "logging_steps": 25,
    "checkpointing_steps": 1000,
    "validation_steps": 1000,
    "validation_labels": [
        ["No Finding"],
        ["Infiltration"],
        ["Effusion"],
        ["Atelectasis"],
        ["Effusion", "Atelectasis"],
    ],
    "validation_inference_steps": 25,
    "validation_guidance_scale": 4.0,
    # Weights & Biases (để wandb_project = null là tắt hẳn)
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_mode": "online",          # online | offline | disabled
    "wandb_log_images": True,
}


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/multilabel.yaml")
    p.add_argument("-o", "--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="ghi đè một khoá config, lặp lại được. "
                        "VD: -o data_root=/data/nih -o train_batch_size=8")
    cli = p.parse_args(argv)

    cfg_path = Path(cli.config)
    if not cfg_path.is_file():
        p.error(f"Không tìm thấy file config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for item in cli.overrides:
        if "=" not in item:
            p.error(f"--set cần dạng KEY=VALUE, nhận '{item}'")
        key, _, raw = item.partition("=")
        key = key.strip()
        # parse bằng YAML để 8 -> int, 1e-4 -> float, true -> bool, null -> None
        config[key] = yaml.safe_load(raw)
        print(f"[config] ghi đè {key} = {config[key]!r}")

    reserved = {"output_dir", "pretrained_model_name_or_path", "device"}
    unknown = sorted(set(config) - set(DEFAULTS) - reserved)
    if unknown:
        print(f"[warn] khoá lạ trong {cfg_path} (bỏ qua): {', '.join(unknown)}")

    args = argparse.Namespace(**{**DEFAULTS, **config})
    for key in ("pretrained_model_name_or_path", "output_dir"):
        if not hasattr(args, key):
            p.error(f"{cfg_path} thiếu '{key}'.")

    args.device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if args.mixed_precision == "bf16" and args.device == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("GPU không hỗ trợ bf16 (vd T4) — dùng mixed_precision: fp16.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(param_groups, args) -> torch.optim.Optimizer:
    kw = dict(betas=(args.adam_beta1, args.adam_beta2),
              weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    return torch.optim.AdamW(param_groups, lr=args.learning_rate, **kw)


def build_lr_lambda(name: str, warmup: int, total: int):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if name == "constant":
            return 1.0
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(name)
    return fn


def cycle(dataloader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in dataloader:
            yield batch


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def diffusion_loss(model_pred, target, timesteps, noise_scheduler, snr_gamma: float):
    """MSE theo mẫu, có trọng số Min-SNR-gamma. Tính ở fp32 cho ổn định dưới AMP."""
    per_sample = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    per_sample = per_sample.mean(dim=list(range(1, model_pred.dim())))
    if snr_gamma and snr_gamma > 0:
        per_sample = per_sample * min_snr_weights(noise_scheduler.compute_snr(timesteps), snr_gamma)
    return per_sample.mean()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    device = args.device
    amp_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    use_amp = args.mixed_precision in ("fp16", "bf16") and device == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16

    output_dir = Path(args.output_dir)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)

    logger = WandbLogger.from_config(args, job_type="multilabel-lora")

    # ---------------- 1. Backbone (không nạp CLIP: điều kiện không còn là text)
    print("[load] U-Net + VAE ...")
    unet = load_unet(args.pretrained_model_name_or_path, variant=args.variant)
    vae = load_vae(args.pretrained_model_name_or_path, variant=args.variant)
    noise_scheduler = NoiseScheduler.from_diffusers_config(
        load_scheduler_config(args.pretrained_model_name_or_path))
    cross_dim = unet.config.cross_attention_dim
    print(f"[load] cross_attention_dim = {cross_dim} | schedule = {noise_scheduler.beta_schedule}")

    if args.vae_decoder_checkpoint:
        from safetensors.torch import load_file
        state = load_file(args.vae_decoder_checkpoint)
        missing, unexpected = vae.load_state_dict(state, strict=False)
        print(f"[vae] nạp decoder tinh chỉnh từ {args.vae_decoder_checkpoint} "
              f"({len(state)} tensor, thừa={len(unexpected)})")

    for m in (vae, unet):
        m.requires_grad_(False)
    vae.eval()

    vae.to(device)                                    # VAE giữ fp32: encode chạy ngoài autocast
    unet.to(device, dtype=amp_dtype if use_amp else None)

    # ---------------- 2. LoRA
    targets = ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"] \
        if args.cross_attention_only else list(args.target_modules)
    injected = inject_lora(unet, target_modules=targets, rank=args.rank,
                           alpha=args.lora_alpha, dropout=args.lora_dropout)
    n_lora = num_trainable_parameters(injected)
    print(f"[lora] {len(injected)} adapter | {n_lora:,} tham số "
          f"| targets = {targets}")
    save_lora_config(LoRAConfig(rank=args.rank, alpha=args.lora_alpha,
                                target_modules=tuple(targets), dropout=args.lora_dropout),
                     str(output_dir / "lora_config.json"))
    if args.resume_lora:
        load_lora_weights_into(injected, args.resume_lora)
        print(f"[resume] LoRA <- {args.resume_lora}")

    # ---------------- 3. Label encoder (train full, luôn ở fp32)
    le_cfg = LabelEncoderConfig(
        num_labels=len(LABEL_NAMES),
        embed_dim=cross_dim,
        tokens_per_label=args.tokens_per_label,
        num_layers=args.label_encoder_layers,
        num_heads=args.label_encoder_heads,
        dropout=args.label_encoder_dropout,
        label_names=tuple(LABEL_NAMES),
    )
    label_encoder = MultiHotLabelEncoder(le_cfg).to(device, dtype=torch.float32)
    label_encoder.train()
    if args.resume_label_encoder:
        load_label_encoder_into(label_encoder, args.resume_label_encoder)
        print(f"[resume] LabelEncoder <- {args.resume_label_encoder}")
    n_le = sum(p.numel() for p in label_encoder.parameters())
    print(f"[label] {le_cfg.num_tokens} token/mẫu | {n_le:,} tham số | nhãn = {list(LABEL_NAMES)}")
    with open(output_dir / "label_encoder_config.json", "w", encoding="utf-8") as f:
        json.dump(le_cfg.to_dict(), f, indent=2, ensure_ascii=False)
    logger.summary({"params/lora": n_lora, "params/label_encoder": n_le,
                    "params/trainable_total": n_lora + n_le,
                    "label_encoder/num_tokens": le_cfg.num_tokens})

    # ---------------- 4. Dataset
    csv_path = args.csv_path or str(Path(args.data_root) / "Data_Entry_2017.csv")
    train_pids = None
    if args.val_ratio and args.val_ratio > 0:
        train_pids, val_pids = patient_level_split(csv_path, args.val_ratio, args.seed)
        print(f"[split] {len(train_pids):,} bệnh nhân train / {len(val_pids):,} val")

    train_dataset = NIHMultiLabelDataset(
        data_root=args.data_root,
        csv_path=csv_path,
        size=args.resolution,
        view_position=args.view_position,
        max_per_label=args.max_per_label,
        max_images=args.max_images,
        patient_ids=train_pids,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    sampler = train_dataset.make_balanced_sampler(beta=args.balance_beta) \
        if args.balance_beta and args.balance_beta > 0 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_multilabel,
        num_workers=args.dataloader_num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    counts = train_dataset.label_counts()
    logger.summary({"data/num_images": len(train_dataset),
                    **{f"data/count_{n.replace(' ', '_')}": int(c)
                       for n, c in zip(LABEL_NAMES, counts)}})

    # ---------------- 5. Optimizer
    lora_params = lora_parameters(injected)
    le_params = list(label_encoder.parameters())
    le_lr = args.label_encoder_lr or args.learning_rate
    optimizer = build_optimizer(
        [{"params": lora_params, "lr": args.learning_rate},
         {"params": le_params, "lr": le_lr}],
        args,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.lr_scheduler, args.lr_warmup_steps, args.max_train_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    all_params = lora_params + le_params

    eff_batch = args.train_batch_size * args.gradient_accumulation_steps
    print(f"[train] batch hiệu dụng {eff_batch} | {args.max_train_steps} step "
          f"| amp={args.mixed_precision if use_amp else 'off'} "
          f"| lr lora={args.learning_rate:g} label={le_lr:g}")

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    batches = cycle(train_dataloader)
    grad_checked = False
    loss_ema = None

    for global_step in range(1, args.max_train_steps + 1):
        for _ in range(args.gradient_accumulation_steps):
            batch = next(batches)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32, non_blocking=True)
            labels = batch["labels"].to(device, dtype=torch.float32, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * VAE_SCALING_FACTOR

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                      (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # LabelEncoder cần gradient -> nằm TRONG autocast, không trong no_grad
                context = label_encoder(labels, drop_prob=args.cond_dropout_prob)
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states=context.to(noisy_latents.dtype))
                loss = diffusion_loss(model_pred, noise, timesteps,
                                      noise_scheduler, args.snr_gamma)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

        if args.max_grad_norm and args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)

        if not grad_checked:
            n_lora_g = sum(1 for p in lora_params if p.grad is not None)
            n_le_g = sum(1 for p in le_params if p.grad is not None)
            if n_lora_g == 0 or n_le_g == 0:
                raise RuntimeError(
                    f"Gradient không chảy (lora={n_lora_g}/{len(lora_params)}, "
                    f"label={n_le_g}/{len(le_params)}). Kiểm tra target_modules / "
                    f"target_modules.")
            print(f"[sanity] gradient tới {n_lora_g} tensor LoRA và {n_le_g} tensor LabelEncoder ✓")
            grad_checked = True

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        l = loss.detach().item()
        loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
        if global_step % args.logging_steps == 0 or global_step == 1:
            lrs = lr_scheduler.get_last_lr()
            print(f"[step {global_step}/{args.max_train_steps}] loss={l:.4f} "
                  f"ema={loss_ema:.4f} lr={lrs[0]:.2e}")
            logger.log({
                "train/loss": l,
                "train/loss_ema": loss_ema,
                "train/lr_lora": lrs[0],
                "train/lr_label_encoder": lrs[-1],
                "train/samples_seen": global_step * eff_batch,
                **gpu_memory_gb(),
            }, step=global_step)

        if global_step % args.checkpointing_steps == 0:
            _save(injected, label_encoder, output_dir, global_step, args)

        if args.validation_steps and global_step % args.validation_steps == 0:
            _validate(unet, vae, label_encoder, args, global_step, output_dir,
                      device, noise_scheduler, use_amp, amp_dtype, logger)

    _save(injected, label_encoder, output_dir, args.max_train_steps, args, final=True)
    logger.summary({"train/final_loss_ema": loss_ema})
    logger.finish()
    print("[done]")


def _save(injected, label_encoder, output_dir: Path, step: int, args, final: bool = False) -> None:
    tag = "final" if final else f"{step}"
    lora_path = output_dir / f"lora-{tag}.safetensors"
    le_path = output_dir / f"label_encoder-{tag}.safetensors"
    save_lora_weights(injected, str(lora_path), alpha=args.lora_alpha, rank=args.rank,
                      extra_metadata={"step": step})
    save_label_encoder(label_encoder, str(le_path), extra_metadata={"step": step})
    print(f"[checkpoint] {lora_path.name} + {le_path.name}")


@torch.no_grad()
def _validate(unet, vae, label_encoder, args, step: int, output_dir: Path, device,
              noise_scheduler, use_amp: bool, amp_dtype, logger=None) -> None:
    was_training = unet.training
    unet.eval()
    label_encoder.eval()
    val_dir = output_dir / "validation" / f"step_{step}"
    val_dir.mkdir(parents=True, exist_ok=True)

    combos = args.validation_labels
    y = batch_multihot(combos, LABEL_NAMES).to(device)
    components = LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            images = sample_from_labels(
                components, y, num_images=len(combos),
                height=args.resolution, width=args.resolution,
                num_inference_steps=args.validation_inference_steps,
                guidance_scale=args.validation_guidance_scale,
                generator=generator, device=device, scheduler=noise_scheduler,
            )
        captions = []
        for combo, img in zip(combos, images):
            name = "+".join(combo).replace(" ", "") if isinstance(combo, (list, tuple)) else str(combo)
            captions.append(name)
            img.save(val_dir / f"{name}.png")
        print(f"[validation] step {step}: {len(images)} ảnh -> {val_dir}")
        if logger is not None:
            logger.log_images(images, captions, step=step)
    except torch.cuda.OutOfMemoryError:
        print("[validation] ⚠ OOM — bỏ qua lượt này.")
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        label_encoder.train()
        if was_training:
            unet.train()


if __name__ == "__main__":
    main()
