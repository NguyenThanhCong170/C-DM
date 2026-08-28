from __future__ import annotations

"""
Tinh chỉnh DECODER của VAE trên ảnh X-quang ngực (encoder đóng băng).

Vì sao phải là một stage riêng: loss khuếch tán chỉ đi qua encoder (ảnh -> latent)
và U-Net. Decoder không nằm trên đồ thị tính toán đó, nên không thể train chung với
`train_multilabel.py`. Ở đây ta train nó bằng loss tái tạo thuần:

        x --(encoder đóng băng)--> z --(decoder train)--> x̂
        L = L1(x, x̂) + w_perc · LPIPS-VGG(x, x̂)

Đóng băng encoder là bắt buộc: nếu encoder đổi, không gian latent đổi theo và mọi
checkpoint LoRA/U-Net đã train sẽ vô nghĩa.

Vì sao đáng làm với X-quang: VAE của SD được train trên ảnh tự nhiên, khi nén ảnh
xám dải động hẹp nó hay bôi mất các chi tiết tần số cao — vân mạch máu, đường Kerley,
bờ tràn dịch mỏng. Chỉ cần decoder học lại đúng miền này là ảnh sinh ra nét hơn rõ rệt.

    python train_vae_decoder.py --config config/vae_decoder.yaml
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset.nih_multilabel import NIHMultiLabelDataset, collate_multilabel
from logging_utils import WandbLogger, gpu_memory_gb
from models.loading import load_vae

DEFAULTS: Dict[str, object] = {
    "variant": None,
    "data_root": "./data/nih",
    "csv_path": None,
    "resolution": 512,
    "crop_size": 256,        # train trên crop ngẫu nhiên -> nhẹ VRAM; decoder là FCN nên vẫn tổng quát
    "view_position": None,
    "max_images": 20000,
    "cache_dir": None,
    "dataloader_num_workers": 4,
    "sample_posterior": False,   # True: lấy mẫu từ posterior thay vì mode -> decoder chịu nhiễu tốt hơn
    "learning_rate": 1e-5,       # decoder pretrain rồi: lr cao sẽ phá texture
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 200,
    "max_train_steps": 6000,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "l1_weight": 1.0,
    "mse_weight": 0.0,
    "perceptual_weight": 0.1,
    "perceptual_layers": [3, 8, 15, 22],   # relu1_2, relu2_2, relu3_3, relu4_3 của VGG16
    "seed": 42,
    "logging_steps": 25,
    "checkpointing_steps": 1000,
    "validation_steps": 500,
    "num_validation_images": 2,
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_mode": "online",
    "wandb_log_images": True,
}


# --------------------------------------------------------------------------
# Perceptual loss (VGG16 ImageNet, đóng băng)
# --------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """LPIPS rút gọn: L1 giữa các feature map VGG16 đã chuẩn hóa theo kênh."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, layers=(3, 8, 15, 22)):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.layers = sorted(int(l) for l in layers)
        self.slices = nn.ModuleList()
        prev = 0
        for l in self.layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l + 1)]))
            prev = l + 1
        self.requires_grad_(False)
        self.eval()
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x trong [-1,1] -> [0,1] -> chuẩn hóa ImageNet
        return ((x.float() + 1.0) / 2.0 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h_p, h_t = self._norm(pred), self._norm(target)
        loss = pred.new_zeros(())
        for s in self.slices:
            h_p, h_t = s(h_p), s(h_t)
            # chuẩn hóa theo kênh (đúng tinh thần LPIPS) trước khi so sánh
            p = h_p / (h_p.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
            t = h_t / (h_t.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
            loss = loss + (p - t).abs().mean()
        return loss / len(self.slices)


# --------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/vae_decoder.yaml")
    p.add_argument("-o", "--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="ghi đè một khoá config, lặp lại được")
    cli = p.parse_args(argv)
    cfg_path = Path(cli.config)
    if not cfg_path.is_file():
        p.error(f"Không tìm thấy config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for item in cli.overrides:
        if "=" not in item:
            p.error(f"--set cần dạng KEY=VALUE, nhận '{item}'")
        key, _, raw = item.partition("=")
        config[key.strip()] = yaml.safe_load(raw)
        print(f"[config] ghi đè {key.strip()} = {config[key.strip()]!r}")
    args = argparse.Namespace(**{**DEFAULTS, **config})
    for key in ("pretrained_model_name_or_path", "output_dir"):
        if not hasattr(args, key):
            p.error(f"{cfg_path} thiếu '{key}'.")
    args.device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_lr_lambda(name: str, warmup: int, total: int):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if name == "constant":
            return 1.0
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        return max(0.0, 1.0 - progress) if name == "linear" else 0.5 * (1 + math.cos(math.pi * progress))
    return fn


def random_crop(x: torch.Tensor, crop: int) -> torch.Tensor:
    """Crop ngẫu nhiên cùng vị trí cho cả batch (đủ dùng, rẻ hơn crop từng ảnh)."""
    _, _, h, w = x.shape
    if crop <= 0 or (crop >= h and crop >= w):
        return x
    top = random.randint(0, max(0, h - crop))
    left = random.randint(0, max(0, w - crop))
    return x[:, :, top:top + crop, left:left + crop]


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = args.device
    amp_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    use_amp = args.mixed_precision in ("fp16", "bf16") and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    output_dir = Path(args.output_dir)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)
    logger = WandbLogger.from_config(args, job_type="vae-decoder")

    vae = load_vae(args.pretrained_model_name_or_path, variant=args.variant)
    vae.to(device, dtype=torch.float32)

    # Đóng băng encoder + quant_conv; chỉ decoder + post_quant_conv được học
    vae.requires_grad_(False)
    vae.encoder.eval()
    trainable = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
    for p_ in trainable:
        p_.requires_grad_(True)
    vae.decoder.train()
    print(f"[vae] train {sum(p.numel() for p in trainable):,} tham số (decoder + post_quant_conv); "
          f"encoder ĐÓNG BĂNG (latent space không đổi)")

    perceptual = None
    if args.perceptual_weight and args.perceptual_weight > 0:
        try:
            perceptual = VGGPerceptualLoss(args.perceptual_layers).to(device)
            print("[loss] perceptual VGG16 bật")
        except Exception as e:   # không tải được weights (máy offline)
            print(f"[warn] không dựng được perceptual loss ({e}) — chỉ dùng L1/MSE.")

    dataset = NIHMultiLabelDataset(
        data_root=args.data_root,
        csv_path=args.csv_path or str(Path(args.data_root) / "Data_Entry_2017.csv"),
        size=args.resolution,
        view_position=args.view_position,
        max_images=args.max_images,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                            collate_fn=collate_multilabel,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=(device == "cuda"), drop_last=True,
                            persistent_workers=args.dataloader_num_workers > 0)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.lr_scheduler, args.lr_warmup_steps, args.max_train_steps))

    def batches():
        while True:
            for b in dataloader:
                yield b
    it = batches()

    print(f"[train] {len(dataset):,} ảnh | crop {args.crop_size} | {args.max_train_steps} step")
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_train_steps + 1):
        logs = {}
        for _ in range(args.gradient_accumulation_steps):
            x = next(it)["pixel_values"].to(device, dtype=torch.float32, non_blocking=True)
            x = random_crop(x, int(args.crop_size))

            with torch.no_grad():
                posterior = vae.encode(x).latent_dist
                z = posterior.sample() if args.sample_posterior else posterior.mode()

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                x_hat = vae.decode(z).sample
                loss = x.new_zeros(())
                if args.l1_weight:
                    l1 = F.l1_loss(x_hat.float(), x.float())
                    loss = loss + args.l1_weight * l1
                    logs["l1"] = l1.detach()
                if args.mse_weight:
                    mse = F.mse_loss(x_hat.float(), x.float())
                    loss = loss + args.mse_weight * mse
                    logs["mse"] = mse.detach()
                if perceptual is not None:
                    lp = perceptual(x_hat, x)
                    loss = loss + args.perceptual_weight * lp
                    logs["perc"] = lp.detach()
            logs["loss"] = loss.detach()
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

        if args.max_grad_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.logging_steps == 0 or step == 1:
            parts = " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
            print(f"[step {step}/{args.max_train_steps}] {parts} lr={lr_scheduler.get_last_lr()[0]:.2e}")
            logger.log({**{f"vae/{k}": v.item() for k, v in logs.items()},
                        "vae/lr": lr_scheduler.get_last_lr()[0],
                        **gpu_memory_gb()}, step=step)

        if args.validation_steps and step % args.validation_steps == 0:
            _validate(vae, dataset, args, step, output_dir, device, logger)

        if step % args.checkpointing_steps == 0 or step == args.max_train_steps:
            _save_decoder(vae, output_dir, step)

    logger.finish()
    print("[done]")


def _save_decoder(vae, output_dir: Path, step: int) -> None:
    """Chỉ lưu decoder + post_quant_conv — nạp bằng load_state_dict(strict=False)."""
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous().float()
             for k, v in vae.state_dict().items()
             if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
    path = output_dir / f"vae_decoder-{step}.safetensors"
    save_file(state, str(path), metadata={"step": str(step), "part": "decoder+post_quant_conv"})
    print(f"[checkpoint] {path.name} ({len(state)} tensor)")


@torch.no_grad()
def _validate(vae, dataset, args, step: int, output_dir: Path, device, logger=None) -> None:
    from PIL import Image

    vae.decoder.eval()
    val_dir = output_dir / "validation" / f"step_{step}"
    val_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(min(args.num_validation_images, len(dataset))):
        x = dataset[i]["pixel_values"].unsqueeze(0).to(device, dtype=torch.float32)
        z = vae.encode(x).latent_dist.mode()
        x_hat = vae.decode(z).sample
        pair = torch.cat([x, x_hat], dim=3)          # ghép gốc | tái tạo
        arr = ((pair.float().clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = arr[0].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr)
        img.save(val_dir / f"recon_{i:02d}.png")
        made.append(img)
    print(f"[validation] step {step}: ảnh gốc|tái tạo -> {val_dir}")
    if logger is not None:
        logger.log_images(made, [f"gốc | tái tạo #{i}" for i in range(len(made))],
                          step=step, key="vae/reconstruction")
    vae.decoder.train()


if __name__ == "__main__":
    main()
