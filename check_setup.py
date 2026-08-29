from __future__ import annotations

"""
Kiểm tra mọi thứ cần thiết TRƯỚC khi chạy train_multilabel.py.

    python check_setup.py --config config/multilabel.yaml

Chạy tuần tự 6 bước; bước nào hỏng sẽ in đúng lệnh cần sửa. Bước cuối chạy thật
1 optimizer step trên GPU để đo VRAM — nếu bước này qua thì run dài cũng chạy được.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


class Fail(Exception):
    pass


def ok(msg): print(f"  ✓ {msg}")
def warn(msg): print(f"  ! {msg}")


def step(i, title):
    print(f"\n[{i}] {title}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/multilabel.yaml")
    p.add_argument("--skip-train-step", action="store_true",
                   help="bỏ qua bước 6 (thử 1 optimizer step)")
    p.add_argument("-o", "--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="ghi đè một khoá config, lặp lại được. "
                        "VD: -o train_batch_size=16 -o data_root=/data/nih")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"✗ Không thấy {cfg_path}")
        return 1
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}

    for item in args.overrides:
        if "=" not in item:
            p.error(f"--set cần dạng KEY=VALUE, nhận '{item}'")
        key, _, raw = item.partition("=")
        cfg[key.strip()] = yaml.safe_load(raw)      # YAML: 16 -> int, 1e-4 -> float, null -> None
        print(f"[config] ghi đè {key.strip()} = {cfg[key.strip()]!r}")

    try:
        # ------------------------------------------------------------------
        step(1, "Thư viện")
        import torch, torchvision, safetensors, numpy, PIL
        ok(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                prop = torch.cuda.get_device_properties(i)
                ok(f"GPU {i}: {prop.name}, {prop.total_memory / 1024**3:.1f} GB, "
                   f"bf16={'có' if torch.cuda.is_bf16_supported() else 'KHÔNG (dùng fp16)'}")
        else:
            warn("Không thấy GPU — train sẽ cực chậm. Bật GPU accelerator trước.")

        mp = cfg.get("mixed_precision", "fp16")
        if mp == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise Fail("config đặt mixed_precision: bf16 nhưng GPU không hỗ trợ → đổi sang fp16")

        # ------------------------------------------------------------------
        step(2, "Base model Stable Diffusion")
        base = Path(cfg["pretrained_model_name_or_path"])
        if not base.is_dir():
            raise Fail(
                f"Không thấy thư mục '{base}'. Tải bằng:\n"
                f"    pip install huggingface_hub\n"
                f"    hf download stable-diffusion-v1-5/stable-diffusion-v1-5 "
                f"--local-dir {base} \\\n"
                f"      scheduler/scheduler_config.json \\\n"
                f"      vae/config.json vae/diffusion_pytorch_model.fp16.safetensors \\\n"
                f"      unet/config.json unet/diffusion_pytorch_model.fp16.safetensors")
        variant = cfg.get("variant")
        suffix = f".{variant}" if variant else ""
        for sub in ("unet", "vae"):
            f_cfg = base / sub / "config.json"
            f_w = base / sub / f"diffusion_pytorch_model{suffix}.safetensors"
            if not f_cfg.is_file():
                raise Fail(f"thiếu {f_cfg}")
            if not f_w.is_file():
                alt = list((base / sub).glob("*.safetensors"))
                raise Fail(f"thiếu {f_w}" + (f" (thấy: {[a.name for a in alt]} → sửa 'variant' "
                                             f"trong config)" if alt else ""))
            ok(f"{sub}/: config.json + {f_w.name} ({f_w.stat().st_size / 1024**2:.0f} MB)")
        if not (base / "scheduler" / "scheduler_config.json").is_file():
            raise Fail(f"thiếu {base}/scheduler/scheduler_config.json")
        ok("scheduler/scheduler_config.json")
        # Nhánh multi-hot KHÔNG cần text_encoder/tokenizer
        if not (base / "text_encoder").is_dir():
            ok("không có text_encoder — đúng, nhánh multi-hot không dùng CLIP")

        # ------------------------------------------------------------------
        step(3, "Dữ liệu NIH")
        from dataset.nih_multilabel import index_image_files, read_data_entry
        data_root = Path(cfg.get("data_root", "./data/nih"))
        csv_path = Path(cfg.get("csv_path") or data_root / "Data_Entry_2017.csv")
        if not data_root.is_dir():
            raise Fail(f"Không thấy '{data_root}'. Trên Kaggle: Add Data → "
                       f"'NIH Chest X-rays' (nih-chest-xrays/data) → đường dẫn "
                       f"thường là /kaggle/input/data")
        if not csv_path.is_file():
            found = [p.name for p in data_root.glob("*.csv")]
            raise Fail(f"Không thấy {csv_path}" + (f" (trong thư mục có: {found})" if found else ""))
        rows = read_data_entry(csv_path)
        ok(f"{csv_path.name}: {len(rows):,} dòng")
        t0 = time.time()
        files = index_image_files(data_root)
        ok(f"quét được {len(files):,} file ảnh trong {time.time() - t0:.1f}s")
        matched = sum(1 for r in rows if (r.get('Image Index') or '') in files)
        if matched == 0:
            raise Fail("CSV và thư mục ảnh không khớp tên file nào — sai data_root?")
        if matched < len(rows) * 0.9:
            warn(f"chỉ khớp {matched:,}/{len(rows):,} dòng CSV — thiếu một phần ảnh")
        else:
            ok(f"khớp {matched:,}/{len(rows):,} dòng CSV với file ảnh")

        # ------------------------------------------------------------------
        step(4, "Dataset + phân phối nhãn")
        from dataset.nih_multilabel import NIHMultiLabelDataset, collate_multilabel
        ds = NIHMultiLabelDataset(
            data_root=data_root, csv_path=csv_path,
            size=cfg.get("resolution", 512),
            view_position=cfg.get("view_position"),
            max_per_label=cfg.get("max_per_label"),
            max_images=cfg.get("max_images"),
            cache_dir=cfg.get("cache_dir"),
            seed=cfg.get("seed", 42),
        )
        t0 = time.time()
        batch = collate_multilabel([ds[i] for i in range(min(4, len(ds)))])
        dt = time.time() - t0
        ok(f"đọc 4 ảnh mất {dt:.2f}s ({dt / 4 * 1000:.0f} ms/ảnh)")
        if dt / 4 > 0.25 and not cfg.get("cache_dir"):
            warn("đọc ảnh chậm — bật 'cache_dir' trong config để cache ảnh đã resize")
        ok(f"pixel_values {tuple(batch['pixel_values'].shape)} "
           f"[{batch['pixel_values'].min():.2f}, {batch['pixel_values'].max():.2f}] | "
           f"labels {tuple(batch['labels'].shape)}")

        # ------------------------------------------------------------------
        step(5, "Nạp U-Net + VAE + LoRA + LabelEncoder")
        from models.label_encoder import LabelEncoderConfig, MultiHotLabelEncoder
        from models.lora import inject_lora, lora_parameters, num_trainable_parameters
        from models.loading import load_scheduler_config, load_unet, load_vae
        from dataset.nih_multilabel import LABEL_NAMES
        from pipeline.inference import NoiseScheduler

        t0 = time.time()
        unet = load_unet(str(base), variant=variant)
        vae = load_vae(str(base), variant=variant)
        sched = NoiseScheduler.from_diffusers_config(load_scheduler_config(str(base)))
        ok(f"nạp xong trong {time.time() - t0:.1f}s | cross_attention_dim="
           f"{unet.config.cross_attention_dim} | schedule={sched.beta_schedule}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp_dtype = {"no": torch.float32, "fp16": torch.float16,
                     "bf16": torch.bfloat16}[mp]
        use_amp = mp != "no" and device == "cuda"

        # Cast TRƯỚC khi inject: adapter luôn tạo ở fp32, nếu cast sau thì
        # .to(dtype=fp16) hạ luôn lora_a/lora_b và GradScaler.unscale_ sẽ ném
        # "Attempting to unscale FP16 gradients".
        vae.to(device)                                   # VAE giữ fp32
        unet.to(device, dtype=amp_dtype if use_amp else None)

        targets = (["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
                   if cfg.get("cross_attention_only") else
                   cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
        unet.requires_grad_(False)
        inj = inject_lora(unet, targets, rank=cfg.get("rank", 64),
                          alpha=cfg.get("lora_alpha", 64))
        if not inj:
            raise Fail(f"inject_lora không khớp module nào với {targets}")
        ok(f"{len(inj)} adapter LoRA | {num_trainable_parameters(inj):,} tham số")

        enc = MultiHotLabelEncoder(LabelEncoderConfig(
            num_labels=len(LABEL_NAMES), embed_dim=unet.config.cross_attention_dim,
            tokens_per_label=cfg.get("tokens_per_label", 2),
            num_layers=cfg.get("label_encoder_layers", 2),
            num_heads=cfg.get("label_encoder_heads", 8),
            label_names=tuple(LABEL_NAMES)))
        ok(f"LabelEncoder {enc.num_tokens} token | "
           f"{sum(p.numel() for p in enc.parameters()):,} tham số")

        # ------------------------------------------------------------------
        if args.skip_train_step:
            print("\nBỏ qua bước 6.")
        else:
            step(6, "Thử 1 optimizer step thật (đo VRAM)")
            import torch.nn.functional as F
            from pipeline.inference import VAE_SCALING_FACTOR, min_snr_weights

            enc.to(device, dtype=torch.float32)

            params = lora_parameters(inj) + list(enc.parameters())
            bad = {p.dtype for p in params} - {torch.float32}
            if bad:
                raise Fail(f"tham số train phải ở fp32, đang có {bad}. "
                           "Kiểm tra thứ tự: phải .to(dtype) TRƯỚC inject_lora.")
            optim = torch.optim.AdamW(params, lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            bs = cfg.get("train_batch_size", 4)
            batch = collate_multilabel([ds[i] for i in range(min(bs, len(ds)))])
            x = batch["pixel_values"].to(device, dtype=torch.float32)
            y = batch["labels"].to(device, dtype=torch.float32)

            t0 = time.time()
            with torch.no_grad():
                lat = vae.encode(x).latent_dist.sample() * VAE_SCALING_FACTOR
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.num_train_timesteps, (lat.shape[0],), device=device)
            noisy = sched.add_noise(lat, noise, t)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                ctx = enc(y, drop_prob=0.1)
                pred = unet(noisy, t, encoder_hidden_states=ctx.to(noisy.dtype))
                per = F.mse_loss(pred.float(), noise.float(), reduction="none").mean((1, 2, 3))
                loss = (per * min_snr_weights(sched.compute_snr(t), 5.0)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            n_grad = sum(1 for p in params if p.grad is not None)
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            if n_grad == 0:
                raise Fail("không có gradient nào — kiểm tra target_modules")
            ok(f"1 step (batch {lat.shape[0]}) chạy {dt:.2f}s, loss={loss.item():.4f}")
            if device == "cuda":
                peak = torch.cuda.max_memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                ok(f"VRAM đỉnh {peak:.1f} / {total:.1f} GB")
                if peak > total * 0.85:
                    warn("sát trần VRAM — giảm train_batch_size hoặc resolution")
                ga = cfg.get("gradient_accumulation_steps", 1)
                steps = cfg.get("max_train_steps", 20000)
                eta = dt * ga * steps / 3600
                print(f"\n  ⏱ Ước tính: {steps:,} step × {ga} micro-batch ≈ {eta:.1f} giờ "
                      f"(chưa tính validation)")

    except Fail as e:
        print(f"\n✗ {e}")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n✗ Lỗi không lường trước: {type(e).__name__}: {e}")
        return 1

    print("\n" + "=" * 70)
    print("TẤT CẢ ĐỀU SẴN SÀNG →  python train_multilabel.py --config " + str(cfg_path))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
