from __future__ import annotations

"""
Sinh ảnh X-quang từ vector nhãn multi-hot.

    python generate_multilabel.py --labels "Effusion|Atelectasis" -n 4
    python generate_multilabel.py --labels "No Finding" --guidance 3.0 --steps 25
    python generate_multilabel.py --vector 0,0,1,0.5,0        # nhãn mềm: Effusion rõ, Others nửa vời
"""

import argparse
import math
import time
from pathlib import Path

import torch

from dataset.nih_multilabel import LABEL_NAMES
from models.label_encoder import labels_to_multihot, load_label_encoder
from models.lora import inject_lora, load_lora_config, load_lora_weights_into
from models.loading import load_scheduler_config, load_unet, load_vae
from pipeline.inference import NoiseScheduler
from pipeline.label_inference import LabelSDComponents, sample_from_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="./sd15", help="thư mục SD 1.5")
    p.add_argument("--lora", default="./out/multilabel/lora-final.safetensors")
    p.add_argument("--label-encoder", default="./out/multilabel/label_encoder-final.safetensors")
    p.add_argument("--lora-config", default="./out/multilabel/lora_config.json")
    p.add_argument("--vae-decoder", default=None, help="checkpoint decoder VAE đã tinh chỉnh")
    p.add_argument("--labels", default="No Finding",
                   help=f"tên nhãn ngăn bằng '|'. Hợp lệ: {list(LABEL_NAMES)}")
    p.add_argument("--vector", default=None, help="thay --labels bằng 5 số, vd 0,1,1,0,0")
    p.add_argument("-n", "--num-images", type=int, default=4,
                   help="TỔNG số ảnh cần sinh")
    p.add_argument("--batch-size", type=int, default=8,
                   help="số ảnh mỗi lượt qua U-Net. CFG nhân đôi con số này. "
                        "Giảm nếu OOM, tăng nếu còn dư VRAM")
    p.add_argument("--grid", action="store_true",
                   help="ghép tất cả ảnh thành một tấm contact sheet để xem nhanh")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="./out/samples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" and args.device == "cuda" else torch.float32

    unet = load_unet(args.base, variant="fp16" if dtype == torch.float16 else None)
    vae = load_vae(args.base, variant="fp16" if dtype == torch.float16 else None)
    scheduler = NoiseScheduler.from_diffusers_config(load_scheduler_config(args.base))

    if args.vae_decoder:
        from safetensors.torch import load_file
        vae.load_state_dict(load_file(args.vae_decoder), strict=False)
        print(f"[vae] decoder tinh chỉnh <- {args.vae_decoder}")

    cfg_path = Path(args.lora_config)
    if cfg_path.is_file():
        lcfg = load_lora_config(str(cfg_path))
        targets, rank, alpha = list(lcfg.target_modules), lcfg.rank, lcfg.alpha
    else:
        targets = ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]
        rank = alpha = 128
        print(f"[warn] không thấy {cfg_path} — dùng mặc định rank={rank}, targets={targets}")

    injected = inject_lora(unet, target_modules=targets, rank=rank, alpha=alpha)
    load_lora_weights_into(injected, args.lora)
    label_encoder = load_label_encoder(args.label_encoder, device=args.device)

    for m in (unet, vae):
        m.to(args.device, dtype=dtype).eval()
    label_encoder.to(args.device, dtype=dtype).eval()

    if args.vector:
        vals = [float(v) for v in args.vector.replace(" ", "").split(",")]
        y = torch.tensor(vals, dtype=torch.float32)
        tag = "vec" + "-".join(f"{v:g}" for v in vals)
    else:
        y = labels_to_multihot(args.labels, LABEL_NAMES)
        tag = args.labels.replace("|", "+").replace(" ", "")
    print(f"[cond] {dict(zip(LABEL_NAMES, y.tolist()))}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # MỘT generator dùng chung cho mọi lô -> mỗi lô lấy nhiễu mới, không lặp lại
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    components = LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder)

    bs = max(1, min(args.batch_size, args.num_images))
    n_batches = math.ceil(args.num_images / bs)
    print(f"[gen] {args.num_images} ảnh | {n_batches} lô x {bs} "
          f"| {args.steps} bước | cfg {args.guidance}")

    all_images, done, t_start = [], 0, time.perf_counter()
    while done < args.num_images:
        k = min(bs, args.num_images - done)
        t0 = time.perf_counter()
        imgs = sample_from_labels(
            components,
            y.unsqueeze(0).expand(k, -1),
            num_images=k,
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator, device=args.device, dtype=dtype, scheduler=scheduler,
        )
        for j, img in enumerate(imgs):
            img.save(outdir / f"{tag}_seed{args.seed}_{done + j:03d}.png")
        all_images.extend(imgs)
        done += k
        dt = time.perf_counter() - t0
        eta = (time.perf_counter() - t_start) / done * (args.num_images - done)
        print(f"[gen] {done}/{args.num_images}  ({dt:.1f}s cho lô này"
              f"{f', còn ~{eta:.0f}s' if done < args.num_images else ''})")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[done] {args.num_images} ảnh trong {time.perf_counter()-t_start:.1f}s -> {outdir}")

    if args.grid:
        path = outdir / f"{tag}_seed{args.seed}_grid.png"
        save_contact_sheet(all_images, path)
        print(f"[grid] {path}")


def save_contact_sheet(images, path, thumb: int = 256, cols: int = None) -> None:
    """Ghép ảnh thành một tấm lưới để duyệt nhanh 50 ảnh cùng lúc."""
    from PIL import Image

    n = len(images)
    cols = cols or min(10, math.ceil(math.sqrt(n)) + 2)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (16, 16, 16))
    for i, im in enumerate(images):
        sheet.paste(im.resize((thumb, thumb), Image.LANCZOS),
                    ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(path)


if __name__ == "__main__":
    main()
