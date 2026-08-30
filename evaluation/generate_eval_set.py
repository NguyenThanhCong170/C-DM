from __future__ import annotations

"""
Sinh tập ảnh synthetic để train classifier TSTR và chấm CAS.

    python -m evaluation.generate_eval_set --config config/evaluation.yaml
    python -m evaluation.generate_eval_set --config config/evaluation.yaml --resume

Phân phối nhãn được LẤY MẪU LẠI từ chính tập train THẬT (kể cả tần suất đồng
mắc), nên TSTR và TRTR train trên cùng một phân phối nhãn — khác biệt duy nhất
là nguồn pixel. Nếu sinh nhãn cân bằng đều thì hiệu số TSTR-TRTR sẽ lẫn cả tác
động của việc đổi phân phối, không còn đo được chất lượng ảnh.

Script có RESUME (bỏ qua ảnh đã tồn tại) và TỰ LÙI BATCH khi OOM.
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

from dataset.nih_multilabel import LABEL_NAMES, NIHMultiLabelDataset
from evaluation.splits import patient_level_split_3way
from models.label_encoder import load_label_encoder
from models.loading import load_scheduler_config, load_unet, load_vae
from models.lora import inject_lora, load_lora_config, load_lora_weights_into
from pipeline.inference import NoiseScheduler
from pipeline.label_inference import LabelSDComponents, sample_from_labels


def parse_args():
    p = argparse.ArgumentParser(description="Sinh tập ảnh synthetic cho TSTR/CAS")
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--resume", action="store_true",
                   help="bỏ qua những ảnh đã có trên đĩa và sinh tiếp")
    p.add_argument("-n", "--num-images", type=int, default=None,
                   help="ghi đè generation.num_images (tiện chạy thử 32 ảnh)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="ghi đè generation.batch_size")
    p.add_argument("--no-autobatch", action="store_true",
                   help="OOM thì báo lỗi luôn thay vì tự giảm batch")
    return p.parse_args()


# ---------------------------------------------------------------------------

def build_label_plan(cfg: dict, num_images: int) -> np.ndarray:
    """
    Lấy mẫu num_images vector nhãn theo ĐÚNG phân phối thực nghiệm của tập
    train thật (bao gồm cả tổ hợp đồng mắc, ví dụ Effusion+Atelectasis).

    Cách làm: bốc ngẫu nhiên CÓ HOÀN LẠI từ chính ma trận nhãn của tập train.
    Đơn giản hơn và chính xác hơn việc mô hình hoá 5 xác suất biên rồi lấy mẫu
    độc lập — cách đó phá vỡ tương quan giữa các bệnh.
    """
    data_root = Path(cfg["data_root"])
    csv_path = Path(cfg["csv_path"]) if cfg.get("csv_path") else data_root / "Data_Entry_2017.csv"

    train_pids, _, _ = patient_level_split_3way(
        csv_path, test_ratio=cfg["split"]["test_ratio"],
        val_ratio=cfg["split"]["val_ratio"], seed=cfg["seed"])

    train_ds = NIHMultiLabelDataset(
        data_root, csv_path, size=cfg["img_size"], patient_ids=train_pids,
        max_per_label=cfg.get("real_train", {}).get("max_per_label"),
        cache_dir=None, seed=cfg["seed"], verbose=True)

    rng = np.random.default_rng(cfg["generation"]["seed"])
    idx = rng.integers(0, len(train_ds.labels), size=num_images)
    plan = train_ds.labels[idx].astype(np.float32)

    print(f"\n[plan] {num_images:,} vector nhãn lấy mẫu từ {len(train_ds.labels):,} ảnh train thật")
    for name, c in zip(LABEL_NAMES, plan.sum(axis=0).astype(int)):
        print(f"[plan]   {name:<13} {c:>7,}  ({100 * c / num_images:5.2f}%)")
    return plan


def save_plan(path: Path, plan: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + list(LABEL_NAMES))
        for i, row in enumerate(plan):
            w.writerow([i] + [f"{v:g}" for v in row])


def load_plan(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[n]) for n in LABEL_NAMES] for r in rows], dtype=np.float32)


def write_manifest(path: Path, outdir: Path, plan: np.ndarray) -> int:
    """Ghi manifest CHỈ gồm những ảnh thật sự tồn tại trên đĩa."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file"] + list(LABEL_NAMES))
        for i, row in enumerate(plan):
            fn = f"syn_{i:06d}.png"
            if (outdir / fn).is_file():
                w.writerow([fn] + [f"{v:g}" for v in row])
                n += 1
    return n


# ---------------------------------------------------------------------------

def load_components(gcfg: dict, device: str, dtype: torch.dtype):
    variant = "fp16" if dtype == torch.float16 else None
    unet = load_unet(gcfg["base"], variant=variant)
    vae = load_vae(gcfg["base"], variant=variant)
    scheduler = NoiseScheduler.from_diffusers_config(load_scheduler_config(gcfg["base"]))

    if gcfg.get("vae_decoder"):
        from safetensors.torch import load_file
        vae.load_state_dict(load_file(gcfg["vae_decoder"]), strict=False)
        print(f"[vae] decoder tinh chỉnh <- {gcfg['vae_decoder']}")

    cfg_path = Path(gcfg["lora_config"])
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Không thấy {cfg_path}. File này ghi rank/alpha/target_modules đã dùng "
            "lúc train — thiếu nó thì không dựng lại đúng kiến trúc LoRA được.")
    lcfg = load_lora_config(str(cfg_path))
    print(f"[lora] rank={lcfg.rank} alpha={lcfg.alpha} targets={list(lcfg.target_modules)}")

    injected = inject_lora(unet, target_modules=list(lcfg.target_modules),
                           rank=lcfg.rank, alpha=lcfg.alpha)
    load_lora_weights_into(injected, gcfg["lora"])
    label_encoder = load_label_encoder(gcfg["label_encoder"], device=device)

    for m in (unet, vae):
        m.to(device, dtype=dtype).eval()
    label_encoder.to(device, dtype=dtype).eval()
    return LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder), scheduler


def _is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and "out of memory" in str(err).lower())


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    gcfg = cfg["generation"]

    num_images = int(args.num_images or gcfg["num_images"])
    batch_size = int(args.batch_size or gcfg["batch_size"])
    device = cfg["device"]
    dtype = (torch.float16 if gcfg.get("dtype", "fp16") == "fp16"
             and str(device).startswith("cuda") else torch.float32)

    outdir = Path(gcfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    plan_path = outdir / "plan.csv"
    manifest_path = outdir / "manifest.csv"

    # --- kế hoạch nhãn (cố định theo seed -> resume an toàn) --------------
    if args.resume and plan_path.is_file():
        plan = load_plan(plan_path)
        print(f"[plan] nạp lại {len(plan):,} vector nhãn từ {plan_path}")
        if len(plan) < num_images:
            raise RuntimeError(
                f"plan.csv chỉ có {len(plan):,} dòng nhưng cần {num_images:,}. "
                "Xoá plan.csv để lập kế hoạch mới (sẽ sinh lại từ đầu).")
        plan = plan[:num_images]
    else:
        plan = build_label_plan(cfg, num_images)
        save_plan(plan_path, plan)

    todo: List[int] = [i for i in range(num_images)
                       if not (outdir / f"syn_{i:06d}.png").is_file()]
    already = num_images - len(todo)
    if already:
        print(f"[resume] {already:,} ảnh đã có, còn {len(todo):,} ảnh cần sinh")
    if not todo:
        n = write_manifest(manifest_path, outdir, plan)
        print(f"[done] đủ ảnh rồi. manifest: {manifest_path} ({n:,} dòng)")
        return

    components, scheduler = load_components(gcfg, device, dtype)

    steps = int(gcfg["steps"])
    guidance = float(gcfg["guidance_scale"])
    res = int(gcfg["resolution"])
    base_seed = int(gcfg["seed"])
    print(f"\n[gen] {len(todo):,} ảnh | batch khởi đầu {batch_size} | {steps} bước "
          f"| cfg {guidance} | {res}x{res} | {dtype}")

    done, t_start, cursor = 0, time.perf_counter(), 0
    while cursor < len(todo):
        chunk = todo[cursor:cursor + batch_size]
        idx_tensor = torch.from_numpy(plan[chunk])          # (k, 5)

        # Generator riêng theo chỉ số ảnh -> resume cho ra ĐÚNG ảnh như chạy liền mạch
        gen = torch.Generator(device=device).manual_seed(base_seed * 1_000_003 + chunk[0])

        try:
            t0 = time.perf_counter()
            images = sample_from_labels(
                components, idx_tensor, num_images=len(chunk),
                height=res, width=res, num_inference_steps=steps,
                guidance_scale=guidance, generator=gen, device=device,
                dtype=dtype, scheduler=scheduler,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if args.no_autobatch or not _is_oom(err) or batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"[oom] hết VRAM -> giảm batch xuống {batch_size} và thử lại")
            continue

        for j, img in enumerate(images):
            img.save(outdir / f"syn_{todo[cursor + j]:06d}.png")

        cursor += len(chunk)
        done += len(chunk)
        dt = time.perf_counter() - t0
        rate = done / (time.perf_counter() - t_start)
        eta = (len(todo) - done) / max(rate, 1e-9)
        print(f"[gen] {done:,}/{len(todo):,}  ({dt:.1f}s/lô, {rate:.2f} ảnh/s, "
              f"còn ~{eta/60:.0f} phút)")

        if (cursor // max(batch_size, 1)) % 20 == 0:
            write_manifest(manifest_path, outdir, plan)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    n = write_manifest(manifest_path, outdir, plan)
    print(f"\n[done] {done:,} ảnh trong {(time.perf_counter()-t_start)/60:.1f} phút")
    print(f"[done] manifest: {manifest_path} ({n:,} dòng)  |  ảnh: {outdir}")


if __name__ == "__main__":
    main()
