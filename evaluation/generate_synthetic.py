"""
Sinh ảnh X-quang synthetic từ vector nhãn multi-hot 5 chiều
(No Finding, Infiltration, Effusion, Atelectasis, Others).

DÙNG CHUNG cho mọi metric cần ảnh synthetic — không gắn với riêng CAS. Ghi lại
ground-truth ĐẦY ĐỦ cả 5 chiều nhãn vào metadata.csv; mỗi metric sau này tự đọc
đúng cột nó cần (xem evaluation/cas/compute_cas.py) thay vì sinh ảnh lại riêng.

Mỗi combo có thể đặt số ảnh riêng qua "count" (ưu tiên hơn "images_per_combo"
chung) — CAS cần đều nhau giữa các combo nên chỉ dùng "images_per_combo", còn
TSTR (evaluation/trtr_tstr/) cần phân phối nhãn lệch (giống phân phối thật) nên
đặt "count" riêng cho từng combo, xem config/evaluation/tstr_generation.yaml.

Cách chạy (từ thư mục gốc repo):
    python -m evaluation.generate_synthetic --config config/evaluation/cas.yaml
"""
import argparse
import csv
import os

import torch
import yaml
from tqdm import tqdm

from evaluation.common.model_loader import load_generation_components
from evaluation.common.versioning import resolve_version_tag
from models.label_encoder import DEFAULT_LABELS as LABEL_NAMES
from models.label_encoder import labels_to_multihot
from pipeline.label_inference import sample_from_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/evaluation/cas.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    device = m["device"]
    dtype = torch.float16 if (m["dtype"] == "fp16" and device.startswith("cuda")) else torch.float32
    if m["dtype"] == "fp16" and not device.startswith("cuda"):
        print("[eval] CPU không chạy tốt fp16 — chuyển sang fp32.")
        dtype = torch.float32

    version_tag = resolve_version_tag(cfg)
    print(f"[eval] version_tag = '{version_tag}'")

    print(f"[eval] Nạp model: base='{m['pretrained_dir']}', lora='{m['lora_path']}'")
    model = load_generation_components(
        base=m["pretrained_dir"],
        lora_path=m["lora_path"],
        lora_config_path=m.get("lora_config_path"),
        label_encoder_path=m["label_encoder_path"],
        device=device,
        dtype=dtype,
        variant=m.get("variant"),
    )

    gen = cfg["generation"]
    out_dir = os.path.join(gen["output_dir"], version_tag)
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, gen.get("metadata_filename", "metadata.csv"))
    print(f"[eval] ảnh + metadata sẽ nằm trong {out_dir}/")

    seed_base = gen["seed"]
    default_n_per_combo = gen.get("images_per_combo")

    rows = []
    img_idx = 0
    for combo in gen["combos"]:
        combo_name = combo["name"]
        label_vector = labels_to_multihot(combo["labels"], LABEL_NAMES)  # (5,)

        # "count" đặt riêng cho từng combo (vd TSTR cần phân phối nhãn lệch, không
        # đều nhau) -> ưu tiên hơn "images_per_combo" chung (vd CAS cần đều nhau).
        n_per_combo = combo.get("count", default_n_per_combo)
        if n_per_combo is None:
            raise ValueError(
                f"Combo '{combo_name}' không có 'count' riêng, và generation.images_per_combo "
                "cũng không đặt trong config -> không biết sinh bao nhiêu ảnh."
            )

        combo_dir = os.path.join(out_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)

        gt_dict = dict(zip(LABEL_NAMES, label_vector.tolist()))
        print(f"[eval] Combo '{combo_name}' | vector = {gt_dict}")

        for i in tqdm(range(n_per_combo)):
            seed = seed_base + img_idx
            generator = torch.Generator(device=device).manual_seed(seed)

            images = sample_from_labels(
                model.components,
                label_vector,
                num_images=1,
                height=gen.get("height", 512),
                width=gen.get("width", 512),
                num_inference_steps=gen.get("num_inference_steps", 25),
                guidance_scale=gen.get("guidance_scale", 4.0),
                generator=generator,
                device=device,
                dtype=dtype,
                scheduler=model.scheduler,
            )
            image = images[0]

            filename = f"{combo_name}_{i:04d}.png"
            filepath = os.path.join(combo_dir, filename)
            image.save(filepath)

            rows.append(
                {
                    "filepath": os.path.relpath(filepath, start=out_dir),
                    "combo": combo_name,
                    "seed": seed,
                    **{f"gt_{name}": v for name, v in gt_dict.items()},
                }
            )
            img_idx += 1

    fieldnames = ["filepath", "combo", "seed"] + [f"gt_{name}" for name in LABEL_NAMES]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[eval] Xong. Đã sinh {img_idx} ảnh. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
