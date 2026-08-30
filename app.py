from __future__ import annotations

"""
Demo Gradio cho C-DM — sinh ảnh X-quang từ VECTOR NHÃN multi-hot (không dùng prompt).

    python app.py
    python app.py --lora out/multilabel_att1_att2/lora-9000.safetensors \
                  --label-encoder out/multilabel_att1_att2/label_encoder-9000.safetensors \
                  --lora-config out/multilabel_att1_att2/lora_config.json
    python app.py --vae-decoder out/vae_decoder/vae_decoder-6000.safetensors   # ảnh nét hơn

Khác bản cũ: không còn CLIP text encoder / prompt. Điều kiện đi vào U-Net qua
MultiHotLabelEncoder, nhánh uncond của CFG là null_tokens học được, nên LoRA và
label encoder PHẢI đi cùng nhau từ một lần train.
"""

import argparse
import gc
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import gradio as gr
import torch

from dataset.nih_multilabel import LABEL_NAMES
from models.label_encoder import load_label_encoder
from models.loading import load_scheduler_config, load_unet, load_vae
from models.lora import LoRAConfig, inject_lora, load_lora_config, load_lora_weights_into
from pipeline.inference import NoiseScheduler
from pipeline.label_inference import LabelSDComponents, sample_from_labels

MAX_SEED = 2**31 - 1
CSS = ".gallery img { background: #000; }"

# Tổ hợp nhãn hay dùng (khớp validation_labels trong config/multilabel.yaml)
PRESET_COMBOS: List[List[str]] = [
    ["No Finding"],
    ["Infiltration"],
    ["Effusion"],
    ["Atelectasis"],
    ["Others"],
    ["Effusion", "Atelectasis"],
    ["Infiltration", "Effusion"],
    ["Infiltration", "Effusion", "Atelectasis"],
]


# --------------------------------------------------------------------------
# Nạp mô hình
# --------------------------------------------------------------------------

class Demo:
    """Giữ model đã nạp sẵn trên GPU để mọi request dùng chung."""

    def __init__(self, pretrained: str, lora_path: str, lora_config_path: Optional[str],
                 label_encoder_path: str, device: str, dtype: torch.dtype,
                 variant: Optional[str] = None, vae_decoder_path: Optional[str] = None,
                 batch_size: int = 4):
        self.device = device
        self.dtype = dtype
        self.batch_size = max(1, batch_size)

        print(f"[app] nạp base model từ '{pretrained}' (device={device}, dtype={dtype}, variant={variant})")
        unet = load_unet(pretrained, variant=variant)
        vae = load_vae(pretrained, variant=variant)
        self.scheduler = NoiseScheduler.from_diffusers_config(load_scheduler_config(pretrained))

        if vae_decoder_path:
            from safetensors.torch import load_file
            vae.load_state_dict(load_file(vae_decoder_path), strict=False)
            print(f"[app] decoder VAE tinh chỉnh <- '{vae_decoder_path}'")

        cfg = _resolve_lora_config(lora_path, lora_config_path)
        _check_lora_compatible(lora_path, unet)
        print(f"[app] inject LoRA rank={cfg.rank} alpha={cfg.alpha} "
              f"| {len(cfg.target_modules)} module đích")
        self.injected: Dict[str, object] = inject_lora(
            unet, cfg.target_modules, cfg.rank, cfg.alpha, dropout=0.0
        )
        load_lora_weights_into(self.injected, lora_path)
        print(f"[app] đã nạp {len(self.injected)} adapter từ '{lora_path}'")

        label_encoder = load_label_encoder(label_encoder_path, device="cpu")
        _check_label_encoder_compatible(label_encoder, unet, label_encoder_path)
        # Tên nhãn lấy từ chính checkpoint -> UI luôn khớp model đang chạy
        self.labels: Sequence[str] = tuple(label_encoder.config.label_names)
        if tuple(self.labels) != tuple(LABEL_NAMES):
            print(f"[app] cảnh báo: nhãn trong checkpoint {list(self.labels)} khác "
                  f"dataset.LABEL_NAMES {list(LABEL_NAMES)} — dùng theo checkpoint.")
        print(f"[app] label encoder: {len(self.labels)} nhãn, "
              f"{label_encoder.num_tokens} token cross-attention")

        for m in (unet, vae):
            m.to(device, dtype=dtype).eval()
        label_encoder.to(device, dtype=dtype).eval()
        self.components = LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder)

    def set_lora_strength(self, strength: float) -> None:
        """scaling = (alpha / rank) * strength; strength=0 tương đương tắt LoRA."""
        for module in self.injected.values():
            module.scaling = (module.alpha / module.rank) * float(strength)


def _check_lora_compatible(lora_path: str, unet) -> None:
    """Bắt sớm trường hợp LoRA train trên base model khác (SD 1.x vs 2.x)."""
    from safetensors import safe_open

    model_dim = getattr(unet.config, "cross_attention_dim", None)
    if model_dim is None:
        return

    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
        # attn2.to_k / to_v nhận encoder_hidden_states -> in_features = cross_attention_dim
        key = next((k for k in keys
                    if k.endswith(".lora_a") and "attn2.to_k" in k), None)
        if key is None:
            key = next((k for k in keys
                        if k.endswith(".lora_a") and "attn2.to_v" in k), None)
        if key is None:
            return
        ckpt_dim = f.get_slice(key).get_shape()[1]

    if ckpt_dim != model_dim:
        raise RuntimeError(
            f"LoRA '{lora_path}' được train trên base model có cross_attention_dim="
            f"{ckpt_dim}, còn checkpoint đang nạp có cross_attention_dim={model_dim} "
            f"({'SD 1.x' if ckpt_dim == 768 else 'SD 2.x'} vs "
            f"{'SD 1.x' if model_dim == 768 else 'SD 2.x'}).\n"
            "  -> Trỏ --pretrained sang đúng base model đã dùng lúc train."
        )


def _check_label_encoder_compatible(label_encoder, unet, path: str) -> None:
    model_dim = getattr(unet.config, "cross_attention_dim", None)
    embed_dim = label_encoder.config.embed_dim
    if model_dim is not None and embed_dim != model_dim:
        raise RuntimeError(
            f"Label encoder '{path}' có embed_dim={embed_dim} nhưng U-Net cần "
            f"cross_attention_dim={model_dim}.\n"
            "  -> Label encoder và base model không cùng một lần train."
        )


def _resolve_lora_config(lora_path: str, lora_config_path: Optional[str] = None) -> LoRAConfig:
    """Ưu tiên lora_config.json do train_multilabel.py ghi ra; không có thì suy từ metadata."""
    if lora_config_path and os.path.isfile(lora_config_path):
        return load_lora_config(lora_config_path)

    from safetensors import safe_open

    print(f"[app] không thấy lora_config '{lora_config_path}' — suy cấu hình từ "
          f"metadata của '{lora_path}'")
    with safe_open(lora_path, framework="pt") as f:
        meta = f.metadata() or {}
        targets = sorted(k[: -len(".lora_a")] for k in f.keys() if k.endswith(".lora_a"))
        if not targets:
            raise ValueError(
                f"'{lora_path}' không chứa key LoRA nào (*.lora_a) — "
                "đây có phải file trọng số LoRA không?"
            )
        rank_from_shape = f.get_slice(targets[0] + ".lora_a").get_shape()[0]

    rank = int(meta.get("rank", rank_from_shape))
    alpha = float(meta.get("alpha", rank))
    return LoRAConfig(rank=rank, alpha=alpha, target_modules=targets, dropout=0.0)


# --------------------------------------------------------------------------
# Hàm sinh ảnh cho UI
# --------------------------------------------------------------------------

def build_ui(demo_state: Demo) -> gr.Blocks:
    labels = list(demo_state.labels)
    n_labels = len(labels)

    def _build_vector(selected: List[str]) -> torch.Tensor:
        vec = torch.zeros(n_labels, dtype=torch.float32)
        for name in selected or []:
            vec[labels.index(name)] = 1.0

        if float(vec.sum()) == 0.0:
            raise gr.Error("Chưa chọn nhãn nào — chọn ít nhất 1 nhãn.")
        if vec[0] > 0 and float(vec[1:].sum()) > 0:
            gr.Warning(
                f"'{labels[0]}' đi kèm nhãn bệnh là tổ hợp không có trong dữ liệu train — "
                "ảnh sinh ra có thể không ổn định."
            )
        return vec

    def generate(selected, num_images, steps, guidance, height, width,
                 seed, randomize_seed, lora_strength, progress=gr.Progress()):
        y = _build_vector(selected)

        if randomize_seed or seed is None or int(seed) < 0:
            seed = random.randint(0, MAX_SEED)
        seed = int(seed)

        demo_state.set_lora_strength(lora_strength)
        generator = torch.Generator(device=demo_state.device).manual_seed(seed)

        num_images = int(num_images)
        bs = max(1, min(demo_state.batch_size, num_images))
        images: List = []
        t0 = time.perf_counter()
        try:
            while len(images) < num_images:
                k = min(bs, num_images - len(images))
                progress(len(images) / num_images, desc=f"Đang sinh ảnh {len(images)+1}/{num_images}…")
                images.extend(sample_from_labels(
                    demo_state.components,
                    y.unsqueeze(0).expand(k, -1),
                    num_images=k,
                    height=int(height), width=int(width),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    generator=generator,
                    device=demo_state.device,
                    dtype=demo_state.dtype,
                    scheduler=demo_state.scheduler,
                ))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise gr.Error(
                "Hết VRAM — giảm 'Số ảnh', giảm kích thước ảnh, hoặc chạy lại app "
                "với --batch-size nhỏ hơn."
            ) from exc
        finally:
            gc.collect()
            if demo_state.device.startswith("cuda"):
                torch.cuda.empty_cache()

        vec_str = " + ".join(n for n, v in zip(labels, y.tolist()) if v > 0)
        info = (f"**{vec_str}** · `[{', '.join(f'{v:g}' for v in y.tolist())}]`  \n"
                f"seed={seed} · steps={int(steps)} · cfg={guidance} · {int(width)}×{int(height)} "
                f"· lora={lora_strength:g} · {time.perf_counter()-t0:.1f}s")
        return images, seed, info

    with gr.Blocks(title="C-DM · Chest X-ray Diffusion Demo") as ui:
        gr.Markdown(
            "# C-DM — Chest X-ray Diffusion (SD 1.5 thuần PyTorch + LoRA)\n"
            "Sinh ảnh X-quang ngực từ **vector nhãn multi-hot** "
            f"(`{', '.join(labels)}`) bằng DPM-Solver++ 2M."
        )

        with gr.Row():
            with gr.Column(scale=3):
                selected = gr.CheckboxGroup(
                    choices=labels, value=[labels[0]], label="Nhãn điều kiện",
                )
                gr.Examples(
                    examples=[[c] for c in PRESET_COMBOS if all(x in labels for x in c)],
                    inputs=[selected], label="Tổ hợp thường dùng",
                )
                run_btn = gr.Button("Sinh ảnh", variant="primary")

                with gr.Accordion("Tham số nâng cao", open=False):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=25, step=1, label="Số bước khử nhiễu")
                        guidance = gr.Slider(
                            1.0, 10.0, value=4.0, step=0.1,
                            label="Guidance scale (CFG) — nhãn thường hợp 3–5",
                        )
                    with gr.Row():
                        height = gr.Slider(256, 768, value=512, step=64, label="Chiều cao")
                        width = gr.Slider(256, 768, value=512, step=64, label="Chiều rộng")
                    num_images = gr.Slider(1, 25, value=1, step=1, label="Số ảnh")
                    lora_strength = gr.Slider(
                        0.0, 1.5, value=1.0, step=0.05,
                        label="Cường độ LoRA (0 = base model, ảnh sẽ không theo nhãn)",
                    )
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        randomize_seed = gr.Checkbox(value=True, label="Seed ngẫu nhiên")

            with gr.Column(scale=4):
                gallery = gr.Gallery(
                    label="Kết quả", columns=2, height=560,
                    object_fit="contain", elem_classes=["gallery"], format="png",
                )
                info = gr.Markdown()

        run_btn.click(
            fn=generate,
            inputs=[selected, num_images, steps, guidance, height, width,
                    seed, randomize_seed, lora_strength],
            outputs=[gallery, seed, info],
            concurrency_limit=1,   # 1 job/GPU để tránh OOM
        )

    return ui


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.abspath(__file__))
    d = lambda *p: os.path.join(root, *p)

    p = argparse.ArgumentParser(description="Gradio demo cho C-DM (điều kiện multi-hot)")
    p.add_argument("--pretrained", default=d("sd15"),
                   help="Thư mục base model SD 1.5 theo cấu trúc HF (cross_attention_dim=768)")
    p.add_argument("--lora", default=d("out", "multilabel_att1_att2", "lora-9000.safetensors"),
                   help="File trọng số LoRA (.safetensors)")
    p.add_argument("--label-encoder", dest="label_encoder",
                   default=d("out", "multilabel_att1_att2", "label_encoder-9000.safetensors"),
                   help="Checkpoint MultiHotLabelEncoder — phải cùng lần train với --lora")
    p.add_argument("--lora-config", dest="lora_config",
                   default=d("out", "multilabel_att1_att2", "lora_config.json"),
                   help="lora_config.json do train_multilabel.py ghi ra; không có thì "
                        "đọc metadata trong file .safetensors")
    p.add_argument("--vae-decoder", dest="vae_decoder", default=None,
                   help="Checkpoint decoder VAE đã tinh chỉnh (train_vae_decoder.py)")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=4,
                   help="Số ảnh mỗi lượt qua U-Net. CFG nhân đôi con số này — giảm nếu OOM")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--variant", default="fp16",
                   help="Hậu tố file trọng số: 'fp16' cho *.fp16.safetensors, "
                        "'' cho file không hậu tố")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Tạo link public *.gradio.live")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.float16 if (args.dtype == "fp16" and args.device.startswith("cuda")) else torch.float32
    if args.dtype == "fp16" and not args.device.startswith("cuda"):
        print("[app] CPU không chạy tốt fp16 — chuyển sang fp32.")

    for path, what, hint in (
        (args.pretrained, "base model", "tải SD 1.5 về thư mục ./sd15"),
        (args.lora, "checkpoint LoRA", "train_multilabel.py --config config/multilabel.yaml"),
        (args.label_encoder, "checkpoint label encoder",
         "cùng lần train với LoRA, ví dụ label_encoder-final.safetensors"),
    ):
        if not os.path.exists(path):
            raise SystemExit(f"[app] không thấy {what}: '{path}'\n  -> {hint}")

    state = Demo(
        pretrained=args.pretrained,
        lora_path=args.lora,
        lora_config_path=args.lora_config,
        label_encoder_path=args.label_encoder,
        device=args.device,
        dtype=dtype,
        variant=args.variant or None,
        vae_decoder_path=args.vae_decoder,
        batch_size=args.batch_size,
    )
    ui = build_ui(state)
    ui.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
