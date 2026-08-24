"""
Chạy:
    python app.py                                   # dùng ./sd15 + checkpoint-4000.safetensors
    python app.py --lora ""                         # chỉ base model, không LoRA

"""

import argparse
import gc
import json
import os
import random
from typing import Dict, List, Optional

import gradio as gr
import torch

from models import (
    LoRAConfig,
    inject_lora,
    load_lora_config,
    load_lora_weights_into,
    load_sd_components,
)
from pipeline.inference import NoiseScheduler, SDComponents, sample

MAX_SEED = 2**31 - 1
CSS = ".gallery img { background: #000; }"

# Prompt mẫu dựng từ concepts_list lúc train (token hiếm "sks").
DEFAULT_CONCEPTS = [
    "a chest x-ray of sks atelectasis",
    "a chest x-ray of sks cardiomegaly",
    "a chest x-ray of sks effusion",
    "a chest x-ray of sks infiltration",
    "a chest x-ray of sks pneumonia",
    "a chest x-ray of normal lungs",
]


# --------------------------------------------------------------------------
# Nạp mô hình
# --------------------------------------------------------------------------

class Demo:
    """Giữ model đã nạp sẵn trên GPU để mọi request dùng chung."""

    def __init__(self, pretrained: str, lora_path: Optional[str], lora_config_path: Optional[str],
                 device: str, dtype: torch.dtype, variant: Optional[str] = None):
        self.device = device
        self.dtype = dtype

        print(f"[app] nạp base model từ '{pretrained}' (device={device}, dtype={dtype}, variant={variant})")
        tok, te, vae, unet, sched_cfg = load_sd_components(
            pretrained, variant=variant, torch_dtype=dtype
        )
        for m in (te, vae, unet):
            m.to(device)
            m.eval()

        self.components = SDComponents(unet=unet, vae=vae, text_encoder=te, tokenizer=tok)
        self.scheduler = NoiseScheduler.from_diffusers_config(sched_cfg)

        self.injected: Dict[str, object] = {}
        self.lora_path = None
        if lora_path:
            self._load_lora(lora_path, lora_config_path)

    def _load_lora(self, lora_path: str, lora_config_path: Optional[str]) -> None:
        cfg = _resolve_lora_config(lora_path, lora_config_path)
        _check_lora_compatible(lora_path, self.components.unet)
        print(f"[app] inject LoRA rank={cfg.rank} alpha={cfg.alpha} "
              f"| {len(cfg.target_modules)} module đích")
        self.injected = inject_lora(
            self.components.unet, cfg.target_modules, cfg.rank, cfg.alpha, dropout=0.0
        )
        load_lora_weights_into(self.injected, lora_path)
        self.lora_path = lora_path
        print(f"[app] đã nạp {len(self.injected)} adapter từ '{lora_path}'")

    def set_lora_strength(self, strength: float) -> None:
        """scaling = (alpha / rank) * strength; strength=0 tương đương tắt LoRA."""
        for module in self.injected.values():
            module.scaling = (module.alpha / module.rank) * float(strength)

    @property
    def has_lora(self) -> bool:
        return bool(self.injected)


def _check_lora_compatible(lora_path: str, unet) -> None:
    from safetensors import safe_open

    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_a"
    with safe_open(lora_path, framework="pt") as f:
        if key not in f.keys():
            return
        ckpt_dim = f.get_slice(key).get_shape()[1]

    model_dim = getattr(unet.config, "cross_attention_dim", None)
    if model_dim is not None and ckpt_dim != model_dim:
        raise RuntimeError(
            f"LoRA '{lora_path}' được train trên base model có cross_attention_dim="
            f"{ckpt_dim}, còn checkpoint đang nạp có cross_attention_dim={model_dim} "
            f"({'SD 1.x' if ckpt_dim == 768 else 'SD 2.x'} vs "
            f"{'SD 1.x' if model_dim == 768 else 'SD 2.x'}).\n"
            "  → Trỏ --pretrained sang đúng base model đã dùng lúc train,\n"
            "  → hoặc chạy base model không LoRA:  python app.py --lora ''"
        )


def _resolve_lora_config(lora_path: str, lora_config_path: Optional[str] = None) -> LoRAConfig:
    if lora_config_path and os.path.isfile(lora_config_path):
        return load_lora_config(lora_config_path)

    from safetensors import safe_open

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


def load_prompt_presets(path: str) -> List[str]:
    if not os.path.isfile(path):
        return DEFAULT_CONCEPTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            concepts = json.load(f)
        prompts, seen = [], set()
        for c in concepts:
            for key in ("instance_prompt", "class_prompt"):
                p = c.get(key)
                if p and p not in seen:
                    seen.add(p)
                    prompts.append(p)
        return prompts or DEFAULT_CONCEPTS
    except (json.JSONDecodeError, AttributeError, TypeError):
        return DEFAULT_CONCEPTS


# --------------------------------------------------------------------------
# Hàm sinh ảnh cho UI
# --------------------------------------------------------------------------

def build_ui(demo_state: Demo, presets: List[str]) -> gr.Blocks:

    def generate(prompt, negative_prompt, num_images, steps, guidance,
                 height, width, seed, randomize_seed, lora_strength,
                 progress=gr.Progress()):
        if not prompt or not prompt.strip():
            raise gr.Error("Prompt không được để trống.")

        if randomize_seed or seed is None or int(seed) < 0:
            seed = random.randint(0, MAX_SEED)
        seed = int(seed)

        if demo_state.has_lora:
            demo_state.set_lora_strength(lora_strength)

        generator = torch.Generator(device=demo_state.device).manual_seed(seed)
        progress(0.1, desc="Đang sinh ảnh…")
        try:
            images = sample(
                components=demo_state.components,
                prompt=prompt.strip(),
                negative_prompt=(negative_prompt or "").strip(),
                num_images=int(num_images),
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
                device=demo_state.device,
                scheduler=demo_state.scheduler,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise gr.Error(
                "Hết VRAM — hãy giảm số ảnh hoặc kích thước ảnh."
            ) from exc
        finally:
            gc.collect()
            if demo_state.device.startswith("cuda"):
                torch.cuda.empty_cache()

        info = (f"seed={seed} · steps={steps} · cfg={guidance} · {width}×{height}"
                f" · lora={lora_strength if demo_state.has_lora else 'off'}")
        return images, seed, info

    with gr.Blocks(title="C-DM · Chest X-ray Diffusion Demo") as ui:
        gr.Markdown(
            "# C-DM — Chest X-ray Diffusion (SD 1.x thuần PyTorch + LoRA)\n"
            "Sinh ảnh X-quang ngực bằng DPM-Solver++ 2M. "
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=presets[0],
                    lines=2,
                    placeholder="a chest x-ray of sks pneumonia",
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                )
                gr.Examples(examples=[[p] for p in presets], inputs=[prompt], label="Concept đã train")
                run_btn = gr.Button("Sinh ảnh", variant="primary")

                with gr.Accordion("Tham số nâng cao", open=False):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=25, step=1, label="Số bước khử nhiễu")
                        guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance scale (CFG)")
                    with gr.Row():
                        height = gr.Slider(256, 768, value=512, step=64, label="Chiều cao")
                        width = gr.Slider(256, 768, value=512, step=64, label="Chiều rộng")
                    num_images = gr.Slider(1, 25, value=1, step=1, label="Số ảnh")
                    lora_strength = gr.Slider(
                        0.0, 1.5, value=1.0, step=0.05,
                        label="Cường độ LoRA (0 = chỉ base model)",
                        interactive=demo_state.has_lora,
                    )
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        randomize_seed = gr.Checkbox(value=True, label="Seed ngẫu nhiên")

            with gr.Column(scale=4):
                gallery = gr.Gallery(
                    label="Kết quả", columns=2, height=560,
                    object_fit="contain", elem_classes=["gallery"],format="png",
                )
                info = gr.Markdown()

        run_btn.click(
            fn=generate,
            inputs=[prompt, negative_prompt, num_images, steps, guidance,
                    height, width, seed, randomize_seed, lora_strength],
            outputs=[gallery, seed, info],
            concurrency_limit=1,   # 1 job/GPU để tránh OOM
        )

    return ui


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Gradio demo cho C-DM")
    p.add_argument("--pretrained", default=os.path.join(root, "sd15"),
                   help="Thư mục base model SD 1.x theo cấu trúc HF (cross_attention_dim=768)")
    p.add_argument("--lora", default=os.path.join(root, "checkpoint-4000.safetensors"),
                   help="File trọng số LoRA (.safetensors); '' để chạy base model")
    p.add_argument("--lora_config", default=None,
                   help="Ghi đè cấu hình LoRA bằng file json; mặc định đọc thẳng "
                        "metadata trong file .safetensors")
    p.add_argument("--concepts", default=os.path.join(root, "concepts.json"))
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

    lora_path = args.lora if args.lora and os.path.isfile(args.lora) else None
    if args.lora and lora_path is None:
        print(f"[app] cảnh báo: không thấy '{args.lora}', chạy base model.")

    state = Demo(args.pretrained, lora_path, args.lora_config, args.device, dtype, args.variant)
    ui = build_ui(state, load_prompt_presets(args.concepts))
    ui.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,        
    )


if __name__ == "__main__":
    main()
