"""
Gradio demo cho nhánh multi-hot: chọn nhãn bằng checkbox thay vì gõ prompt.

    python app_multilabel.py
    python app_multilabel.py --lora ./out/multilabel/lora-final.safetensors \
                             --label_encoder ./out/multilabel/label_encoder-final.safetensors

`app.py` (bản prompt/DreamBooth) vẫn giữ nguyên, hai app chạy độc lập.
"""

import argparse
import gc
import os
from typing import Dict, List, Optional

import gradio as gr
import torch

from dataset.nih_multilabel import LABEL_NAMES
from models.label_encoder import load_label_encoder
from models.lora import LoRAConfig, inject_lora, load_lora_config, load_lora_weights_into
from models.loading import load_scheduler_config, load_unet, load_vae
from pipeline.inference import NoiseScheduler
from pipeline.label_inference import LabelSDComponents, sample_from_labels

MAX_SEED = 2**31 - 1
CSS = ".gallery img { background: #000; }"

# Tổ hợp hay dùng — bấm là điền sẵn checkbox
PRESETS: List[List[str]] = [
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

class Demo:
    """Model nạp một lần, dùng chung cho mọi request."""

    def __init__(self, pretrained: str, lora_path: Optional[str], lora_config_path: Optional[str],
                 label_encoder_path: str, vae_decoder_path: Optional[str],
                 device: str, dtype: torch.dtype, variant: Optional[str]):
        self.device = device
        self.dtype = dtype

        print(f"[app] nạp U-Net + VAE từ '{pretrained}' (device={device}, dtype={dtype})")
        unet = load_unet(pretrained, variant=variant)
        vae = load_vae(pretrained, variant=variant)
        self.scheduler = NoiseScheduler.from_diffusers_config(load_scheduler_config(pretrained))

        if vae_decoder_path:
            from safetensors.torch import load_file
            vae.load_state_dict(load_file(vae_decoder_path), strict=False)
            print(f"[app] decoder VAE tinh chỉnh <- {vae_decoder_path}")

        self.injected: Dict[str, object] = {}
        if lora_path:
            cfg = _resolve_lora_config(lora_path, lora_config_path)
            print(f"[app] inject LoRA rank={cfg.rank} alpha={cfg.alpha} "
                  f"| {len(cfg.target_modules)} module đích")
            self.injected = inject_lora(unet, cfg.target_modules, cfg.rank, cfg.alpha, dropout=0.0)
            load_lora_weights_into(self.injected, lora_path)
            print(f"[app] đã nạp {len(self.injected)} adapter từ '{lora_path}'")

        if not os.path.isfile(label_encoder_path):
            raise FileNotFoundError(
                f"Không thấy label encoder '{label_encoder_path}'. Nhánh multi-hot bắt buộc "
                f"phải có nó — U-Net không còn nhận điều kiện từ text.")
        label_encoder = load_label_encoder(label_encoder_path)
        print(f"[app] LabelEncoder: {label_encoder.num_labels} nhãn, "
              f"{label_encoder.num_tokens} token/mẫu")

        if label_encoder.config.embed_dim != unet.config.cross_attention_dim:
            raise RuntimeError(
                f"LabelEncoder có embed_dim={label_encoder.config.embed_dim} nhưng base model "
                f"có cross_attention_dim={unet.config.cross_attention_dim} — sai base model.")

        for m in (unet, vae, label_encoder):
            m.to(device, dtype=dtype).eval()

        self.label_names = list(label_encoder.config.label_names) or list(LABEL_NAMES)
        self.components = LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder)

    def set_lora_strength(self, strength: float) -> None:
        for module in self.injected.values():
            module.scaling = (module.alpha / module.rank) * float(strength)

    @property
    def has_lora(self) -> bool:
        return bool(self.injected)


def _resolve_lora_config(lora_path: str, lora_config_path: Optional[str] = None) -> LoRAConfig:
    if lora_config_path and os.path.isfile(lora_config_path):
        return load_lora_config(lora_config_path)
    from safetensors import safe_open
    with safe_open(lora_path, framework="pt") as f:
        meta = f.metadata() or {}
        targets = sorted(k[: -len(".lora_a")] for k in f.keys() if k.endswith(".lora_a"))
        if not targets:
            raise ValueError(f"'{lora_path}' không chứa key LoRA nào (*.lora_a).")
        rank_from_shape = f.get_slice(targets[0] + ".lora_a").get_shape()[0]
    rank = int(meta.get("rank", rank_from_shape))
    return LoRAConfig(rank=rank, alpha=float(meta.get("alpha", rank)),
                      target_modules=targets, dropout=0.0)


# --------------------------------------------------------------------------

def build_ui(state: Demo) -> gr.Blocks:
    names = state.label_names
    n = len(names)

    def _vector(checked, use_soft, *soft_vals) -> torch.Tensor:
        if use_soft:
            vec = [float(v) for v in soft_vals[:n]]
        else:
            vec = [1.0 if name in (checked or []) else 0.0 for name in names]
        if sum(vec) == 0:
            raise gr.Error("Chọn ít nhất một nhãn (hoặc kéo một slider lên > 0).")
        return torch.tensor(vec, dtype=torch.float32)

    def generate(checked, use_soft, num_images, steps, guidance, height, width,
                 seed, randomize_seed, lora_strength, *soft_vals):
        import random as _random
        y = _vector(checked, use_soft, *soft_vals)

        if randomize_seed or seed is None or int(seed) < 0:
            seed = _random.randint(0, MAX_SEED)
        seed = int(seed)

        if state.has_lora:
            state.set_lora_strength(lora_strength)

        generator = torch.Generator(device=state.device).manual_seed(seed)
        try:
            images = sample_from_labels(
                state.components,
                y.unsqueeze(0).expand(int(num_images), -1),
                num_images=int(num_images),
                height=int(height), width=int(width),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator, device=state.device,
                dtype=state.dtype, scheduler=state.scheduler,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise gr.Error("Hết VRAM — giảm số ảnh hoặc kích thước ảnh.") from exc
        finally:
            gc.collect()
            if state.device.startswith("cuda"):
                torch.cuda.empty_cache()

        vec_txt = ", ".join(f"{nm}={v:g}" for nm, v in zip(names, y.tolist()))
        info = (f"`[{vec_txt}]` · seed={seed} · steps={steps} · cfg={guidance} "
                f"· {width}×{height} · lora={lora_strength if state.has_lora else 'off'}")
        return images, seed, info

    with gr.Blocks(title="C-DM · Multi-hot Chest X-ray") as ui:
        gr.Markdown(
            "# C-DM — Sinh X-quang ngực theo nhãn multi-hot\n"
            "Điều kiện là vector nhãn, không phải prompt. Chọn nhiều nhãn cùng lúc "
            "để mô phỏng ca đồng mắc."
        )

        with gr.Row():
            with gr.Column(scale=3):
                checked = gr.CheckboxGroup(
                    choices=names, value=[names[0]], label="Nhãn",
                    info="'No Finding' nên đứng một mình; 4 nhãn còn lại kết hợp tự do.",
                )
                gr.Examples(examples=[[p] for p in PRESETS], inputs=[checked],
                            label="Tổ hợp mẫu")
                run_btn = gr.Button("Sinh ảnh", variant="primary")

                with gr.Accordion("Nhãn mềm (nội suy mức độ)", open=False):
                    use_soft = gr.Checkbox(
                        value=False, label="Dùng slider thay cho checkbox",
                        info="0 = chắc chắn không có, 1 = có rõ. Giá trị giữa cho biểu hiện nhẹ.",
                    )
                    soft = [gr.Slider(0.0, 1.0, value=1.0 if i == 0 else 0.0, step=0.05, label=nm)
                            for i, nm in enumerate(names)]

                with gr.Accordion("Tham số nâng cao", open=False):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=25, step=1, label="Số bước khử nhiễu")
                        guidance = gr.Slider(1.0, 15.0, value=4.0, step=0.1,
                                             label="Guidance scale (CFG)",
                                             info="Điều kiện nhãn nhẹ hơn text — 3–5 thường đủ.")
                    with gr.Row():
                        height = gr.Slider(256, 768, value=512, step=64, label="Chiều cao")
                        width = gr.Slider(256, 768, value=512, step=64, label="Chiều rộng")
                    num_images = gr.Slider(1, 16, value=2, step=1, label="Số ảnh")
                    lora_strength = gr.Slider(
                        0.0, 1.5, value=1.0, step=0.05,
                        label="Cường độ LoRA (0 = chỉ base model)",
                        interactive=state.has_lora)
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        randomize_seed = gr.Checkbox(value=True, label="Seed ngẫu nhiên")

            with gr.Column(scale=4):
                gallery = gr.Gallery(label="Kết quả", columns=2, height=560,
                                     object_fit="contain", elem_classes=["gallery"], format="png")
                info = gr.Markdown()

        gr.Markdown(
            "> Ảnh sinh ra là **dữ liệu tổng hợp**, không phải ảnh chụp thật và không "
            "dùng được cho mục đích chẩn đoán."
        )

        run_btn.click(
            fn=generate,
            inputs=[checked, use_soft, num_images, steps, guidance, height, width,
                    seed, randomize_seed, lora_strength] + soft,
            outputs=[gallery, seed, info],
            concurrency_limit=1,
        )

    return ui


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Gradio demo multi-hot cho C-DM")
    p.add_argument("--pretrained", default=os.path.join(root, "sd15"))
    p.add_argument("--lora", default=os.path.join(root, "out", "multilabel", "lora-final.safetensors"))
    p.add_argument("--lora_config", default=os.path.join(root, "out", "multilabel", "lora_config.json"))
    p.add_argument("--label_encoder",
                   default=os.path.join(root, "out", "multilabel", "label_encoder-final.safetensors"))
    p.add_argument("--vae_decoder", default=None,
                   help="checkpoint decoder VAE đã tinh chỉnh (tuỳ chọn)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--variant", default="fp16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if (args.dtype == "fp16" and args.device.startswith("cuda")) else torch.float32
    if args.dtype == "fp16" and not args.device.startswith("cuda"):
        print("[app] CPU không chạy tốt fp16 — chuyển sang fp32.")

    lora_path = args.lora if args.lora and os.path.isfile(args.lora) else None
    if args.lora and lora_path is None:
        print(f"[app] cảnh báo: không thấy '{args.lora}', chạy base model (ảnh sẽ chưa ra X-quang).")

    state = Demo(args.pretrained, lora_path, args.lora_config, args.label_encoder,
                 args.vae_decoder, args.device, dtype, args.variant or None)
    build_ui(state).queue(max_size=8).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        show_error=True, css=CSS,
    )


if __name__ == "__main__":
    main()
