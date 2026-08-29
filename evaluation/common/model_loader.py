import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from models.label_encoder import load_label_encoder
from models.loading import load_scheduler_config, load_unet, load_vae
from models.lora import DEFAULT_TARGET_MODULES, inject_lora, load_lora_config, load_lora_weights_into
from pipeline.inference import NoiseScheduler
from pipeline.label_inference import LabelSDComponents


@dataclass
class LoadedLabelModel:
    components: LabelSDComponents
    scheduler: NoiseScheduler
    injected: Dict[str, object]


def load_generation_components(
    base: str,
    lora_path: str,
    lora_config_path: Optional[str],
    label_encoder_path: str,
    device: str,
    dtype: torch.dtype,
    variant: Optional[str] = None,
) -> LoadedLabelModel:
    unet = load_unet(base, variant=variant, torch_dtype=dtype)
    vae = load_vae(base, variant=variant, torch_dtype=dtype)
    scheduler = NoiseScheduler.from_diffusers_config(load_scheduler_config(base))

    if lora_config_path and os.path.isfile(lora_config_path):
        lcfg = load_lora_config(lora_config_path)
        targets, rank, alpha, dropout = list(lcfg.target_modules), lcfg.rank, lcfg.alpha, lcfg.dropout
    else:
        targets, rank, alpha, dropout = list(DEFAULT_TARGET_MODULES), 128, 128.0, 0.0
        print(
            f"[eval][cảnh báo] không thấy lora_config '{lora_config_path}' — dùng mặc định "
            f"rank={rank} alpha={alpha} targets={targets}. Nếu checkpoint train với rank/target "
            "khác, load_lora_weights_into() bên dưới sẽ raise lỗi rõ ràng (key không khớp)."
        )

    injected = inject_lora(unet, target_modules=targets, rank=rank, alpha=alpha, dropout=dropout)
    load_lora_weights_into(injected, lora_path)
    print(f"[eval] đã nạp {len(injected)} adapter LoRA từ '{lora_path}'")

    label_encoder = load_label_encoder(label_encoder_path, device=device)

    for m in (unet, vae):
        m.to(device, dtype=dtype).eval()
    label_encoder.to(device, dtype=dtype).eval()

    components = LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder)
    return LoadedLabelModel(components=components, scheduler=scheduler, injected=injected)
