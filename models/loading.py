from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .text_encoder import CLIPTextModel
from .tokenizer import CLIPTokenizer
from .unet import UNet2DConditionModel
from .vae import AutoencoderKL


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_folder(pretrained_path: str, subfolder: str) -> str:
    folder = os.path.join(pretrained_path, subfolder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Không thấy thư mục '{subfolder}' trong {pretrained_path}.")
    return folder


def _load_state_dict(folder: str, variant: Optional[str] = None) -> Dict[str, torch.Tensor]:
    suffix = f".{variant}" if variant else ""
    candidates = [
        f"diffusion_pytorch_model{suffix}.safetensors",
        f"model{suffix}.safetensors",
        f"diffusion_pytorch_model{suffix}.bin",
        f"pytorch_model{suffix}.bin",
    ]
    index_candidates = [
        f"diffusion_pytorch_model{suffix}.safetensors.index.json",
        f"model{suffix}.safetensors.index.json",
    ]

    for name in index_candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            from safetensors.torch import load_file
            index = _read_json(path)["weight_map"]
            state: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(index.values())):
                state.update(load_file(os.path.join(folder, shard)))
            return state

    for name in candidates:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"Không thấy file trọng số trong {folder}")


def _assert_supported_unet(cfg: Dict[str, Any]) -> None:
    problems = []
    if cfg.get("class_embed_type") is not None:
        problems.append(f"class_embed_type={cfg['class_embed_type']}")
    if cfg.get("addition_embed_type") is not None:
        problems.append(f"addition_embed_type={cfg['addition_embed_type']} (SDXL)")
    if cfg.get("dual_cross_attention", False):
        problems.append("dual_cross_attention=True")
    if cfg.get("resnet_time_scale_shift", "default") != "default":
        problems.append(f"resnet_time_scale_shift={cfg['resnet_time_scale_shift']}")
    if cfg.get("mid_block_type", "UNetMidBlock2DCrossAttn") != "UNetMidBlock2DCrossAttn":
        problems.append(f"mid_block_type={cfg['mid_block_type']}")
    if cfg.get("time_embedding_type", "positional") != "positional":
        problems.append(f"time_embedding_type={cfg['time_embedding_type']}")
    for t in cfg.get("down_block_types", []):
        if t not in ("CrossAttnDownBlock2D", "DownBlock2D"):
            problems.append(f"down_block_type={t}")
    for t in cfg.get("up_block_types", []):
        if t not in ("CrossAttnUpBlock2D", "UpBlock2D"):
            problems.append(f"up_block_type={t}")
    if problems:
        raise NotImplementedError("U-Net checkpoint dùng tính năng chưa hỗ trợ:\n  - " + "\n  - ".join(problems))


def _assert_supported_vae(cfg: Dict[str, Any]) -> None:
    for t in cfg.get("down_block_types", []):
        if t != "DownEncoderBlock2D":
            raise NotImplementedError(f"VAE down_block_type={t} chưa hỗ trợ")
    for t in cfg.get("up_block_types", []):
        if t != "UpDecoderBlock2D":
            raise NotImplementedError(f"VAE up_block_type={t} chưa hỗ trợ")


def load_unet(pretrained_path: str, subfolder: str = "unet",
              variant: Optional[str] = None,
              torch_dtype: Optional[torch.dtype] = None) -> UNet2DConditionModel:
    folder = _resolve_folder(pretrained_path, subfolder)
    cfg = _read_json(os.path.join(folder, "config.json"))
    _assert_supported_unet(cfg)

    model = UNet2DConditionModel(
        sample_size=cfg.get("sample_size", 64),
        in_channels=cfg.get("in_channels", 4),
        out_channels=cfg.get("out_channels", 4),
        layers_per_block=cfg.get("layers_per_block", 2),
        block_out_channels=tuple(cfg.get("block_out_channels", (320, 640, 1280, 1280))),
        down_block_types=tuple(cfg["down_block_types"]),
        up_block_types=tuple(cfg["up_block_types"]),
        cross_attention_dim=cfg.get("cross_attention_dim", 768),
        attention_head_dim=cfg.get("attention_head_dim", 8),
        num_attention_heads=cfg.get("num_attention_heads", None),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        norm_eps=cfg.get("norm_eps", 1e-5),
        downsample_padding=cfg.get("downsample_padding", 1),
        transformer_layers_per_block=cfg.get("transformer_layers_per_block", 1),
        use_linear_projection=cfg.get("use_linear_projection", False),
    )
    state = _load_state_dict(folder, variant)
    _strict_load(model, state, "U-Net")
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model


def load_vae(pretrained_path: str, subfolder: str = "vae",
             variant: Optional[str] = None,
             torch_dtype: Optional[torch.dtype] = None) -> AutoencoderKL:
    folder = _resolve_folder(pretrained_path, subfolder)
    cfg = _read_json(os.path.join(folder, "config.json"))
    _assert_supported_vae(cfg)

    model = AutoencoderKL(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        block_out_channels=tuple(cfg.get("block_out_channels", (128, 256, 512, 512))),
        layers_per_block=cfg.get("layers_per_block", 2),
        latent_channels=cfg.get("latent_channels", 4),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        sample_size=cfg.get("sample_size", 512),
        scaling_factor=cfg.get("scaling_factor", 0.18215),
    )
    state = _load_state_dict(folder, variant)
    _strict_load(model, state, "VAE")
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model


def load_text_encoder(pretrained_path: str, subfolder: str = "text_encoder",
                      variant: Optional[str] = None,
                      torch_dtype: Optional[torch.dtype] = None) -> CLIPTextModel:
    folder = _resolve_folder(pretrained_path, subfolder)
    cfg = _read_json(os.path.join(folder, "config.json"))
    if "text_config" in cfg:
        cfg = {**cfg, **cfg["text_config"]}

    # XỬ LÝ LỖI MAPPING Ở ĐÂY: Quét cả 2 key "hidden_act" và "hidden_activation"
    hidden_act = cfg.get("hidden_act", cfg.get("hidden_activation", "quick_gelu"))

    model = CLIPTextModel(
        vocab_size=cfg.get("vocab_size", 49408),
        hidden_size=cfg.get("hidden_size", 768),
        num_hidden_layers=cfg.get("num_hidden_layers", 12),
        num_attention_heads=cfg.get("num_attention_heads", 12),
        intermediate_size=cfg.get("intermediate_size", 3072),
        max_position_embeddings=cfg.get("max_position_embeddings", 77),
        hidden_act=hidden_act,
        layer_norm_eps=cfg.get("layer_norm_eps", 1e-5),
    )
    state = _load_state_dict(folder, variant)
    state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
    if not any(k.startswith("text_model.") for k in state):
        state = {f"text_model.{k}": v for k, v in state.items()}
    state.pop("text_projection.weight", None)
    state.pop("logit_scale", None)

    _strict_load(model, state, "text encoder")
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    return model


def load_tokenizer(pretrained_path: str, subfolder: str = "tokenizer") -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(pretrained_path, subfolder=subfolder)


def load_scheduler_config(pretrained_path: str, subfolder: str = "scheduler") -> Dict[str, Any]:
    folder = _resolve_folder(pretrained_path, subfolder)
    return _read_json(os.path.join(folder, "scheduler_config.json"))


_LEGACY_ATTENTION_KEYS = {
    ".query.": ".to_q.",
    ".key.": ".to_k.",
    ".value.": ".to_v.",
    ".proj_attn.": ".to_out.0.",
}


def _convert_legacy_attention_keys(
    state: Dict[str, torch.Tensor], verbose: bool = True, what: str = ""
) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    n_renamed = 0
    n_squeezed = 0
    linear_suffixes = (".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight")

    for key, tensor in state.items():
        new_key = key
        if "attentions." in key:
            for old, new in _LEGACY_ATTENTION_KEYS.items():
                if old in new_key:
                    new_key = new_key.replace(old, new)
                    n_renamed += 1
                    break
            if new_key.endswith(linear_suffixes) and tensor.dim() == 4 and tensor.shape[-2:] == (1, 1):
                tensor = tensor.squeeze(-1).squeeze(-1)
                n_squeezed += 1
        converted[new_key] = tensor

    if verbose and (n_renamed or n_squeezed):
        detail = f"{n_renamed} key attention đổi tên"
        if n_squeezed:
            detail += f", {n_squeezed} tensor conv1x1 ép về linear"
        print(f"[load] {what}: checkpoint định dạng cũ — {detail}")
    return converted


def _strict_load(model: torch.nn.Module, state: Dict[str, torch.Tensor], what: str) -> None:
    state = _convert_legacy_attention_keys(state, what=what)
    missing, unexpected = model.load_state_dict(state, strict=False)
    shape_errors = []
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) != tuple(v.shape):
            shape_errors.append(f"{k}: checkpoint {tuple(v.shape)} != model {tuple(model_state[k].shape)}")

    if missing or unexpected or shape_errors:
        raise RuntimeError(
            f"Nạp {what} không khớp — kiến trúc tự viết và checkpoint lệch nhau.\n"
            f"  thiếu ({len(missing)}): {list(missing)[:5]}\n"
            f"  thừa  ({len(unexpected)}): {list(unexpected)[:5]}\n"
            f"  lệch shape ({len(shape_errors)}): {shape_errors[:5]}\n"
        )


def load_sd_components(
    pretrained_path: str,
    variant: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> Tuple[CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel, Dict[str, Any]]:
    if not os.path.isdir(pretrained_path):
        raise FileNotFoundError(f"'{pretrained_path}' không phải thư mục.")

    tokenizer = load_tokenizer(pretrained_path)
    text_encoder = load_text_encoder(pretrained_path, variant=variant, torch_dtype=torch_dtype)
    vae = load_vae(pretrained_path, variant=variant, torch_dtype=torch_dtype)
    unet = load_unet(pretrained_path, variant=variant, torch_dtype=torch_dtype)
    scheduler_config = load_scheduler_config(pretrained_path)

    if verbose:
        n = lambda m: sum(p.numel() for p in m.parameters())
        print(f"[load] tokenizer vocab={len(tokenizer)} | text_encoder {n(text_encoder)/1e6:.1f}M "
              f"| vae {n(vae)/1e6:.1f}M | unet {n(unet)/1e6:.1f}M")
    return tokenizer, text_encoder, vae, unet, scheduler_config