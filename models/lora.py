"""
LoRA (Low-Rank Adaptation) cho U-Net của Stable Diffusion — implementation thuần PyTorch.

Điểm khác so với bản trước:
  * KHÔNG có bias riêng trong adapter (bản cũ copy bias của base layer rồi cộng lần
    thứ hai vào output -> bias bị nhân đôi ngay từ step 0).
  * Adapter được tạo trực tiếp trên device của base layer -> inject trước hay sau
    `.to(device)` đều đúng.
  * Tham số LoRA luôn giữ float32 (chuẩn cho AMP), forward tự cast theo base output.
  * Hỗ trợ cả nn.Linear và nn.Conv2d.
  * API: inject_lora / lora_parameters / num_trainable_parameters /
    save_lora_weights / load_lora_weights_into / save_lora_config / merge_and_unload.

Paper: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "to_q",       # query projection
    "to_k",       # key projection
    "to_v",       # value projection
    "to_out.0",   # output projection của attention
)

# Gợi ý nếu muốn tăng capacity (áp cho cả conv của ResNet block):
#   --target_modules to_q to_k to_v to_out.0 proj_in proj_out conv1 conv2
CONV_TARGET_MODULES: Tuple[str, ...] = ("conv1", "conv2", "conv_shortcut")


@dataclass
class LoRAConfig:
    """Cấu hình LoRA. `scaling = alpha / rank`."""

    rank: int = 4
    alpha: float = 1.0
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "LoRAConfig":
        return cls(
            rank=int(d["rank"]),
            alpha=float(d["alpha"]),
            target_modules=tuple(d.get("target_modules", DEFAULT_TARGET_MODULES)),
            dropout=float(d.get("dropout", 0.0)),
        )


# --------------------------------------------------------------------------
# Adapter modules
# --------------------------------------------------------------------------


class _LoRABase(nn.Module):
    """Phần dùng chung: giữ base layer đóng băng + cờ merged."""

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank phải > 0, nhận {rank}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRALinear(_LoRABase):
    """nn.Linear đóng băng + nhánh low-rank: y = W x + b + (alpha/r) * B A x."""

    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        w = base_layer.weight
        # LoRA params luôn fp32, tạo sẵn trên device của base layer.
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, device=w.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base_layer.out_features, self.rank, device=w.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))  # B = 0 -> delta W = 0 lúc khởi tạo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
        h = self.lora_dropout(x).to(self.lora_a.dtype)
        lora_out = (h @ self.lora_a.T) @ self.lora_b.T
        return base_out + (lora_out * self.scaling).to(base_out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling


class LoRAConv2d(_LoRABase):
    """nn.Conv2d đóng băng + nhánh low-rank (conv k×k rank chiều -> conv 1×1)."""

    def __init__(self, base_layer: nn.Conv2d, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConv2d chưa hỗ trợ grouped convolution")
        w = base_layer.weight
        self.lora_a = nn.Parameter(
            torch.empty(self.rank, base_layer.in_channels, *base_layer.kernel_size,
                        device=w.device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base_layer.out_channels, self.rank, 1, 1, device=w.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
        h = self.lora_dropout(x).to(self.lora_a.dtype)
        h = F.conv2d(
            h, self.lora_a,
            stride=self.base_layer.stride,
            padding=self.base_layer.padding,
            dilation=self.base_layer.dilation,
        )
        h = F.conv2d(h, self.lora_b)
        return base_out + (h * self.scaling).to(base_out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        delta = self.lora_b.flatten(1) @ self.lora_a.flatten(1)  # (out, in*k*k)
        return delta.view_as(self.base_layer.weight) * self.scaling


LoRAModule = Union[LoRALinear, LoRAConv2d]


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------


def _matches(full_name: str, targets: Sequence[str]) -> bool:
    # So khớp theo hậu tố đường dẫn module để tránh bắt nhầm tên chứa substring.
    return any(full_name == t or full_name.endswith("." + t) for t in targets)


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    verbose: bool = False,
) -> Dict[str, LoRAModule]:
    """
    Đóng băng toàn bộ `model` rồi thay các layer khớp `target_modules` bằng adapter LoRA.

    Trả về dict {đường_dẫn_module: adapter} — dùng làm "handle" cho optimizer,
    save/load và merge. Adapter được tạo trên đúng device của layer gốc nên có thể
    gọi trước hoặc sau `model.to(device)` đều được.
    """
    target_modules = tuple(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    injected: Dict[str, LoRAModule] = {}

    def _recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (LoRALinear, LoRAConv2d)):
                continue  # đã inject rồi, không lồng thêm

            if _matches(full_name, target_modules):
                if isinstance(child, nn.Linear):
                    adapter: LoRAModule = LoRALinear(child, rank, alpha, dropout)
                elif isinstance(child, nn.Conv2d):
                    adapter = LoRAConv2d(child, rank, alpha, dropout)
                else:
                    _recurse(child, full_name)
                    continue
                setattr(module, name, adapter)
                injected[full_name] = adapter
                if verbose:
                    print(f"  ✓ {full_name} ({type(child).__name__})")
                continue

            _recurse(child, full_name)

    _recurse(model)

    if not injected:
        raise RuntimeError(
            f"Không inject được adapter nào với target_modules={target_modules}. "
            "Kiểm tra lại tên module (ví dụ: to_q to_k to_v to_out.0)."
        )

    for adapter in injected.values():
        adapter.lora_a.requires_grad_(True)
        adapter.lora_b.requires_grad_(True)
    return injected


def lora_parameters(injected: Dict[str, LoRAModule]) -> List[nn.Parameter]:
    """Danh sách param trainable, thứ tự ổn định (dùng cho optimizer và clip_grad)."""
    params: List[nn.Parameter] = []
    for _, adapter in sorted(injected.items()):
        params.append(adapter.lora_a)
        params.append(adapter.lora_b)
    return params


def num_trainable_parameters(injected: Dict[str, LoRAModule]) -> int:
    return sum(p.numel() for p in lora_parameters(injected))


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------


def save_lora_config(config: LoRAConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_lora_config(path: str) -> LoRAConfig:
    with open(path, "r", encoding="utf-8") as f:
        return LoRAConfig.from_dict(json.load(f))


def save_lora_weights(
    injected: Dict[str, LoRAModule],
    output_path: str,
    alpha: Optional[float] = None,
    rank: Optional[int] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Lưu lora_a/lora_b (fp32, trên CPU) ra safetensors kèm metadata rank/alpha."""
    state_dict = {}
    for name, adapter in injected.items():
        state_dict[f"{name}.lora_a"] = adapter.lora_a.detach().float().cpu().contiguous()
        state_dict[f"{name}.lora_b"] = adapter.lora_b.detach().float().cpu().contiguous()

    metadata = {"format": "cdm-lora-v1"}
    if rank is not None:
        metadata["rank"] = str(rank)
    if alpha is not None:
        metadata["alpha"] = str(alpha)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})

    save_file(state_dict, output_path, metadata=metadata)


def load_lora_weights_into(
    injected: Dict[str, LoRAModule],
    checkpoint_path: str,
    strict: bool = True,
) -> None:
    """Nạp checkpoint LoRA vào các adapter đã inject sẵn (khớp theo tên module)."""
    state_dict = load_file(checkpoint_path)

    expected = set()
    for name in injected:
        expected.add(f"{name}.lora_a")
        expected.add(f"{name}.lora_b")

    missing = expected - set(state_dict)
    unexpected = set(state_dict) - expected
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint không khớp cấu hình LoRA hiện tại.\n"
            f"  thiếu {len(missing)} key (ví dụ: {sorted(missing)[:3]})\n"
            f"  thừa  {len(unexpected)} key (ví dụ: {sorted(unexpected)[:3]})\n"
            f"  -> kiểm tra --rank / --target_modules có giống lúc train không."
        )

    with torch.no_grad():
        for name, adapter in injected.items():
            for suffix, param in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
                key = f"{name}.{suffix}"
                if key not in state_dict:
                    continue
                tensor = state_dict[key]
                if tensor.shape != param.shape:
                    raise RuntimeError(f"{key}: shape checkpoint {tuple(tensor.shape)} "
                                       f"!= shape model {tuple(param.shape)}")
                param.copy_(tensor.to(param.device, param.dtype))


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """
    Cộng delta LoRA vào weight gốc và gỡ adapter, trả model về kiến trúc chuẩn.

    Sau bước này state_dict lại đúng format gốc của diffusers (không còn tiền tố
    `.base_layer.`), nên `unet.save_pretrained(...)` dùng được bình thường.
    """
    def _recurse(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRALinear, LoRAConv2d)):
                base = child.base_layer
                delta = child.delta_weight().to(device=base.weight.device, dtype=base.weight.dtype)
                base.weight.data += delta
                setattr(module, name, base)
            else:
                _recurse(child)

    _recurse(model)
    return model


if __name__ == "__main__":
    print("Default target modules:", DEFAULT_TARGET_MODULES)