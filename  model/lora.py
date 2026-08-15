"""
Custom Low-Rank Adaptation (LoRA) implementation — pure PyTorch, no `peft`.

For a frozen linear layer W0 (d_out x d_in), LoRA adds a trainable low-rank
update:

    h = W0 x + (alpha / r) * B A x

    A in R^{r x d_in}   -- Kaiming-uniform initialized
    B in R^{d_out x r}  -- zero initialized

Zero-initializing B means the wrapped layer is numerically identical to the
original one at step 0 (LoRA starts as a mathematical no-op), which is the
standard trick from the LoRA paper (Hu et al., 2021) that lets training start
from the pretrained model's exact behavior.

This module also implements:
  - `inject_lora`: recursively finds target `nn.Linear` submodules (by dotted
    path suffix, e.g. "attn1.to_q", "attn2.to_out.0") inside an arbitrary
    module tree (typically a diffusers `UNet2DConditionModel`) and replaces
    them in-place with `LoRALinear` wrappers.
  - `save_lora_weights` / `load_lora_weights_into`: serialize/deserialize the
    A/B matrices to a single `.safetensors` file using the same dotted-key
    naming convention Diffusers/PEFT use for UNet LoRA adapters
    (`unet.<module.path>.lora_A.weight` / `...lora_B.weight`), so the file
    can also be loaded by `StableDiffusionPipeline.load_lora_weights(...)`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

# The four linear projections DreamBooth/LoRA fine-tuning conventionally
# targets inside a diffusers `Attention` module (self- and cross-attention
# share the same submodule names: `attn1` = self-attn, `attn2` = cross-attn).
DEFAULT_TARGET_MODULES: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")


@dataclass
class LoRAConfig:
    rank: int = 24
    alpha: float = 24.0
    target_modules: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_MODULES)

    def to_dict(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha, "target_modules": list(self.target_modules)}


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank update."""

    def __init__(self, base_layer: nn.Linear, rank: int = 24, alpha: float = 24.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear can only wrap nn.Linear, got {type(base_layer)}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Runtime multiplier on top of `scaling`, e.g. for inference-time
        # LoRA-strength sweeps (0.0 = base model only, 1.0 = full adapter).
        self.multiplier = 1.0

        # Base weights stay a plain nn.Linear (bias included) so forward
        # semantics and state_dict shape exactly match the original layer.
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B left at zero -> forward(x) == base_layer(x) at init.

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self._merged or self.multiplier == 0.0:
            return base_out
        # (x @ A^T) @ B^T avoids ever materializing the full d_out x d_in delta.
        lora_out = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + (self.scaling * self.multiplier) * lora_out.to(base_out.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold A/B into base_layer.weight in-place — for fast, LoRA-free inference."""
        if self._merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base_layer.weight.add_(delta.to(self.base_layer.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base_layer.weight.sub_(delta.to(self.base_layer.weight.dtype))
        self._merged = False

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.rank}, alpha={self.alpha}"


def _resolve_parent(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    """Walk `root` along `dotted_name` and return (parent_module, last_attr).

    Handles both attribute children (`attn1`) and indexed children inside an
    `nn.ModuleList` / `nn.Sequential` (`to_out.0`).
    """
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _get_submodule(parent: nn.Module, attr: str) -> nn.Module:
    return parent[int(attr)] if attr.isdigit() else getattr(parent, attr)


def _set_submodule(parent: nn.Module, attr: str, value: nn.Module) -> None:
    if attr.isdigit():
        parent[int(attr)] = value
    else:
        setattr(parent, attr, value)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    rank: int = 24,
    alpha: float = 24.0,
) -> Dict[str, LoRALinear]:
    """Recursively replace target `nn.Linear` submodules of `model` with `LoRALinear`.

    A module is targeted when its dotted path (as returned by
    `model.named_modules()`) equals, or ends with ".<t>", one of the entries
    in `target_modules`. This matches only linear layers by design (asserted
    below) — it will *not* accidentally match e.g. `to_out.1` (the Dropout
    that follows `to_out.0`).

    Returns a dict {dotted_name: LoRALinear} — use it to build the optimizer
    param group (`lora_parameters`) and for save/load.

    NOTE: does not touch `requires_grad` of anything outside the injected
    layers. Callers are still responsible for `model.requires_grad_(False)`
    (or equivalent) *before* calling this, so any parameters that are not
    part of a target Linear (e.g. GroupNorm, conv layers, timestep/embedding
    projections) stay frozen. `LoRALinear.__init__` freezes the wrapped
    `base_layer` itself either way.
    """
    matches: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == t or name.endswith("." + t) for t in target_modules):
            matches.append(name)

    if not matches:
        raise ValueError(
            f"inject_lora found no nn.Linear submodules matching {list(target_modules)}. "
            "Check the target module names against this model's architecture."
        )

    injected: Dict[str, LoRALinear] = {}
    for name in matches:
        parent, attr = _resolve_parent(model, name)
        base_linear = _get_submodule(parent, attr)
        lora_layer = LoRALinear(base_linear, rank=rank, alpha=alpha)
        _set_submodule(parent, attr, lora_layer)
        injected[name] = lora_layer

    return injected


def lora_parameters(injected: Dict[str, LoRALinear]) -> List[nn.Parameter]:
    """Flat list of trainable (lora_A, lora_B) parameters, for the optimizer."""
    params: List[nn.Parameter] = []
    for layer in injected.values():
        params.append(layer.lora_A)
        params.append(layer.lora_B)
    return params


def set_lora_scale(model_or_injected, scale: float) -> None:
    """Set the runtime multiplier on every LoRALinear (e.g. for a scale sweep
    at inference time: 0.0 = base model, 1.0 = full adapter)."""
    modules = (
        model_or_injected.values()
        if isinstance(model_or_injected, dict)
        else (m for m in model_or_injected.modules() if isinstance(m, LoRALinear))
    )
    for layer in modules:
        layer.multiplier = scale


def num_trainable_parameters(injected: Dict[str, LoRALinear]) -> int:
    return sum(p.numel() for p in lora_parameters(injected))


# --------------------------------------------------------------------------
# Serialization — Diffusers/PEFT-compatible key naming.
#
# Newer Diffusers releases (PEFT backend) save/load UNet LoRA adapters with
# keys shaped like:
#   unet.<block>....attn2.to_q.lora_A.weight
#   unet.<block>....attn2.to_q.lora_B.weight
# i.e. `{prefix}.{dotted_module_path}.lora_A.weight` / `lora_B.weight`.
# We reuse that convention here so a file saved by this module can be handed
# to `StableDiffusionPipeline.load_lora_weights(path)` directly. Diffusers
# key conventions have moved before across versions — if `load_lora_weights`
# ever rejects the file, load it with `load_lora_weights_into` from this
# module instead (it doesn't depend on Diffusers' internal parsing at all).
# --------------------------------------------------------------------------


def save_lora_weights(
    injected: Dict[str, LoRALinear],
    save_path: str,
    prefix: str = "unet",
    alpha: Optional[float] = None,
    rank: Optional[int] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Serialize every LoRALinear's A/B matrices to a single .safetensors file."""
    state_dict: Dict[str, torch.Tensor] = {}
    for name, layer in injected.items():
        state_dict[f"{prefix}.{name}.lora_A.weight"] = layer.lora_A.detach().cpu().contiguous()
        state_dict[f"{prefix}.{name}.lora_B.weight"] = layer.lora_B.detach().cpu().contiguous()

    metadata = {"format": "pt"}
    if alpha is not None:
        metadata["lora_alpha"] = str(alpha)
    if rank is not None:
        metadata["lora_rank"] = str(rank)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})

    save_file(state_dict, save_path, metadata=metadata)


def load_lora_weights_into(
    injected: Dict[str, LoRALinear],
    load_path: str,
    prefix: str = "unet",
    strict: bool = True,
    device: str = "cpu",
) -> None:
    """Load a .safetensors file written by `save_lora_weights` back into an
    already-injected model (i.e. call `inject_lora` with the same
    target_modules/rank first, then load weights into the result)."""
    state_dict = load_file(load_path, device=device)
    missing: List[str] = []
    for name, layer in injected.items():
        ka, kb = f"{prefix}.{name}.lora_A.weight", f"{prefix}.{name}.lora_B.weight"
        if ka not in state_dict or kb not in state_dict:
            missing.append(name)
            continue
        with torch.no_grad():
            layer.lora_A.copy_(state_dict[ka].to(layer.lora_A.device, layer.lora_A.dtype))
            layer.lora_B.copy_(state_dict[kb].to(layer.lora_B.device, layer.lora_B.dtype))
    if strict and missing:
        raise KeyError(f"load_lora_weights_into: missing weights for {len(missing)} layer(s): {missing[:5]}...")


def save_lora_config(config: LoRAConfig, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_lora_config(path: str) -> LoRAConfig:
    with open(path) as f:
        d = json.load(f)
    return LoRAConfig(rank=d["rank"], alpha=d["alpha"], target_modules=tuple(d["target_modules"]))