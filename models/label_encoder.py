from __future__ import annotations

"""
Bộ mã hóa nhãn multi-hot thay cho CLIP text encoder.

Ý tưởng: U-Net của Stable Diffusion nhận điều kiện qua cross-attention với một
CHUỖI vector (B, L, cross_attention_dim). Ta không cần chuỗi đó phải đến từ text —
chỉ cần nó mang đúng thông tin nhãn. Module này biến vector multi-hot y ∈ {0,1}^K
thành chuỗi token có cùng chiều với CLIP, nên toàn bộ trọng số cross-attention đã
pretrain vẫn dùng lại được (chỉ tinh chỉnh qua LoRA).

Mỗi nhãn k được cấp 2 bộ token học được:
    present_emb[k]  — dùng khi y_k = 1
    absent_emb[k]   — dùng khi y_k = 0
Token thứ k = y_k · present + (1 − y_k) · absent  →  nhãn vắng mặt cũng là một tín
hiệu rõ ràng (khác hẳn việc cắt token đi), và công thức nội suy này cho phép truyền
nhãn mềm (y ∈ [0,1]) lúc suy diễn để trộn mức độ biểu hiện bệnh.

Thêm 1 token toàn cục (kiểu BOS) + positional embedding, rồi vài lớp Transformer
để các nhãn "nói chuyện" với nhau (đồng mắc Effusion+Atelectasis khác tổng hai bệnh
riêng lẻ).

Classifier-Free Guidance: có sẵn `null_tokens` học được, dùng cho nhánh uncond.
Lúc train, `forward(..., drop_prob=0.1)` sẽ thay ngẫu nhiên 10% mẫu bằng null.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

# Thứ tự nhãn mặc định (5 chiều). Chỉ số này phải khớp giữa dataset / train / suy diễn.
DEFAULT_LABELS: tuple = ("No Finding", "Infiltration", "Effusion", "Atelectasis", "Others")


@dataclass
class LabelEncoderConfig:
    num_labels: int = 5
    embed_dim: int = 768          # = unet.config.cross_attention_dim
    tokens_per_label: int = 2
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    label_names: tuple = DEFAULT_LABELS

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label_names"] = list(self.label_names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEncoderConfig":
        d = dict(d)
        d["label_names"] = tuple(d.get("label_names", DEFAULT_LABELS))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def num_tokens(self) -> int:
        return 1 + self.num_labels * self.tokens_per_label


class MultiHotLabelEncoder(nn.Module):
    """multi-hot (B, K) -> encoder_hidden_states (B, L, D) cho cross-attention."""

    def __init__(self, config: Optional[LabelEncoderConfig] = None, **kwargs):
        super().__init__()
        self.config = config or LabelEncoderConfig(**kwargs)
        cfg = self.config
        K, D, T = cfg.num_labels, cfg.embed_dim, cfg.tokens_per_label

        # Token cho từng nhãn ở hai trạng thái có/không
        self.present_emb = nn.Parameter(torch.randn(K, T, D) * 0.02)
        self.absent_emb = nn.Parameter(torch.randn(K, T, D) * 0.02)
        # Token toàn cục — chỗ để model gom ngữ cảnh chung "đây là ảnh X-quang ngực"
        self.global_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        # Vị trí: chuỗi ngắn và có thứ tự cố định nên dùng positional học được
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.num_tokens, D) * 0.02)
        # Chuỗi rỗng cho nhánh uncond của CFG
        self.null_tokens = nn.Parameter(torch.randn(1, cfg.num_tokens, D) * 0.02)

        if cfg.num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=D,
                nhead=cfg.num_heads,
                dim_feedforward=int(D * cfg.mlp_ratio),
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=cfg.num_layers, enable_nested_tensor=False
            )
        else:
            self.transformer = None

        self.final_norm = nn.LayerNorm(D)
        # Scale khởi tạo gần với biên độ last_hidden_state của CLIP (~1) để U-Net
        # không bị sốc phân phối ở bước đầu.
        self.out_scale = nn.Parameter(torch.ones(1))

    # ------------------------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return self.config.num_tokens

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    def _build_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        """labels (B, K) float trong [0,1] -> (B, L, D) trước transformer."""
        B, K = labels.shape
        if K != self.config.num_labels:
            raise ValueError(f"labels có {K} chiều, encoder cấu hình cho {self.config.num_labels}")

        y = labels.to(self.present_emb.dtype).view(B, K, 1, 1)      # (B,K,1,1)
        tok = y * self.present_emb + (1.0 - y) * self.absent_emb     # (B,K,T,D)
        tok = tok.reshape(B, K * self.config.tokens_per_label, -1)   # (B,K*T,D)

        g = self.global_token.expand(B, -1, -1)                      # (B,1,D)
        return torch.cat([g, tok], dim=1) + self.pos_emb             # (B,L,D)

    def forward(
        self,
        labels: torch.Tensor,
        drop_prob: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        labels: (B, K) multi-hot (hoặc nhãn mềm).
        drop_prob: xác suất thay cả mẫu bằng null_tokens (huấn luyện CFG).
        """
        h = self._build_tokens(labels)
        if self.transformer is not None:
            h = self.transformer(h)
        h = self.final_norm(h) * self.out_scale

        if drop_prob > 0.0:
            B = h.shape[0]
            rand = torch.rand(B, 1, 1, device=h.device, generator=generator)
            null = self.null_embedding(B).to(h.dtype)
            h = torch.where(rand < drop_prob, null, h)
        return h

    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """Chuỗi uncond cho CFG — đi qua đúng final_norm để cùng thang với nhánh cond."""
        null = self.null_tokens.expand(batch_size, -1, -1)
        return self.final_norm(null) * self.out_scale

    # ------------------------------------------------------------------
    def encode_for_cfg(self, labels: torch.Tensor, do_cfg: bool = True) -> torch.Tensor:
        """Trả về [uncond; cond] ghép theo batch, đúng thứ tự pipeline sample() dùng."""
        cond = self.forward(labels, drop_prob=0.0)
        if not do_cfg:
            return cond
        uncond = self.null_embedding(cond.shape[0]).to(cond.dtype)
        return torch.cat([uncond, cond], dim=0)


# ----------------------------------------------------------------------
# Lưu / nạp
# ----------------------------------------------------------------------

def save_label_encoder(encoder: MultiHotLabelEncoder, path: str,
                       extra_metadata: Optional[dict] = None) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    state = {k: v.detach().cpu().contiguous().float() for k, v in encoder.state_dict().items()}
    meta = {"label_encoder_config": json.dumps(encoder.config.to_dict())}
    for k, v in (extra_metadata or {}).items():
        meta[str(k)] = str(v)
    save_file(state, path, metadata=meta)


def load_label_encoder(path: str, device: Union[str, torch.device] = "cpu",
                       config: Optional[LabelEncoderConfig] = None) -> MultiHotLabelEncoder:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}

    if config is None:
        raw = meta.get("label_encoder_config")
        if raw is None:
            raise ValueError(
                f"{path} không có metadata 'label_encoder_config' — truyền config thủ công.")
        config = LabelEncoderConfig.from_dict(json.loads(raw))

    encoder = MultiHotLabelEncoder(config)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict lệch: thiếu={missing}, thừa={unexpected}")
    return encoder.to(device)


def load_label_encoder_into(encoder: MultiHotLabelEncoder, path: str) -> None:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    encoder.load_state_dict(state, strict=True)


# ----------------------------------------------------------------------
# Tiện ích: tên nhãn -> vector multi-hot
# ----------------------------------------------------------------------

def labels_to_multihot(
    names: Union[str, Sequence[str]],
    label_names: Sequence[str] = DEFAULT_LABELS,
) -> torch.Tensor:
    """
    >>> labels_to_multihot(["Effusion", "Atelectasis"])
    tensor([0., 0., 1., 1., 0.])
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split("|") if n.strip()]
    lookup = {n.lower(): i for i, n in enumerate(label_names)}
    vec = torch.zeros(len(label_names), dtype=torch.float32)
    for n in names:
        key = n.strip().lower()
        if key in ("normal", "no_finding", "nofinding"):
            key = "no finding"
        if key not in lookup:
            raise KeyError(f"Nhãn '{n}' không thuộc {list(label_names)}")
        vec[lookup[key]] = 1.0
    return vec


def batch_multihot(
    combos: Sequence[Union[str, Sequence[str]]],
    label_names: Sequence[str] = DEFAULT_LABELS,
) -> torch.Tensor:
    return torch.stack([labels_to_multihot(c, label_names) for c in combos])
