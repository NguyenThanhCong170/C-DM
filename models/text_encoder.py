from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Hàm QuickGELU đặc trưng của OpenAI CLIP."""
    return x * torch.sigmoid(1.702 * x)

# Dictionary ánh xạ các hàm kích hoạt chuẩn
ACT2FN = {
    "quick_gelu": quick_gelu,
    "gelu": F.gelu,
    "relu": F.relu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh")
}


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Viết tay Attention (khớp bit 100% với Hugging Face)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().reshape(b, seq, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        # Nạp động theo cấu hình file config
        self.act = ACT2FN.get(hidden_act, F.gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, hidden_act: str, layer_norm_eps: float):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), mask)
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_act: str,
        layer_norm_eps: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, num_attention_heads, intermediate_size, hidden_act, layer_norm_eps)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        hidden_act: str,
        layer_norm_eps: float
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = CLIPEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, hidden_act, layer_norm_eps
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        bsz, seq_len = input_ids.shape
        
        # 1. Tạo Causal Mask cốt lõi (tam giác trên = -inf)
        causal_mask = torch.empty(bsz, 1, seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask.fill_(torch.finfo(hidden_states.dtype).min)
        causal_mask.triu_(1)
        
        # 2. Xử lý Padding Attention Mask được truyền từ Tokenizer
        if attention_mask is not None:
            # attention_mask: 1 là từ thật, 0 là pad.
            expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(hidden_states.dtype)
            inverted_mask = 1.0 - expanded_mask
            pad_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(hidden_states.dtype).min)
            causal_mask = causal_mask + pad_mask
            
        hidden_states = self.encoder(hidden_states, causal_mask)
        return self.final_layer_norm(hidden_states)


class CLIPTextOutput:
    """Wrapper đầu ra để khớp với API của Hugging Face Transformers."""
    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, idx: int) -> torch.Tensor:
        return (self.last_hidden_state,)[idx]


class CLIPTextModel(nn.Module):
    """Cấu trúc khớp 100% với file state_dict."""
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 77,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ):
        super().__init__()
        self.text_model = CLIPTextTransformer(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> CLIPTextOutput:
        last_hidden_state = self.text_model(input_ids, attention_mask)
        return CLIPTextOutput(last_hidden_state=last_hidden_state)