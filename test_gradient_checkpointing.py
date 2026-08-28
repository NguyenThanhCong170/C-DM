"""
Kiểm chứng gradient checkpointing của models/unet.py không làm sai kết quả.

    python test_gradient_checkpointing.py

Bật/tắt checkpointing phải cho ra output và gradient GIỐNG HỆT nhau — nếu lệch thì
mọi con số loss trong log đều không tin được.
"""

import torch

from models.lora import inject_lora, lora_parameters
from models.unet import UNet2DConditionModel


def build():
    torch.manual_seed(0)
    u = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4,
        block_out_channels=(32, 64), layers_per_block=2,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=64, attention_head_dim=4, norm_num_groups=32)
    u.requires_grad_(False)
    inj = inject_lora(u, ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
                      rank=4, alpha=4)
    return u, inj


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([10, 700])
    ctx = torch.randn(2, 11, 64)

    res = {}
    for use_ckpt in (False, True):
        unet, inj = build()
        if use_ckpt:
            unet.enable_gradient_checkpointing()
        unet.train()
        c = ctx.clone().detach().requires_grad_(True)
        out = unet(x, t, encoder_hidden_states=c).sample
        out.sum().backward()
        grads = torch.cat([p.grad.flatten() for p in lora_parameters(inj) if p.grad is not None])
        res[use_ckpt] = (out.detach().clone(), grads, c.grad.clone())

    o0, g0, c0 = res[False]
    o1, g1, c1 = res[True]

    assert torch.allclose(o0, o1, atol=1e-6), (o0 - o1).abs().max()
    print(f"  ✓ output khớp (sai lệch tối đa {(o0 - o1).abs().max():.2e})")

    assert g0.numel() == g1.numel() and g0.numel() > 0
    assert torch.allclose(g0, g1, atol=1e-5), (g0 - g1).abs().max()
    print(f"  ✓ gradient LoRA khớp ({g0.numel()} phần tử, "
          f"sai lệch {(g0 - g1).abs().max():.2e})")

    assert torch.allclose(c0, c1, atol=1e-5) and c1.abs().sum() > 0
    print(f"  ✓ gradient chảy về encoder_hidden_states (LabelEncoder) và khớp "
          f"(|grad| = {c1.abs().sum():.4f})")

    unet, _ = build()
    unet.enable_gradient_checkpointing()
    unet.eval()
    with torch.no_grad():
        unet(x, t, encoder_hidden_states=ctx)
    print("  ✓ chạy được dưới no_grad — suy diễn không bị ảnh hưởng")

    print("\nGRADIENT CHECKPOINTING PASS")


if __name__ == "__main__":
    main()
