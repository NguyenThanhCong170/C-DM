"""
Smoke test cho pipeline multi-hot (không cần GPU, không cần dữ liệu thật).

    python smoke_test_multilabel.py

Kiểm tra: ánh xạ nhãn NIH -> 5 chiều, LabelEncoder (shape/CFG/dropout/nhãn mềm),
save-load safetensors, gradient chảy đúng vào LoRA + LabelEncoder trong khi U-Net
gốc vẫn đóng băng, vòng lặp DPM-Solver++ với CFG null-label, Dataset đọc layout
Kaggle, và stage tinh chỉnh decoder VAE.
"""
import torch, numpy as np
import yaml, os
from models.vae import AutoencoderKL
from train_vae_decoder import VGGPerceptualLoss, random_crop, _save_decoder, DEFAULTS as VD
from train_multilabel import DEFAULTS as MD, diffusion_loss, build_lr_lambda
from pathlib import Path

torch.manual_seed(0)

from models.label_encoder import (MultiHotLabelEncoder, LabelEncoderConfig,
                                  labels_to_multihot, batch_multihot,
                                  save_label_encoder, load_label_encoder)
from models.lora import inject_lora, lora_parameters, num_trainable_parameters
from models.unet import UNet2DConditionModel
from dataset.nih_multilabel import (finding_string_to_multihot, LABEL_NAMES,
                                    collate_multilabel, NIHMultiLabelDataset)
from pipeline.inference import NoiseScheduler, min_snr_weights
from pipeline.label_inference import LabelSDComponents, sample_from_labels

ok = lambda m: print("  ✓", m)

# ---------- 1. ánh xạ nhãn
print("[1] label mapping")
assert (finding_string_to_multihot("No Finding") == np.array([1,0,0,0,0],'f')).all()
assert (finding_string_to_multihot("Effusion|Infiltration") == np.array([0,1,1,0,0],'f')).all()
assert (finding_string_to_multihot("Cardiomegaly|Nodule") == np.array([0,0,0,0,1],'f')).all()
assert (finding_string_to_multihot("Atelectasis|Pneumothorax") == np.array([0,0,0,1,1],'f')).all()
assert (labels_to_multihot("Effusion|Atelectasis") == torch.tensor([0.,0,1,1,0])).all()
assert (labels_to_multihot("normal") == torch.tensor([1.,0,0,0,0])).all()
ok("14 nhãn NIH -> 5 chiều, đồng mắc giữ đủ bit")

# ---------- 2. label encoder
print("[2] label encoder")
D = 320
enc = MultiHotLabelEncoder(LabelEncoderConfig(num_labels=5, embed_dim=D, tokens_per_label=2,
                                              num_layers=2, num_heads=8))
y = batch_multihot([["No Finding"], ["Effusion","Atelectasis"], ["Infiltration"]], LABEL_NAMES)
h = enc(y)
assert h.shape == (3, 11, D), h.shape
ok(f"multi-hot (3,5) -> context {tuple(h.shape)} (1 global + 5 nhãn × 2 token)")

# nhãn khác nhau -> context khác nhau; nhãn giống nhau -> giống nhau
h2 = enc(batch_multihot([["No Finding"],["No Finding"],["Effusion","Atelectasis"]], LABEL_NAMES))
assert torch.allclose(h[0], h2[0], atol=1e-6) and torch.allclose(h2[0], h2[1], atol=1e-6)
assert not torch.allclose(h[0], h[1], atol=1e-3)
ok("deterministic theo nhãn, và các nhãn khác nhau cho context khác nhau")

cfg_ctx = enc.encode_for_cfg(y, do_cfg=True)
assert cfg_ctx.shape == (6, 11, D)
assert torch.allclose(cfg_ctx[:3], enc.null_embedding(3), atol=1e-6)
assert torch.allclose(cfg_ctx[3:], h, atol=1e-6)
ok("encode_for_cfg -> [uncond(null); cond] đúng thứ tự pipeline")

torch.manual_seed(1)
hd = enc(y.repeat(200,1), drop_prob=0.5)
null = enc.null_embedding(1)[0]
frac = torch.isclose(hd, null.expand_as(hd), atol=1e-6).all(dim=(1,2)).float().mean().item()
assert 0.4 < frac < 0.6, frac
ok(f"cond dropout hoạt động (drop_prob=0.5 -> {frac:.0%} mẫu là null)")

soft = enc(torch.tensor([[0.,0,0.5,0,0]]))
assert soft.shape == (1,11,D) and torch.isfinite(soft).all()
ok("nhãn mềm y=0.5 chạy được (nội suy present/absent)")

# ---------- 3. save / load
print("[3] save & load")
save_label_encoder(enc, "/tmp/le.safetensors", extra_metadata={"step": 1})
enc2 = load_label_encoder("/tmp/le.safetensors")
assert torch.allclose(enc2(y), h, atol=1e-5)
ok("safetensors round-trip, config đọc lại từ metadata")

# ---------- 4. U-Net nhỏ + LoRA + gradient
print("[4] U-Net tí hon + LoRA + gradient")
unet = UNet2DConditionModel(
    sample_size=8, in_channels=4, out_channels=4,
    block_out_channels=(32, 64), layers_per_block=1,
    down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
    up_block_types=("UpBlock2D","CrossAttnUpBlock2D"),
    cross_attention_dim=D, attention_head_dim=4, norm_num_groups=32,
)
unet.requires_grad_(False)
inj = inject_lora(unet, ["attn2.to_q","attn2.to_k","attn2.to_v","attn2.to_out.0"], rank=8, alpha=8)
assert len(inj) > 0
ok(f"{len(inj)} adapter LoRA trên cross-attention ({num_trainable_parameters(inj):,} tham số)")

sched = NoiseScheduler()
lat = torch.randn(2,4,8,8)
t = torch.randint(0, 1000, (2,))
noise = torch.randn_like(lat)
noisy = sched.add_noise(lat, noise, t)
ctx = enc(batch_multihot([["Effusion"],["No Finding"]], LABEL_NAMES), drop_prob=0.1)
pred = unet(noisy, t, encoder_hidden_states=ctx).sample
assert pred.shape == lat.shape
w = min_snr_weights(sched.compute_snr(t), 5.0)
loss = (torch.nn.functional.mse_loss(pred, noise, reduction="none").mean((1,2,3)) * w).mean()
loss.backward()
lp = lora_parameters(inj); ep = list(enc.parameters())
g_lora = sum(1 for p in lp if p.grad is not None)
nz_lora = sum(1 for p in lp if p.grad is not None and p.grad.abs().sum() > 0)
g_enc  = sum(1 for p in ep if p.grad is not None and p.grad.abs().sum() > 0)
assert g_lora == len(lp), f"{g_lora}/{len(lp)}"
# lora_b khởi tạo = 0 nên grad của lora_a bằng 0 ở bước đầu — đúng thiết kế LoRA
assert nz_lora == len(lp)//2, f"{nz_lora}/{len(lp)}"
assert g_enc > 0, "label encoder không nhận gradient!"
ok(f"loss Min-SNR backward: gradient tới {g_lora}/{len(lp)} tensor LoRA "
   f"({nz_lora} khác 0 — lora_b=0 lúc init) và {g_enc}/{len(ep)} tensor LabelEncoder")

# base weights vẫn đóng băng
frozen = [n for n,p in unet.named_parameters() if p.requires_grad and "lora_" not in n]
assert not frozen, frozen[:3]
ok("trọng số gốc U-Net vẫn đóng băng")

# ---------- 5. sampler end-to-end (VAE giả)
print("[5] vòng lặp sinh ảnh")
class FakeVAE(torch.nn.Module):
    class Out:
        def __init__(s, x): s.sample = x
    def decode(self, z):
        return self.Out(torch.nn.functional.interpolate(z[:, :3], scale_factor=8))
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))
imgs = sample_from_labels(
    LabelSDComponents(unet=unet, vae=FakeVAE(), label_encoder=enc),
    labels_to_multihot("Effusion|Atelectasis"), num_images=2, height=64, width=64,
    num_inference_steps=4, guidance_scale=4.0, device="cpu", dtype=torch.float32,
    generator=torch.Generator().manual_seed(0), scheduler=sched)
assert len(imgs) == 2 and imgs[0].size == (64,64)
ok(f"DPM-Solver++ 4 bước + CFG(null-label) -> {len(imgs)} ảnh {imgs[0].size}")

# ---------- 6. dataset trên NIH giả
print("[6] dataset đọc layout Kaggle")
import csv, os
from PIL import Image
root = "/tmp/nih_fake"
os.makedirs(f"{root}/images_001/images", exist_ok=True)
os.makedirs(f"{root}/images_002/images", exist_ok=True)
rows = [("00000001_000.png","No Finding","1","PA","images_001"),
        ("00000002_000.png","Effusion|Atelectasis","2","PA","images_001"),
        ("00000003_000.png","Cardiomegaly","3","AP","images_002"),
        ("00000004_000.png","Infiltration","4","PA","images_002"),
        ("00000005_000.png","No Finding","5","PA","images_002")]
for name,_,_,_,folder in rows:
    Image.fromarray((np.random.rand(64,64)*65535).astype(np.uint16)).save(f"{root}/{folder}/images/{name}")
with open(f"{root}/Data_Entry_2017.csv","w",newline="",encoding="utf-8") as f:
    w_ = csv.writer(f); w_.writerow(["Image Index","Finding Labels","Patient ID","View Position"])
    for name,lab,pid,vp,_ in rows: w_.writerow([name,lab,pid,vp])

ds = NIHMultiLabelDataset(root, size=32, cache_dir="/tmp/nih_cache", verbose=False)
assert len(ds) == 5
b = collate_multilabel([ds[i] for i in range(len(ds))])
assert b["pixel_values"].shape == (5,3,32,32) and b["labels"].shape == (5,5)
assert b["pixel_values"].min() >= -1.001 and b["pixel_values"].max() <= 1.001
ok(f"5 ảnh, pixel_values {tuple(b['pixel_values'].shape)} trong [-1,1], labels {tuple(b['labels'].shape)}")
_ = ds[0]  # lần 2 đọc từ cache
assert os.path.exists("/tmp/nih_cache/00000001_000_32.png")
ok("cache ảnh resize hoạt động")

ds_pa = NIHMultiLabelDataset(root, size=32, view_position="PA", verbose=False)
assert len(ds_pa) == 4
ok("lọc View Position=PA -> 4/5 ảnh")

ds_cap = NIHMultiLabelDataset(root, size=32, max_per_label={"No Finding": 1}, verbose=False)
assert int(ds_cap.labels[:,0].sum()) == 1 and len(ds_cap) == 4
ok("max_per_label cắt đúng lớp áp đảo, giữ nguyên ảnh đa nhãn")

s = ds.make_balanced_sampler(beta=0.5)
assert len(list(iter(s))) == len(ds)
ok("WeightedRandomSampler dựng được")




print("[7] config YAML khớp DEFAULTS")
for path, defaults, reserved in [("config/multilabel.yaml", MD, {"output_dir","pretrained_model_name_or_path","device"}),
                                 ("config/vae_decoder.yaml", VD, {"output_dir","pretrained_model_name_or_path","device"})]:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    unknown = set(cfg) - set(defaults) - reserved
    assert not unknown, (path, unknown)
    assert "pretrained_model_name_or_path" in cfg and "output_dir" in cfg
    ok(f"{path}: {len(cfg)} khoá, không có khoá lạ")
cfg = yaml.safe_load(open("config/multilabel.yaml", encoding="utf-8"))
assert cfg["max_per_label"] == {"No Finding": 15000, "Others": 20000}, cfg["max_per_label"]
ok("YAML không nuốt 'No Finding' thành boolean")

print("[8] VAE decoder: đóng băng encoder, loss, checkpoint")
vae = AutoencoderKL(block_out_channels=(32,64), layers_per_block=1, norm_num_groups=32)
vae.requires_grad_(False)
trainable = list(vae.decoder.parameters()) + list(vae.post_quant_conv.parameters())
for p in trainable: p.requires_grad_(True)
enc_train = [n for n,p in vae.named_parameters() if p.requires_grad and n.startswith("encoder.")]
assert not enc_train and not any(p.requires_grad for p in vae.quant_conv.parameters())
ok(f"encoder + quant_conv đóng băng; {sum(p.numel() for p in trainable):,} tham số decoder được train")

x = torch.randn(2,3,64,64)
xc = random_crop(x, 32); assert xc.shape == (2,3,32,32)
ok("random_crop giữ đúng shape (2,3,32,32)")
with torch.no_grad():
    z = vae.encode(xc).latent_dist.mode()
xh = vae.decode(z).sample
assert xh.shape == xc.shape
loss = torch.nn.functional.l1_loss(xh, xc); loss.backward()
assert all(p.grad is not None for p in trainable)
assert all(p.grad is None for p in vae.encoder.parameters())
ok("L1 backward: gradient chỉ chảy vào decoder, encoder sạch")
os.makedirs("/tmp/vout", exist_ok=True)
_save_decoder(vae, Path("/tmp/vout"), 1)
from safetensors import safe_open
with safe_open("/tmp/vout/vae_decoder-1.safetensors","pt") as f:
    keys = list(f.keys())
assert keys and all(k.startswith(("decoder.","post_quant_conv.")) for k in keys)
vae2 = AutoencoderKL(block_out_channels=(32,64), layers_per_block=1, norm_num_groups=32)
from safetensors.torch import load_file
missing, unexpected = vae2.load_state_dict(load_file("/tmp/vout/vae_decoder-1.safetensors"), strict=False)
assert not unexpected and all(m.startswith(("encoder.","quant_conv.")) for m in missing)
ok(f"checkpoint chỉ chứa {len(keys)} tensor decoder, nạp lại strict=False sạch sẽ")

print("[9] loss + lr schedule")
sched = NoiseScheduler()
t = torch.randint(0,1000,(4,))
l = diffusion_loss(torch.randn(4,4,8,8), torch.randn(4,4,8,8), t, sched, 5.0)
assert torch.isfinite(l) and l > 0
ok(f"diffusion_loss (Min-SNR γ=5) = {l.item():.4f}")
f = build_lr_lambda("cosine", 100, 1000)
assert abs(f(0)) < 1e-9 and abs(f(100)-1) < 1e-9 and f(1000) < 1e-6 and f(550) > 0.4
ok("cosine + warmup: 0 -> 1 tại step 100 -> ~0 tại step cuối")

print("\nTẤT CẢ SMOKE TEST ĐỀU PASS")
