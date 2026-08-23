# C-DM — Sinh ảnh X-quang ngực bằng Stable Diffusion thuần PyTorch

Stable Diffusion 1.5 + LoRA (DreamBooth multi-concept) cho 5 bệnh lý phổi trên bộ NIH ChestX-ray14.
Toàn bộ kiến trúc (U-Net, VAE, CLIP text encoder, tokenizer, sampler DPM-Solver++ 2M) được viết
lại bằng `torch` gốc — **không dùng `diffusers` hay `transformers`**.


---

## Setup

Cần GPU ~5GB VRAM (512×512, fp16).

```bash
# 1. Clone và cài thư viện
git clone https://github.com/NguyenThanhCong170/C-DM && cd C-DM
pip install -r requirements.txt

# 2. Base model SD 1.5 (bản fp16, ~2GB) — LoRA trong repo được train trên đúng model này
pip install huggingface_hub
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./sd15 \
  scheduler/scheduler_config.json \
  tokenizer/vocab.json tokenizer/merges.txt \
  tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json \
  text_encoder/config.json text_encoder/model.fp16.safetensors \
  vae/config.json vae/diffusion_pytorch_model.fp16.safetensors \
  unet/config.json unet/diffusion_pytorch_model.fp16.safetensors
```

Trọng số LoRA `checkpoint-4000.safetensors` (12MB) đã có sẵn trong repo — `rank`, `alpha` và
danh sách module đích đọc thẳng từ metadata của file, không cần file config kèm theo.

Cấu trúc sau khi setup:

```
C-DM/
├── sd15/                          # base model SD 1.5 (cross_attention_dim = 768), tự tải
├── checkpoint-4000.safetensors    # LoRA rank 16, 4000 step — có sẵn trong repo
├── concepts.json                  # khai báo concept, chỉ cần khi train lại
└── app.py
```

---

## Sinh ảnh

### Giao diện web

```bash
python app.py
```

Mở `http://127.0.0.1:7860`. Thêm `--share` để tạo link public tạm thời.

### Prompt

Token `sks` là identifier đã học lúc train — **bắt buộc có** thì mới ra ảnh X-quang đúng.

```
a chest x-ray of sks atelectasis
a chest x-ray of sks cardiomegaly
a chest x-ray of sks effusion
a chest x-ray of sks infiltration
a chest x-ray of sks pneumonia
a chest x-ray of normal lungs
```

Tham số mặc định (25 bước, CFG 7.5) cho kết quả tốt. Thanh **Cường độ LoRA** kéo về `0`
sẽ tắt LoRA để đối chiếu với SD 1.5 gốc.

### Gọi từ Python

```python
import torch
from models import load_sd_components, inject_lora, load_lora_weights_into
from pipeline.inference import NoiseScheduler, SDComponents, sample

tok, te, vae, unet, sched_cfg = load_sd_components("./sd15", variant="fp16",
                                                   torch_dtype=torch.float16)
for m in (te, vae, unet):
    m.to("cuda").eval()
sched = NoiseScheduler.from_diffusers_config(sched_cfg)

# rank/alpha/target_modules của checkpoint đi kèm repo
inj = inject_lora(unet, ["to_q", "to_k", "to_v", "to_out.0"], rank=16, alpha=16)
load_lora_weights_into(inj, "checkpoint-4000.safetensors")

imgs = sample(
    components=SDComponents(unet, vae, te, tok),
    prompt="a chest x-ray of sks pneumonia",
    negative_prompt="blurry, artifact",
    num_images=4,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(0),
    device="cuda",
    scheduler=sched,
)
imgs[0].save("out.png")
```

---

## Train lại

```bash
python train.py \
  --pretrained_model_name_or_path ./sd15 \
  --variant fp16 \
  --concepts_list concepts.json \
  --class_data_dir ./data/class_no_finding \
  --with_prior_preservation --prior_loss_weight 1.0 \
  --output_dir ./out/lora \
  --rank 16 --lora_alpha 16 \
  --max_train_steps 4000 \
  --train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --lr_scheduler cosine --lr_warmup_steps 200 \
  --snr_gamma 5.0 \
  --mixed_precision fp16 --gradient_checkpointing \
  --checkpointing_steps 500 \
  --validation_prompt "a chest x-ray of sks pneumonia" --validation_steps 250
```

Đây là cấu hình đã sinh ra `checkpoint-4000.safetensors` (5×1000 ảnh bệnh + 1000 ảnh
"No Finding" làm prior preservation, ~4h trên Tesla T4).

`concepts.json` khai báo 5 concept, đường dẫn tương đối so với thư mục gốc repo. Dữ liệu ảnh
(`data/`) không kèm trong repo — tải NIH ChestX-ray14 rồi tiền xử lý về 512×512 grayscale,
padding viền đen giữ tỉ lệ giải phẫu.

---
