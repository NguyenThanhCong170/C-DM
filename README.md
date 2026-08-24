# C-DM — Chest X-ray Generation with Stable Diffusion in Pure PyTorch

Stable Diffusion 1.5 + LoRA (multi-concept DreamBooth) for 5 lung conditions on the NIH
ChestX-ray14 dataset. The entire architecture includes U-Net, VAE, CLIP text encoder, tokenizer,
DPM-Solver++ 2M sampler.

---

## Setup

Requires a GPU with ~5GB VRAM (512×512, fp16).

```bash
# 1. Clone and install dependencies
git clone https://github.com/NguyenThanhCong170/C-DM && cd C-DM
pip install -r requirements.txt

# 2. SD 1.5 base model (fp16 build, ~2GB) — the LoRA in this repo was trained on this exact model
pip install huggingface_hub
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./sd15 \
  scheduler/scheduler_config.json \
  tokenizer/vocab.json tokenizer/merges.txt \
  tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json \
  text_encoder/config.json text_encoder/model.fp16.safetensors \
  vae/config.json vae/diffusion_pytorch_model.fp16.safetensors \
  unet/config.json unet/diffusion_pytorch_model.fp16.safetensors
```

The LoRA weights `checkpoint-4000.safetensors` (12MB) are already included in the repo — `rank`,
`alpha`, and the list of target modules are read directly from the file's metadata, no separate
config file needed.

Directory layout after setup:

```
C-DM/
├── sd15/                          # SD 1.5 base model (cross_attention_dim = 768), downloaded by you
├── checkpoint-4000.safetensors    # LoRA, rank 16, 4000 steps — included in the repo
├── concepts.json                  # concept definitions, only needed for retraining
└── app.py
```

---

## Generating images

### Web UI

```bash
python app.py
```

Open `http://127.0.0.1:7860`. Add `--share` to create a temporary public link.

### Prompt

The `sks` token is the identifier learned during training — **required** to get correct
chest X-ray output.

```
a chest x-ray of sks atelectasis
a chest x-ray of sks cardiomegaly
a chest x-ray of sks effusion
a chest x-ray of sks infiltration
a chest x-ray of sks pneumonia
a chest x-ray of normal lungs
```

The default parameters (25 steps, CFG 7.5) give good results. Dragging the **LoRA strength**
slider to `0` disables LoRA, useful for comparing against the base SD 1.5.

### Calling from Python

```python
import torch
from models import load_sd_components, inject_lora, load_lora_weights_into
from pipeline.inference import NoiseScheduler, SDComponents, sample

tok, te, vae, unet, sched_cfg = load_sd_components("./sd15", variant="fp16",
                                                   torch_dtype=torch.float16)
for m in (te, vae, unet):
    m.to("cuda").eval()
sched = NoiseScheduler.from_diffusers_config(sched_cfg)

# rank/alpha/target_modules matching the checkpoint shipped with the repo
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

## Retraining

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

This is the exact configuration that produced `checkpoint-4000.safetensors` (5×1000 disease
images plus 1000 "No Finding" images for prior preservation, ~4h on a Tesla T4).

`concepts.json` declares the 5 concepts, with paths relative to the repo root. Image data
(`data/`) is not included in the repo — download NIH ChestX-ray14 and preprocess it to
512×512 grayscale, with black-border padding to preserve anatomical proportions.

---
