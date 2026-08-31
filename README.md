# C-DM — Label-Conditioned Chest X-ray Generation

Fine-tuning Stable Diffusion 1.5 to generate chest X-rays from a **5-dim multi-hot label vector**, trained on NIH ChestX-ray14.

```
labels (B,5) --MultiHotLabelEncoder--> (B,11,768) --cross-attn--> U-Net (+LoRA) --> latents --VAE--> image
```

Trained parts: the **U-Net LoRA adapters** + the **whole MultiHotLabelEncoder**. The VAE and the
base U-Net weights stay frozen. The VAE decoder is fine-tuned separately in stage 2.

The 5 labels (`dataset/nih_multilabel.py` → `LABEL_NAMES`):

| # | Label | Note |
|---|---|---|
| 0 | `No Finding` | no abnormality reported |
| 1 | `Infiltration` | |
| 2 | `Effusion` | |
| 3 | `Atelectasis` | |
| 4 | `Others` | the remaining 11 NIH-14 findings collapsed together (`OTHERS_MEMBERS`) |

No `diffusers` dependency: `models/unet.py` and `models/vae.py` are hand-written re-implementations
of `UNet2DConditionModel` / `AutoencoderKL`, and `models/loading.py` loads stock HF SD 1.5 weights
into them.

---

## 1. Setup

### 1.1 Dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires `torch>=2.1`. A GPU is not needed for the smoke test, but is required for training.
`wandb` must be **installed** (`smoke_test_multilabel.py` and `train_vae_decoder.py` import it at
module level), though you never have to log in: setting `wandb_mode: disabled` (or
`wandb_project: null`) in the YAML turns logging off entirely.

### 1.2 SD 1.5 base model

Only `unet`, `vae` and `scheduler` are needed — **no** `text_encoder`/`tokenizer`, since the
multi-hot branch does not use CLIP:

```bash
pip install huggingface_hub
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./sd15 \
    scheduler/scheduler_config.json \
    unet/config.json unet/diffusion_pytorch_model.fp16.safetensors \
    vae/config.json  vae/diffusion_pytorch_model.fp16.safetensors
```

Expected layout:

```
sd15/
├── scheduler/scheduler_config.json
├── unet/{config.json, diffusion_pytorch_model.fp16.safetensors}
└── vae/{config.json, diffusion_pytorch_model.fp16.safetensors}
```

If you download the files **without** the `.fp16` suffix, set `variant: null` in the config.

### 1.3 NIH ChestX-ray14 data

Kaggle layout (`nih-chest-xrays/data`):

```
data/nih/
├── Data_Entry_2017.csv
├── images_001/images/*.png
├── images_002/images/*.png
└── ... images_012/
```

Point `data_root` at that directory. The scripts scan every `images_*/images/` folder (or
`images/`, or `data_root` itself) and match files against the CSV's `Image Index` column.

---

## 2. Preflight checks

Two layers, in this order:

```bash
# (a) Logic smoke test — NO GPU, NO data required
python smoke_test_multilabel.py

# (b) Real preflight — checks sd15, the NIH data, the dataset and model loading,
#     then runs one real optimizer step to measure VRAM and estimate training time
python check_setup.py --config config/multilabel.yaml
python check_setup.py --config config/multilabel.yaml --skip-train-step   # skip the last step
```

`check_setup.py` prints peak VRAM and an hour estimate for `max_train_steps` — use those numbers
to tune `train_batch_size` before kicking off a long run.

Then do a short run on real data (~30 steps, 400 images):

```bash
python train_multilabel.py --config config/multilabel_smoke.yaml
```

---

## 3. Stage 1 — Label-conditioned diffusion training

```bash
python train_multilabel.py --config config/multilabel.yaml
```

Config overrides from the command line (repeatable, values parsed as YAML):

```bash
python train_multilabel.py --config config/multilabel.yaml \
    -o data_root=/kaggle/input/data -o train_batch_size=8 -o max_train_steps=12000
```

Unknown YAML keys only produce a **warning**, they do not stop the run — so watch for the
`[warn] khoá lạ` line after editing a config. The `DEFAULTS` dict at the top of
`train_multilabel.py` is the schema of record for every key.

### 3.1 The keys that matter most

| Key | Default (config/multilabel.yaml) | Meaning |
|---|---|---|
| `rank`, `lora_alpha` | 64 / 64 | LoRA capacity |
| `cross_attention_only` | `false` | `true` → inject into `attn2` only (the blocks that see labels), much lighter |
| `target_modules` | `to_q,to_k,to_v,to_out.0` | used when `cross_attention_only: false` |
| `tokens_per_label` | 2 | tokens per label → `1 + 5*2 = 11` tokens total |
| `cond_dropout_prob` | 0.1 | fraction of samples whose labels are replaced by the null label. **This is what makes CFG work at sampling time** |
| `label_encoder_lr` | 1e-4 | separate lr for the encoder (a module trained from scratch) |
| `max_per_label` | `No Finding: 15000`, `Others: 20000` | caps on the dominant classes |
| `balance_beta` | 0.5 | 0 = keep the original distribution, 1 = fully balanced |
| `val_ratio` | 0.05 | split **by Patient ID**, to avoid anatomy leakage |
| `cache_dir` | `./cache512` | cache of resized images; ~25–30 GB for 112k images @512 |
| `snr_gamma` | 5.0 | Min-SNR-γ loss weighting |
| `mixed_precision` | `fp16` | `bf16` errors out on GPUs that don't support it (e.g. T4) |
| `resume_lora` / `resume_label_encoder` | `null` | resume from a checkpoint — **pass both** |

### 3.2 What lands in `output_dir`

```
out/multilabel_att1_att2/
├── lora-1000.safetensors ... lora-final.safetensors
├── label_encoder-1000.safetensors ... label_encoder-final.safetensors
├── lora_config.json               # rank/alpha/targets — every loader reads them back from here
├── label_encoder_config.json
└── validation/step_1000/*.png     # validation images for validation_labels
```

**The LoRA and the label encoder are an inseparable pair**: the unconditional branch of CFG is the
learned `null_tokens` living inside the label encoder, so mixing files from two different training
runs produces garbage. Always use the same step for both.

---

## 4. Stage 2 — VAE decoder fine-tune (optional)

The encoder stays frozen; only the decoder + `post_quant_conv` are trained, on random 256px crops
with an L1 + perceptual (VGG16) loss. Purpose: sharper decoded images, less smearing of the
parenchymal texture.

```bash
python train_vae_decoder.py --config config/vae_decoder.yaml
```

Output: `out/vae_decoder/vae_decoder-{step}.safetensors`.

That checkpoint can be reused in several places:

- training-time validation: `vae_decoder_checkpoint:` in `config/multilabel.yaml`
- generation CLI: `--vae-decoder`
- demo: `--vae-decoder`
- evaluation: `generation.vae_decoder` in `config/evaluation.yaml`

Set `perceptual_weight: 0` if the machine cannot download the VGG16 weights.

---

## 5. Inference

### 5.1 CLI

```bash
# by label name, separated by '|'
python generate_multilabel.py \
    --lora           out/multilabel_att1_att2/lora-9000.safetensors \
    --label-encoder  out/multilabel_att1_att2/label_encoder-9000.safetensors \
    --lora-config    out/multilabel_att1_att2/lora_config.json \
    --labels "Effusion|Atelectasis" -n 8 --grid

# soft labels: Effusion fully on, Others halfway
python generate_multilabel.py --vector 0,0,1,0.5,0 -n 4
```

The script's defaults point at `./out/multilabel/`, which is **not** the existing checkpoint
directory, so in practice you always pass `--lora / --label-encoder / --lora-config`.

Commonly used flags:

| Flag | Default | Note |
|---|---|---|
| `--steps` | 25 | DPM-Solver++(2M) steps |
| `--guidance` | 4.0 | CFG; the working range is **3–5**. 7.5 (the text-to-image default) over-contrasts the image and destroys parenchymal detail |
| `--batch-size` | 8 | CFG **doubles** the batch through the U-Net — lower it if you OOM |
| `--seed` | 0 | one shared generator across batches, so batches never repeat the same noise |
| `--grid` | | assemble a contact sheet for quick browsing |
| `--dtype` | fp16 | use `fp32` on CPU |

Images are written to `--outdir` (default `./out/samples`) as `{tag}_seed{seed}_{i:03d}.png`.

### 5.2 Gradio demo

```bash
python app.py \
    --lora          out/multilabel_att1_att2/lora-9000.safetensors \
    --label-encoder out/multilabel_att1_att2/label_encoder-9000.safetensors \
    --lora-config   out/multilabel_att1_att2/lora_config.json \
    --vae-decoder   out/vae_decoder/vae_decoder-6000.safetensors    # optional
```

Open `http://127.0.0.1:7860`. The UI has label checkboxes, presets for the common combinations,
and sliders for: denoising steps (1–100, default 25), CFG (1–10, default 4.0), image size
(256–768), number of images, **LoRA strength** (0–1.5; 0 = plain base model, the image will not
follow the labels), and the seed.

There is no prompt box and no negative prompt — the label vector is the only conditioning.

`app.py`'s defaults already point at `out/multilabel_att1_att2/*-9000`, so with that checkpoint
plain `python app.py` is enough. `--share` creates a public `*.gradio.live` link.

### 5.3 As a library

Every generation path (training validation, CLI, demo, evaluation) goes through a single function,
so a change there propagates everywhere at once:

```python
from pipeline.label_inference import LabelSDComponents, sample_from_labels
from models.label_encoder import labels_to_multihot

y = labels_to_multihot("Effusion|Atelectasis").unsqueeze(0)   # (1,5)
images = sample_from_labels(
    LabelSDComponents(unet=unet, vae=vae, label_encoder=label_encoder),
    y, num_images=1, num_inference_steps=25, guidance_scale=4.0, device="cuda",
)
```

`guidance_scale <= 1.0` disables CFG entirely (only the conditional branch runs — twice as fast,
but the labels barely show).

---

## 6. Evaluation

Three metrics — **TRTR / TSTR / CAS** — produced by two 5-label DenseNet-121 classifiers.

```bash
# 0) edit the checkpoint paths in config/evaluation.yaml (the `generation` section)
python -m evaluation.train_classifier --config config/evaluation.yaml --mode real       # TRTR
python -m evaluation.generate_eval_set --config config/evaluation.yaml                  # 5000 synthetic images
python -m evaluation.train_classifier --config config/evaluation.yaml --mode synthetic  # TSTR
python -m evaluation.compute_cas       --config config/evaluation.yaml                  # CAS
python -m evaluation.report            --config config/evaluation.yaml                  # merge into a report
```

Results: `out/eval/{trtr,tstr,cas}.json` + `out/eval/report.md`.

DenseNet-121 appears **only** here; it is not part of the generation pipeline and does not score
anything inside the Gradio demo.

The details (what each metric means, how to read the numbers, and traps such as "hamming accuracy
alone is meaningless" and "CAS is blind to mode collapse") are in
[evaluation/README.md](evaluation/README.md) — read that file before reporting any results.

---

## 7. Project layout

```
config/            # YAML per script; the DEFAULTS dict in each script is the schema of record
dataset/           # NIHMultiLabelDataset, label mapping, patient-level split, image cache
models/
├── unet.py, vae.py            # hand-written SD 1.5 (no diffusers)
├── small_block/, middle_block/
├── loading.py                 # loads HF checkpoints, fails loudly on any key mismatch
├── label_encoder.py           # MultiHotLabelEncoder + null_tokens for CFG
└── lora.py                    # self-contained LoRA (no peft)
pipeline/
├── inference.py               # NoiseScheduler, the DPM-Solver++(2M) step, VAE decode
└── label_inference.py         # sample_from_labels — the ONLY entry point for generation
evaluation/        # TRTR / TSTR / CAS
train_multilabel.py, train_vae_decoder.py
generate_multilabel.py, app.py
check_setup.py, smoke_test_multilabel.py
```

---
