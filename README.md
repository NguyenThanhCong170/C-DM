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
├── config/training.yaml           # training settings, only needed for retraining
└── app.py
```

---

## Generating images

### Web UI
Using Gradio to deloy model

```bash
pip install gradio
```

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

Training is driven by a YAML config instead of command-line flags:

```bash
python train.py                              # reads config/training.yaml
```

`config/training.yaml` holds the settings that produced `checkpoint-4000.safetensors`
(5×1000 disease images plus 1000 "No Finding" images for prior preservation, ~4h on a Tesla T4):


`concepts.json` declares the 5 concepts, with paths relative to the repo root. Image data
(`data/`) is not included in the repo — download NIH ChestX-ray14 and preprocess it to
512×512 grayscale, with black-border padding to preserve anatomical proportions.

---

## Bắt đầu nhanh (nhánh multi-hot) — chạy trên server

```bash
# 0. Môi trường
pip install -r requirements.txt
wandb login                       # bỏ qua nếu để wandb_project: null trong config

# 1. Base model — nhánh multi-hot KHÔNG cần text_encoder/tokenizer (~1.9 GB)
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir ./sd15 \
  scheduler/scheduler_config.json \
  vae/config.json  vae/diffusion_pytorch_model.fp16.safetensors \
  unet/config.json unet/diffusion_pytorch_model.fp16.safetensors

# 2. Trỏ tới dữ liệu — sửa `data_root` trong config, hoặc symlink cho gọn:
ln -s /đường/dẫn/tới/nih ./data/nih      # thư mục chứa Data_Entry_2017.csv

# 3. Kiểm tra: 6 bước, bước cuối chạy 1 optimizer step thật để đo VRAM + ETA
python check_setup.py --config config/multilabel.yaml

# 4. Chạy thử 30 step trên dữ liệu thật
python train_multilabel.py --config config/multilabel_smoke.yaml

# 5. Train thật, trong tmux
./run_train.sh config/multilabel.yaml
```

Không cần sửa YAML cho mỗi thí nghiệm — `-o KEY=VALUE` ghi đè bất kỳ khoá nào
(lặp lại được, giá trị parse bằng YAML nên `8` ra int, `1e-4` ra float):

```bash
python train_multilabel.py --config config/multilabel.yaml \
    -o data_root=/data/nih -o train_batch_size=8 -o learning_rate=5e-5 \
    -o wandb_run_name=rank256-bs8 -o output_dir=./out/exp2
```

### Chạy dài trong tmux

`./run_train.sh` tạo sẵn session `cdm` với hai cửa sổ: `train` (có `tee` ra
`logs/train_<timestamp>.log`) và `gpu` (`watch nvidia-smi`).

| | |
|---|---|
| `tmux attach -t cdm` | gắn vào session đang chạy |
| `Ctrl-b d` | thoát ra, training vẫn chạy tiếp |
| `Ctrl-b n` | chuyển giữa cửa sổ train / gpu |
| `tail -f logs/train_*.log` | theo dõi log từ shell khác |
| `tmux kill-session -t cdm` | dừng hẳn |
| `CUDA_VISIBLE_DEVICES=1 ./run_train.sh` | chạy trên GPU khác |
| `CDM_SESSION=exp2 ./run_train.sh ... -o output_dir=./out/exp2` | thí nghiệm thứ hai song song |

Muốn train tiếp từ checkpoint cũ thì thêm vào config (hoặc truyền bằng `-o`):

```yaml
resume_lora: "./out/multilabel/lora-10000.safetensors"
resume_label_encoder: "./out/multilabel/label_encoder-10000.safetensors"
```

> Lưu ý: resume chỉ nạp lại **trọng số**, không nạp lại trạng thái optimizer và
> lịch learning rate — lr sẽ khởi động lại từ đầu lịch cosine. Với LoRA thì
> chấp nhận được, nhưng đừng coi nó tương đương một run liền mạch.

### Theo dõi bằng Weights & Biases

Bật sẵn trong `config/multilabel.yaml` (`wandb_project: "c-dm-multilabel"`).
Mỗi `logging_steps` sẽ log `train/loss`, `train/loss_ema`, learning rate của cả hai
nhóm tham số, số ảnh đã thấy và VRAM; mỗi `validation_steps` đẩy luôn ảnh sinh ra
kèm caption là tổ hợp nhãn. Phân phối nhãn và số tham số nằm ở tab Summary.

```yaml
wandb_project: "c-dm-multilabel"
wandb_mode: "online"       # "offline" nếu server không ra được mạng, "disabled" để tắt
wandb_run_name: null
wandb_log_images: true
```

Server không ra được mạng thì để `wandb_mode: "offline"`, xong run thì đồng bộ:

```bash
wandb sync out/multilabel/wandb/offline-run-*
```

Không cài wandb, hoặc `wandb_project: null` → script in một dòng cảnh báo rồi chạy
bình thường, không có gì hỏng.

Hai script kiểm tra chạy được không cần GPU lẫn dữ liệu thật:
`python smoke_test_multilabel.py` và `python test_gradient_checkpointing.py`.

Nếu chạy trên Kaggle thay vì server, dùng `notebooks/kaggle_multilabel.ipynb`.

---

## Điều kiện bằng nhãn multi-hot (không dùng prompt)

Nhánh thứ hai của repo: thay vì mô tả bệnh bằng câu prompt (`"a chest x-ray of sks
pneumonia"`), ảnh được sinh trực tiếp từ vector multi-hot 5 chiều. CLIP text encoder
bị bỏ hẳn khỏi pipeline.

```
labels (B,5) ──► MultiHotLabelEncoder ──► (B,11,768) ──cross-attention──► U-Net (+LoRA)
```

| Chiều | Nhãn |
|---|---|
| 0 | No Finding (bình thường) |
| 1 | Infiltration |
| 2 | Effusion |
| 3 | Atelectasis |
| 4 | Others (gộp 11 bệnh còn lại của NIH) |

Chiều 1–4 là multi-hot thật: ảnh đồng mắc Effusion + Atelectasis cho `[0,0,1,1,0]`.

### File mới

| File | Vai trò |
|---|---|
| `models/label_encoder.py` | `MultiHotLabelEncoder`: mỗi nhãn có token *present* / *absent* học được, cộng 1 token toàn cục, đi qua 2 lớp Transformer. Có `null_tokens` cho nhánh uncond của CFG. |
| `dataset/nih_multilabel.py` | Đọc `Data_Entry_2017.csv` + `images_001..012/images/`, ánh xạ 14 → 5 nhãn, undersample theo trần mỗi lớp, `WeightedRandomSampler`, cache ảnh đã resize, tách train/val theo `Patient ID`. |
| `pipeline/label_inference.py` | `sample_from_labels()` — DPM-Solver++ 2M + CFG với null-label thay negative prompt. |
| `train_multilabel.py` | Train LoRA (U-Net) + LabelEncoder. Min-SNR γ=5, label dropout 10% để học CFG. |
| `train_vae_decoder.py` | Stage riêng: tinh chỉnh decoder VAE bằng L1 + perceptual, **encoder đóng băng**. |
| `generate_multilabel.py` | CLI sinh ảnh từ tên nhãn hoặc vector số. |
| `smoke_test_multilabel.py` | Kiểm tra toàn bộ pipeline, không cần GPU và dữ liệu thật. |
| `check_setup.py` | 6 bước kiểm tra trước khi train: thư viện/GPU → base model → dữ liệu → dataset → nạp model → 1 optimizer step thật (in VRAM đỉnh + ETA). |
| `test_gradient_checkpointing.py` | Chứng minh bật/tắt gradient checkpointing cho output và gradient giống hệt nhau. |
| `logging_utils.py` | Logger bọc quanh wandb — no-op an toàn khi wandb tắt/không cài/mất mạng. |
| `run_train.sh` | Khởi động training trong tmux, log ra `logs/`, kèm cửa sổ `nvidia-smi`. |
| `notebooks/kaggle_multilabel.ipynb` | Notebook Kaggle (chỉ cần nếu chạy trên Kaggle). |
| `app_multilabel.py` | Gradio demo chọn nhãn bằng checkbox (tuỳ chọn, `app.py` bản prompt vẫn giữ nguyên). |

### Chạy

```bash
# (tuỳ chọn, chạy trước) tinh chỉnh decoder VAE cho miền ảnh X-quang
python train_vae_decoder.py --config config/vae_decoder.yaml

# huấn luyện chính
python train_multilabel.py --config config/multilabel.yaml

# sinh ảnh
python generate_multilabel.py --labels "Effusion|Atelectasis" -n 4 --guidance 4.0
python generate_multilabel.py --vector 0,0,1,0.5,0        # nhãn mềm
```

Sửa `data_root` trong hai file config trỏ tới thư mục chứa `Data_Entry_2017.csv`
(trên Kaggle là `/kaggle/input/data`).

### Vì sao decoder VAE phải train riêng

Loss khuếch tán chỉ đi qua **encoder** (ảnh → latent) và U-Net; decoder không nằm
trên đồ thị tính toán đó nên không thể tối ưu chung một lượt. `train_vae_decoder.py`
train nó bằng loss tái tạo thuần và **đóng băng encoder** — nếu encoder đổi thì không
gian latent đổi theo và mọi checkpoint LoRA đã train sẽ vô nghĩa.

### Ghi chú kỹ thuật

`models/unet.py` giờ có `enable_gradient_checkpointing()` thật (trước đây `train.py`
chỉ in cảnh báo rồi bỏ qua). Nó bọc từng cặp resnet/attention của 5 loại block bằng
`torch.utils.checkpoint(..., use_reentrant=False)` — đổi ~30% tốc độ lấy ~40–50% VRAM,
nhờ đó chạy được `train_batch_size: 4` ở 512×512 trên một T4.

`use_reentrant=False` là bắt buộc: bản reentrant cũ đòi ít nhất một input
`requires_grad`, mà khi backbone đóng băng (chỉ LoRA học) điều đó không đúng ở mọi
block — gradient sẽ âm thầm biến mất thay vì báo lỗi.
