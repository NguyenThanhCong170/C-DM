 # C-DM: DreamBooth & LoRA thuần PyTorch cho ảnh X-quang ngực

## Tổng quan
Dự án triển khai mô hình Stable Diffusion 1.x hoàn toàn bằng thư viện PyTorch gốc. Toàn bộ quy trình từ huấn luyện (Train) đến sinh ảnh (Inference) **không phụ thuộc** vào `diffusers` hay `transformers`. Chỉ sử dụng `torch`, `safetensors`, `numpy` và `Pillow`.

Các thành phần cốt lõi được xây dựng độc lập bao gồm:
- **U-Net:** `UNet2DConditionModel`
- **VAE:** `AutoencoderKL`
- **Text Encoder & Tokenizer:** Mạng CLIP text tower và CLIP BPE tokenizer (đọc trực tiếp `vocab.json` + `merges.txt`).
- **Pipeline:**
- **PEFT:** Kiến trúc LoRALinear / LoRAConv2d, tích hợp cơ chế inject, merge và save/load.

## Đặc điểm Kỹ thuật
- **Tương thích `state_dict` tuyệt đối:** Kiến trúc mạng được đặt tên biến (module name) trùng khớp 100% với thư viện `diffusers`. Trọng số được nạp thẳng với cờ `strict=True`, loại bỏ hoàn toàn các bảng ánh xạ key phức tạp.
- **Độ chính xác chuẩn toán học:** Sai số tối đa (maxdiff) khi nạp chéo trọng số giữa C-DM và bản chuẩn `diffusers` là $0.0$ (trùng khớp đến từng bit) cho U-Net, VAE và CLIPTextModel.
- **Khả năng kế thừa:** Hàm load tự động nhận diện dạng checkpoint cũ (legacy attention keys như `query/key/value/proj_attn`) và chặn các config không thuộc kiến trúc SD 1.x (như `SD 2.x`, `v_prediction`, `SDXL`).

---

## Chuẩn bị môi trường và Dữ liệu

Base checkpoint yêu cầu phải được lưu thành thư mục cục bộ (local directory) theo cấu trúc chuẩn của Hugging Face.

```bash
# Cài đặt thư viện nền tảng
pip install -r requirements.txt
pip install huggingface_hub

# Tải pre-trained checkpoint
huggingface-cli download danyalmalik/stable-diffusion-chest-xray --local-dir ./ckpt/cxr

python train.py \
  --pretrained_model_name_or_path ./ckpt/cxr \
  --concepts_list concepts.json \
  --class_data_dir ./data/class_no_finding \
  --with_prior_preservation \
  --output_dir ./out/lora_multiconcept \
  --rank 64 \
  --lora_alpha 64 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 4000 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --snr_gamma 5.0 \
  --validation_prompt "a chest x-ray of sks pneumonia" \
  --validation_steps 500

## Suy luận và Sinh ảnh (Inference)

## Pipeline sinh ảnh tự viết cho phép nạp trực tiếp file pytorch_lora_weights.safetensors vào mạng U-Net và khử nhiễu.
```Bash 
import torch
from models import load_sd_components, inject_lora, load_lora_config, load_lora_weights_into
from pipeline.inference import NoiseScheduler, SDComponents, sample

# 1. Nạp Base Model và lập lịch nhiễu (Noise Scheduler)
tok, te, vae, unet, sched_cfg = load_sd_components("./ckpt/cxr")
[m.to("cuda") for m in (te, vae, unet)]
sched = NoiseScheduler.from_diffusers_config(sched_cfg)

# 2. Tiêm (Inject) và nạp trọng số LoRA vào U-Net
cfg = load_lora_config("out/lora_multiconcept/lora_config.json")
inj = inject_lora(unet, cfg.target_modules, cfg.rank, cfg.alpha)
load_lora_weights_into(inj, "out/lora_multiconcept/pytorch_lora_weights.safetensors")

# 3. Lấy mẫu sinh ảnh (DDIM Sampling)
imgs = sample(
    components=SDComponents(unet, vae, te, tok),
    prompt="a chest x-ray of hta atelectasis",
    negative_prompt="blurry, artifact",
    num_images=4,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(0),
    device="cuda",
    scheduler=sched
)