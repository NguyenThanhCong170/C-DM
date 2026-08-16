# C-DM: Pure PyTorch DreamBooth + LoRA for Chest X-ray Generation

Triển khai thuần **PyTorch** huấn luyện **DreamBooth + LoRA** trên nền tảng **Latent Diffusion (SD 1.5)** để sinh ảnh X-quang lồng ngực có kiểm soát bệnh học lâm sàng (Pneumonia, Effusion, Normal,...).

---

## 📂 Cấu trúc dự án

```text
C-DM/
├── models/lora.py          # Custom LoRALinear, inject_lora, save/load safetensors
├── dataset/xray_dataset.py # Tiền xử lý X-quang + DreamBooth Dual-sampling DataLoader
├── pipeline/inference.py   # NoiseScheduler, DDIM Sampler, CFG & VAE Decoder
├── train.py                # Script huấn luyện chính (AMP fp16, GradScaler, Min-SNR)
├── app.py                  # Giao diện Web UI Gradio phục vụ Demo
└── requirements.txt        # Thư viện phụ thuộc

python train.py \
    --pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --instance_data_dir "./processed_data/pneumonia" \
    --instance_prompt "a chest x-ray of sks pneumonia" \
    --class_data_dir "./processed_data/normal" \
    --class_prompt "a chest x-ray" \
    --with_prior_preservation \
    --output_dir "./lora_output_pneumonia" \
    --resolution 512 \
    --rank 24 --lora_alpha 24.0 --snr_gamma 5.0 \
    --train_batch_size 2 --gradient_accumulation_steps 2 \
    --max_train_steps 1500 --learning_rate 1e-4 \
    --mixed_precision fp16 --gradient_checkpointing