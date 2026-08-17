# Model Architecture Quick Reference

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DreamBooth + LoRA Training                      │
└─────────────────────────────────────────────────────────────────┘

INPUT STAGE
───────────
Chest X-ray (PNG/JPG)          Text Prompts
    │                          │
    └──→ [Preprocess]          └──→ [Tokenize]
         (512×512)                   (max_length=77)
         Normalize                   
         RGB Duplicate              
    │                          │
    └──→ [VAE Encode]          └──→ [CLIP Encode]
         Frozen                     Frozen
         (3,512,512)                (77,768)
         │                          │
         ├────→ Latent z            ├────→ Text Embedding
         │      (4,64,64)           │      
         │      scaled by 0.18215   │
         │                          │
         └─┬────────────────────────┘
           │
NOISE DIFFUSION STAGE
─────────────────────
           │
           ├─→ Sample timestep t ∈ [0, 1000]
           │
           ├─→ Encode noise to latent: 
           │   z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε
           │
DENOISING STAGE
───────────────
           │
           ├─→ [U-Net Denoiser]
           │   ├─ Input: (z_t, t, text_embedding)
           │   │
           │   ├─ Self-Attention Blocks (LoRA⚡)
           │   │   ├─ to_q: query projection
           │   │   ├─ to_k: key projection
           │   │   ├─ to_v: value projection
           │   │   └─ to_out: output projection
           │   │
           │   ├─ Cross-Attention Blocks (LoRA⚡)
           │   │   └─ Text-Image interaction
           │   │
           │   └─ Output: Predicted noise ε̂_θ
           │
LOSS COMPUTATION
────────────────
           │
           ├─→ Instance Loss:
           │   L_inst = MSE(ε, ε̂_θ)
           │
           ├─→ Prior Preservation Loss:
           │   L_prior = MSE(ε_prior, ε̂_θ_prior)
           │
           └─→ Total Loss:
               L_total = L_inst + λ · L_prior
               (Optimize: LoRA parameters only)

OUTPUT STAGE
────────────
           ├─→ Backward pass (LoRA gradients only)
           │
           └─→ Optimizer step (AdamW, 8-bit AdamW)
               Update LoRA weights
```

---

## 📊 Component Specifications

### Stable Diffusion Base Model
| Component | Type | Size | Status | Purpose |
|-----------|------|------|--------|---------|
| **CLIP Text Encoder** | ViT-L/14 | 123M | 🔒 Frozen | Text → Embeddings |
| **VAE Encoder** | AutoEncoder | 84M | 🔒 Frozen | Image → Latents |
| **U-Net Denoiser** | Diffusion | 860M | 🔄 LoRA | Noise Prediction |
| **DDPM Scheduler** | Noise Schedule | - | 🔒 Frozen | Noise Timeline |

### LoRA Configuration (Recommended)
```
Rank (r):           32
Alpha (α):          32
Scaling:            α/r = 1.0
Dropout:            0.05
Target Modules:     [to_q, to_k, to_v, to_out.0]
```

### Parameter Count
```
U-Net Total:        860,000,000 parameters
LoRA Trainable:       2,500,000 parameters (0.29%)
LoRA Checkpoint:           15 MB
vs Full Fine-tune:    3,400 MB (227x smaller)
```

### Memory Requirements
```
Model Loading:      ~12 GB (fp32) / ~6 GB (fp16)
Training (batch=2): ~16 GB (fp32) / ~8 GB (fp16)
Inference:          ~6 GB (fp16)
```

---

## 🔧 LoRA Injection Points

### U-Net Architecture Blocks
```
Depth 0: Input (4, 64, 64)
├─ ResBlock
├─ AttentionBlock [LoRA⚡] ← Inject here
│  ├─ self-attention (LoRA)
│  └─ cross-attention (LoRA)
└─ Downsample

Depth 1: (4, 32, 32)
├─ ResBlock + AttentionBlock [LoRA⚡]
└─ Downsample

Middle: (4, 16, 16)
├─ ResBlock
├─ AttentionBlock [LoRA⚡] ← Important for semantics
└─ ResBlock

Depth 2: (4, 32, 32)
├─ Upsample
├─ ResBlock + AttentionBlock [LoRA⚡]
└─ ...

Depth 3: (4, 64, 64)
├─ Upsample
├─ ResBlock + AttentionBlock [LoRA⚡]
└─ Output Projection [LoRA⚡]
```

---

## 🎯 Training Data Flow

```
Instance Batch (Pathology-Specific)          Class Batch (General)
┌────────────────────────────┐              ┌──────────────────────┐
│ X-ray Images (3, 512, 512) │              │ Normal X-rays (...)  │
│ Prompts: "sks pneumonia"   │              │ Prompts: "x-ray"     │
└────────────────────────────┘              └──────────────────────┘
           │                                         │
           └──────────────────┬──────────────────────┘
                              │
                    [Concatenate in batch]
                              │
                    Latent z: (8, 4, 64, 64)
                    Prompts: (8, 77, 768)
                              │
                    Sample timestep t
                              │
                    Add Noise ε
                              │
                    Forward to U-Net
                              │
    ┌───────────────┬─────────┴──────────┬──────────────┐
    │               │                    │              │
Instance Loss   Prior Loss             SNR Weighting  Backward
  (4 samples)    (4 samples)         (timestep weight) (LoRA only)
    │               │                    │              │
    └───────────────┴────────────────────┴──────────────┘
                              │
                   Update LoRA Parameters
```

---

## 📈 Training Dynamics

### Loss Function
```
L = Σ_t [w(t) · || ε_t - ε̂_θ(z_t, t, c) ||²₂]
           └─ Min-SNR-γ weighting

Where:
  ε_t = true noise
  ε̂_θ = U-Net prediction (with LoRA)
  c = conditioning (text prompt)
  w(t) = schedule weight (e.g., min-snr-gamma)
```

### Learning Rate Schedule
```
Warmup Phase (0-10% steps):
  lr(t) = initial_lr · (t / warmup_steps)

Constant Phase (10-100% steps):
  lr(t) = initial_lr

Recommended:
  initial_lr = 1e-4 (with α/r = 1.0 scaling)
```

### Gradient Accumulation
```
For memory-limited GPUs:
  gradient_accumulation_steps = 2-4
  
Effective batch size = batch_size × accumulation_steps
Memory cost: ~1/2 to 1/4 of batch_size
```

---

## 🚀 Model Inference

### Sampling Process
```
Step 0: Random Noise
  z_T ~ N(0, I)

Step 1-1000: Iterative Denoising
  z_{t-1} = (z_t - √(1-ᾱ_{t-1}) · ε̂_θ(z_t, t, c)) / √ᾱ_{t-1}
                                   ↑
                        U-Net with trained LoRA

Where:
  ε̂_θ = denoised prediction at step t
  c = text conditioning (prompt embeddings)

Step 1000: Final Image
  x_0 = VAE.decode(z_0)
```

### Generation Hyperparameters
```
Inference Steps:      50-100 (quality vs speed tradeoff)
Guidance Scale:       7.5 (how much to weight text prompt)
Negative Guidance:    "blurry, low quality, artifacts"
Seed:                 42 (for reproducibility)
```

---

## 💾 Checkpoint Management

### Saving LoRA Weights
```python
save_lora_weights(model.unet, "checkpoint.safetensors")
# Output: ~15-20 MB

Structure:
{
  "unet.down_blocks.0.attentions.0.to_q.lora_a": tensor,
  "unet.down_blocks.0.attentions.0.to_q.lora_b": tensor,
  "unet.down_blocks.0.attentions.0.to_k.lora_a": tensor,
  ...
}
```

### Loading LoRA Weights
```python
load_lora_weights(model.unet, "checkpoint.safetensors")
# Ready for inference or continued training
```

### Merging LoRA (for Deployment)
```python
merge_lora_into_base(model.unet)
# Creates single model file (~3.3 GB)
# No separate LoRA weights needed
```

---

## 🎛️ Hyperparameter Tuning

### For Better Results:
| Parameter | ↓ Quality | → Balanced | ↑ Quality |
|-----------|----------|-----------|-----------|
| **Rank** | 4 | **32** | 64 |
| **Alpha** | 4 | **32** | 64 |
| **Learning Rate** | 5e-5 | **1e-4** | 5e-4 |
| **Batch Size** | 1 | **2** | 4 |
| **Num Steps** | 500 | **1500** | 3000 |
| **Prior Weight** | 0.05 | **0.1** | 0.25 |

### Dropout Effects:
```
lora_dropout=0.0  → Low regularization, faster overfitting
lora_dropout=0.05 → Recommended, good balance
lora_dropout=0.1  → Heavy regularization, slower convergence
```

---

## 🐛 Debugging Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| Blurry outputs | Under-trained | ↑ training steps, ↑ rank |
| Mode collapse | Overfitting | ↑ prior weight, ↑ dropout |
| Out of memory | Large batch | ↓ batch_size, enable checkpointing |
| Slow convergence | Low LR | ↑ learning_rate, ↑ rank |
| Training NaN | Gradient explosion | ↓ learning_rate, enable gradient clipping |

---

## 📚 Key Files

```
models/
├── __init__.py          # Public API
├── lora.py              # LoRA injection & utilities (800+ lines)
└── base_model.py        # Model loading & wrapping (400+ lines)

ARCHITECTURE.md          # Detailed documentation
architecture_quick_reference.md  # This file
```

---

## 🔗 Integration with Other Components

```
Dataset (xray_dataset.py)
    ↓ (images, prompts)
    ↓
Training Loop (train.py)
    ↓ uses
    ↓
Base Model (base_model.py)
    ↓ injects LoRA into
    ↓
U-Net (via lora.py)
    ↓ during training
    ↓
Checkpoints (safetensors)
    ↓ loaded in
    ↓
Inference Pipeline (pipeline/inference.py)
    ↓
Generated X-ray Images
```

---

**Version:** 1.0  
**Last Updated:** 2024  
**Framework:** PyTorch + Diffusers  
**Model Base:** Stable Diffusion v1.5
