"""
Base model loading utilities for Stable Diffusion with LoRA adaptation.
Handles loading pretrained checkpoints and setting up the model architecture.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .lora import LoRAConfig, inject_lora_into_unet


class StableDiffusionXrayModel:
    """
    Wrapper for Stable Diffusion components with LoRA adaptation for chest X-ray fine-tuning.
    
    Manages:
    - Text encoder (CLIP): frozen
    - VAE: frozen  
    - U-Net: LoRA-adapted
    - Noise scheduler: for diffusion training
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        lora_config: Optional[LoRAConfig] = None,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        """
        Initialize Stable Diffusion model with optional LoRA injection.
        
        Args:
            pretrained_model_name_or_path: Hugging Face model ID or local path.
                Can be a base SD checkpoint or chest-X-ray-pretrained model.
            lora_config: LoRA configuration. If None, no LoRA injection.
            torch_dtype: Precision for model weights (float32, float16, bfloat16).
            device: Device to load models on ("cpu", "cuda", etc.).
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_config = lora_config
        
        print(f"Loading pretrained model from: {pretrained_model_name_or_path}")
        
        # Load text encoder (frozen)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        self.text_encoder.requires_grad_(False)
        print("✓ Loaded text encoder (frozen)")
        
        # Load VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        self.vae.requires_grad_(False)
        print("✓ Loaded VAE encoder-decoder (frozen)")
        
        # Load U-Net (will be LoRA-adapted)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )
        print("✓ Loaded U-Net")
        
        # Inject LoRA if config provided
        self.lora_layers = None
        if lora_config is not None:
            self.lora_layers = inject_lora_into_unet(self.unet, lora_config)
        else:
            # Make all U-Net parameters trainable (full fine-tune)
            self.unet.requires_grad_(True)
            print("⚠ No LoRA config: U-Net is fully trainable (full fine-tune mode)")
        
        # Noise scheduler for training (DDPM)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        print("✓ Loaded DDPM noise scheduler")
    
    def encode_prompt(
        self,
        prompts: list[str],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode text prompts to embedding space using CLIP text encoder.
        
        Args:
            prompts: List of text prompts.
            device: Device to compute on (default: self.device).
            
        Returns:
            Prompt embeddings (batch_size, 77, 768).
        """
        device = device or self.device
        
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            embeddings = self.text_encoder(
                inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
        
        return embeddings.last_hidden_state
    
    def encode_images_to_latent(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to VAE latent space.
        
        Args:
            images: Image tensor (batch_size, 3, height, width) in [-1, 1].
            
        Returns:
            Latent tensor (batch_size, 4, height//8, width//8).
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # VAE scaling factor
        
        return latents
    
    def decode_latents_to_images(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode VAE latents back to image space.
        
        Args:
            latents: Latent tensor (batch_size, 4, height//8, width//8).
            
        Returns:
            Image tensor (batch_size, 3, height, width) in [-1, 1].
        """
        with torch.no_grad():
            latents = latents / 0.18215
            images = self.vae.decode(latents).sample
        
        return images
    
    def get_trainable_parameters(self):
        """
        Return iterator over trainable parameters (LoRA or full U-Net).
        
        Yields:
            Parameter tensors that require gradients.
        """
        for param in self.unet.parameters():
            if param.requires_grad:
                yield param
    
    def get_trainable_param_count(self) -> Tuple[int, int]:
        """
        Count trainable vs total parameters in U-Net.
        
        Returns:
            (trainable_count, total_count)
        """
        trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.unet.parameters())
        return trainable, total
    
    def to(self, device: str):
        """Move models to device."""
        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)
        self.device = device
        return self
    
    def __repr__(self) -> str:
        trainable, total = self.get_trainable_param_count()
        lora_info = f" (LoRA rank={self.lora_config.rank})" if self.lora_config else " (full fine-tune)"
        return (
            f"StableDiffusionXrayModel{lora_info}\n"
            f"  Text Encoder: frozen | VAE: frozen | U-Net: {trainable:,}/{total:,} trainable"
        )


if __name__ == "__main__":
    # Example: initialize model with LoRA
    config = LoRAConfig(rank=32, alpha=32)
    model = StableDiffusionXrayModel(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        lora_config=config,
    )
    print(model)
    print(f"\nTrainable params: {model.get_trainable_param_count()}")
