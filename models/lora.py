"""
LoRA (Low-Rank Adaptation) modules for efficient fine-tuning of Stable Diffusion U-Net.

LoRA injects low-rank trainable adapters into attention/projection layers while keeping
the base model frozen. This drastically reduces trainable parameters while maintaining
good fine-tuning performance.

Paper: https://arxiv.org/abs/2106.09714
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection into model layers."""
    
    rank: int = 4
    """Inner dimension of LoRA matrices (A, B)."""
    
    alpha: float = 1.0
    """Scaling factor for LoRA outputs. Effective LR = (alpha/rank) * base_lr."""
    
    target_modules: List[str] = field(default_factory=lambda: DEFAULT_TARGET_MODULES)
    """Which module names to inject LoRA into (substring matching)."""
    
    lora_dropout: float = 0.0
    """Dropout rate applied to LoRA inputs."""
    
    bias: str = "none"
    """Whether to train biases: "none", "all", or "lora_only"."""
    
    modules_to_save: Optional[List[str]] = None
    """Optionally save (and fine-tune) whole modules alongside LoRA."""
    
    inference_mode: bool = False
    """If True, merge LoRA into the base model weights (reduces memory, no gradient tracking)."""


DEFAULT_TARGET_MODULES = [
    "to_q",  # Query projection in attention
    "to_k",  # Key projection in attention
    "to_v",  # Value projection in attention
    "to_out.0",  # Output projection in attention
]


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear layer.
    
    Wraps a frozen nn.Linear with trainable low-rank matrices A ∈ ℝ^(r × in),
    B ∈ ℝ^(out × r). Output = base(x) + scale * B @ A @ x
    
    This keeps base weights frozen while training only O(2 * rank * in * out) parameters
    instead of O(in * out) for a full fine-tune.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base frozen linear layer (wrapped, not created)
        self.base_layer = None  # Will be set after wrapping
        
        # LoRA matrices
        self.lora_a = nn.Parameter(
            torch.randn(rank, in_features) / (in_features**0.5)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        
        # Dropout and bias
        self.lora_dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA adaptation.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Base forward pass (frozen)
        base_out = self.base_layer(x)
        
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_out = self.lora_dropout(x) @ self.lora_a.T  # (..., rank)
        lora_out = lora_out @ self.lora_b.T  # (..., out_features)
        lora_out = lora_out * self.scaling
        
        # Add LoRA bias if applicable
        if self.bias is not None:
            lora_out = lora_out + self.bias
        
        return base_out + lora_out


class LoRAWrapper:
    """
    Helper class to wrap a nn.Linear layer with LoRA adaptation.
    Replaces the forward method while keeping the original layer accessible.
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        self.linear_layer = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        
        # Store original forward
        self.original_forward = linear_layer.forward
    
    def apply(self) -> LoRALinear:
        """Create a LoRA-adapted version of the linear layer."""
        lora_linear = LoRALinear(
            in_features=self.linear_layer.in_features,
            out_features=self.linear_layer.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            bias=self.bias and (self.linear_layer.bias is not None),
        )
        
        # Attach original layer as base_layer
        lora_linear.base_layer = self.linear_layer
        
        # Copy original bias if applicable
        if self.linear_layer.bias is not None and lora_linear.bias is not None:
            with torch.no_grad():
                lora_linear.bias.copy_(self.linear_layer.bias)
        
        return lora_linear


def inject_lora_into_unet(
    unet: nn.Module,
    config: LoRAConfig,
) -> Dict[str, LoRALinear]:
    """
    Inject LoRA adapters into a Stable Diffusion U-Net.
    
    Recursively finds all nn.Linear layers matching target_modules and wraps them
    with LoRA. Freezes the base model.
    
    Args:
        unet: The Stable Diffusion U-Net module.
        config: LoRAConfig specifying rank, alpha, target modules, etc.
        
    Returns:
        Dictionary of injected LoRA layers for reference/checkpointing.
    """
    lora_layers = {}
    
    # Freeze all base parameters
    for param in unet.parameters():
        param.requires_grad = False
    
    # Recursively inject LoRA
    def inject_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this is a Linear layer to adapt
            if isinstance(child, nn.Linear):
                # Check if name matches any target module
                if any(target in full_name for target in config.target_modules):
                    # Replace with LoRA-wrapped version
                    lora_linear = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.lora_dropout,
                        bias=child.bias is not None,
                    )
                    
                    lora_linear.base_layer = child
                    
                    # Copy original bias if exists
                    if child.bias is not None:
                        with torch.no_grad():
                            lora_linear.bias.copy_(child.bias)
                    
                    # Replace the layer in parent module
                    setattr(module, name, lora_linear)
                    
                    lora_layers[full_name] = lora_linear
                    print(f"✓ Injected LoRA into {full_name} (rank={config.rank}, α={config.alpha})")
            else:
                # Recurse into child modules
                inject_recursive(child, full_name)
    
    inject_recursive(unet)
    
    print(f"\n✓ Successfully injected LoRA into {len(lora_layers)} layers")
    return lora_layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract all trainable LoRA parameters from a model.
    
    Args:
        model: Model containing LoRA layers.
        
    Returns:
        List of LoRA parameter tensors.
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_a)
            params.append(module.lora_b)
            if module.bias is not None:
                params.append(module.bias)
    return params


def save_lora_weights(
    model: nn.Module,
    output_path: str,
    save_base: bool = False,
) -> None:
    """
    Save LoRA weights to disk using safetensors format.
    
    Args:
        model: Model with LoRA layers.
        output_path: Path to save weights (e.g., 'model.safetensors').
        save_base: If True, also save base model weights (large, usually not needed).
    """
    state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Save LoRA matrices
            state_dict[f"{name}.lora_a"] = module.lora_a.data.cpu()
            state_dict[f"{name}.lora_b"] = module.lora_b.data.cpu()
            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.data.cpu()
    
    # Optionally save base layer weights (only LoRA-adapted ones)
    if save_base:
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and module.base_layer is not None:
                state_dict[f"{name}.base_weight"] = module.base_layer.weight.data.cpu()
                if module.base_layer.bias is not None:
                    state_dict[f"{name}.base_bias"] = module.base_layer.bias.data.cpu()
    
    save_file(state_dict, output_path)
    print(f"✓ Saved {len(state_dict)} tensors to {output_path}")


def load_lora_weights(
    model: nn.Module,
    checkpoint_path: str,
) -> None:
    """
    Load LoRA weights from disk.
    
    Args:
        model: Model with LoRA layers to load weights into.
        checkpoint_path: Path to saved LoRA weights.
    """
    state_dict = load_file(checkpoint_path)
    
    # Flatten model state to match checkpoint
    model_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            model_state[f"{name}.lora_a"] = module.lora_a
            model_state[f"{name}.lora_b"] = module.lora_b
            if module.bias is not None:
                model_state[f"{name}.bias"] = module.bias
    
    # Load weights
    for key, value in state_dict.items():
        if key in model_state:
            model_state[key].data.copy_(value)
            print(f"✓ Loaded {key}")
        else:
            print(f"⚠ Checkpoint key {key} not found in model")
    
    print(f"✓ Loaded {len([k for k in state_dict.keys() if k in model_state])} weights")


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adapters into the base model weights permanently.
    Useful for inference when you want a single model file without separate LoRA weights.
    
    Args:
        model: Model with LoRA layers.
        
    Returns:
        Model with LoRA merged (LoRA layers become regular Linear layers).
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge: W_merged = W_base + (α/r) * B @ A
            merged_weight = module.base_layer.weight.data.clone()
            merged_weight += module.scaling * module.lora_b @ module.lora_a
            
            # Create merged linear layer
            merged_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.base_layer.bias is not None,
            )
            merged_linear.weight.data.copy_(merged_weight)
            
            if module.base_layer.bias is not None:
                merged_linear.bias.data.copy_(module.base_layer.bias)
            
            # Replace LoRA layer with merged linear
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            parent_module = model
            for part in parent_name.split("."):
                if part:
                    parent_module = getattr(parent_module, part)
            
            setattr(parent_module, child_name, merged_linear)
            print(f"✓ Merged LoRA into {name}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("LoRA module loaded successfully!")
    print(f"Default target modules: {DEFAULT_TARGET_MODULES}")
