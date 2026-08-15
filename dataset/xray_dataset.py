"""
Preprocessing + PyTorch Dataset/DataLoader for DreamBooth-style fine-tuning
on chest X-ray images (built for NIH ChestX-ray14, but works on any folder
of single-channel medical X-ray images).

Two pieces:
  1. `preprocess_xray_to_rgb` / `percentile_normalize` — the actual pixel
     pipeline: percentile intensity windowing -> LANCZOS resize -> grayscale
     duplicated into 3 channels. Idempotent-safe to run on images that were
     already preprocessed this way (as in the original notebook's offline
     preprocessing step) — re-applying it is a near no-op.
  2. `DreamBoothXrayDataset` — dual sampling of *instance* images (the
     target pathology, e.g. "sks pneumonia") and *class* images (prior
     preservation, e.g. real "No Finding" images), with a `collate_fn` that
     concatenates both into one batch so a single forward pass produces both
     the instance and the prior-preservation predictions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# Pixel-level preprocessing
# --------------------------------------------------------------------------


def percentile_normalize(arr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Clip `arr` to its [lo, hi] percentile range and rescale to uint8 [0, 255].

    Chest X-rays frequently have heavy-tailed intensity histograms (e.g. a
    few near-saturated pixels from portable-machine markers or borders);
    windowing on percentiles instead of hard min/max keeps those outliers
    from crushing the contrast of the anatomy that actually matters.
    """
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    if p_hi <= p_lo:
        return arr.astype(np.uint8)
    clipped = np.clip(arr, p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255.0).astype(np.uint8)


def preprocess_xray_to_rgb(
    path: Union[str, Path],
    size: int = 512,
    use_percentile_norm: bool = True,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
) -> Image.Image:
    """Load an X-ray image file and return a preprocessed RGB PIL.Image.

    Steps: force single-channel ("L") -> optional percentile intensity
    normalization -> LANCZOS resize to (size, size), no cropping (X-ray
    source images are already square; cropping would risk cutting into the
    costophrenic angles / lung apices, exactly where many findings live) ->
    duplicate the single channel into 3 identical RGB channels (the VAE
    expects 3-channel input; duplicating avoids inventing color information
    a colormap would).
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)

    if use_percentile_norm:
        arr = percentile_normalize(arr, percentile_lo, percentile_hi)

    img = Image.fromarray(arr, mode="L").resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


def list_images(directory: Union[str, Path]) -> List[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def build_image_transform(size: int) -> T.Compose:
    """Tensor pipeline applied *after* `preprocess_xray_to_rgb`: resize
    safety-net (no-op if already `size`), to-tensor, normalize to [-1, 1]
    (matches the range `vae.encode` expects for SD's AutoencoderKL).
    """
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


# --------------------------------------------------------------------------
# DreamBooth dataset: dual sampling of instance + class(prior) images
# --------------------------------------------------------------------------


class DreamBoothXrayDataset(Dataset):
    """Dual-sampling dataset for DreamBooth-style fine-tuning.

    * Instance data: images of the target pathology, all paired with the
      single fixed `instance_prompt` (e.g. "a chest x-ray of sks pneumonia").
    * Class data (optional, for prior preservation): real "normal" images,
      all paired with `class_prompt` (e.g. "a chest x-ray").

    `__len__` is `max(num_instance, num_class)` so one epoch is defined by
    the larger of the two sets; the smaller set is cycled with modulo
    indexing (standard DreamBooth behavior — this is what makes prior
    preservation "free" in terms of extra epochs).

    Each `__getitem__` call re-runs `preprocess_xray_to_rgb` from the raw
    file (percentile norm + LANCZOS resize) rather than assuming the files
    on disk are already exactly `size x size` — safe/no-op if they are.
    """

    def __init__(
        self,
        instance_data_root: Union[str, Path],
        instance_prompt: str,
        tokenizer,
        class_data_root: Optional[Union[str, Path]] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        use_percentile_norm: bool = True,
        random_flip: bool = False,
        tokenizer_max_length: Optional[int] = None,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length or getattr(tokenizer, "model_max_length", 77)
        self.use_percentile_norm = use_percentile_norm

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.is_dir():
            raise FileNotFoundError(f"instance_data_root does not exist: {instance_data_root}")
        self.instance_images_path = list_images(self.instance_data_root)
        if not self.instance_images_path:
            raise ValueError(f"No images found under {instance_data_root}")
        self.instance_prompt = instance_prompt
        self.num_instance_images = len(self.instance_images_path)

        self.with_prior_preservation = class_data_root is not None
        if self.with_prior_preservation:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = list_images(self.class_data_root)
            if not self.class_images_path:
                raise ValueError(
                    f"with_prior_preservation=True but no images found under {class_data_root}. "
                    "Generate/copy class images first (see generate_class_images in the training script)."
                )
            self.class_prompt = class_prompt or "a photo"
            self.num_class_images = len(self.class_images_path)
        else:
            self.class_images_path = []
            self.num_class_images = 0

        self._length = max(self.num_instance_images, self.num_class_images)

        transforms = [T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS)]
        if random_flip:
            # NOTE: default False on purpose. Horizontal flip mirrors a chest
            # X-ray's heart position (situs inversus is <0.01% of the real
            # population) -- it silently teaches the model anatomy that
            # isn't real. Only enable this if that's a tradeoff you want.
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms += [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        self.image_transforms = T.Compose(transforms)

    def __len__(self) -> int:
        return self._length

    def _load_tensor(self, path: Path) -> torch.Tensor:
        img = preprocess_xray_to_rgb(path, size=self.size, use_percentile_norm=self.use_percentile_norm)
        return self.image_transforms(img)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example: Dict[str, torch.Tensor] = {}

        instance_path = self.instance_images_path[index % self.num_instance_images]
        example["instance_images"] = self._load_tensor(instance_path)
        example["instance_prompt_ids"] = self._tokenize(self.instance_prompt)

        if self.with_prior_preservation:
            class_path = self.class_images_path[index % self.num_class_images]
            example["class_images"] = self._load_tensor(class_path)
            example["class_prompt_ids"] = self._tokenize(self.class_prompt)

        return example


def collate_fn(examples: Sequence[Dict[str, torch.Tensor]], with_prior_preservation: bool) -> Dict[str, torch.Tensor]:
    """Stack a list of `DreamBoothXrayDataset` examples into a batch.

    When prior preservation is on, instance and class tensors are
    concatenated along the batch dimension (instance first, then class) —
    this is the standard DreamBooth trick: one forward pass computes both
    the instance and the prior predictions, and the training loop later
    slices the batch back in half to apply `prior_loss_weight` separately.
    """
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]

    if with_prior_preservation:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack(input_ids)

    return {"pixel_values": pixel_values, "input_ids": input_ids}


class PromptDataset(Dataset):
    """Trivial (prompt, index) dataset — used only when generating extra
    class images on the fly (PRIOR_MODE="generated" in the original
    notebook) to batch the generation loop via a DataLoader."""

    def __init__(self, prompt: str, num_samples: int):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        return {"prompt": self.prompt, "index": index}