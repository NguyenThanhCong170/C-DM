"""
Multi-Concept Preprocessing + PyTorch Dataset/DataLoader for DreamBooth-style fine-tuning
on chest X-ray images (Trains all pathologies into a single LoRA checkpoint).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# 1. Pixel-level preprocessing
# --------------------------------------------------------------------------


def percentile_normalize(arr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Cắt đuôi histogram [lo, hi] để tránh bẹp độ tương phản do các marker/vùng bão hòa sáng."""
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
    """Nạp ảnh X-quang, chuẩn hóa mức xám, resize Lanczos và nhân bản thành 3 kênh RGB cho VAE."""
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
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


# --------------------------------------------------------------------------
# 2. Multi-Concept DreamBooth Dataset (1-Checkpoint Unified Training)
# --------------------------------------------------------------------------


class DreamBoothXrayDataset(Dataset):
    """
    Dataset gom toàn bộ các bệnh (Multi-Concept) vào huấn luyện 1 checkpoint LoRA duy nhất.
    
    Cấu trúc `concepts_list` truyền vào:
    [
        {"instance_data_root": "./data/pneumonia",    "instance_prompt": "a chest x-ray of sks1 pneumonia"},
        {"instance_data_root": "./data/effusion",     "instance_prompt": "a chest x-ray of sks2 effusion"},
        {"instance_data_root": "./data/pneumothorax", "instance_prompt": "a chest x-ray of sks3 pneumothorax"},
        {"instance_data_root": "./data/atelectasis",  "instance_prompt": "a chest x-ray of sks4 atelectasis"},
        {"instance_data_root": "./data/cardiomegaly", "instance_prompt": "a chest x-ray of sks5 cardiomegaly"},
    ]
    """

    def __init__(
        self,
        tokenizer,
        concepts_list: Optional[List[Dict[str, str]]] = None,
        instance_data_root: Optional[Union[str, Path]] = None,
        instance_prompt: Optional[str] = None,
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

        # 1. Gom danh sách ảnh và prompt cho từng concept
        self.instance_items: List[Tuple[Path, str]] = []

        if concepts_list is not None:
            # Multi-concept mode: gom toàn bộ ảnh từ các thư mục bệnh khác nhau
            for concept in concepts_list:
                c_dir = Path(concept["instance_data_root"])
                c_prompt = concept["instance_prompt"]
                imgs = list_images(c_dir)
                if not imgs:
                    raise ValueError(f"Không tìm thấy ảnh trong thư mục: {c_dir}")
                for img_p in imgs:
                    self.instance_items.append((img_p, c_prompt))
            print(f"[Dataset] Đã nạp thành công {len(concepts_list)} concepts với tổng cộng {len(self.instance_items)} ảnh.")
        elif instance_data_root is not None and instance_prompt is not None:
            # Single-concept legacy mode
            imgs = list_images(instance_data_root)
            if not imgs:
                raise ValueError(f"Không tìm thấy ảnh trong thư mục: {instance_data_root}")
            for img_p in imgs:
                self.instance_items.append((img_p, instance_prompt))
        else:
            raise ValueError("Cần cung cấp concepts_list hoặc instance_data_root + instance_prompt.")

        # Xáo trộn toàn bộ mẫu để mỗi mini-batch nhận ngẫu nhiên các loại bệnh khác nhau
        random.shuffle(self.instance_items)
        self.num_instance_images = len(self.instance_items)

        # 2. Xử lý tập đối chứng Prior Preservation (Class Images chung)
        self.with_prior_preservation = class_data_root is not None
        if self.with_prior_preservation:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = list_images(self.class_data_root)
            if not self.class_images_path:
                raise ValueError(f"with_prior_preservation=True nhưng không thấy ảnh tại {class_data_root}")
            self.class_prompt = class_prompt or "a chest x-ray"
            self.num_class_images = len(self.class_images_path)
            print(f"[Dataset] Đã nạp {self.num_class_images} ảnh class cho Prior Preservation.")
        else:
            self.class_images_path = []
            self.num_class_images = 0

        self._length = max(self.num_instance_images, self.num_class_images)

        # 3. Pipeline xử lý ảnh
        transforms = [T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS)]
        if random_flip:
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

        # Lấy ảnh bệnh và prompt tương ứng của concept đó
        instance_path, prompt_text = self.instance_items[index % self.num_instance_images]
        example["instance_images"] = self._load_tensor(instance_path)
        example["instance_prompt_ids"] = self._tokenize(prompt_text)

        # Lấy ảnh đối chứng ngực chuẩn
        if self.with_prior_preservation:
            class_path = self.class_images_path[index % self.num_class_images]
            example["class_images"] = self._load_tensor(class_path)
            example["class_prompt_ids"] = self._tokenize(self.class_prompt)

        return example


# --------------------------------------------------------------------------
# 3. Collate function
# --------------------------------------------------------------------------


def collate_fn(examples: Sequence[Dict[str, torch.Tensor]], with_prior_preservation: bool = True) -> Dict[str, torch.Tensor]:
    """Ghép nối instance và class vào chung 1 batch tensor để U-Net forward 1 lần duy nhất."""
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]

    if with_prior_preservation and "class_images" in examples[0]:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack(input_ids)

    return {"pixel_values": pixel_values, "input_ids": input_ids}