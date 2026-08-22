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
# 1. Tiền xử lý mức pixel
# --------------------------------------------------------------------------


def percentile_normalize(arr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Cắt đuôi histogram [lo, hi] để tránh bẹp tương phản do marker / vùng bão hòa sáng."""
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    if p_hi <= p_lo:
        return _to_uint8_minmax(arr)
    clipped = np.clip(arr, p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255.0).round().astype(np.uint8)


def _to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max <= a_min:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - a_min) / (a_max - a_min) * 255.0).round().astype(np.uint8)


def load_grayscale_array(path: Union[str, Path]) -> np.ndarray:
    """Đọc ảnh về mảng 2D float32, giữ nguyên dải động gốc (kể cả PNG 16-bit)."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I", "F"):
        arr = np.asarray(img)  # giữ nguyên 16-bit, không để PIL cắt về 8-bit
    else:
        arr = np.asarray(img.convert("L"))
    if arr.ndim == 3:  # RGBA/RGB lọt qua
        arr = arr[..., 0]
    return arr.astype(np.float32)


def preprocess_xray_to_rgb(
    path: Union[str, Path],
    size: int = 512,
    use_percentile_norm: bool = True,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
) -> Image.Image:
    """Nạp ảnh X-quang, chuẩn hóa mức xám, resize LANCZOS, nhân bản 3 kênh cho VAE."""
    arr = load_grayscale_array(path)
    arr8 = (
        percentile_normalize(arr, percentile_lo, percentile_hi)
        if use_percentile_norm
        else _to_uint8_minmax(arr)
    )
    img = Image.fromarray(arr8, mode="L").resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


def list_images(directory: Union[str, Path]) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


# --------------------------------------------------------------------------
# 2. Multi-Concept DreamBooth Dataset
# --------------------------------------------------------------------------


class DreamBoothXrayDataset(Dataset):
    """
    Gom toàn bộ các bệnh (multi-concept) để train chung 1 checkpoint LoRA.

    concepts_list:
    [
        {"instance_data_root": "./data/pneumonia", "instance_prompt": "a chest x-ray of sks pneumonia"},
        {"instance_data_root": "./data/effusion",  "instance_prompt": "a chest x-ray of ohwx effusion"},
        ...
    ]

    Lưu ý y khoa: `random_flip` mặc định tắt. Lật ngang ảnh ngực làm tim đổi sang
    bên phải (dextrocardia giả), gây nhiễu khi học cardiomegaly.
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
        seed: int = 42,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length or getattr(tokenizer, "model_max_length", 77)
        self.use_percentile_norm = use_percentile_norm

        # 1. Gom ảnh + prompt của từng concept
        self.instance_items: List[Tuple[Path, str]] = []

        if concepts_list:
            for concept in concepts_list:
                c_dir = Path(concept["instance_data_root"])
                c_prompt = concept["instance_prompt"]
                imgs = list_images(c_dir)
                if not imgs:
                    raise ValueError(f"Không tìm thấy ảnh trong thư mục: {c_dir}")
                for img_p in imgs:
                    self.instance_items.append((img_p, c_prompt))
                print(f"[Dataset]   concept '{c_prompt}': {len(imgs)} ảnh")
            print(f"[Dataset] {len(concepts_list)} concepts, tổng {len(self.instance_items)} ảnh.")
        elif instance_data_root is not None and instance_prompt is not None:
            imgs = list_images(instance_data_root)
            if not imgs:
                raise ValueError(f"Không tìm thấy ảnh trong thư mục: {instance_data_root}")
            for img_p in imgs:
                self.instance_items.append((img_p, instance_prompt))
            print(f"[Dataset] single-concept: {len(imgs)} ảnh.")
        else:
            raise ValueError("Cần cung cấp concepts_list hoặc (instance_data_root + instance_prompt).")

        # Trộn để mini-batch nhận ngẫu nhiên nhiều loại bệnh — RNG riêng để tái lập.
        random.Random(seed).shuffle(self.instance_items)
        self.num_instance_images = len(self.instance_items)

        # 2. Tập đối chứng cho Prior Preservation
        self.with_prior_preservation = class_data_root is not None
        if self.with_prior_preservation:
            self.class_images_path = list_images(class_data_root)
            if not self.class_images_path:
                raise ValueError(f"with_prior_preservation=True nhưng không thấy ảnh tại {class_data_root}")
            self.class_prompt = class_prompt or "a chest x-ray"
            self.num_class_images = len(self.class_images_path)
            print(f"[Dataset] {self.num_class_images} ảnh class cho Prior Preservation.")
        else:
            self.class_images_path = []
            self.num_class_images = 0

        self._length = max(self.num_instance_images, self.num_class_images)

        # 3. Pipeline biến đổi ảnh (resize đã làm trong preprocess)
        transforms: List[object] = []
        if random_flip:
            print("[Dataset] ⚠ random_flip đang BẬT — cân nhắc tắt với ảnh X-quang ngực.")
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
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

        instance_path, prompt_text = self.instance_items[index % self.num_instance_images]
        example["instance_images"] = self._load_tensor(instance_path)
        example["instance_prompt_ids"] = self._tokenize(prompt_text)

        if self.with_prior_preservation:
            class_path = self.class_images_path[index % self.num_class_images]
            example["class_images"] = self._load_tensor(class_path)
            example["class_prompt_ids"] = self._tokenize(self.class_prompt)

        return example


# --------------------------------------------------------------------------
# 3. Collate
# --------------------------------------------------------------------------


def collate_fn(
    examples: Sequence[Dict[str, torch.Tensor]],
    with_prior_preservation: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Ghép instance và class vào chung 1 batch để U-Net forward một lần.

    Thứ tự: [instance_0..instance_{B-1}, class_0..class_{B-1}] — train.py dựa vào
    đúng thứ tự này để tách nửa đầu / nửa sau khi tính prior loss.
    """
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]

    if with_prior_preservation:
        if "class_images" not in examples[0]:
            raise ValueError("with_prior_preservation=True nhưng dataset không sinh class_images")
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack(input_ids)

    return {"pixel_values": pixel_values, "input_ids": input_ids}