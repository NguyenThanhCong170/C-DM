from __future__ import annotations

"""
Dataset đọc ảnh synthetic từ manifest.csv do `generate_eval_set.py` sinh ra.

Cùng interface với `NIHMultiLabelDataset` (__getitem__ trả {"pixel_values",
"labels"}, có make_balanced_sampler) để `classifier.train_one` dùng được cả hai
mà không cần rẽ nhánh.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler

from dataset.nih_multilabel import LABEL_NAMES, preprocess_xray_to_rgb


def balanced_sampler_from_labels(labels: np.ndarray, beta: float = 0.5,
                                 num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Bản sao logic sample_weights của NIHMultiLabelDataset, để hai nguồn dữ
    liệu được cân bằng y hệt nhau — nếu khác, so sánh TRTR/TSTR mất công bằng."""
    counts = np.maximum(labels.sum(axis=0), 1.0)
    inv = (counts.max() / counts) ** beta
    num_active = np.maximum(labels.sum(axis=1, keepdims=True), 1.0)
    w = (labels * inv).sum(axis=1) / num_active.squeeze(1)
    return WeightedRandomSampler(torch.from_numpy(w.astype(np.float64)),
                                 num_samples=num_samples or len(labels),
                                 replacement=True)


class SyntheticManifestDataset(Dataset):
    """
    manifest.csv: cột `file` (đường dẫn tương đối so với thư mục chứa manifest,
    hoặc tuyệt đối) + 5 cột nhãn đúng tên trong LABEL_NAMES.

    Nhãn ở đây là nhãn ĐIỀU KIỆN đã đưa vào mô hình sinh — tức là nhãn mà ảnh
    "lẽ ra phải có". CAS chính là phép đo xem ảnh có thật sự mang nhãn đó không.
    """

    def __init__(
        self,
        manifest: Union[str, Path],
        size: int = 512,
        use_percentile_norm: bool = True,
        max_images: Optional[int] = None,
        verbose: bool = True,
    ):
        manifest = Path(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Không thấy manifest: {manifest}\n"
                "Chạy `python -m evaluation.generate_eval_set --config config/evaluation.yaml` trước."
            )
        self.root = manifest.parent
        self.size = int(size)
        self.use_percentile_norm = use_percentile_norm

        paths: List[Path] = []
        labels: List[np.ndarray] = []
        with open(manifest, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing_cols = [c for c in ("file",) + tuple(LABEL_NAMES) if c not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"manifest thiếu cột {missing_cols}; có {reader.fieldnames}")
            for row in reader:
                p = Path(row["file"])
                if not p.is_absolute():
                    p = self.root / p
                if not p.is_file():
                    continue
                paths.append(p)
                labels.append(np.array([float(row[n]) for n in LABEL_NAMES], dtype=np.float32))

        if not paths:
            raise RuntimeError(f"Không có ảnh nào tồn tại trong {manifest}")
        if max_images is not None:
            paths, labels = paths[:max_images], labels[:max_images]

        self.paths = paths
        self.labels = np.stack(labels)                    # (N, 5)
        self.image_transforms = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

        if verbose:
            counts = self.labels.sum(axis=0).astype(int)
            print(f"[Synthetic] {len(self.paths):,} ảnh từ {manifest}")
            for name, c in zip(LABEL_NAMES, counts):
                print(f"[Synthetic]   {name:<13} {c:>7,}  ({100 * c / len(self.paths):5.2f}%)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img = preprocess_xray_to_rgb(self.paths[index], size=self.size,
                                     use_percentile_norm=self.use_percentile_norm)
        return {
            "pixel_values": self.image_transforms(img),
            "labels": torch.from_numpy(self.labels[index].copy()),
        }

    def label_counts(self) -> np.ndarray:
        return self.labels.sum(axis=0)

    def make_balanced_sampler(self, num_samples: Optional[int] = None,
                              beta: float = 0.5) -> WeightedRandomSampler:
        return balanced_sampler_from_labels(self.labels, beta=beta, num_samples=num_samples)
