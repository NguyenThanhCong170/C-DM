from __future__ import annotations

"""
Dataset đọc ảnh synthetic do evaluation/generate_synthetic.py xuất ra.

FORMAT THẬT của metadata.csv (đã đối chiếu với evaluation/generate_synthetic.py,
KHÔNG còn là hợp đồng tạm định nghĩa nữa):

    <output_dir>/<version_tag>/
    ├── <combo_name>/*.png
    └── metadata.csv   -- cột: filepath, combo, seed, "gt_No Finding",
                           "gt_Infiltration", "gt_Effusion", "gt_Atelectasis",
                           "gt_Others" (tên cột nhãn có tiền tố "gt_", đúng thứ
                           tự dataset.nih_multilabel.LABEL_NAMES, giá trị 0/1 —
                           có thể là nhãn mềm trong [0,1] nếu sinh bằng vector
                           không phải one-hot).

`filepath` trong CSV là đường dẫn TƯƠNG ĐỐI so với thư mục CHỨA metadata.csv
(không phải tương đối CWD lúc chạy script) — _read_manifest() tự resolve theo
đúng quy ước này.
"""

import csv
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler

from dataset.nih_multilabel import LABEL_NAMES, preprocess_xray_to_rgb


def _read_manifest(manifest_path: Union[str, Path]) -> list[dict]:
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{manifest_path} rỗng.")

    label_cols = [f"gt_{name}" for name in LABEL_NAMES]
    missing_cols = [c for c in ["filepath", *label_cols] if c not in rows[0]]
    if missing_cols:
        raise ValueError(
            f"{manifest_path} thiếu cột {missing_cols}.\n"
            f"Cần đủ: filepath, {label_cols}\n"
            "-> Đây có phải metadata.csv do evaluation/generate_synthetic.py xuất ra không? "
            "Nếu bạn tự tạo file thủ công, đặt đúng tên cột như trên."
        )

    base_dir = manifest_path.parent
    for r in rows:
        r["_resolved_path"] = base_dir / r["filepath"]
    return rows


class SyntheticManifestDataset(Dataset):
    """Cùng interface với NIHMultiLabelDataset (__getitem__ trả {"pixel_values","labels"},
    có make_balanced_sampler) để classifier.train_one() dùng chung được cho cả TRTR/TSTR."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        size: int = 512,
        # Mặc định KHÔNG percentile-normalize: hàm này vốn để sửa dải sáng lệch của ảnh
        # X-quang DICOM 16-bit thật; ảnh synthetic đã ở dạng 8-bit "bình thường" sau khi
        # decode qua VAE nên áp lại percentile normalize dễ làm lệch tương phản không cần thiết.
        # Đổi thành True nếu muốn khớp tuyệt đối pipeline tiền xử lý ảnh thật.
        use_percentile_norm: bool = False,
        verbose: bool = True,
    ):
        rows = _read_manifest(manifest_path)
        self.size = size
        self.use_percentile_norm = use_percentile_norm
        self.paths = [r["_resolved_path"] for r in rows]
        self.labels = np.array(
            [[float(r[f"gt_{name}"]) for name in LABEL_NAMES] for r in rows], dtype=np.float32
        )
        self.image_transforms = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

        if verbose:
            counts = self.labels.sum(axis=0).astype(int)
            print(f"[SyntheticDataset] {len(self.paths):,} ảnh synthetic (từ {manifest_path})")
            for name, c in zip(LABEL_NAMES, counts):
                print(f"[SyntheticDataset]   {name:<13} {c:>7,}  ({100 * c / len(self.paths):5.2f}%)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = preprocess_xray_to_rgb(self.paths[index], size=self.size,
                                     use_percentile_norm=self.use_percentile_norm)
        label = torch.from_numpy(self.labels[index].copy())
        return {"pixel_values": self.image_transforms(img), "labels": label}

    def sample_weights(self, mode: str = "inverse_freq", beta: float = 0.5) -> torch.Tensor:
        counts = np.maximum(self.labels.sum(axis=0), 1.0)
        inv = (counts.max() / counts) ** beta
        active = self.labels
        num_active = np.maximum(active.sum(axis=1, keepdims=True), 1.0)
        w = (active * inv).sum(axis=1) / num_active.squeeze(1)
        return torch.from_numpy(w.astype(np.float64))

    def make_balanced_sampler(self, num_samples: Optional[int] = None,
                              beta: float = 0.5) -> WeightedRandomSampler:
        w = self.sample_weights(beta=beta)
        return WeightedRandomSampler(w, num_samples=num_samples or len(self), replacement=True)
