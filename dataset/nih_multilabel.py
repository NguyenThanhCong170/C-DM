'''
Multilabel dataset: Sử dụng multi-hot encoding 
'''


from __future__ import annotations
import csv
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# Tiền xử lý mức pixel

def percentile_normalize(arr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Cắt đuôi histogram [lo, hi] rồi trải về 0-255. Min-max thuần bị marker
    'L'/'R' và viền collimator chiếm hết dải sáng, làm bẹt tương phản nhu mô."""
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
    """Đọc về mảng 2D float32, giữ nguyên dải động gốc (kể cả PNG 16-bit)."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I", "F"):
        arr = np.asarray(img)
    else:
        arr = np.asarray(img.convert("L"))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def preprocess_xray_to_rgb(
    path: Union[str, Path],
    size: int = 512,
    use_percentile_norm: bool = True,
    percentile_lo: float = 0.5,
    percentile_hi: float = 99.5,
) -> Image.Image:
    """Chuẩn hóa mức xám, resize LANCZOS, nhân bản 3 kênh cho VAE."""
    arr = load_grayscale_array(path)
    arr8 = (percentile_normalize(arr, percentile_lo, percentile_hi)
            if use_percentile_norm else _to_uint8_minmax(arr))
    img = Image.fromarray(arr8, mode="L").resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


# Ánh xạ nhãn
LABEL_NAMES: Tuple[str, ...] = ("No Finding", "Infiltration", "Effusion", "Atelectasis", "Others")
NIH_14 = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
)

_EXPLICIT = {"Infiltration": 1, "Effusion": 2, "Atelectasis": 3}
OTHERS_MEMBERS = tuple(n for n in NIH_14 if n not in _EXPLICIT)

# Chuyển categorical sang multi-hot
def finding_string_to_multihot(finding: str) -> np.ndarray:
    vec = np.zeros(len(LABEL_NAMES), dtype=np.float32)
    parts = [p.strip() for p in finding.split("|") if p.strip()]
    if not parts or parts == ["No Finding"]:
        vec[0] = 1.0
        return vec
    for p in parts:
        if p == "No Finding":
            continue
        idx = _EXPLICIT.get(p)
        if idx is not None:
            vec[idx] = 1.0
        else:
            vec[4] = 1.0
    if vec.sum() == 0:
        vec[0] = 1.0
    return vec


def index_image_files(root: Union[str, Path]) -> Dict[str, Path]:
    """
    Trả về map {tên_file: đường_dẫn}
    """
    root = Path(root)
    mapping: Dict[str, Path] = {}
    candidates: List[Path] = []

    for sub in sorted(root.glob("images_*")):
        inner = sub / "images"
        candidates.append(inner if inner.is_dir() else sub)
    if (root / "images").is_dir():
        candidates.append(root / "images")
    if not candidates:
        candidates.append(root)

    for folder in candidates:
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in IMG_EXTENSIONS:
                mapping.setdefault(p.name, p)
    return mapping


def read_data_entry(csv_path: Union[str, Path]) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# Note: Cần patient id vì chia theo patient"
class NIHMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        csv_path: Optional[Union[str, Path]] = None,
        size: int = 512,
        view_position: Optional[str] = None,
        max_per_label: Optional[Dict[str, int]] = None,
        max_images: Optional[int] = None,
        patient_ids: Optional[Sequence[str]] = None,
        exclude_patient_ids: Optional[Sequence[str]] = None,
        use_percentile_norm: bool = True,
        cache_dir: Optional[Union[str, Path]] = None, #Lưu trữ ảnh đã được xử lý
        seed: int = 42,
        verbose: bool = True,
    ):
        self.size = int(size)
        self.use_percentile_norm = use_percentile_norm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        data_root = Path(data_root)
        csv_path = Path(csv_path) if csv_path else data_root / "Data_Entry_2017.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Không thấy {csv_path}")

        file_index = index_image_files(data_root)
        if not file_index:
            raise FileNotFoundError(f"Không thấy ảnh nào dưới {data_root}")

        if view_position is not None:
            _vp = str(view_position).strip().upper()
            if _vp in ("", "NONE", "NULL"):
                view_position = None            # YAML viết None/"" -> coi như không lọc
            elif _vp not in ("PA", "AP"):
                raise ValueError(
                    f"view_position phải là 'PA', 'AP' hoặc null — nhận {view_position!r}. "
                    "Trong YAML, null viết là `null` hoặc `~`, KHÔNG phải `None`.")

        rows = read_data_entry(csv_path)
        rng = random.Random(seed)

        keep_ids = set(map(str, patient_ids)) if patient_ids is not None else None
        drop_ids = set(map(str, exclude_patient_ids)) if exclude_patient_ids else set()

        items: List[Tuple[Path, np.ndarray, str]] = []
        missing = 0

        # Item = [(path, labels, patient id)]
        for r in rows:
            name = r.get("Image Index")
            pid = str(r.get("Patient ID", "")).strip()
            if keep_ids is not None and pid not in keep_ids:
                continue
            if pid in drop_ids:
                continue
            if view_position:
                vp = (r.get("View Position") or "").strip().upper()
                if vp != view_position.strip().upper():
                    continue
            path = file_index.get(name)
            if path is None:
                missing += 1
                continue
            items.append((path, finding_string_to_multihot(r.get("Finding Labels", "")), pid))

        if not items:
            raise RuntimeError("Không có mẫu nào sau khi lọc")

        # Check maximum theo label
        if max_per_label:
            items = self._apply_caps(items, max_per_label, rng)

        rng.shuffle(items)

        # Check maximum theo số ảnh
        if max_images is not None:
            items = items[:max_images]

        # Chia theo labels, paths, patient_ids
        self.items = items
        self.labels = np.stack([lab for _, lab, _ in items])          # (N, 5)
        self.paths = [p for p, _, _ in items]
        self.patient_ids = [pid for _, _, pid in items]

        # Chuẩn hóa, đưa vào tensor
        transforms: List[object] = []
        transforms += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.image_transforms = T.Compose(transforms)

        # In ra để kiểm tra
        if verbose:
            counts = self.labels.sum(axis=0).astype(int)
            print(f"[Dataset] {len(self.items):,} ảnh"
                  + (f" (bỏ {missing:,} dòng CSV không thấy file)" if missing else ""))
            for name, c in zip(LABEL_NAMES, counts):
                print(f"[Dataset]   {name:<13} {c:>7,}  ({100 * c / len(self.items):5.2f}%)")
            multi = int((self.labels[:, 1:].sum(axis=1) >= 2).sum())
            print(f"[Dataset]   đồng mắc >=2 bệnh: {multi:,}")

    # Giữ ảnh nhiều nhãn thay vì ảnh ít nhãn
    @staticmethod
    def _apply_caps(items, max_per_label: Dict[str, int], rng: random.Random):
        caps = {LABEL_NAMES.index(k): v for k, v in max_per_label.items() if k in LABEL_NAMES}
        if not caps:
            return items
        order = list(range(len(items)))
        rng.shuffle(order)                     
        # ảnh nhiều nhãn xếp trước -> được giữ, ảnh đơn nhãn bị cắt trước
        order.sort(key=lambda i: -items[i][1].sum())
        used = Counter()
        kept = []
        for i in order:
            lab = items[i][1]
            active = np.nonzero(lab)[0]
            if any(idx in caps and used[idx] >= caps[idx] for idx in active):
                continue
            kept.append(items[i])
            for idx in active:
                used[idx] += 1
        return kept

    def __len__(self) -> int:
        return len(self.items)

    def _cached_path(self, path: Path) -> Path:
        return self.cache_dir / f"{path.stem}_{self.size}.png"

    def _save_cache(self, img: Image.Image, cached: Path) -> None:
        """
        Ghi cache NGUYÊN TỬ: ra file tạm rồi os.replace().

        Với num_workers > 1 và WeightedRandomSampler (lấy mẫu có hoàn lại), hai
        worker có thể cùng ghi một file. `Image.save()` thẳng vào đích không nguyên
        tử, nên worker khác đọc phải file ghi dở -> UnidentifiedImageError giữa run.
        os.replace() là nguyên tử trên cùng filesystem, nên file đích luôn hoặc
        chưa tồn tại, hoặc hoàn chỉnh.
        """
        tmp = cached.with_name(f".{cached.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
        try:
            img.convert("L").save(tmp, format="PNG", optimize=False)
            os.replace(tmp, cached)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _load_image(self, path: Path):
        if self.cache_dir is not None:
            cached = self._cached_path(path)
            if cached.is_file():
                try:
                    return Image.open(cached).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    pass        # cache hỏng từ lần chạy trước -> dựng lại từ ảnh gốc
            img = preprocess_xray_to_rgb(path, size=self.size,
                                         use_percentile_norm=self.use_percentile_norm)
            self._save_cache(img, cached)
            return img
        return preprocess_xray_to_rgb(path, size=self.size,
                                      use_percentile_norm=self.use_percentile_norm)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label, _ = self.items[index]
        img = self._load_image(path)
        return {
            "pixel_values": self.image_transforms(img),
            "labels": torch.from_numpy(label.copy()),
        }

    def label_counts(self) -> np.ndarray:
        return self.labels.sum(axis=0)

    def sample_weights(self, mode: str = "inverse_freq", beta: float = 0.5) -> torch.Tensor:
        """
        Trọng số cho WeightedRandomSampler. Với multi-label, trọng số của một ảnh =
        trung bình nghịch đảo tần suất các nhãn đang bật, làm mềm bằng lũy thừa
        `beta` (beta=1 cân bằng hoàn toàn, beta=0 giữ nguyên phân phối gốc; 0.5 là
        thỏa hiệp — vẫn thấy đủ ảnh No Finding để học giải phẫu bình thường).
        """
        counts = np.maximum(self.labels.sum(axis=0), 1.0)
        inv = (counts.max() / counts) ** beta                     # (K,)
        active = self.labels                                      # (N,K)
        num_active = np.maximum(active.sum(axis=1, keepdims=True), 1.0)
        if mode == "max":
            w = (active * inv).max(axis=1)
        else:
            w = (active * inv).sum(axis=1) / num_active.squeeze(1)
        return torch.from_numpy(w.astype(np.float64))

    def make_balanced_sampler(self, num_samples: Optional[int] = None,
                              beta: float = 0.5) -> WeightedRandomSampler:
        w = self.sample_weights(beta=beta)
        return WeightedRandomSampler(w, num_samples=num_samples or len(self), replacement=True)




def collate_multilabel(examples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    labels = torch.stack([e["labels"] for e in examples]).float()
    return {"pixel_values": pixel_values, "labels": labels}


def patient_level_split(csv_path: Union[str, Path], val_ratio: float = 0.05,
                        seed: int = 42) -> Tuple[List[str], List[str]]:
    """Tách theo Patient ID — cùng một bệnh nhân không được nằm cả hai bên."""
    rows = read_data_entry(csv_path)
    pids = sorted({str(r.get("Patient ID", "")).strip() for r in rows if r.get("Patient ID")})
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_ratio))
    return pids[n_val:], pids[:n_val]
