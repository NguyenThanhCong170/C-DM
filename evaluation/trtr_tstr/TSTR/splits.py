from __future__ import annotations

"""
Mở rộng `patient_level_split` (2-way: train/val) trong dataset/nih_multilabel.py
thành 3-way (train/val/test) để TRTR và TSTR có chung một test_real cố định,
không bao giờ đổi giữa 2 cách train.
"""

import random
from pathlib import Path
from typing import List, Tuple, Union

from dataset.nih_multilabel import read_data_entry


def patient_level_split_3way(
    csv_path: Union[str, Path],
    test_ratio: float = 0.2,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Tách Patient ID thành 3 tập rời nhau: (train_pids, val_pids, test_pids).

    Giữ đúng tinh thần patient_level_split gốc (shuffle bằng random.Random(seed)
    rồi cắt theo tỉ lệ) nhưng thêm test_ratio để có test_real riêng biệt, cố định
    xuyên suốt cả TRTR và TSTR.
    """
    rows = read_data_entry(csv_path)
    pids = sorted({str(r.get("Patient ID", "")).strip() for r in rows if r.get("Patient ID")})

    rng = random.Random(seed)
    rng.shuffle(pids)

    n_test = max(1, int(len(pids) * test_ratio))
    test_pids = pids[:n_test]
    rest = pids[n_test:]

    n_val = max(1, int(len(rest) * val_ratio))
    val_pids = rest[:n_val]
    train_pids = rest[n_val:]

    assert not (set(train_pids) & set(test_pids))
    assert not (set(val_pids) & set(test_pids))
    assert not (set(train_pids) & set(val_pids))

    return train_pids, val_pids, test_pids
