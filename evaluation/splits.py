from __future__ import annotations

"""Chia dữ liệu NIH theo BỆNH NHÂN thành 3 phần train / val / test."""

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
    """
    Trả về (train_pids, val_pids, test_pids).

    Vì sao chia theo Patient ID chứ không theo ảnh: NIH có ~30k bệnh nhân cho
    112k ảnh, trung bình ~3.7 ảnh/người. Chia ngẫu nhiên theo ảnh sẽ đặt ảnh
    chụp tháng 1 và tháng 6 của cùng một người vào hai phía — classifier học
    thuộc giải phẫu người đó (hình dạng xương sườn, sẹo cũ) rồi ăn điểm trên
    test. macro-AUC bị thổi lên vài điểm mà không phản ánh khả năng tổng quát.

    val_ratio tính trên phần CÒN LẠI sau khi cắt test, không phải trên toàn bộ.
    """
    rows = read_data_entry(csv_path)
    pids = sorted({str(r.get("Patient ID", "")).strip() for r in rows if r.get("Patient ID")})
    if not pids:
        raise RuntimeError(f"Không đọc được Patient ID nào từ {csv_path}")

    rng = random.Random(seed)
    rng.shuffle(pids)

    n_test = max(1, int(len(pids) * test_ratio))
    test_pids = pids[:n_test]
    rest = pids[n_test:]

    n_val = max(1, int(len(rest) * val_ratio))
    val_pids = rest[:n_val]
    train_pids = rest[n_val:]

    assert not (set(train_pids) & set(val_pids))
    assert not (set(train_pids) & set(test_pids))
    assert not (set(val_pids) & set(test_pids))
    return train_pids, val_pids, test_pids
