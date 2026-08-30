from __future__ import annotations

"""
Chỉ số cho bài toán multi-label 5 nhãn, dùng chung cho TRTR / TSTR / CAS.

Nguyên tắc: **AUC là chỉ số chính**, F1/accuracy chỉ là phụ.
AUC không phụ thuộc ngưỡng nên so sánh được giữa các mô hình; F1 thì phụ thuộc,
và trên nhãn thưa nó nói nhiều về ngưỡng hơn là về mô hình.
"""

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from dataset.nih_multilabel import LABEL_NAMES


# ---------------------------------------------------------------------------
# Ngưỡng
# ---------------------------------------------------------------------------

def calibrate_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                         grid: int = 199, verbose: bool = True) -> np.ndarray:
    """
    Với mỗi nhãn, quét ngưỡng và chọn giá trị tối đa hoá F1.

    Vì sao 0.5 gần như luôn sai: BCE trên nhãn thưa (positive rate ~15-25%)
    đẩy toàn bộ phân phối xác suất về phía thấp — mô hình "đúng" khi đoán 0.3
    cho một ca dương tính, vì đó vẫn là kỳ vọng hậu nghiệm hợp lý. Cắt ở 0.5
    biến gần hết dự đoán thành âm tính: F1 tụt thảm hại trong khi AUC không đổi.
    Ngưỡng PHẢI hiệu chỉnh trên tập validation THẬT, không bao giờ trên test.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidates = np.linspace(0.01, 0.99, grid)
    out = np.full(y_true.shape[1], 0.5, dtype=np.float64)

    for i, name in enumerate(LABEL_NAMES[: y_true.shape[1]]):
        if len(np.unique(y_true[:, i])) < 2:
            if verbose:
                print(f"  {name:<14} chỉ một lớp trong tập hiệu chỉnh -> giữ 0.5")
            continue
        f1s = [f1_score(y_true[:, i], (y_prob[:, i] >= c).astype(int), zero_division=0)
               for c in candidates]
        best = int(np.argmax(f1s))
        out[i] = float(candidates[best])
        if verbose:
            print(f"  {name:<14} ngưỡng={out[i]:.3f}  F1={f1s[best]:.4f}"
                  f"   (nếu dùng 0.5: F1={f1s[grid // 2]:.4f})")
    return out


# ---------------------------------------------------------------------------
# Đánh giá
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray,
             thresholds: Optional[Sequence[float]] = None,
             label_names: Sequence[str] = LABEL_NAMES) -> Dict:
    """
    Trả về dict đầy đủ chỉ số. `thresholds` None -> 0.5 cho mọi nhãn.

    Có kèm `hamming_baseline_all_negative`: điểm mà một mô hình đoán TOÀN ÂM
    TÍNH đạt được. Với positive rate ~25% thì baseline này là 0.75 — bất kỳ
    hamming accuracy nào dưới con số đó đều tệ hơn việc không làm gì cả. Luôn
    đọc hai số này cạnh nhau.
    """
    y_true = np.asarray(y_true).astype(np.int8)      # sklearn muốn nhãn nhị phân
    y_prob = np.asarray(y_prob, dtype=np.float64)
    k = y_true.shape[1]
    names = list(label_names[:k])

    th = (np.full(k, 0.5) if thresholds is None
          else np.asarray(thresholds, dtype=np.float64))
    y_pred = (y_prob >= th[None, :]).astype(np.int8)

    per_auc, per_f1 = {}, {}
    for i, name in enumerate(names):
        col = y_true[:, i]
        per_auc[name] = (float(roc_auc_score(col, y_prob[:, i]))
                         if len(np.unique(col)) > 1 else float("nan"))
        per_f1[name] = float(f1_score(col, y_pred[:, i], zero_division=0))

    valid = [v for v in per_auc.values() if not np.isnan(v)]
    pos_rate = y_true.mean()

    return {
        "n_samples": int(y_true.shape[0]),
        "macro_auc": float(np.mean(valid)) if valid else float("nan"),
        "per_label_auc": per_auc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label_f1": per_f1,
        "hamming_accuracy": float((y_true == y_pred).mean()),
        "hamming_baseline_all_negative": float(1.0 - pos_rate),
        "exact_match_ratio": float((y_true == y_pred).all(axis=1).mean()),
        "positive_rate_true": float(pos_rate),
        "positive_rate_pred": float(y_pred.mean()),
        "thresholds": {n: float(t) for n, t in zip(names, th)},
    }


# ---------------------------------------------------------------------------
# In ấn
# ---------------------------------------------------------------------------

def print_metrics(title: str, m: Dict) -> None:
    print(f"\n{'=' * 68}\n{title}   (n = {m['n_samples']:,})\n{'=' * 68}")
    print(f"macro-AUC          {m['macro_auc']:.4f}        <- CHỈ SỐ CHÍNH")
    print(f"macro-F1           {m['macro_f1']:.4f}")
    print(f"micro-F1           {m['micro_f1']:.4f}")
    print(f"hamming accuracy   {m['hamming_accuracy']:.4f} "
          f"(baseline đoán toàn âm: {m['hamming_baseline_all_negative']:.4f})")
    print(f"exact match        {m['exact_match_ratio']:.4f}")
    print(f"\n{'nhãn':<16}{'AUC':>9}{'F1':>9}{'ngưỡng':>9}")
    print("-" * 43)
    for name in m["per_label_auc"]:
        auc = m["per_label_auc"][name]
        print(f"{name:<16}{auc:>9.4f}{m['per_label_f1'][name]:>9.4f}"
              f"{m['thresholds'][name]:>9.3f}")


def print_comparison(results: Dict[str, Dict]) -> None:
    """So sánh nhiều giao thức cạnh nhau. `results` = {"TRTR": m1, "TSTR": m2, ...}"""
    names = list(results)
    if not names:
        return
    labels = list(next(iter(results.values()))["per_label_auc"])

    width = max(12, max(len(n) for n in names) + 2)
    print(f"\n{'=' * (18 + width * len(names))}")
    print("SO SÁNH  (macro-AUC và AUC từng nhãn)")
    print("=" * (18 + width * len(names)))
    print(f"{'':<18}" + "".join(f"{n:>{width}}" for n in names))
    print("-" * (18 + width * len(names)))
    print(f"{'macro-AUC':<18}" + "".join(f"{results[n]['macro_auc']:>{width}.4f}" for n in names))
    for lab in labels:
        print(f"{'  ' + lab:<18}"
              + "".join(f"{results[n]['per_label_auc'][lab]:>{width}.4f}" for n in names))
    print("-" * (18 + width * len(names)))
    print(f"{'macro-F1':<18}" + "".join(f"{results[n]['macro_f1']:>{width}.4f}" for n in names))
    print(f"{'n mẫu':<18}" + "".join(f"{results[n]['n_samples']:>{width},}" for n in names))

    if "TRTR" in results and "TSTR" in results:
        d = results["TSTR"]["macro_auc"] - results["TRTR"]["macro_auc"]
        rel = 100.0 * results["TSTR"]["macro_auc"] / results["TRTR"]["macro_auc"]
        print(f"\nTSTR - TRTR = {d:+.4f}  ({rel:.1f}% của TRTR)")
        print("Đây là con số quan trọng nhất: ảnh sinh giữ lại được bao nhiêu phần")
        print("giá trị huấn luyện của ảnh thật. Khoảng cách nhỏ = mô hình sinh tốt.")
