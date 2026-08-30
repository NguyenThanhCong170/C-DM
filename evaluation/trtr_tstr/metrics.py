from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from dataset.nih_multilabel import LABEL_NAMES


def macro_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    aucs = []
    for i in range(labels_np.shape[1]):
        if len(np.unique(labels_np[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(labels_np[:, i], probs[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_model(model, data_loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"]
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    preds = (probs >= threshold).astype(int)

    per_label_auc = {}
    for i, name in enumerate(LABEL_NAMES):
        if len(np.unique(labels_np[:, i])) < 2:
            per_label_auc[name] = float("nan")
        else:
            per_label_auc[name] = roc_auc_score(labels_np[:, i], probs[:, i])

    macro_auc = float(np.nanmean(list(per_label_auc.values())))
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    return {"per_label_auc": per_label_auc, "macro_auc": macro_auc, "macro_f1": macro_f1}


def _aggregate(runs: list[dict]) -> dict:
    macro_aucs = [r["macro_auc"] for r in runs]
    per_label = {
        label: float(np.mean([r["per_label_auc"][label] for r in runs]))
        for label in LABEL_NAMES
    }
    return {
        "macro_auc_mean": float(np.mean(macro_aucs)),
        "macro_auc_std": float(np.std(macro_aucs)),
        "per_label_auc_mean": per_label,
        "n_seeds": len(runs),
    }


def print_comparison(results: dict[str, list[dict]]) -> dict:
    """results = {"TRTR": [metrics_seed1, ...], "TSTR": [metrics_seed1, ...]}
    In bảng ra stdout, trả về dict đã tổng hợp (dùng lại được để lưu JSON/markdown)."""
    aggregated = {setup: _aggregate(runs) for setup, runs in results.items()}

    header = f"{'Setup':10s} {'macro-AUC (mean ± std)':25s} " + \
             " ".join(f"{l:>18s}" for l in LABEL_NAMES)
    print(header)
    for setup, agg in aggregated.items():
        mean_str = f"{agg['macro_auc_mean']:.4f} ± {agg['macro_auc_std']:.4f}"
        row = " ".join(f"{agg['per_label_auc_mean'][l]:18.4f}" for l in LABEL_NAMES)
        print(f"{setup:10s} {mean_str:25s} {row}")

    if "TRTR" in aggregated and "TSTR" in aggregated:
        delta = aggregated["TRTR"]["macro_auc_mean"] - aggregated["TSTR"]["macro_auc_mean"]
        print(f"\nΔ macro-AUC (TRTR - TSTR) = {delta:.4f}  "
              f"(càng gần 0 -> ảnh synthetic càng hữu dụng cho downstream task)")
        aggregated["delta_macro_auc"] = delta

    return aggregated
