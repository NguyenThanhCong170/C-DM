from __future__ import annotations

"""
DenseNet-121 5 nhãn + vòng train/eval DÙNG CHUNG cho TRTR và TSTR.

Điểm mấu chốt về mặt phương pháp: hai giao thức chỉ khác nhau ở MỘT biến duy
nhất — nguồn ảnh train (thật hay sinh). Kiến trúc, optimizer, lịch lr, sampler
cân bằng, số epoch, tập val, tập test đều giống hệt. Nếu để lệch bất kỳ thứ gì
khác thì hiệu số TSTR − TRTR không còn quy được về chất lượng ảnh sinh.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121

from dataset.nih_multilabel import LABEL_NAMES, collate_multilabel
from evaluation.metrics import calibrate_thresholds, evaluate, print_metrics

NUM_CLASSES = len(LABEL_NAMES)


# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """DenseNet-121 với đầu ra num_classes logit (KHÔNG sigmoid — dùng BCEWithLogits).

    Pretrained ImageNet vẫn hữu ích cho X-quang dù miền ảnh khác hẳn: các tầng
    đầu học cạnh/kết cấu, thứ dùng chung được. Đây cũng là khởi tạo của CheXNet.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _raw(model: nn.Module) -> nn.Module:
    """Gỡ nn.DataParallel để state_dict không dính tiền tố 'module.'"""
    return model.module if isinstance(model, nn.DataParallel) else model


def _make_scaler(enabled: bool):
    try:                                   # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):    # torch 2.1 - 2.2
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str,
                  use_amp: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (probs, labels), cả hai shape (N, 5)."""
    model.eval()
    probs, trues = [], []
    amp_on = use_amp and str(device).startswith("cuda")
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(batch["labels"].numpy())
    return np.concatenate(probs), np.concatenate(trues)


def make_loader(ds, batch_size: int, num_workers: int, sampler=None,
                shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers, collate_fn=collate_multilabel,
        pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------

def train_one(
    run_name: str,
    train_ds,
    val_ds,
    test_ds,
    output_dir: Path,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    balance_beta: float = 0.5,
    pretrained: bool = True,
    use_amp: bool = True,
    early_stop_patience: int = 4,
    use_data_parallel: bool = True,
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    """
    Train, chọn checkpoint tốt nhất theo macro-AUC trên VAL, hiệu chỉnh ngưỡng
    trên VAL, rồi đánh giá một lần duy nhất trên TEST.

    Không bao giờ chạm vào TEST trước bước cuối — chọn epoch hay ngưỡng theo
    test là rò rỉ, và con số báo cáo sẽ lạc quan giả.
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    thr_path = ckpt_dir / f"{run_name}_thresholds.json"

    n_gpu = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    print(f"\n[{run_name}] device={device}  GPU={n_gpu}  "
          f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    sampler = train_ds.make_balanced_sampler(beta=balance_beta)
    train_loader = make_loader(train_ds, batch_size, num_workers, sampler=sampler)
    val_loader = make_loader(val_ds, batch_size, num_workers)
    test_loader = make_loader(test_ds, batch_size, num_workers)

    model = build_model(pretrained=pretrained).to(device)
    if use_data_parallel and n_gpu > 1:
        print(f"[{run_name}] bọc nn.DataParallel trên {n_gpu} GPU")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(train_loader)))
    amp_on = use_amp and str(device).startswith("cuda")
    scaler = _make_scaler(amp_on)

    best_auc, best_epoch, bad_epochs = -1.0, -1, 0
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        t_epoch = time.perf_counter()

        for batch in train_loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.detach())
            n_batch += 1

        train_loss = running / max(1, n_batch)
        val_prob, val_true = predict_probs(model, val_loader, device, use_amp)
        val_m = evaluate(val_true, val_prob)          # ngưỡng 0.5, chỉ để theo dõi
        val_auc = val_m["macro_auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auc": val_auc})

        mark = ""
        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            torch.save(_raw(model).state_dict(), ckpt_path)
            mark = "  <- tốt nhất, đã lưu"
        else:
            bad_epochs += 1

        print(f"[{run_name}] epoch {epoch:>2}/{epochs}  loss={train_loss:.4f}  "
              f"val macro-AUC={val_auc:.4f}  ({time.perf_counter()-t_epoch:.0f}s){mark}")

        if early_stop_patience and bad_epochs >= early_stop_patience:
            print(f"[{run_name}] dừng sớm: {bad_epochs} epoch không cải thiện")
            break

    print(f"[{run_name}] xong sau {(time.perf_counter()-t_start)/60:.1f} phút, "
          f"epoch tốt nhất = {best_epoch} (val macro-AUC {best_auc:.4f})")

    # --- nạp lại checkpoint tốt nhất -------------------------------------
    eval_model = build_model(pretrained=False).to(device)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- hiệu chỉnh ngưỡng trên VAL --------------------------------------
    print(f"\n[{run_name}] hiệu chỉnh ngưỡng trên tập VAL thật:")
    val_prob, val_true = predict_probs(eval_model, val_loader, device, use_amp)
    thresholds = calibrate_thresholds(val_true, val_prob)
    thr_path.write_text(
        json.dumps({n: float(t) for n, t in zip(LABEL_NAMES, thresholds)}, indent=2),
        encoding="utf-8")

    # --- đánh giá TEST một lần duy nhất ----------------------------------
    test_prob, test_true = predict_probs(eval_model, test_loader, device, use_amp)
    metrics = evaluate(test_true, test_prob, thresholds)
    metrics["run_name"] = run_name
    metrics["best_epoch"] = best_epoch
    metrics["best_val_macro_auc"] = float(best_auc)
    metrics["history"] = history
    metrics["checkpoint"] = str(ckpt_path)
    metrics["n_train"] = len(train_ds)

    print_metrics(f"{run_name}  —  đánh giá trên TEST thật", metrics)
    return metrics
