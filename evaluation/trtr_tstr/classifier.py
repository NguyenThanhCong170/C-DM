from __future__ import annotations

import gc
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121

from dataset.nih_multilabel import LABEL_NAMES, collate_multilabel

from .metrics import evaluate_model, macro_auc_from_logits

NUM_CLASSES = len(LABEL_NAMES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model


def _raw(model: nn.Module) -> nn.Module:
    """Gỡ wrapper nn.DataParallel (nếu có) để state_dict không dính tiền tố 'module.'"""
    return model.module if isinstance(model, nn.DataParallel) else model


def train_one(
    run_name: str,
    train_ds,
    val_ds,
    test_ds,
    output_dir: Path,
    balance_beta: float = 0.5,
    batch_size: int = 16,
    num_workers: int = 2,
    epochs: int = 15,
    lr: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    use_amp: bool = True,
    use_data_parallel: bool = True,
) -> dict:
   
    set_seed(seed)
    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"[{run_name}] device={device} | GPU khả dụng={n_gpu}")

    sampler = train_ds.make_balanced_sampler(beta=balance_beta)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=True,
                               collate_fn=collate_multilabel, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_multilabel, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_multilabel, persistent_workers=False)

    model = build_model(pretrained=pretrained).to(device)
    if use_data_parallel and n_gpu > 1:
        print(f"[{run_name}] Bọc nn.DataParallel để dùng cả {n_gpu} GPU")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_loss = float("inf")
    ckpt_path = checkpoint_dir / f"{run_name}.pt"
    log_lines = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_outputs.append(outputs.float().cpu())
                all_labels.append(labels.cpu())
        val_loss /= len(val_ds)
        val_auc = macro_auc_from_logits(torch.cat(all_outputs), torch.cat(all_labels))
        lr_sched.step(val_loss)

        elapsed = time.time() - t0
        line = (f"[{run_name}] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_macro_auc={val_auc:.4f}  ({elapsed:.0f}s)")
        print(line)
        log_lines.append(line)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(_raw(model).state_dict(), ckpt_path)

        del all_outputs, all_labels
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (logs_dir / f"{run_name}.log").write_text("\n".join(log_lines))

    eval_model = build_model(pretrained=False).to(device)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate_model(eval_model, test_loader, device)
    test_metrics["run_name"] = run_name
    test_metrics["seed"] = seed
    print(f"[{run_name}] TEST macro-AUC = {test_metrics['macro_auc']:.4f}")
    return test_metrics
