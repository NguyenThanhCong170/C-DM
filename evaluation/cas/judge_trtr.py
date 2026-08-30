from __future__ import annotations

"""
Judge dùng chính classifier TRTR (DenseNet-121, 5 lớp) thay cho torchxrayvision.

Vì sao tốt hơn cho CAS:
  * Không gian nhãn KHỚP CHÍNH XÁC 5 chiều của mô hình sinh — không phải gộp
    thủ công 18 nhãn về 5, không có nhãn ngưỡng NaN bị âm tính âm thầm.
  * Tiền xử lý dùng LẠI `preprocess_xray_to_rgb` — đúng hàm mà classifier thấy
    lúc train, nên ảnh synthetic đi qua cùng một đường ống.
  * `No Finding` là một đầu ra thật, không phải suy ra từ "không bệnh nào vượt ngưỡng".

Ngưỡng: BCE trên dữ liệu mất cân bằng khiến 0.5 thường quá cao. Chạy
    python -m evaluation.cas.judge_trtr --calibrate --config <cas.yaml>
để hiệu chỉnh ngưỡng tối ưu F1 trên tập validation THẬT, ghi ra thresholds.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torchvision.transforms as T

from dataset.nih_multilabel import LABEL_NAMES, preprocess_xray_to_rgb

from ..trtr_tstr.classifier import build_model


class TRTRJudge:
    """Cùng interface với `evaluation.cas.judge.Judge` để compute_cas.py dùng thay được."""

    def __init__(
        self,
        checkpoint: Union[str, Path],
        img_size: int = 512,
        thresholds: Optional[Union[str, Path, Dict[str, float]]] = None,
        threshold_overrides: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        use_percentile_norm: bool = True,
    ):
        self.device = device
        self.img_size = int(img_size)
        self.use_percentile_norm = use_percentile_norm
        self.pathologies = list(LABEL_NAMES)

        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Không thấy checkpoint TRTR: {ckpt}\n"
                "Chạy `python -m evaluation.trtr_tstr.compute_trtr --config "
                "config/evaluation/trtr_tstr.yaml` trước."
            )
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(device).eval()
        print(f"[judge-trtr] {ckpt.name} | {len(self.pathologies)} nhãn: {self.pathologies}")

        self.thresholds = self._resolve_thresholds(thresholds, threshold_overrides)
        print("[judge-trtr] ngưỡng: "
              + ", ".join(f"{n}={t:.3f}" for n, t in zip(self.pathologies, self.thresholds)))

        # ĐÚNG pipeline mà classifier thấy lúc train (xem NIHMultiLabelDataset)
        self._to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    # ------------------------------------------------------------------
    def _resolve_thresholds(self, thresholds, overrides) -> np.ndarray:
        out = np.full(len(self.pathologies), 0.5, dtype=np.float64)
        if isinstance(thresholds, (str, Path)):
            p = Path(thresholds)
            if p.is_file():
                thresholds = json.loads(p.read_text(encoding="utf-8"))
                print(f"[judge-trtr] nạp ngưỡng hiệu chỉnh từ {p}")
            else:
                print(f"[judge-trtr] ⚠ không thấy {p} — dùng 0.5 cho mọi nhãn. "
                      f"Chạy `--calibrate` để hiệu chỉnh.")
                thresholds = None
        for src in (thresholds, overrides):
            for name, value in (src or {}).items():
                if name in self.pathologies:
                    out[self.pathologies.index(name)] = float(value)
        return out

    def _preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        img = preprocess_xray_to_rgb(image_path, size=self.img_size,
                                     use_percentile_norm=self.use_percentile_norm)
        return self._to_tensor(img).unsqueeze(0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, image_paths: Sequence[Union[str, Path]]) -> np.ndarray:
        """(N, 5) xác suất. Nhận cả một ảnh lẫn một lô — lô nhanh hơn nhiều."""
        batch = torch.cat([self._preprocess(p) for p in image_paths]).to(self.device)
        return torch.sigmoid(self.model(batch)).cpu().numpy()

    def predict(self, image_path: Union[str, Path]) -> dict:
        probs = self.predict_probs([image_path])[0]
        return {
            name: {"prob": float(p), "positive": bool(p >= t)}
            for name, p, t in zip(self.pathologies, probs, self.thresholds)
        }

    def predict_subset(self, image_path: Union[str, Path], labels: List[str]) -> dict:
        full = self.predict(image_path)
        missing = [l for l in labels if l not in full]
        if missing:
            raise ValueError(
                f"Nhãn {missing} không nằm trong {self.pathologies}. "
                "Kiểm tra `target_labels` trong config."
            )
        return {l: full[l] for l in labels}


# ----------------------------------------------------------------------
# Hiệu chỉnh ngưỡng trên tập validation THẬT
# ----------------------------------------------------------------------

def calibrate_thresholds(judge: TRTRJudge, dataset, batch_size: int = 32,
                         grid: int = 199) -> Dict[str, float]:
    """
    Với mỗi nhãn, quét ngưỡng và chọn giá trị tối đa hoá F1 trên dữ liệu THẬT.

    0.5 gần như luôn quá cao: BCE trên nhãn thưa đẩy xác suất về thấp, nên ngưỡng
    cố định làm mô hình dưới-dự-đoán và F1 tụt dù AUC vẫn tốt.
    """
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader

    from dataset.nih_multilabel import collate_multilabel

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_multilabel, num_workers=2)
    probs, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(judge.device)
            probs.append(torch.sigmoid(judge.model(x)).cpu().numpy())
            trues.append(batch["labels"].numpy())
    probs, trues = np.concatenate(probs), np.concatenate(trues)

    candidates = np.linspace(0.01, 0.99, grid)
    result = {}
    for i, name in enumerate(judge.pathologies):
        if len(np.unique(trues[:, i])) < 2:
            result[name] = 0.5
            print(f"  {name:<14} chỉ một lớp trong tập val -> giữ 0.5")
            continue
        f1s = [f1_score(trues[:, i], (probs[:, i] >= c).astype(int), zero_division=0)
               for c in candidates]
        best = int(np.argmax(f1s))
        result[name] = float(candidates[best])
        print(f"  {name:<14} ngưỡng={candidates[best]:.3f}  F1={f1s[best]:.4f}  "
              f"(so với 0.5: F1={f1s[grid // 2]:.4f})")
    return result


def _main_calibrate():
    import argparse

    import yaml

    from dataset.nih_multilabel import NIHMultiLabelDataset
    from ..trtr_tstr.TSTR.splits import patient_level_split_3way

    p = argparse.ArgumentParser(description="Hiệu chỉnh ngưỡng judge TRTR trên val thật")
    p.add_argument("--config", default="config/evaluation/cas.yaml")
    p.add_argument("--trtr-config", default="config/evaluation/trtr_tstr.yaml")
    args = p.parse_args()

    cas_cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    trtr_cfg = yaml.safe_load(open(args.trtr_config, encoding="utf-8"))
    jc = cas_cfg["judge"]

    judge = TRTRJudge(
        checkpoint=jc["checkpoint"],
        img_size=jc.get("img_size", trtr_cfg.get("img_size", 512)),
        thresholds=None,
        device=cas_cfg["model"]["device"],
    )

    data_root = Path(trtr_cfg["data_root"])
    csv_path = data_root / "Data_Entry_2017.csv"
    _, val_pids, _ = patient_level_split_3way(
        csv_path, test_ratio=trtr_cfg["test_ratio"],
        val_ratio=trtr_cfg["val_ratio"], seed=trtr_cfg["seed"])
    val_ds = NIHMultiLabelDataset(data_root, csv_path, size=judge.img_size,
                                  patient_ids=val_pids, max_per_label=None,
                                  cache_dir=trtr_cfg.get("cache_dir"),
                                  seed=trtr_cfg["seed"], verbose=False)
    print(f"[calibrate] {len(val_ds):,} ảnh THẬT trong tập validation")

    thresholds = calibrate_thresholds(judge, val_ds)
    out = Path(jc.get("thresholds_path", "out/trtr/checkpoints/thresholds.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calibrate] đã ghi {out}")


if __name__ == "__main__":
    _main_calibrate()
