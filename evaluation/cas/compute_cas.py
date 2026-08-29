import argparse
import csv
import json
import os

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from evaluation.cas.judge import Judge
from evaluation.cas.metrics import (
    bootstrap_ci,
    derive_no_finding,
    exact_match_ratio,
    hamming_accuracy,
    per_label_metrics,
)
from evaluation.common.versioning import resolve_version_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/evaluation/cas.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target_labels = cfg["target_labels"]
    gen = cfg["generation"]

    version_tag = resolve_version_tag(cfg)
    metadata_path = os.path.join(gen["output_dir"], version_tag, gen.get("metadata_filename", "metadata.csv"))
    results_dir = os.path.join(cfg["evaluation"]["results_dir"], version_tag)
    os.makedirs(results_dir, exist_ok=True)
    print(f"[CAS] version_tag = '{version_tag}' -> đọc {metadata_path}, ghi kết quả vào {results_dir}/")

    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(
            f"Không có dữ liệu trong {metadata_path}. "
            "Chạy `python -m evaluation.generate_synthetic --config <cùng file config>` trước."
        )

    base_dir = os.path.dirname(metadata_path)

    judge = Judge(
        model_name=cfg["judge"]["model_name"],
        use_op_threshs=cfg["judge"]["use_op_threshs"],
        threshold_overrides=cfg["judge"].get("threshold_overrides") or {},
        device=cfg["model"]["device"],
    )

    y_true, y_prob, y_pred, gt_no_finding = [], [], [], []
    detail_rows = []

    for row in tqdm(rows, desc="Chấm điểm bằng judge model"):
        image_path = os.path.join(base_dir, row["filepath"])
        pred = judge.predict_subset(image_path, target_labels)

        gt_vec = [int(round(float(row[f"gt_{lbl}"]))) for lbl in target_labels]
        prob_vec = [pred[lbl]["prob"] for lbl in target_labels]
        pred_vec = [int(pred[lbl]["positive"]) for lbl in target_labels]

        y_true.append(gt_vec)
        y_prob.append(prob_vec)
        y_pred.append(pred_vec)
        
        gt_no_finding.append(int(round(float(row["gt_No Finding"]))))

        detail_rows.append(
            {
                "filepath": row["filepath"],
                "combo": row["combo"],
                **{f"gt_{lbl}": v for lbl, v in zip(target_labels, gt_vec)},
                **{f"prob_{lbl}": v for lbl, v in zip(target_labels, prob_vec)},
                **{f"pred_{lbl}": v for lbl, v in zip(target_labels, pred_vec)},
            }
        )

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    gt_no_finding = np.array(gt_no_finding)
    pred_no_finding = derive_no_finding(y_pred)
    for i, r in enumerate(detail_rows):
        r["gt_no_finding"] = int(gt_no_finding[i])
        r["pred_no_finding"] = int(pred_no_finding[i])

    report = {
        "n_images": len(rows),
        "target_labels": target_labels,
        "hamming_accuracy": hamming_accuracy(y_true, y_pred),
        "exact_match_ratio": exact_match_ratio(y_true, y_pred),
        "per_label": per_label_metrics(y_true, y_prob, y_pred, target_labels),
        "no_finding": {"accuracy": float((gt_no_finding == pred_no_finding).mean())},
    }

    n_boot = cfg["evaluation"].get("bootstrap_iterations", 1000)
    for i, lbl in enumerate(target_labels):
        if len(np.unique(y_true[:, i])) == 2:
            lo, hi = bootstrap_ci(
                y_true[:, i], y_prob[:, i],
                lambda yt, yp: roc_auc_score(yt, yp),
                n_iterations=n_boot,
            )
            report["per_label"][lbl]["auc_roc_ci95"] = [lo, hi]

    report_path = os.path.join(results_dir, "cas_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    detail_path = os.path.join(results_dir, "cas_predictions.csv")
    fieldnames = list(detail_rows[0].keys())
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"[CAS] Hamming accuracy : {report['hamming_accuracy']:.4f}")
    print(f"[CAS] Exact match ratio: {report['exact_match_ratio']:.4f}")
    print(f"[CAS] No-finding acc.  : {report['no_finding']['accuracy']:.4f}")
    for lbl, m in report["per_label"].items():
        auc_str = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "N/A"
        print(f"[CAS]   {lbl:15s} AUC={auc_str}  F1={m['f1']:.4f}  (n+={m['n_positive']}/{m['n_total']})")
    print(f"[CAS] Báo cáo đầy đủ: {report_path}")
    print(f"[CAS] Chi tiết từng ảnh: {detail_path}")


if __name__ == "__main__":
    main()
