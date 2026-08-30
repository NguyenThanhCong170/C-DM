from __future__ import annotations

"""
CAS — Classification Accuracy Score.

    python -m evaluation.compute_cas --config config/evaluation.yaml

Ý tưởng: lấy classifier đã train trên ảnh THẬT (chính là mô hình của TRTR) làm
"giám khảo", rồi hỏi từng ảnh SINH: ảnh này có thật sự mang cái nhãn mà ta đã
điều kiện cho mô hình sinh không? Đây là phép đo ĐỘ TRUNG THỰC NHÃN.

Trong ma trận 2x2 train/test:

                    test ảnh thật      test ảnh sinh
    train ảnh thật      TRTR               CAS
    train ảnh sinh      TSTR                -

Vì sao PHẢI đọc CAS cạnh TRTR: bản thân con số CAS không có thang đo. CAS = 0.82
là tốt hay tệ hoàn toàn phụ thuộc vào việc chính giám khảo đó đạt bao nhiêu trên
ảnh thật. Script này vì thế LUÔN chấm cả hai và in cạnh nhau.

Và vì sao CAS KHÔNG đủ một mình: nó mù trước mode collapse. Nếu mô hình chỉ sinh
đúng một ảnh Effusion hoàn hảo lặp lại 2000 lần, CAS gần như tuyệt đối, trong
khi TSTR sẽ sụp đổ vì classifier không có gì để học ngoài một mẫu duy nhất.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset.nih_multilabel import LABEL_NAMES, NIHMultiLabelDataset
from evaluation.classifier import build_model, make_loader, predict_probs
from evaluation.metrics import evaluate, print_comparison, print_metrics
from evaluation.splits import patient_level_split_3way
from evaluation.synthetic_dataset import SyntheticManifestDataset


def parse_args():
    p = argparse.ArgumentParser(description="CAS: giám khảo train trên ảnh thật chấm ảnh sinh")
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="mặc định out/eval/checkpoints/real_seed{seed}.pt")
    p.add_argument("--thresholds", default=None,
                   help="mặc định <checkpoint>_thresholds.json (hiệu chỉnh trên val thật)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--no-real-baseline", action="store_true",
                   help="bỏ qua việc chấm lại tập test thật (không khuyến khích)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ccfg = cfg["classifier"]
    device = cfg["device"]
    img_size = int(cfg["img_size"])
    seed = int(cfg["seed"])
    output_dir = Path(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"

    ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_dir / f"real_seed{seed}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Không thấy checkpoint giám khảo: {ckpt}\n"
            "Chạy `python -m evaluation.train_classifier --mode real` trước.")

    thr_path = Path(args.thresholds) if args.thresholds else \
        ckpt.with_name(ckpt.stem + "_thresholds.json")
    if thr_path.is_file():
        thr_map = json.loads(thr_path.read_text(encoding="utf-8"))
        thresholds = [float(thr_map.get(n, 0.5)) for n in LABEL_NAMES]
        print(f"[cas] ngưỡng hiệu chỉnh <- {thr_path}")
    else:
        thresholds = [0.5] * len(LABEL_NAMES)
        print(f"[cas] ⚠ không thấy {thr_path} — dùng 0.5 cho mọi nhãn. "
              "F1 sẽ bị đánh giá thấp; AUC không ảnh hưởng.")

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[cas] giám khảo <- {ckpt}")

    bs, nw = int(ccfg["batch_size"]), int(ccfg["num_workers"])
    use_amp = bool(ccfg["use_amp"])

    # --- chấm ảnh SINH ---------------------------------------------------
    manifest = Path(args.manifest) if args.manifest else \
        Path(cfg["generation"]["outdir"]) / "manifest.csv"
    syn_ds = SyntheticManifestDataset(
        manifest, size=img_size,
        use_percentile_norm=bool(cfg.get("synthetic_percentile_norm", True)))
    syn_prob, syn_true = predict_probs(model, make_loader(syn_ds, bs, nw), device, use_amp)
    cas = evaluate(syn_true, syn_prob, thresholds)
    cas["protocol"] = "CAS"
    cas["judge_checkpoint"] = str(ckpt)
    cas["manifest"] = str(manifest)
    print_metrics("CAS  —  giám khảo (train ảnh thật) chấm ảnh SINH", cas)

    report = {"CAS": cas}

    # --- baseline: cùng giám khảo chấm ảnh THẬT --------------------------
    if not args.no_real_baseline:
        data_root = Path(cfg["data_root"])
        csv_path = Path(cfg["csv_path"]) if cfg.get("csv_path") else data_root / "Data_Entry_2017.csv"
        _, _, test_pids = patient_level_split_3way(
            csv_path, test_ratio=cfg["split"]["test_ratio"],
            val_ratio=cfg["split"]["val_ratio"], seed=seed)
        real_ds = NIHMultiLabelDataset(
            data_root, csv_path, size=img_size, patient_ids=test_pids,
            max_per_label=None,
            max_images=(cfg.get("eval_sets") or {}).get("test_max_images"),
            cache_dir=cfg.get("cache_dir"), seed=seed, verbose=False)
        real_prob, real_true = predict_probs(model, make_loader(real_ds, bs, nw), device, use_amp)
        base = evaluate(real_true, real_prob, thresholds)
        base["protocol"] = "TRTR (baseline cùng giám khảo)"
        report["TRTR"] = base
        print_metrics("BASELINE  —  cùng giám khảo chấm ảnh THẬT (tập test)", base)

        print_comparison({"ảnh THẬT": base, "ảnh SINH": cas})
        gap = cas["macro_auc"] - base["macro_auc"]
        print(f"\nCAS - baseline = {gap:+.4f}")
        print("Gần 0  -> ảnh sinh mang nhãn rõ ràng ngang ảnh thật.")
        print("Âm nhiều -> ảnh sinh không thể hiện được bệnh đã điều kiện.")
        print("Dương nhiều -> ảnh sinh 'dễ' bất thường: mô hình sinh có thể đang lặp")
        print("               vài mẫu điển hình quá sạch. Kiểm tra chéo bằng TSTR.")

    out = output_dir / "cas.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
