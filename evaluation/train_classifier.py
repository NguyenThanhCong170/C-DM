from __future__ import annotations

"""
Tinh chỉnh DenseNet-121 thành 5 nhãn, từ ảnh THẬT hoặc ảnh SINH.

    # TRTR — train ảnh thật, test ảnh thật
    python -m evaluation.train_classifier --config config/evaluation.yaml --mode real

    # TSTR — train ảnh sinh, test ảnh thật (cùng tập test)
    python -m evaluation.train_classifier --config config/evaluation.yaml --mode synthetic

Tập VAL và TEST LUÔN là ảnh thật ở cả hai chế độ. Đó là điểm mấu chốt: TSTR chỉ
có nghĩa khi thước đo cuối cùng là dữ liệu thật.
"""

import argparse
import json
from pathlib import Path

import yaml

from dataset.nih_multilabel import NIHMultiLabelDataset
from evaluation.classifier import train_one
from evaluation.splits import patient_level_split_3way
from evaluation.synthetic_dataset import SyntheticManifestDataset

PROTOCOL = {"real": "TRTR", "synthetic": "TSTR"}


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier cho TRTR hoặc TSTR")
    p.add_argument("--config", default="config/evaluation.yaml")
    p.add_argument("--mode", required=True, choices=["real", "synthetic"],
                   help="nguồn ảnh TRAIN. val/test luôn là ảnh thật")
    p.add_argument("--seed", type=int, default=None, help="ghi đè seed")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--manifest", default=None,
                   help="đường dẫn manifest.csv (mặc định lấy từ generation.outdir)")
    p.add_argument("--limit-train", type=int, default=None,
                   help="giới hạn số ảnh TRAIN. Dùng để so sánh CÔNG BẰNG: tập thật "
                        "có ~41k ảnh còn tập sinh chỉ 10k, nên TRTR mặc định được lợi "
                        "về số lượng. Chạy thêm TRTR với --limit-train 10000 để tách "
                        "riêng ảnh hưởng của chất lượng ảnh khỏi ảnh hưởng của cỡ dữ liệu.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ccfg = dict(cfg["classifier"])

    seed = int(args.seed if args.seed is not None else cfg["seed"])
    if args.epochs is not None:
        ccfg["epochs"] = args.epochs
    if args.batch_size is not None:
        ccfg["batch_size"] = args.batch_size

    data_root = Path(cfg["data_root"])
    csv_path = Path(cfg["csv_path"]) if cfg.get("csv_path") else data_root / "Data_Entry_2017.csv"
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    img_size = int(cfg["img_size"])

    train_pids, val_pids, test_pids = patient_level_split_3way(
        csv_path, test_ratio=cfg["split"]["test_ratio"],
        val_ratio=cfg["split"]["val_ratio"], seed=cfg["seed"])
    print(f"Bệnh nhân: train={len(train_pids):,}  val={len(val_pids):,}  test={len(test_pids):,}")
    (output_dir / "split_pids.json").write_text(
        json.dumps({"train": train_pids, "val": val_pids, "test": test_pids}), encoding="utf-8")

    def real_ds(pids, caps=None, verbose=True, max_images=None):
        return NIHMultiLabelDataset(
            data_root, csv_path, size=img_size, patient_ids=pids,
            max_per_label=caps, max_images=max_images, cache_dir=cfg.get("cache_dir"),
            seed=cfg["seed"], verbose=verbose)

    esets = cfg.get("eval_sets") or {}
    print("\n--- tập VAL (thật) ---")
    val_ds = real_ds(val_pids, max_images=esets.get("val_max_images"))
    print("\n--- tập TEST (thật) ---")
    test_ds = real_ds(test_pids, max_images=esets.get("test_max_images"))

    if args.mode == "real":
        print("\n--- tập TRAIN (thật) ---")
        train_ds = real_ds(train_pids, cfg.get("real_train", {}).get("max_per_label"),
                           max_images=args.limit_train)
    else:
        manifest = Path(args.manifest) if args.manifest else \
            Path(cfg["generation"]["outdir"]) / "manifest.csv"
        print("\n--- tập TRAIN (sinh) ---")
        train_ds = SyntheticManifestDataset(
            manifest, size=img_size, max_images=args.limit_train,
            use_percentile_norm=bool(cfg.get("synthetic_percentile_norm", True)))

    protocol = PROTOCOL[args.mode]
    metrics = train_one(
        run_name=(f"{args.mode}_seed{seed}"
                  + (f"_n{args.limit_train}" if args.limit_train else "")),
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        output_dir=output_dir,
        batch_size=int(ccfg["batch_size"]), num_workers=int(ccfg["num_workers"]),
        epochs=int(ccfg["epochs"]), lr=float(ccfg["lr"]),
        weight_decay=float(ccfg["weight_decay"]),
        balance_beta=float(ccfg["balance_beta"]),
        pretrained=bool(ccfg["pretrained"]), use_amp=bool(ccfg["use_amp"]),
        early_stop_patience=int(ccfg["early_stop_patience"]),
        use_data_parallel=bool(ccfg.get("use_data_parallel", True)),
        seed=seed, device=cfg["device"],
    )
    metrics["protocol"] = protocol
    metrics["train_source"] = args.mode
    metrics["limit_train"] = args.limit_train

    suffix = f"_n{args.limit_train}" if args.limit_train else ""
    out = output_dir / f"{protocol.lower()}{suffix}.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
