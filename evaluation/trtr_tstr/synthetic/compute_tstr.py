from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dataset.nih_multilabel import NIHMultiLabelDataset

from .classifier import train_one
from .metrics import print_comparison
from .synthetic_dataset import SyntheticManifestDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/evaluation/trtr_tstr.yaml")
    p.add_argument("--synthetic-manifest", required=True,
                   help="CSV: image_path + 1 cột/label, xem synthetic_dataset.py")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_root = Path(cfg["data_root"])
    csv_path = data_root / "Data_Entry_2017.csv"
    output_dir = Path(cfg["output_dir"])

    test_pids_path = output_dir / "test_pids.json"
    val_pids_path = output_dir / "val_pids.json"
    if not test_pids_path.exists() or not val_pids_path.exists():
        raise FileNotFoundError(
            f"Không thấy {test_pids_path} / {val_pids_path}.\n"
            "Chạy compute_trtr.py TRƯỚC để chốt test_real/val_real "
            "(TSTR phải test trên đúng test_real mà TRTR đã dùng, không tự split lại)."
        )
    test_pids = json.loads(test_pids_path.read_text())
    val_pids = json.loads(val_pids_path.read_text())
    print(f"Dùng lại test_real ({len(test_pids)} patient) và val_real ({len(val_pids)} patient) từ TRTR.")

    val_ds = NIHMultiLabelDataset(data_root, csv_path, size=cfg["img_size"],
                                   patient_ids=val_pids, max_per_label=None,
                                   cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])
    test_ds = NIHMultiLabelDataset(data_root, csv_path, size=cfg["img_size"],
                                    patient_ids=test_pids, max_per_label=None,
                                    cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])
    train_ds = SyntheticManifestDataset(args.synthetic_manifest, size=cfg["img_size"])

    runs = []
    for i in range(cfg.get("num_seeds", 1)):
        seed = cfg["seed"] + i
        metrics = train_one(
            run_name=f"tstr_seed{seed}",
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            output_dir=output_dir,
            balance_beta=cfg["balance_beta"],
            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            epochs=cfg["epochs"], lr=cfg["lr"], seed=seed,
            use_amp=cfg.get("use_amp", True),
            use_data_parallel=cfg.get("use_data_parallel", True),
        )
        runs.append(metrics)

    aggregated = print_comparison({"TSTR": runs})

    with open(output_dir / "tstr.json", "w") as f:
        json.dump({"runs": runs, "aggregated": aggregated["TSTR"]}, f, indent=2)
    print(f"\nĐã lưu: {output_dir / 'tstr.json'}")


if __name__ == "__main__":
    main()
