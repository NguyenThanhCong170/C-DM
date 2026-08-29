from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dataset.nih_multilabel import NIHMultiLabelDataset

from .classifier import train_one
from .metrics import print_comparison
from .splits import patient_level_split_3way


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/evaluation/trtr_tstr.yaml")
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
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pids, val_pids, test_pids = patient_level_split_3way(
        csv_path, test_ratio=cfg["test_ratio"], val_ratio=cfg["val_ratio"], seed=cfg["seed"],
    )
    print(f"Patients: train={len(train_pids)}  val={len(val_pids)}  test={len(test_pids)}")

    val_ds = NIHMultiLabelDataset(data_root, csv_path, size=cfg["img_size"],
                                   patient_ids=val_pids, max_per_label=None,
                                   cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])
    test_ds = NIHMultiLabelDataset(data_root, csv_path, size=cfg["img_size"],
                                    patient_ids=test_pids, max_per_label=None,
                                    cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])
    train_ds = NIHMultiLabelDataset(data_root, csv_path, size=cfg["img_size"],
                                     patient_ids=train_pids,
                                     max_per_label=cfg.get("max_per_label_train"),
                                     cache_dir=cfg.get("cache_dir"), seed=cfg["seed"])

    (output_dir / "test_pids.json").write_text(json.dumps(test_pids))
    (output_dir / "val_pids.json").write_text(json.dumps(val_pids))

    runs = []
    for i in range(cfg.get("num_seeds", 1)):
        seed = cfg["seed"] + i
        metrics = train_one(
            run_name=f"trtr_seed{seed}",
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            output_dir=output_dir,
            balance_beta=cfg["balance_beta"],
            batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            epochs=cfg["epochs"], lr=cfg["lr"], seed=seed,
            use_amp=cfg.get("use_amp", True),
            use_data_parallel=cfg.get("use_data_parallel", True),
        )
        runs.append(metrics)

    aggregated = print_comparison({"TRTR": runs})

    with open(output_dir / "trtr.json", "w") as f:
        json.dump({"runs": runs, "aggregated": aggregated["TRTR"]}, f, indent=2)
    print(f"\nĐã lưu: {output_dir / 'trtr.json'}")


if __name__ == "__main__":
    main()
