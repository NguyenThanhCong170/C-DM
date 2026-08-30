from __future__ import annotations

"""
So sánh kết quả TRTR vs TSTR đã chạy trước đó (compute_trtr.py, compute_tstr.py).

    python -m evaluation.trtr_tstr.compare \\
        --trtr out/eval/trtr_tstr/trtr.json --tstr out/eval/trtr_tstr/tstr.json
"""

import argparse
import json
from pathlib import Path

from .metrics import print_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trtr", required=True)
    p.add_argument("--tstr", required=True)
    p.add_argument("--out", default=None, help="lưu bảng so sánh JSON (mặc định: cạnh --trtr)")
    return p.parse_args()


def main():
    args = parse_args()
    trtr_runs = json.loads(Path(args.trtr).read_text())["runs"]
    tstr_runs = json.loads(Path(args.tstr).read_text())["runs"]

    aggregated = print_comparison({"TRTR": trtr_runs, "TSTR": tstr_runs})

    out_path = Path(args.out) if args.out else Path(args.trtr).parent / "trtr_vs_tstr.json"
    out_path.write_text(json.dumps(aggregated, indent=2))
    print(f"\nĐã lưu bảng so sánh: {out_path}")


if __name__ == "__main__":
    main()
