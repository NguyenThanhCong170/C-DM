from __future__ import annotations

"""
Gộp trtr.json / tstr.json / cas.json thành một bảng và một file report.md.

    python -m evaluation.report --config config/evaluation.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from evaluation.metrics import print_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/evaluation.yaml")
    return p.parse_args()


def _md_table(results: Dict[str, dict]) -> str:
    names = list(results)
    labels = list(next(iter(results.values()))["per_label_auc"])
    lines = ["| Chỉ số | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]
    lines.append("| **macro-AUC** | " +
                 " | ".join(f"**{results[n]['macro_auc']:.4f}**" for n in names) + " |")
    for lab in labels:
        lines.append(f"| AUC {lab} | " +
                     " | ".join(f"{results[n]['per_label_auc'][lab]:.4f}" for n in names) + " |")
    lines.append("| macro-F1 | " +
                 " | ".join(f"{results[n]['macro_f1']:.4f}" for n in names) + " |")
    lines.append("| hamming acc | " +
                 " | ".join(f"{results[n]['hamming_accuracy']:.4f}" for n in names) + " |")
    lines.append("| (baseline toàn âm) | " +
                 " | ".join(f"{results[n]['hamming_baseline_all_negative']:.4f}" for n in names) + " |")
    lines.append("| n mẫu | " +
                 " | ".join(f"{results[n]['n_samples']:,}" for n in names) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])

    results: Dict[str, dict] = {}
    for key, fname in (("TRTR", "trtr.json"), ("TSTR", "tstr.json")):
        p = out_dir / fname
        if p.is_file():
            results[key] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[report] thiếu {p} — bỏ qua {key}")

    cas_path = out_dir / "cas.json"
    if cas_path.is_file():
        cas_blob = json.loads(cas_path.read_text(encoding="utf-8"))
        results["CAS"] = cas_blob["CAS"]

    if not results:
        raise SystemExit("Không có kết quả nào để gộp. Chạy các bước train trước.")

    print_comparison(results)

    md = ["# Kết quả đánh giá mô hình sinh X-quang", "",
          "Tất cả AUC đều tính trên nhãn multi-hot 5 chiều; **macro-AUC là chỉ số chính** "
          "vì nó không phụ thuộc ngưỡng.", "",
          _md_table(results), "",
          "## Đọc bảng này thế nào", "",
          "| Giao thức | Train | Test | Trả lời câu hỏi |",
          "|---|---|---|---|",
          "| TRTR | ảnh thật | ảnh thật | Trần trên. Bài toán 5 nhãn này khó tới đâu? |",
          "| TSTR | ảnh **sinh** | ảnh thật | Ảnh sinh có thay được ảnh thật để TRAIN không? |",
          "| CAS | ảnh thật | ảnh **sinh** | Ảnh sinh có thật sự mang nhãn đã điều kiện không? |",
          ""]

    if "TRTR" in results and "TSTR" in results:
        d = results["TSTR"]["macro_auc"] - results["TRTR"]["macro_auc"]
        rel = 100.0 * results["TSTR"]["macro_auc"] / results["TRTR"]["macro_auc"]
        md += [f"**TSTR − TRTR = {d:+.4f}** ({rel:.1f}% của TRTR). ",
               "Đây là con số kết luận: phần giá trị huấn luyện mà ảnh sinh giữ lại được.", ""]
    if "CAS" in results and "TRTR" in results:
        d = results["CAS"]["macro_auc"] - results["TRTR"]["macro_auc"]
        md += [f"**CAS − TRTR = {d:+.4f}**. ",
               "CAS cao mà TSTR thấp là dấu hiệu kinh điển của mode collapse: ảnh sinh "
               "dễ phân loại vì chúng quá giống nhau, nên vô dụng khi dùng làm dữ liệu train.", ""]

    path = out_dir / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nĐã lưu {path}")


if __name__ == "__main__":
    main()
