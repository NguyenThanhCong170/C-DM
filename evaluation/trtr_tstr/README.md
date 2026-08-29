# TRTR vs TSTR

Đo mức độ "hữu dụng cho downstream task" của ảnh synthetic bằng cách train một
classifier multi-label 100% trên ảnh sinh, rồi test trên cùng một tập ảnh thật
(`test_real`) mà baseline TRTR dùng.

- **TRTR** (Train Real, Test Real) không cần ảnh synthetic.
- **TSTR** (Train Synthetic, Test Real) cần ảnh synthetic từ
  `evaluation/generate_synthetic.py` (chạy với
  `config/evaluation/tstr_generation.yaml`, KHÔNG dùng chung batch ảnh của CAS).

Chênh lệch macro-AUC giữa 2 setup càng nhỏ → ảnh synthetic càng giữ đúng tín
hiệu bệnh lý mà nhãn điều kiện yêu cầu (bổ khuyết cho FID/IS vốn chỉ đo độ
"giống ảnh thật" chứ không đo đúng nội dung y khoa).

## File

| File | Vai trò |
|---|---|
| `classifier.py` | DenseNet-121 + `train_one()` — vòng train/eval DÙNG CHUNG cho cả TRTR và TSTR |
| `metrics.py` | macro-AUC theo từng nhãn, `print_comparison()` in bảng + tính Δ |
| `splits.py` | `patient_level_split_3way()` — mở rộng `patient_level_split` gốc trong `dataset/nih_multilabel.py` thành 3 phần (train/val/test), tránh rò rỉ Patient ID |
| `synthetic_dataset.py` | `SyntheticManifestDataset` — đọc `metadata.csv` do `evaluation/generate_synthetic.py` xuất ra, cùng interface với `NIHMultiLabelDataset` |
| `compute_trtr.py` | Entry point TRTR — chạy được ngay, không cần ảnh synthetic |
| `compute_tstr.py` | Entry point TSTR — cần `--synthetic-manifest`, bắt buộc chạy `compute_trtr.py` trước |
| `compare.py` | Đọc `trtr.json` + `tstr.json`, in bảng so sánh, lưu `trtr_vs_tstr.json` |

Nhãn lấy trực tiếp từ `dataset.nih_multilabel.LABEL_NAMES` (hiện tại: `No Finding`,
`Infiltration`, `Effusion`, `Atelectasis`, `Others`) — không định nghĩa lại ở đây,
tự động khớp nếu nhóm đổi định nghĩa nhãn.

## Cách chạy

```bash
# 1. TRTR trước (bắt buộc, để chốt test_real/val_real dùng chung)
python -m evaluation.trtr_tstr.compute_trtr --config config/evaluation/trtr_tstr.yaml
#    -> log lúc load train_ds in ra số ảnh THẬT mỗi nhãn, dùng số này để cập
#       nhật "count" trong config/evaluation/tstr_generation.yaml (xem comment
#       trong file đó) trước khi sinh ảnh TSTR.

# 2. Sinh ảnh synthetic RIÊNG cho TSTR (không dùng chung batch ảnh của CAS)
python -m evaluation.generate_synthetic --config config/evaluation/tstr_generation.yaml
#    -> in ra version_tag, vd "multilabel_20260829-1530"
#    -> ảnh + metadata.csv nằm ở data/synthetic_tstr/<version_tag>/

# 3. TSTR — trỏ thẳng vào metadata.csv vừa sinh
python -m evaluation.trtr_tstr.compute_tstr \
    --config config/evaluation/trtr_tstr.yaml \
    --synthetic-manifest data/synthetic_tstr/<version_tag>/metadata.csv

# 4. So sánh
python -m evaluation.trtr_tstr.compare \
    --trtr out/eval/trtr_tstr/trtr.json --tstr out/eval/trtr_tstr/tstr.json
```

## Format `metadata.csv` mà `synthetic_dataset.py` đọc

Do `evaluation/generate_synthetic.py` xuất ra (dùng chung cho mọi metric, xem
`evaluation/README.md`) — **không phải file bạn tự tạo tay**:

```
filepath,combo,seed,gt_No Finding,gt_Infiltration,gt_Effusion,gt_Atelectasis,gt_Others
no_finding/no_finding_0000.png,no_finding,1000,1,0,0,0,0
infiltration_effusion/infiltration_effusion_0000.png,infiltration_effusion,1600,0,1,1,0,0
...
```

- `filepath` là đường dẫn **tương đối so với thư mục chứa `metadata.csv`**
  (`_read_manifest()` trong `synthetic_dataset.py` tự resolve đúng theo quy ước
  này — không phải tương đối thư mục đang chạy lệnh).
- Cột nhãn có tiền tố `gt_`, đúng tên & thứ tự `LABEL_NAMES`, giá trị 0/1 (có
  thể là nhãn mềm trong `[0,1]` nếu sinh bằng vector không phải one-hot).
- Số lượng & phân phối nhãn của ảnh synthetic nên **xấp xỉ phân phối thật của
  `train_real`** để so sánh công bằng — xem comment chi tiết trong
  `config/evaluation/tstr_generation.yaml` (số hiện tại trong file là tỉ lệ ước
  lượng, cần thay bằng số đo thật từ log của `compute_trtr.py`).


