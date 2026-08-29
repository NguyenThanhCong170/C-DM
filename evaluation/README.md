# Evaluation

Đánh giá chất lượng ảnh X-quang synthetic sinh ra từ LoRA (điều kiện bằng vector
nhãn multi-hot 5 chiều: `No Finding, Infiltration, Effusion, Atelectasis, Others`).

## Cấu trúc

```
config/evaluation/          # 1 file config yaml / metric
evaluation/
├── generate_synthetic.py   # sinh ảnh + ground truth — DÙNG CHUNG cho mọi metric
├── common/                 # code dùng chung, không phụ thuộc metric nào
│   ├── model_loader.py     #   load base model + LoRA + label encoder
│   └── versioning.py       #   tách kết quả theo từng lần train (version_tag)
└── <tên_metric>/           # 1 thư mục / metric, chỉ chứa logic riêng của nó
    └── ...
```

**Nguyên tắc**: chỉ có 1 SCRIPT sinh ảnh (`generate_synthetic.py`) cho mọi metric
— không script nào tự viết lại logic load model/sinh ảnh riêng. Nhưng mỗi
metric có thể cần 1 BỘ ẢNH khác nhau (số lượng, phân phối nhãn khác nhau), nên
chạy `generate_synthetic.py` với config riêng của từng metric (không nhất
thiết dùng chung 1 lần chạy). Ground-truth luôn ghi đủ 5 chiều nhãn vào
`metadata.csv`, mỗi metric tự đọc đúng cột nó cần.

## Các metric hiện có

| Metric | Thư mục | Trả lời câu hỏi |
|---|---|---|
| **CAS** (Classification Accuracy Score) | [`cas/`](cas/README.md) | Ảnh sinh ra có thực sự chứa đúng đặc trưng bệnh lý đã yêu cầu không (theo 1 model chẩn đoán độc lập)? |
| **TRTR vs TSTR** | [`trtr_tstr/`](trtr_tstr/README.md) | Ảnh synthetic có hữu dụng để TRAIN một classifier thật không (so với train trên ảnh thật)? |

## Quy trình chung

```bash
# 1. Sinh ảnh (mỗi metric có thể cần 1 bộ ảnh khác nhau -> 1 file config riêng,
#    xem bảng dưới)
python -m evaluation.generate_synthetic --config config/evaluation/cas.yaml
# hoặc: python -m evaluation.generate_synthetic --config config/evaluation/tstr_generation.yaml

# 2. Chạy metric cụ thể, đọc lại đúng bộ ảnh vừa sinh
python -m evaluation.cas.compute_cas --config config/evaluation/cas.yaml
```

Xem README riêng của từng metric để biết chính xác cần config nào và bộ ảnh
nào (không phải metric nào cũng dùng chung 1 bộ ảnh — CAS cần 4 combo đơn-bệnh
số lượng đều nhau, TSTR cần phân phối nhãn lệch giống thật, xem
[`trtr_tstr/README.md`](trtr_tstr/README.md)).

Xem [`UPDATING_MODEL.md`](UPDATING_MODEL.md) để biết cách tự sửa/tự chạy khi
checkpoint model đổi.

## Thêm 1 metric mới

1. Tạo thư mục `evaluation/<ten_metric>/` (`__init__.py` + code riêng).
2. Đọc `metadata.csv` do `generate_synthetic.py` ghi ra (cột `filepath`,
   `combo`, `seed`, `gt_<Tên nhãn>` cho cả 5 nhãn) — không tự sinh ảnh lại.
3. Tạo `config/evaluation/<ten_metric>.yaml` nếu cần tham số riêng (có thể trỏ
   `generation.output_dir`/`version_tag` vào đúng bộ ảnh đã sinh ở bước 1 để
   khỏi sinh lại).
4. Thêm 1 dòng vào bảng "Các metric hiện có" ở trên.
