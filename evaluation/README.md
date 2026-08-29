# Evaluation

Đánh giá chất lượng ảnh X-quang synthetic sinh ra từ LoRA (điều kiện bằng vector
nhãn multi-hot 5 chiều: `No Finding, Infiltration, Effusion, Atelectasis, Others`).

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

