# CAS (Classification Accuracy Score)

Đánh giá xem ảnh X-quang synthetic sinh ra từ LoRA có thực sự chứa đúng đặc trưng
bệnh lý đã yêu cầu hay không, bằng một model chẩn đoán độc lập
(`densenet121-res224-nih`, TorchXRayVision) chưa từng biết vector nhãn gốc.

## Phạm vi đánh giá

3/5 nhãn: `Atelectasis`, `Infiltration`, `Effusion` (đúng tên `model.pathologies`
của torchxrayvision). `No Finding` được đối chiếu bằng cột `gt_No Finding` có sẵn
trong `metadata.csv` (không suy ra), so với dự đoán no-finding = không nhãn bệnh
nào trong 3 nhãn trên vượt ngưỡng (`Others` không nằm trong phạm vi đánh giá).

## Cách chạy

```bash
# 1. Sinh 500 ảnh synthetic (dùng chung — xem evaluation/README.md)
python -m evaluation.generate_synthetic --config config/evaluation/cas.yaml

# 2. Chấm điểm bằng judge model + xuất báo cáo
python -m evaluation.cas.compute_cas --config config/evaluation/cas.yaml
```

Kết quả nằm ở `reports/cas/<version_tag>/`:
- `cas_report.json` — Hamming accuracy, exact match ratio, AUC-ROC/F1 từng nhãn (kèm CI 95%), no-finding accuracy.
- `cas_predictions.csv` — dự đoán chi tiết từng ảnh, dùng để debug case sai.

`<version_tag>` tự đặt theo checkpoint đang chấm (xem `evaluation/common/versioning.py`),
nên chạy lại với checkpoint mới không đè mất kết quả checkpoint cũ.



