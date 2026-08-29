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

## Giới hạn cần biết khi đọc kết quả

- Judge model và LoRA của bạn đều "học" từ nguồn gốc NIH ChestX-ray14 — nhãn NIH
  có nhiễu (~10%, gán tự động bằng NLP từ báo cáo), và một số nhãn (đặc biệt
  Pneumothorax) từng bị ghi nhận là gắn với shortcut hình ảnh (chest tube) chứ
  không hoàn toàn phản ánh bệnh lý thật. CAS cao không loại trừ khả năng cả hai
  model cùng dựa vào cùng 1 đặc trưng giả.
- `op_threshs` được hiệu chỉnh trên ảnh X-quang thật, không phải ảnh synthetic
  → có domain shift, nên đọc AUC-ROC (không phụ thuộc threshold) song song với
  Hamming/exact match.
- n ≈ 125 ảnh/tổ hợp → AUC-ROC riêng từng nhãn có phương sai đáng kể, luôn xem
  cùng khoảng tin cậy 95% (`auc_roc_ci95`) trong báo cáo, đừng chỉ nhìn điểm số
  trung tâm.
- Nên cân nhắc chạy chéo thêm 1 judge khác domain (vd. `densenet121-res224-chex`)
  để kiểm tra CAS có tổng quát hay chỉ đúng với riêng model NIH.
