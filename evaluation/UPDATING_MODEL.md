# Cập nhật checkpoint model

Hướng dẫn thao tác khi nhóm có checkpoint LoRA / label encoder mới, và cách tự
sửa các trường hợp phổ biến mà không cần đụng vào code đánh giá.

## 1. Quy trình chuẩn mỗi khi có checkpoint mới

```bash
# 1. Train xong, checkpoint nằm ở (mặc định theo multilabel.yaml):
#    out/multilabel/lora-final.safetensors
#    out/multilabel/label_encoder-final.safetensors
#    out/multilabel/lora_config.json
#    (nếu output_dir trong multilabel.yaml khác, sửa lại 3 đường dẫn tương ứng
#     trong config/evaluation/cas.yaml — không cần sửa gì khác)

# 2. Sinh ảnh (version_tag tự đặt theo thời điểm sửa checkpoint gần nhất,
#    không đè lên kết quả của lần train trước)
python -m evaluation.generate_synthetic --config config/evaluation/cas.yaml

# 3. Chấm CAS
python -m evaluation.cas.compute_cas --config config/evaluation/cas.yaml

# 4. Kết quả:
#    data/synthetic/<version_tag>/              (ảnh + metadata.csv, KHÔNG commit)
#    reports/cas/<version_tag>/cas_report.json  (commit để so sánh giữa các lần train)
#    reports/cas/<version_tag>/cas_predictions.csv
```

`<version_tag>` được in ra ở dòng đầu log của cả 2 lệnh trên, dạng
`multilabel_20260829-1530` — ghép tên thư mục cha của checkpoint với thời điểm
sửa file gần nhất.

## 2. Luôn chạy thử nhanh trước khi chạy full 500 ảnh

Sửa tạm `images_per_combo` trong `config/evaluation/cas.yaml` xuống 2-3, chạy
hết 2 lệnh ở mục 1 để bắt lỗi sớm (sai đường dẫn checkpoint, LoRA lệch rank,
model load lỗi...) trước khi tốn thời gian sinh 500 ảnh thật. Nếu chạy xong
không lỗi (số liệu lúc này chưa có ý nghĩa vì cỡ mẫu quá nhỏ) → đổi lại `125`
và chạy full.

## 3. Các thay đổi thường gặp và cần sửa ở đâu

| Bạn đổi cái gì | Sửa ở đâu | Có cần sửa code không |
|---|---|---|
| Checkpoint mới (cùng kiến trúc, cùng 5 nhãn) | Không cần sửa gì — `lora_path`/`label_encoder_path` trong `cas.yaml` đã trỏ sẵn vào `lora-final.safetensors` (bị ghi đè mỗi lần train) | Không |
| Base model đổi (`sd15` đổi tên/vị trí) | `model.pretrained_dir` trong `cas.yaml` | Không |
| Muốn giữ kết quả của 1 checkpoint cụ thể, không bị lần train sau tính chung version_tag tự động | Đặt tay `model.version_tag` trong `cas.yaml` | Không |
| `rank`/`lora_alpha`/`cross_attention_only` trong `multilabel.yaml` đổi | Không cần sửa — `lora_config.json` được train tự ghi ra, `evaluation/common/model_loader.py` đọc lại đúng từ đó | Không |
| Muốn đánh giá thêm nhãn (hiện chỉ 3/5: Atelectasis, Infiltration, Effusion) | Thêm tên vào `target_labels` trong `cas.yaml` (phải là tên `torchxrayvision` biết, vd `Cardiomegaly`) | Không |
| `num_inference_steps`/`guidance_scale` đổi trong `generate_multilabel.py`/demo thật | Sửa lại 2 giá trị tương ứng trong `cas.yaml` cho khớp | Không |
| `models/loading.py` hoặc `models/lora.py` đổi API (đổi tên hàm/tham số) | `evaluation/common/model_loader.py` | **Có** — đồng bộ tay |
| `models/label_encoder.py` đổi số nhãn / thứ tự nhãn | `evaluation/generate_synthetic.py` (import `DEFAULT_LABELS`), và `target_labels` trong `cas.yaml` | Thường không — `LABEL_NAMES` import trực tiếp từ `label_encoder.py`, chỉ cần kiểm tra lại |

## 4. Việc không cần đụng vào code

- `evaluation/cas/judge.py`, `metrics.py`, `compute_cas.py` — không phụ thuộc
  cách sinh ảnh, chỉ đọc ảnh + `metadata.csv`. Cập nhật checkpoint xong chạy
  lại nguyên trạng.
- `evaluation/common/model_loader.py`, `versioning.py` — chỉ sửa khi API load
  model trong `models/` thực sự đổi chữ ký hàm.

## 5. Lỗi thường gặp

| Thông báo lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Checkpoint không khớp cấu hình LoRA hiện tại` (từ `load_lora_weights_into`) | `lora_config.json` không khớp `lora-final.safetensors` thật (rank/target_modules lệch) | Kiểm tra lại 2 file này có phải cùng 1 lần train hay không |
| `RuntimeError: Không inject được adapter nào với target_modules=...` | `lora_config.json` chứa tên module không tồn tại trong UNet | Xem lại `cross_attention_only` lúc train có khớp `target_modules` đang dùng không |
| `FileNotFoundError: Không thấy checkpoint LoRA` (từ `versioning.py`) | `lora_path` trong `cas.yaml` trỏ sai, hoặc train chưa lưu checkpoint nào | Kiểm tra lại đường dẫn và trạng thái training |
| Nhãn trong `target_labels` không có trong `model.pathologies` của torchxrayvision (lỗi từ `judge.py`) | Sai chính tả tên pathology | Dùng đúng chính tả gốc của torchxrayvision, vd `Atelectasis` (không phải `atelectasis`) |

