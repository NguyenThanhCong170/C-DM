# Đánh giá mô hình sinh: TRTR / TSTR / CAS

## Ma trận 2×2

|                     | test **ảnh thật** | test **ảnh sinh** |
|---------------------|-------------------|-------------------|
| train **ảnh thật**  | **TRTR** — trần trên | **CAS** — độ trung thực nhãn |
| train **ảnh sinh**  | **TSTR** — giá trị huấn luyện | (vô nghĩa) |

Cả ba chỉ số đến từ **hai** classifier, không phải ba:

- `real_seed42.pt` (train ảnh thật) → chấm test thật = **TRTR**, chấm ảnh sinh = **CAS**
- `synthetic_seed42.pt` (train ảnh sinh) → chấm test thật = **TSTR**

## Các file

| File | Vai trò |
|---|---|
| `splits.py` | Chia train/val/test theo **Patient ID** (không theo ảnh — tránh rò rỉ) |
| `metrics.py` | macro-AUC, AUC từng nhãn, F1, hiệu chỉnh ngưỡng, baseline hamming |
| `synthetic_dataset.py` | Đọc `manifest.csv`, cùng interface với `NIHMultiLabelDataset` |
| `classifier.py` | DenseNet-121 5 nhãn + `train_one()` dùng chung cho cả hai chế độ |
| `generate_eval_set.py` | Sinh ảnh synthetic khớp phân phối nhãn tập train thật |
| `train_classifier.py` | Entry point TRTR (`--mode real`) và TSTR (`--mode synthetic`) |
| `compute_cas.py` | CAS + baseline ảnh thật của **cùng** giám khảo |
| `report.py` | Gộp 3 file JSON thành bảng + `report.md` |

## Thứ tự chạy

```bash
# 0) Cấu hình: sửa config/evaluation.yaml (đường dẫn lora / label_encoder / data_root)

# 1) TRTR — cũng tạo luôn giám khảo cho CAS   (~40-60 phút trên A40 @512)
python -m evaluation.train_classifier --config config/evaluation.yaml --mode real

# 2) Sinh 5.000 ảnh synthetic                 (~1-1.5 giờ; có --resume nếu đứt)
python -m evaluation.generate_eval_set --config config/evaluation.yaml

# 3) TSTR — train trên ảnh sinh, test ảnh thật
python -m evaluation.train_classifier --config config/evaluation.yaml --mode synthetic

# 4) CAS — giám khảo bước 1 chấm ảnh bước 2
python -m evaluation.compute_cas --config config/evaluation.yaml

# 5) Gộp báo cáo
python -m evaluation.report --config config/evaluation.yaml
```

Chạy trong tmux. Mỗi bước ghi JSON riêng vào `out/eval/`, nên đứt bước nào chạy
lại đúng bước đó, không mất bước trước.

## Chạy thử nhanh trước khi tốn vài tiếng

```bash
python -m evaluation.generate_eval_set --config config/evaluation.yaml -n 64
python -m evaluation.train_classifier --config config/evaluation.yaml \
    --mode real --epochs 1 --limit-train 200
```

## Cỡ dữ liệu

Cấu hình mặc định: **1000 ảnh/nhãn** ở tập train thật và **5000** ảnh sinh, cố ý
để hai tập train xấp xỉ bằng nhau — TSTR và TRTR khi đó chỉ khác nhau ở nguồn
pixel, không lẫn ảnh hưởng của cỡ dữ liệu.

Con số thật sẽ **nhỏ hơn 5000**. Đây là bài toán multi-label: một ảnh
`Effusion|Atelectasis` tiêu tốn hạn mức của cả hai nhãn cùng lúc, nên tổng ảnh
luôn ít hơn tổng trần. Xem dòng `[Dataset] ... ảnh` script in ra để biết con số
chính xác, rồi chỉnh `generation.num_images` cho khớp nếu muốn cân tuyệt đối.

Bộ lọc ưu tiên giữ ảnh NHIỀU nhãn trước (`_apply_caps` trong
`dataset/nih_multilabel.py`), nên các tổ hợp đồng mắc hiếm không bị cắt mất —
đó là thứ khó học nhất, cắt đi thì cả TRTR lẫn TSTR đều mất ý nghĩa.

`--limit-train N` giới hạn thêm số ảnh train ở cả hai chế độ, tiện khi muốn
quét xem cỡ dữ liệu ảnh hưởng thế nào:

```bash
python -m evaluation.train_classifier --config config/evaluation.yaml \
    --mode real --limit-train 2000
# -> out/eval/trtr_n2000.json, checkpoint real_seed42_n2000.pt
```

Lưu ý: `compute_cas.py` mặc định tìm giám khảo tên `real_seed42.pt`. Dùng
checkpoint có hậu tố thì phải chỉ rõ:

```bash
python -m evaluation.compute_cas --config config/evaluation.yaml \
    --checkpoint out/eval/checkpoints/real_seed42_n2000.pt
```

Tập **val** bị cắt còn 3000 ảnh (`eval_sets.val_max_images`) vì nó bị chấm lại
sau MỖI epoch. Tập **test** để nguyên (`test_max_images: null`) — nó chỉ chạy
đúng một lần và là con số đi vào báo cáo.

## Ba cái bẫy khi đọc số

**1. Hamming accuracy tự nó vô nghĩa.** Với tỉ lệ dương tính ~25%, một mô hình
đoán TOÀN ÂM TÍNH đã đạt 0.75. Mọi báo cáo ở đây in kèm
`hamming_baseline_all_negative` — luôn đọc hai số cạnh nhau.

**2. CAS một mình không kết luận được gì.** Không có baseline của cùng giám khảo
trên ảnh thật thì CAS = 0.82 là vô nghĩa. `compute_cas.py` vì thế luôn chấm cả
hai. Và CAS **mù trước mode collapse**: sinh một ảnh Effusion hoàn hảo lặp 2000
lần thì CAS gần tuyệt đối trong khi TSTR sụp đổ. Luôn báo cáo cặp CAS + TSTR.

**3. Ngưỡng 0.5 gần như luôn sai.** BCE trên nhãn thưa đẩy xác suất về thấp;
ngưỡng được hiệu chỉnh tối ưu F1 trên tập **val thật** (không bao giờ trên test)
và lưu cạnh checkpoint. AUC không phụ thuộc ngưỡng nên vẫn là chỉ số chính.

## Vì sao `__init__.py` để rỗng

Lần trước file đó re-export từ `.metrics` / `.splits` / `.classifier`, nên mọi
`import evaluation.<gì đó>` đều kéo theo torch + torchvision + sklearn, và một
file bị đổi chỗ là cả gói sập. Cứ import thẳng module cần dùng.
