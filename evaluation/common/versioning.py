"""
Xác định "version_tag" — nhãn định danh 1 lần train/checkpoint — dùng để tách
thư mục output của các lần chạy generate_synthetic.py / compute_*.py, tránh lần
chạy sau (checkpoint mới) ghi đè mất kết quả của checkpoint cũ.

LƯU Ý: pipeline train hiện tại luôn lưu checkpoint cuối với TÊN FILE CỐ ĐỊNH
("lora-final.safetensors", "label_encoder-final.safetensors") — khác với thiết
kế cũ (checkpoint-4000.safetensors, có số step trong tên). Vì vậy KHÔNG thể suy
version_tag từ tên file như trước; thay vào đó dùng thời điểm sửa đổi (mtime)
của file checkpoint, kết hợp tên thư mục cha, để mỗi lần train mới (ghi đè file)
tự động ra 1 version_tag khác.
"""
import os
import time


def resolve_version_tag(cfg: dict) -> str:
    """Ưu tiên model.version_tag nếu bạn đặt tay trong config (vd để so sánh
    nhiều cấu hình sinh ảnh khác nhau trên CÙNG 1 checkpoint). Mặc định (null)
    thì tự suy ra, không cần bạn nhớ đổi tay mỗi lần train lại."""
    tag = cfg["model"].get("version_tag")
    if tag:
        return str(tag)

    lora_path = cfg["model"]["lora_path"]
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(
            f"Không thấy checkpoint LoRA: {lora_path}\n"
            "-> Kiểm tra lại model.lora_path trong file config, hoặc train chưa lưu checkpoint."
        )

    parent = os.path.basename(os.path.dirname(os.path.abspath(lora_path))) or "model"
    mtime = time.strftime("%Y%m%d-%H%M", time.localtime(os.path.getmtime(lora_path)))
    return f"{parent}_{mtime}"
