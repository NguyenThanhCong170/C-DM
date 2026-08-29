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
