from __future__ import annotations

"""
Logger mỏng bọc quanh Weights & Biases.

Mục tiêu: train_multilabel.py / train_vae_decoder.py gọi cùng một API dù wandb có
được cài hay không, có mạng hay không. Mọi lỗi của wandb đều bị nuốt và in cảnh báo
— một run 20k step không được phép chết vì logger.

    logger = WandbLogger.from_config(args, job_type="diffusion")
    logger.log({"loss": 0.31, "lr": 1e-4}, step=120)
    logger.log_images(pil_images, ["No Finding", "Effusion"], step=1000)
    logger.finish()
"""

import os
from typing import Dict, List, Optional, Sequence


class WandbLogger:
    """No-op hoàn toàn khi wandb tắt/không cài — mọi method vẫn gọi được."""

    def __init__(self, run=None, log_images: bool = True):
        self._run = run
        self._log_images = log_images

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, args, job_type: str = "train",
                    extra_config: Optional[dict] = None) -> "WandbLogger":
        project = getattr(args, "wandb_project", None)
        mode = str(getattr(args, "wandb_mode", "online") or "online")

        if not project or mode == "disabled":
            print("[wandb] tắt (không đặt 'wandb_project' hoặc wandb_mode: disabled)")
            return cls(None)

        try:
            import wandb
        except ImportError:
            print("[wandb] ⚠ chưa cài — chạy tiếp không log. Cài bằng: pip install wandb")
            return cls(None)

        if mode == "online" and not (os.environ.get("WANDB_API_KEY") or _has_netrc()):
            print("[wandb] ⚠ không thấy WANDB_API_KEY và ~/.netrc — chuyển sang mode 'offline'.\n"
                  "        Đồng bộ sau bằng:  wandb sync <thư mục wandb/offline-run-*>")
            mode = "offline"

        config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        config.update(extra_config or {})

        try:
            run = wandb.init(
                project=project,
                entity=getattr(args, "wandb_entity", None) or None,
                name=getattr(args, "wandb_run_name", None) or None,
                job_type=job_type,
                mode=mode,
                config=config,
                dir=str(getattr(args, "output_dir", ".")),
                # offline không resume được — truyền vào chỉ tạo warning thừa
                **({"resume": "allow"} if mode == "online" else {}),
            )
        except Exception as e:
            print(f"[wandb] ⚠ init lỗi ({type(e).__name__}: {e}) — chạy tiếp không log.")
            return cls(None)

        # offline run không có .name, rơi về id
        label = getattr(run, "name", None) or getattr(run, "id", "?")
        url = getattr(run, "url", None) if mode == "online" else None
        print(f"[wandb] mode={mode} | project={project} | run={label}"
              + (f"\n[wandb] {url}" if url else ""))
        return cls(run, log_images=bool(getattr(args, "wandb_log_images", True)))

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as e:
            print(f"[wandb] ⚠ log lỗi: {e}")

    def log_images(self, images: Sequence, captions: Sequence[str],
                   step: Optional[int] = None, key: str = "validation") -> None:
        if self._run is None or not self._log_images or not images:
            return
        try:
            import wandb
            payload = [wandb.Image(im, caption=str(c)) for im, c in zip(images, captions)]
            self._run.log({key: payload}, step=step)
        except Exception as e:
            print(f"[wandb] ⚠ log ảnh lỗi: {e}")

    def summary(self, values: Dict[str, object]) -> None:
        if self._run is None:
            return
        try:
            for k, v in values.items():
                self._run.summary[k] = v
        except Exception as e:
            print(f"[wandb] ⚠ summary lỗi: {e}")

    def watch_files(self, *paths: str) -> None:
        """Lưu checkpoint/config lên wandb artifact storage (tuỳ chọn, dung lượng lớn)."""
        if self._run is None:
            return
        try:
            for p in paths:
                if p and os.path.isfile(p):
                    self._run.save(p, policy="now")
        except Exception as e:
            print(f"[wandb] ⚠ save file lỗi: {e}")

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass


def _has_netrc() -> bool:
    for name in (".netrc", "_netrc"):
        p = os.path.join(os.path.expanduser("~"), name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    if "api.wandb.ai" in f.read():
                        return True
            except OSError:
                pass
    return False


def gpu_memory_gb() -> Dict[str, float]:
    """VRAM hiện tại/đỉnh, dùng để log. Trả dict rỗng nếu không có CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "vram/allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "vram/peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    except Exception:
        return {}
