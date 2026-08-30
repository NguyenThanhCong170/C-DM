import numpy as np
import skimage.io
import torch
import torchvision.transforms
import torchxrayvision as xrv


class Judge:
    def __init__(self, model_name: str = "densenet121-res224-nih",
                 use_op_threshs: bool = True,
                 threshold_overrides: dict | None = None,
                 device: str = "cpu"):
        self.device = device
        # get_model(weights, **kwargs) đẩy thẳng kwargs vào DenseNet.__init__,
        # vốn KHÔNG có tham số from_hf_hub — trọng số tải từ model_urls, không qua HF Hub.
        self.model = xrv.models.get_model(model_name)
        self.model.to(device).eval()

        self.pathologies = list(self.model.pathologies)

        # Ngưỡng dương tính do tác giả TorchXRayVision hiệu chỉnh sẵn (op_threshs).
        # Đây là ngưỡng calibrate trên ẢNH THẬT — cần lưu ý domain shift khi áp
        # dụng cho ảnh synthetic (xem cas/README.md phần "Giới hạn của CAS").
        op = getattr(self.model, "op_threshs", None)
        if use_op_threshs and op is not None:
            # op_threshs là BUFFER đăng ký -> .to(device) đẩy nó lên GPU theo model,
            # nên phải .cpu() trước khi sang numpy.
            if torch.is_tensor(op):
                op = op.detach().cpu().numpy()
            self.thresholds = np.asarray(op, dtype=np.float64)
        else:
            self.thresholds = np.full(len(self.pathologies), 0.5, dtype=np.float64)

        # Model chỉ hiệu chỉnh ngưỡng cho các nhãn mà bộ dữ liệu của nó có.
        # Nhãn thiếu -> threshold = NaN -> `prob >= NaN` luôn False, tức KHÔNG BAO GIỜ
        # dương tính. Báo ra để khỏi lặng lẽ mất nhãn khi gộp về 5 chiều.
        nan_labels = [n for n, t in zip(self.pathologies, self.thresholds) if np.isnan(t)]
        if nan_labels:
            print(f"[judge] {model_name}: {len(nan_labels)} nhãn không có ngưỡng hiệu chỉnh "
                  f"(luôn âm tính): {nan_labels}")

        threshold_overrides = threshold_overrides or {}
        for label, value in threshold_overrides.items():
            if label in self.pathologies:
                self.thresholds[self.pathologies.index(label)] = value

        # Ảnh synthetic 512x512
        # của model (224) sau khi center-crop vuông.
        self._transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim > 2:
            img = img[:, :, 0]
        img = img[None, :, :]
        img = self._transform(img)
        return torch.from_numpy(img).unsqueeze(0).float()

    def predict(self, image_path: str) -> dict:
        """Trả về dict {pathology_name: (prob, is_positive)} cho TẤT CẢ 18 nhãn
        mà model biết. Bên compute_cas.py sẽ chỉ lấy subset cần dùng."""
        tensor = self._preprocess(image_path).to(self.device)
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy()[0]

        result = {}
        for name, prob, thresh in zip(self.pathologies, probs, self.thresholds):
            result[name] = {"prob": float(prob), "positive": bool(prob >= thresh)}
        return result

    def predict_subset(self, image_path: str, labels: list[str]) -> dict:
        full = self.predict(image_path)
        missing = [l for l in labels if l not in full]
        if missing:
            raise ValueError(
                f"Nhãn {missing} không nằm trong danh sách pathology của model "
                f"({self.pathologies}). Kiểm tra lại tên nhãn trong config."
            )
        return {l: full[l] for l in labels}
