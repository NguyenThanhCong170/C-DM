import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% số bit đúng trên toàn bộ ma trận (n_samples x n_labels)."""
    return float((y_true == y_pred).mean())


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% số ảnh mà TOÀN BỘ vector dự đoán khớp 100% ground truth."""
    return float((y_true == y_pred).all(axis=1).mean())


def per_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                       label_names: list[str]) -> dict:

    out = {}
    for i, name in enumerate(label_names):
        yt, yp, ypred = y_true[:, i], y_prob[:, i], y_pred[:, i]
        auc = None
        if len(np.unique(yt)) == 2:
            try:
                auc = float(roc_auc_score(yt, yp))
            except ValueError:
                auc = None
        f1 = float(f1_score(yt, ypred, zero_division=0))
        out[name] = {"auc_roc": auc, "f1": f1, "n_positive": int(yt.sum()), "n_total": len(yt)}
    return out


def bootstrap_ci(values_true: np.ndarray, values_pred_or_prob: np.ndarray,
                  metric_fn, n_iterations: int = 1000, seed: int = 0) -> tuple[float, float]:

    rng = np.random.default_rng(seed)
    n = len(values_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(values_true[idx], values_pred_or_prob[idx]))
        except ValueError:
            continue
    if not scores:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def derive_no_finding(y_pred_diseases: np.ndarray) -> np.ndarray:
    return (y_pred_diseases.sum(axis=1) == 0).astype(int)
