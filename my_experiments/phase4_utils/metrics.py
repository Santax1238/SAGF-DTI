from __future__ import annotations

import numpy as np


def _safe_metric(fn, default: float = float("nan")) -> float:
    try:
        return float(fn())
    except Exception:
        return default


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5, y_score=None) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
        balanced_accuracy_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    # AUROC/AUPR only require a monotone decision score. Prefer the raw
    # float32 binary logit so sigmoid saturation cannot create artificial ties.
    y_rank = y_prob if y_score is None else np.asarray(y_score).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "AUROC": _safe_metric(lambda: roc_auc_score(y_true, y_rank)),
        "AUPR": _safe_metric(lambda: average_precision_score(y_true, y_rank)),
        "ACC": _safe_metric(lambda: accuracy_score(y_true, y_pred), 0.0),
        "BACC": _safe_metric(lambda: balanced_accuracy_score(y_true, y_pred), 0.0),
        "F1": _safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0), 0.0),
        "Precision": _safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0), 0.0),
        "Recall": _safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0), 0.0),
        "MCC": _safe_metric(lambda: matthews_corrcoef(y_true, y_pred), 0.0),
    }
