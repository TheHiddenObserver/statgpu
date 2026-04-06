"""Binary classification metrics used by model-level evaluation APIs."""

from typing import Dict, Tuple

import numpy as np


def _as_binary_labels(y, *, name: str) -> np.ndarray:
    """Validate and normalize a binary label array encoded as 0/1."""
    y_arr = np.asarray(y).reshape(-1)
    unique = np.unique(y_arr)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{name} must contain only binary labels encoded as 0/1")
    return y_arr.astype(int)


def binary_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Compute binary confusion matrix with layout [[TN, FP], [FN, TP]].
    """
    y_true_arr = _as_binary_labels(y_true, name="y_true")
    y_pred_arr = _as_binary_labels(y_pred, name="y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def binary_classification_table(y_true, y_pred) -> Dict[str, float]:
    """Compute a compact binary classification summary table."""
    cm = binary_confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "support_negative": tn + fp,
        "support_positive": fn + tp,
    }


def binary_roc_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve arrays (fpr, tpr, thresholds) for binary labels.

    Returns thresholds in descending order with an initial ``np.inf`` entry,
    matching sklearn's convention.
    """
    y_true_arr = _as_binary_labels(y_true, name="y_true")
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    positives = np.sum(y_true_arr == 1)
    negatives = np.sum(y_true_arr == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC is undefined when y_true has only one class")

    order = np.argsort(y_score_arr, kind="mergesort")[::-1]
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score_sorted[threshold_indices]]

    tpr = tps / positives
    fpr = fps / negatives

    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def binary_roc_auc_score(y_true, y_score) -> float:
    """Compute binary ROC-AUC using trapezoidal integration."""
    fpr, tpr, _ = binary_roc_curve(y_true, y_score)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def binary_precision_recall_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve arrays (precision, recall, thresholds).

    Thresholds are returned in descending order with an initial ``np.inf``
    entry, where precision is defined as 1.0 and recall is 0.0.
    """
    y_true_arr = _as_binary_labels(y_true, name="y_true")
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    positives = np.sum(y_true_arr == 1)
    if positives == 0:
        raise ValueError("Precision-recall is undefined when y_true has no positive class")

    order = np.argsort(y_score_arr, kind="mergesort")[::-1]
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    precision = np.divide(
        tps,
        tps + fps,
        out=np.ones_like(tps, dtype=float),
        where=(tps + fps) != 0,
    )
    recall = tps / positives
    thresholds = y_score_sorted[threshold_indices]

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[np.inf, thresholds]

    return precision.astype(float), recall.astype(float), thresholds.astype(float)


def binary_average_precision_score(y_true, y_score) -> float:
    """Compute average precision from the precision-recall curve."""
    precision, recall, _ = binary_precision_recall_curve(y_true, y_score)
    recall_diff = np.diff(recall)
    return float(np.sum(recall_diff * precision[1:]))
