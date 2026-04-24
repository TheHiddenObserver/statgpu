"""Backend-agnostic binary classification evaluation utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from statgpu.backends import _resolve_backend


def _as_binary_labels_numpy(y, *, name: str) -> np.ndarray:
    y_arr = np.asarray(y).reshape(-1)
    unique = np.unique(y_arr)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{name} must contain only binary labels encoded as 0/1")
    return y_arr.astype(np.int64)


def _as_binary_labels_cupy(y, *, name: str):
    import cupy as cp

    y_arr = cp.asarray(y).reshape(-1)
    unique = cp.unique(y_arr)
    is_binary = cp.all((unique == 0) | (unique == 1))
    if not bool(is_binary.item()):
        raise ValueError(f"{name} must contain only binary labels encoded as 0/1")
    return y_arr.astype(cp.int64)


def _as_binary_labels_torch(y, *, name: str):
    import torch

    y_arr = torch.as_tensor(y).reshape(-1)
    unique = torch.unique(y_arr)
    is_binary = torch.all((unique == 0) | (unique == 1))
    if not bool(is_binary.item()):
        raise ValueError(f"{name} must contain only binary labels encoded as 0/1")
    return y_arr.to(dtype=torch.int64)


def _binary_confusion_numpy(y_true, y_pred):
    y_true_arr = _as_binary_labels_numpy(y_true, name="y_true")
    y_pred_arr = _as_binary_labels_numpy(y_pred, name="y_pred")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _binary_confusion_cupy(y_true, y_pred):
    import cupy as cp

    y_true_arr = _as_binary_labels_cupy(y_true, name="y_true")
    y_pred_arr = _as_binary_labels_cupy(y_pred, name="y_pred")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tn = cp.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = cp.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = cp.sum((y_true_arr == 1) & (y_pred_arr == 0))
    tp = cp.sum((y_true_arr == 1) & (y_pred_arr == 1))
    return cp.array([[tn, fp], [fn, tp]], dtype=cp.int64)


def _binary_confusion_torch(y_true, y_pred):
    import torch

    y_true_arr = _as_binary_labels_torch(y_true, name="y_true")
    y_pred_arr = _as_binary_labels_torch(y_pred, name="y_pred")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tn = torch.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = torch.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = torch.sum((y_true_arr == 1) & (y_pred_arr == 0))
    tp = torch.sum((y_true_arr == 1) & (y_pred_arr == 1))
    return torch.stack(
        [torch.stack([tn, fp]), torch.stack([fn, tp])]
    ).to(dtype=torch.int64)


def _classification_table_numpy(y_true, y_pred):
    cm = _binary_confusion_numpy(y_true, y_pred)
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


def _classification_table_cupy(y_true, y_pred):
    import cupy as cp

    cm = _binary_confusion_cupy(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    zero = cp.asarray(0.0, dtype=cp.float64)
    tp_f = tp.astype(cp.float64)
    tn_f = tn.astype(cp.float64)
    fp_f = fp.astype(cp.float64)
    fn_f = fn.astype(cp.float64)
    total_f = total.astype(cp.float64)

    precision = cp.where((tp + fp) > 0, tp_f / (tp_f + fp_f), zero)
    recall = cp.where((tp + fn) > 0, tp_f / (tp_f + fn_f), zero)
    specificity = cp.where((tn + fp) > 0, tn_f / (tn_f + fp_f), zero)
    f1 = cp.where((precision + recall) > 0, 2.0 * precision * recall / (precision + recall), zero)
    accuracy = cp.where(total > 0, (tp_f + tn_f) / total_f, zero)
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


def _classification_table_torch(y_true, y_pred):
    import torch

    cm = _binary_confusion_torch(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    zero = torch.tensor(0.0, device=cm.device, dtype=torch.float64)
    tp_f = tp.to(torch.float64)
    tn_f = tn.to(torch.float64)
    fp_f = fp.to(torch.float64)
    fn_f = fn.to(torch.float64)
    total_f = total.to(torch.float64)

    precision = torch.where((tp + fp) > 0, tp_f / (tp_f + fp_f), zero)
    recall = torch.where((tp + fn) > 0, tp_f / (tp_f + fn_f), zero)
    specificity = torch.where((tn + fp) > 0, tn_f / (tn_f + fp_f), zero)
    f1 = torch.where((precision + recall) > 0, 2.0 * precision * recall / (precision + recall), zero)
    accuracy = torch.where(total > 0, (tp_f + tn_f) / total_f, zero)
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


def _roc_curve_numpy(y_true, y_score):
    y_true_arr = _as_binary_labels_numpy(y_true, name="y_true")
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not np.all(np.isfinite(y_score_arr)):
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the ROC curve."
        )

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


def _roc_curve_cupy(y_true, y_score):
    import cupy as cp

    y_true_arr = _as_binary_labels_cupy(y_true, name="y_true")
    y_score_arr = cp.asarray(y_score, dtype=cp.float64).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not cp.all(cp.isfinite(y_score_arr)).item():
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the ROC curve."
        )

    positives = cp.sum(y_true_arr == 1)
    negatives = cp.sum(y_true_arr == 0)
    if int(positives.item()) == 0 or int(negatives.item()) == 0:
        raise ValueError("ROC is undefined when y_true has only one class")

    order = cp.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]
    distinct_value_indices = cp.where(cp.diff(y_score_sorted) != 0)[0]
    threshold_indices = cp.concatenate(
        [distinct_value_indices, cp.asarray([y_true_sorted.size - 1], dtype=distinct_value_indices.dtype)]
    )

    tps = cp.cumsum(y_true_sorted)[threshold_indices]
    fps = (threshold_indices + 1) - tps
    tps = cp.concatenate([cp.asarray([0], dtype=tps.dtype), tps])
    fps = cp.concatenate([cp.asarray([0], dtype=fps.dtype), fps])
    thresholds = cp.concatenate([cp.asarray([cp.inf], dtype=y_score_sorted.dtype), y_score_sorted[threshold_indices]])

    tpr = tps.astype(cp.float64) / positives.astype(cp.float64)
    fpr = fps.astype(cp.float64) / negatives.astype(cp.float64)
    return fpr, tpr, thresholds


def _roc_curve_torch(y_true, y_score):
    import torch

    y_true_arr = _as_binary_labels_torch(y_true, name="y_true")
    y_score_arr = torch.as_tensor(y_score, dtype=torch.float64, device=y_true_arr.device).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not torch.all(torch.isfinite(y_score_arr)).item():
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the ROC curve."
        )

    positives = torch.sum(y_true_arr == 1)
    negatives = torch.sum(y_true_arr == 0)
    if int(positives.item()) == 0 or int(negatives.item()) == 0:
        raise ValueError("ROC is undefined when y_true has only one class")

    order = torch.argsort(y_score_arr, descending=True)
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    diff = y_score_sorted[1:] - y_score_sorted[:-1]
    distinct_value_indices = torch.nonzero(diff != 0, as_tuple=False).reshape(-1)
    threshold_indices = torch.cat(
        [
            distinct_value_indices,
            torch.tensor([y_true_sorted.numel() - 1], device=y_true_sorted.device, dtype=torch.long),
        ]
    )

    tps = torch.cumsum(y_true_sorted, dim=0)[threshold_indices]
    fps = (threshold_indices + 1) - tps
    tps = torch.cat([torch.zeros(1, device=tps.device, dtype=tps.dtype), tps])
    fps = torch.cat([torch.zeros(1, device=fps.device, dtype=fps.dtype), fps])
    thresholds = torch.cat(
        [
            torch.tensor([float("inf")], device=y_score_sorted.device, dtype=y_score_sorted.dtype),
            y_score_sorted[threshold_indices],
        ]
    )

    tpr = tps.to(torch.float64) / positives.to(torch.float64)
    fpr = fps.to(torch.float64) / negatives.to(torch.float64)
    return fpr, tpr, thresholds


def _roc_auc_from_curve(backend: str, fpr, tpr):
    if backend == "numpy":
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(tpr, fpr))
        return float(np.trapz(tpr, fpr))
    if backend == "cupy":
        import cupy as cp

        if hasattr(cp, "trapezoid"):
            return cp.trapezoid(tpr, fpr)
        return cp.trapz(tpr, fpr)

    import torch

    if hasattr(torch, "trapezoid"):
        return torch.trapezoid(tpr, fpr)
    return torch.trapz(tpr, fpr)


def _precision_recall_curve_numpy(y_true, y_score):
    y_true_arr = _as_binary_labels_numpy(y_true, name="y_true")
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not np.all(np.isfinite(y_score_arr)):
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the precision-recall curve."
        )

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
    precision = np.divide(tps, tps + fps, out=np.ones_like(tps, dtype=float), where=(tps + fps) != 0)
    recall = tps / positives
    thresholds = y_score_sorted[threshold_indices]

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[np.inf, thresholds]
    return precision.astype(float), recall.astype(float), thresholds.astype(float)


def _precision_recall_curve_cupy(y_true, y_score):
    import cupy as cp

    y_true_arr = _as_binary_labels_cupy(y_true, name="y_true")
    y_score_arr = cp.asarray(y_score, dtype=cp.float64).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not cp.all(cp.isfinite(y_score_arr)).item():
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the precision-recall curve."
        )

    positives = cp.sum(y_true_arr == 1)
    if int(positives.item()) == 0:
        raise ValueError("Precision-recall is undefined when y_true has no positive class")

    order = cp.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]
    distinct_value_indices = cp.where(cp.diff(y_score_sorted) != 0)[0]
    threshold_indices = cp.concatenate(
        [distinct_value_indices, cp.asarray([y_true_sorted.size - 1], dtype=distinct_value_indices.dtype)]
    )

    tps = cp.cumsum(y_true_sorted)[threshold_indices]
    fps = (threshold_indices + 1) - tps
    denom = (tps + fps).astype(cp.float64)
    safe_denom = cp.where(denom != 0, denom, cp.asarray(1.0, dtype=cp.float64))
    precision = tps.astype(cp.float64) / safe_denom
    precision = cp.where(denom != 0, precision, cp.ones_like(precision))
    recall = tps.astype(cp.float64) / positives.astype(cp.float64)
    thresholds = y_score_sorted[threshold_indices]

    precision = cp.concatenate([cp.asarray([1.0], dtype=cp.float64), precision])
    recall = cp.concatenate([cp.asarray([0.0], dtype=cp.float64), recall])
    thresholds = cp.concatenate([cp.asarray([cp.inf], dtype=y_score_sorted.dtype), thresholds])
    return precision, recall, thresholds


def _precision_recall_curve_torch(y_true, y_score):
    import torch

    y_true_arr = _as_binary_labels_torch(y_true, name="y_true")
    y_score_arr = torch.as_tensor(y_score, dtype=torch.float64, device=y_true_arr.device).reshape(-1)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    if not torch.all(torch.isfinite(y_score_arr)).item():
        raise ValueError(
            "y_score contains non-finite values (NaN or inf). "
            "All scores must be finite to compute the precision-recall curve."
        )

    positives = torch.sum(y_true_arr == 1)
    if int(positives.item()) == 0:
        raise ValueError("Precision-recall is undefined when y_true has no positive class")

    order = torch.argsort(y_score_arr, descending=True)
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    diff = y_score_sorted[1:] - y_score_sorted[:-1]
    distinct_value_indices = torch.nonzero(diff != 0, as_tuple=False).reshape(-1)
    threshold_indices = torch.cat(
        [
            distinct_value_indices,
            torch.tensor([y_true_sorted.numel() - 1], device=y_true_sorted.device, dtype=torch.long),
        ]
    )

    tps = torch.cumsum(y_true_sorted, dim=0)[threshold_indices]
    fps = (threshold_indices + 1) - tps
    denom = (tps + fps).to(torch.float64)
    safe_denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    precision = tps.to(torch.float64) / safe_denom
    precision = torch.where(denom != 0, precision, torch.ones_like(precision))
    recall = tps.to(torch.float64) / positives.to(torch.float64)
    thresholds = y_score_sorted[threshold_indices]

    precision = torch.cat([torch.tensor([1.0], device=precision.device, dtype=precision.dtype), precision])
    recall = torch.cat([torch.tensor([0.0], device=recall.device, dtype=recall.dtype), recall])
    thresholds = torch.cat(
        [torch.tensor([float("inf")], device=thresholds.device, dtype=thresholds.dtype), thresholds]
    )
    return precision, recall, thresholds


def _average_precision_from_curve(backend: str, precision, recall):
    if backend == "numpy":
        return float(np.sum(np.diff(recall) * precision[1:]))
    if backend == "cupy":
        import cupy as cp

        return cp.sum(cp.diff(recall) * precision[1:])

    import torch

    return torch.sum((recall[1:] - recall[:-1]) * precision[1:])


def binary_confusion_matrix(y_true, y_pred, backend: str = "auto"):
    backend_name = _resolve_backend(backend, y_true, y_pred)
    if backend_name == "numpy":
        return _binary_confusion_numpy(y_true, y_pred)
    if backend_name == "cupy":
        return _binary_confusion_cupy(y_true, y_pred)
    return _binary_confusion_torch(y_true, y_pred)


def binary_classification_table(y_true, y_pred, backend: str = "auto") -> Dict[str, Any]:
    backend_name = _resolve_backend(backend, y_true, y_pred)
    if backend_name == "numpy":
        return _classification_table_numpy(y_true, y_pred)
    if backend_name == "cupy":
        return _classification_table_cupy(y_true, y_pred)
    return _classification_table_torch(y_true, y_pred)


def binary_roc_curve(y_true, y_score, backend: str = "auto"):
    backend_name = _resolve_backend(backend, y_true, y_score)
    if backend_name == "numpy":
        return _roc_curve_numpy(y_true, y_score)
    if backend_name == "cupy":
        return _roc_curve_cupy(y_true, y_score)
    return _roc_curve_torch(y_true, y_score)


def binary_roc_auc_score(y_true, y_score, backend: str = "auto"):
    backend_name = _resolve_backend(backend, y_true, y_score)
    fpr, tpr, _ = binary_roc_curve(y_true, y_score, backend=backend_name)
    return _roc_auc_from_curve(backend_name, fpr, tpr)


def binary_precision_recall_curve(y_true, y_score, backend: str = "auto"):
    backend_name = _resolve_backend(backend, y_true, y_score)
    if backend_name == "numpy":
        return _precision_recall_curve_numpy(y_true, y_score)
    if backend_name == "cupy":
        return _precision_recall_curve_cupy(y_true, y_score)
    return _precision_recall_curve_torch(y_true, y_score)


def binary_average_precision_score(y_true, y_score, backend: str = "auto"):
    backend_name = _resolve_backend(backend, y_true, y_score)
    precision, recall, _ = binary_precision_recall_curve(y_true, y_score, backend=backend_name)
    return _average_precision_from_curve(backend_name, precision, recall)


def evaluate_binary_classification(
    y_true,
    y_score,
    threshold: float = 0.5,
    include_curves: bool = True,
    backend: str = "auto",
) -> Dict[str, Any]:
    """
    One-shot binary evaluation from external class-1 probabilities.

    Parameters
    ----------
    y_true : array-like
        Binary labels encoded as 0/1.
    y_score : array-like
        Predicted probabilities for positive class.
    threshold : float, default=0.5
        Threshold used for hard predictions in confusion/table metrics.
    include_curves : bool, default=True
        Whether to include full ROC/PR curve arrays.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Backend selection. ``'auto'`` is inferred from input arrays.

    Returns
    -------
    dict
        Batch evaluation dictionary.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    backend_name = _resolve_backend(y_true, y_score, backend)
    if backend_name == "numpy":
        y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
        if not np.all(np.isfinite(y_score_arr)):
            raise ValueError(
                "y_score contains non-finite values (NaN or inf). "
                "Ensure all predicted probabilities are finite before calling evaluate_binary_classification."
            )
        y_pred = (y_score_arr >= threshold).astype(np.int64)
    elif backend_name == "cupy":
        import cupy as cp

        y_score_arr = cp.asarray(y_score, dtype=cp.float64).reshape(-1)
        if not cp.all(cp.isfinite(y_score_arr)).item():
            raise ValueError(
                "y_score contains non-finite values (NaN or inf). "
                "Ensure all predicted probabilities are finite before calling evaluate_binary_classification."
            )
        y_pred = (y_score_arr >= threshold).astype(cp.int64)
    else:
        import torch

        y_true_t = torch.as_tensor(y_true)
        y_score_arr = torch.as_tensor(y_score, dtype=torch.float64, device=y_true_t.device).reshape(-1)
        if not torch.all(torch.isfinite(y_score_arr)).item():
            raise ValueError(
                "y_score contains non-finite values (NaN or inf). "
                "Ensure all predicted probabilities are finite before calling evaluate_binary_classification."
            )
        y_pred = (y_score_arr >= threshold).to(dtype=torch.int64)

    result: Dict[str, Any] = {
        "threshold": float(threshold),
        "confusion_matrix": binary_confusion_matrix(y_true, y_pred, backend=backend_name),
        "classification_table": binary_classification_table(y_true, y_pred, backend=backend_name),
    }

    fpr, tpr, roc_thresholds = binary_roc_curve(y_true, y_score_arr, backend=backend_name)
    precision, recall, pr_thresholds = binary_precision_recall_curve(y_true, y_score_arr, backend=backend_name)
    result["roc_auc"] = _roc_auc_from_curve(backend_name, fpr, tpr)
    result["average_precision"] = _average_precision_from_curve(backend_name, precision, recall)

    if include_curves:
        result["roc_curve"] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        }
        result["precision_recall_curve"] = {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
        }

    return result
