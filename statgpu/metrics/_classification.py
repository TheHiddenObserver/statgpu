"""Binary classification metrics - numpy-only public API.

All computation is delegated to :mod:`statgpu.evaluation._classification`
(numpy backend) so that there is a single implementation to maintain.
"""

from typing import Dict, Tuple, Union

import numpy as np

from ..evaluation._classification import (
    binary_average_precision_score as _eval_average_precision_score,
    binary_classification_table as _eval_classification_table,
    binary_confusion_matrix as _eval_confusion_matrix,
    binary_precision_recall_curve as _eval_precision_recall_curve,
    binary_roc_auc_score as _eval_roc_auc_score,
    binary_roc_curve as _eval_roc_curve,
)


def binary_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Compute binary confusion matrix with layout [[TN, FP], [FN, TP]].
    """
    return _eval_confusion_matrix(y_true, y_pred, backend="numpy")


def binary_classification_table(y_true, y_pred) -> Dict[str, Union[int, float]]:
    """Compute a compact binary classification summary table."""
    return _eval_classification_table(y_true, y_pred, backend="numpy")


def binary_roc_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve arrays (fpr, tpr, thresholds) for binary labels.

    Returns thresholds in descending order with an initial ``np.inf`` entry,
    matching sklearn's convention.
    """
    return _eval_roc_curve(y_true, y_score, backend="numpy")


def binary_roc_auc_score(y_true, y_score) -> float:
    """Compute binary ROC-AUC using trapezoidal integration."""
    return _eval_roc_auc_score(y_true, y_score, backend="numpy")


def binary_precision_recall_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve arrays (precision, recall, thresholds).

    Thresholds are returned in descending order with an initial ``np.inf``
    entry, where precision is defined as 1.0 and recall is 0.0.
    """
    return _eval_precision_recall_curve(y_true, y_score, backend="numpy")


def binary_average_precision_score(y_true, y_score) -> float:
    """Compute average precision from the precision-recall curve."""
    return _eval_average_precision_score(y_true, y_score, backend="numpy")
