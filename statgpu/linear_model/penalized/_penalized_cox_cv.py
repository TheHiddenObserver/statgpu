"""Survival-aware cross-validation for :class:`PenalizedGLM_CV`.

This module is intentionally separate from the scalar-response GLM CV engine.
It preserves the two-column right-censored target, scores held-out folds with
unpenalized Cox partial likelihood, and refits ``PenalizedCoxPHModel`` without
an intercept.
"""

from __future__ import annotations

import copy
import numbers
import warnings

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, get_backend
from statgpu.backends._utils import _require_real_array
from statgpu.cross_validation._base import folds_are_complete, kfold_indices
from statgpu.solvers import ConvergenceWarning


def _shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    return tuple(int(dimension) for dimension in shape)


def _slice_rows(value, indices, backend):
    index = backend.asarray(indices, dtype=backend.int64)
    return value[index]


def _slice_target(target, indices, backend):
    return _slice_rows(target, indices, backend)


def _target_event(target):
    return target[:, 1]


def _target_shape_contract(target, n_samples):
    if isinstance(target, dict):
        raise ValueError(
            "Cox CV requires y with shape (n_samples, 2); dictionary targets "
            "are not supported by the final PenalizedCoxPHModel refit"
        )
    _require_real_array(target, "y")
    if _shape(target) != (n_samples, 2):
        raise ValueError(
            "Cox CV requires y with shape (n_samples, 2) and columns "
            "[time, event]"
        )


def _backend_contract(cv_device):
    name = cv_device.value if isinstance(cv_device, Device) else str(cv_device).lower()
    if name in {"cuda", "cupy"}:
        return "cupy", "cuda", "cuda"
    if name == "torch":
        return "torch", "torch", "cuda"
    if name in {"cpu", "numpy"}:
        return "numpy", "cpu", "cpu"
    raise ValueError(f"unsupported Cox CV device {cv_device!r}")


def _to_backend_target(target, backend):
    return backend.asarray(target, dtype=backend.float64)


def _coerce_folds(cv_splits, n_samples, cv, random_state):
    if cv_splits is None:
        folds = kfold_indices(
            n_samples,
            n_splits=cv,
            random_state=random_state,
            shuffle=True,
        )
    else:
        folds = list(cv_splits)
    if not folds:
        raise ValueError("cv_splits must contain at least one fold")

    normalized = []
    all_indices = np.arange(n_samples, dtype=np.int64)
    for fold_index, pair in enumerate(folds):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"cv_splits fold {fold_index} must be a (train, validation) pair"
            )
        train = np.asarray(pair[0], dtype=np.int64)
        validation = np.asarray(pair[1], dtype=np.int64)
        if train.ndim != 1 or validation.ndim != 1:
            raise ValueError("CV fold indices must be one-dimensional")
        if train.size == 0 or validation.size == 0:
            raise ValueError("CV train and validation folds must be non-empty")
        if (
            np.unique(train).size != train.size
            or np.unique(validation).size != validation.size
        ):
            raise ValueError("CV fold indices must not contain duplicates")
        if (
            np.any(train < 0)
            or np.any(validation < 0)
            or np.any(train >= n_samples)
            or np.any(validation >= n_samples)
        ):
            raise ValueError("CV fold indices are out of bounds")
        if np.intersect1d(train, validation).size:
            raise ValueError("CV train and validation folds must be disjoint")
        expected_train = np.setdiff1d(
            all_indices, validation, assume_unique=False
        )
        if not np.array_equal(np.sort(train), expected_train):
            raise ValueError(
                "each Cox CV train fold must be the complement of its "
                "validation fold"
            )
        normalized.append((train, validation))

    if not folds_are_complete(normalized, n_samples):
        raise ValueError(
            "Cox CV validation folds must cover every sample exactly once"
        )
    return normalized


def _penalty_name(penalty):
    return str(getattr(penalty, "name", penalty)).lower().strip()


def _penalty_for_alpha(penalty, alpha):
    from statgpu.penalties import Penalty

    if not isinstance(penalty, Penalty):
        return penalty
    candidate = copy.deepcopy(penalty)
    candidate.alpha = float(alpha)
    return candidate


def _validate_alpha_grid(alpha_grid, penalty_name):
    grid = np.asarray(alpha_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha_grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
        raise ValueError("alpha_grid must contain finite non-negative values")
    if penalty_name in {"scad", "mcp"} and np.any(grid <= 0.0):
        raise ValueError("SCAD/MCP alpha_grid values must be strictly positive")
    return grid


def _alpha_grid_from_zero_score(
    loss, X_preprocessed, y_preprocessed, n_alphas, backend
):
    zero = backend.zeros(
        (int(X_preprocessed.shape[1]),), dtype=X_preprocessed.dtype
    )
    gradient = loss.gradient(X_preprocessed, y_preprocessed, zero)
    xp = backend.xp
    alpha_max = _to_float_scalar(xp.max(xp.abs(gradient)))
    if not np.isfinite(alpha_max) or alpha_max <= 0.0:
        alpha_max = 1.0
    return np.geomspace(
        alpha_max,
        max(alpha_max * 1e-4, 1e-12),
        int(n_alphas),
    )

def _finite_column_mean(scores):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    counts = np.sum(finite, axis=0)
    totals = np.sum(np.where(finite, scores, 0.0), axis=0)
    means = np.full(scores.shape[1], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means, counts


def _select_supported_alpha(mean_scores, alpha_grid, valid_counts, required_count):
    eligible = np.isfinite(mean_scores) & (valid_counts == int(required_count))
    if not np.any(eligible):
        raise RuntimeError(
            "Penalized Cox CV produced no alpha with finite evidence from every "
            "evaluable fold; no regularization parameter was selected."
        )
    best = float(np.min(mean_scores[eligible]))
    tolerance = max(1e-12, abs(best) * 1e-10)
    candidates = np.flatnonzero(
        eligible & (mean_scores <= best + tolerance)
    )
    return int(candidates[np.argmax(alpha_grid[candidates])])



def fit_penalized_cox_cv(estimator, X, y, sample_weight=None):
    """Fit the survival-specific branch of ``PenalizedGLM_CV``."""
    from statgpu.linear_model.penalized._penalized_cox import (
        PenalizedCoxPHModel,
    )
    from statgpu.losses import CoxPartialLikelihoodLoss

    if sample_weight is not None:
        raise NotImplementedError("Penalized Cox CV does not support sample_weight")
    if str(estimator.cv_strategy).lower() != "strict":
        raise NotImplementedError(
            "Penalized Cox CV supports cv_strategy='strict' only; two-stage "
            "scalar-response screening is not valid for survival targets."
        )

    if not hasattr(X, "shape"):
        X = np.asarray(X)
    if not isinstance(y, dict) and not hasattr(y, "shape"):
        y = np.asarray(y)
    x_shape = _shape(X)
    if len(x_shape) != 2 or x_shape[1] < 1:
        raise ValueError("X must have shape (n_samples, n_features)")
    n_samples, n_features = x_shape
    _require_real_array(X, "X")
    _target_shape_contract(y, n_samples)

    penalty_name = _penalty_name(estimator.penalty)
    PenalizedCoxPHModel._validate_supported_penalty(estimator.penalty)
    loss_kwargs = dict(getattr(estimator, "_loss_kwargs", {}) or {})
    unsupported_loss_kwargs = set(loss_kwargs) - {"ties"}
    if unsupported_loss_kwargs:
        raise ValueError(
            "Cox CV loss_kwargs supports only 'ties'; unsupported keys: "
            + ", ".join(sorted(unsupported_loss_kwargs))
        )
    ties = str(loss_kwargs.get("ties", "breslow")).lower()
    if ties not in {"breslow", "efron"}:
        raise ValueError("Penalized Cox CV ties must be 'breslow' or 'efron'")

    if estimator._alpha_grid_input is None:
        if isinstance(estimator.n_alphas, (bool, np.bool_)) or not isinstance(
            estimator.n_alphas, numbers.Integral
        ) or int(estimator.n_alphas) < 1:
            raise ValueError("n_alphas must be a positive integer")
        requested_n_alphas = int(estimator.n_alphas)
        alpha_grid = None
    else:
        alpha_grid = _validate_alpha_grid(
            estimator._alpha_grid_input, penalty_name
        )
        requested_n_alphas = int(alpha_grid.size)
    cv_device = estimator._effective_cv_device(
        X, penalty_name, requested_n_alphas
    )
    backend_name, model_device, backend_device = _backend_contract(cv_device)
    backend = get_backend(backend=backend_name, device=backend_device)
    X_backend = backend.asarray(X, dtype=backend.float64)
    y_backend = _to_backend_target(y, backend)

    validation_loss = CoxPartialLikelihoodLoss(ties=ties)
    X_preprocessed = None
    try:
        X_preprocessed, y_preprocessed = validation_loss.preprocess(
            X_backend, y_backend
        )
        if validation_loss._n_events < 1:
            raise ValueError("at least one observed event is required")
        if estimator._alpha_grid_input is None:
            alpha_grid = _alpha_grid_from_zero_score(
                validation_loss,
                X_preprocessed,
                y_preprocessed,
                estimator.n_alphas,
                backend,
            )
    finally:
        validation_loss.release_fit_cache()
    if alpha_grid is None:
        raise RuntimeError("automatic Cox alpha-grid construction failed")
    alpha_grid = _validate_alpha_grid(alpha_grid, penalty_name)

    folds = _coerce_folds(
        estimator.cv_splits,
        n_samples,
        estimator.cv,
        estimator.random_state,
    )
    event_host = np.asarray(
        _to_numpy(_target_event(y_backend)), dtype=np.float64
    )
    train_event_counts = np.asarray(
        [int(np.sum(event_host[train])) for train, _ in folds],
        dtype=np.int64,
    )
    validation_event_counts = np.asarray(
        [int(np.sum(event_host[validation])) for _, validation in folds],
        dtype=np.int64,
    )
    fold_valid = (train_event_counts > 0) & (validation_event_counts > 0)
    n_effective_folds = int(np.sum(fold_valid))
    if n_effective_folds == 0:
        raise RuntimeError(
            "Penalized Cox CV could not evaluate any fold: training and "
            "validation partitions each require at least one event."
        )

    scores = np.full((len(folds), len(alpha_grid)), np.nan, dtype=np.float64)
    failure_path = np.empty(scores.shape, dtype=object)
    failure_path.fill(None)
    cv_solver = estimator._solver_for_cv(cv_device, X=X)

    for fold_index, (train, validation) in enumerate(folds):
        if not fold_valid[fold_index]:
            reason = (
                "missing_training_event"
                if train_event_counts[fold_index] == 0
                else "missing_validation_event"
            )
            failure_path[fold_index, :] = reason
            continue
        X_train = _slice_rows(X_backend, train, backend)
        y_train = _slice_target(y_backend, train, backend)
        X_validation = _slice_rows(X_backend, validation, backend)
        y_validation = _slice_target(y_backend, validation, backend)
        previous_coef = None
        for alpha_index in np.argsort(-alpha_grid):
            alpha = float(alpha_grid[alpha_index])
            model = PenalizedCoxPHModel(
                penalty=_penalty_for_alpha(estimator.penalty, alpha),
                alpha=alpha,
                ties=ties,
                solver=cv_solver,
                max_iter=estimator.max_iter,
                tol=estimator.tol,
                fit_intercept=False,
                l1_ratio=estimator.l1_ratio,
                penalty_kwargs=dict(getattr(estimator, "_penalty_kwargs", {}) or {}),
                device=model_device,
                compute_inference=False,
                loss_kwargs=loss_kwargs,
                gpu_memory_cleanup=False,
            )
            if previous_coef is not None:
                model._init_coef = previous_coef.copy()
            try:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(X_train, y_train)
                except ConvergenceWarning:
                    failure_path[fold_index, alpha_index] = (
                        "solver_not_converged"
                    )
                    continue
                coef = np.asarray(model.coef_, dtype=np.float64)
                if coef.shape != (n_features,) or not np.all(np.isfinite(coef)):
                    raise FloatingPointError(
                        "candidate produced non-finite Cox coefficients"
                    )
                heldout_loss = CoxPartialLikelihoodLoss(ties=ties)
                try:
                    value = heldout_loss.value(
                        X_validation, y_validation, coef
                    )
                finally:
                    heldout_loss.release_fit_cache()
                if not np.isfinite(value):
                    raise FloatingPointError(
                        "candidate produced non-finite held-out Cox loss"
                    )
                scores[fold_index, alpha_index] = float(value)
                previous_coef = coef
            except FloatingPointError as exc:
                failure_path[fold_index, alpha_index] = (
                    f"{type(exc).__name__}: {exc}"
                )

    mean_scores, valid_score_counts = _finite_column_mean(scores)
    best_index = _select_supported_alpha(
        mean_scores,
        alpha_grid,
        valid_score_counts,
        n_effective_folds,
    )
    best_alpha = float(alpha_grid[best_index])

    final_model = PenalizedCoxPHModel(
        penalty=_penalty_for_alpha(estimator.penalty, best_alpha),
        alpha=best_alpha,
        ties=ties,
        solver=cv_solver,
        max_iter=estimator.max_iter,
        tol=estimator.tol,
        fit_intercept=False,
        l1_ratio=estimator.l1_ratio,
        penalty_kwargs=dict(getattr(estimator, "_penalty_kwargs", {}) or {}),
        device=model_device,
        compute_inference=False,
        loss_kwargs=loss_kwargs,
        gpu_memory_cleanup=False,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            final_model.fit(X_backend, y_backend)
    except ConvergenceWarning as exc:
        raise RuntimeError(
            "Penalized Cox CV selected an alpha, but the final refit did not "
            "converge; fitted state was not published."
        ) from exc

    estimator.alpha_ = best_alpha
    estimator.alpha_grid_ = alpha_grid.copy()
    estimator.best_score_ = -float(mean_scores[best_index])
    estimator.cv_strategy_ = "strict"
    estimator.cv_selected_device_ = model_device
    estimator.cv_results_ = {
        "alpha": alpha_grid.copy(),
        "mean_score": mean_scores,
        "mean_test_score": -mean_scores,
        "all_scores": scores,
        "valid_score_counts": valid_score_counts,
        "required_valid_score_count": n_effective_folds,
        "failure_path": failure_path,
        "fold_indices": [(train.copy(), validation.copy()) for train, validation in folds],
        "train_event_counts": train_event_counts,
        "validation_event_counts": validation_event_counts,
        "fold_valid": fold_valid,
        "n_effective_folds": n_effective_folds,
        "scoring": "negative_partial_log_likelihood_per_row",
        "ties": ties,
        "fit_intercept": False,
        "final_refit_class": "PenalizedCoxPHModel",
        "cv_strategy_": "strict",
        "cv_selected_device_": model_device,
        "mean_score_stage1": None,
        "all_scores_stage1": None,
        "refined_mask": np.ones(len(alpha_grid), dtype=bool),
    }
    estimator.estimator_ = final_model
    estimator.coef_ = np.asarray(final_model.coef_, dtype=np.float64).copy()
    estimator.intercept_ = 0.0
    estimator._fitted = True
    return estimator


__all__ = ["fit_penalized_cox_cv"]