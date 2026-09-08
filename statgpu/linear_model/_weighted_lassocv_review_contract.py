"""Weighted LassoCV objective closure found by the canonical PR #138 review.

The dedicated LassoCV path solvers divide their Gram gradients by an integer row
count.  Analytic-weight Lasso, however, is defined with average loss normalized
by ``sum(sample_weight)`` and weighted centering on the original observations.
The historical weighted CV slow path mixed those conventions: GPU folds applied
``sqrt(w)`` before ordinary centering, while CPU/GPU path solvers received a
truncated ``int(sum(w))`` normalizer.  The default alpha grid also ignored
weights.

For weighted calls only, this contract maps every training fold to the exactly
equivalent unweighted row-count problem

    sqrt(w * n_train / sum(w)) * (X - weighted_mean(X))

(and the analogous response).  Existing CPU/CuPy/Torch path solvers can then
continue dividing by ``n_train`` without changing their unweighted fast paths.
Validation MSE remains evaluated on the raw validation observations with the
original analytic weights.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model.cv._device import (
    resolve_cv_backend,
    validate_cv_sample_weight,
)
from statgpu.linear_model.wrappers import _lasso as _lasso_impl


_WEIGHTED_LASSOCV_MARKER = "__statgpu_pr138_weighted_lassocv_objective__"
_ORIGINAL_SELECT = _lasso_impl._select_lasso_alpha_cv


def _weight_sum(value) -> float:
    total = float(np.asarray(_to_numpy(value), dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("each weighted CV training fold must have positive weight sum")
    return total


def _weighted_working_numpy(X, y, weight, *, fit_intercept: bool):
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    n = int(X_arr.shape[0])
    w_sum = float(np.sum(w, dtype=np.float64))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("each weighted CV training fold must have positive weight sum")

    if fit_intercept:
        X_mean = np.sum(X_arr * w[:, None], axis=0) / w_sum
        y_mean = float(np.sum(y_arr * w) / w_sum)
        X_centered = X_arr - X_mean
        y_centered = y_arr - y_mean
    else:
        X_mean = np.zeros(X_arr.shape[1], dtype=np.float64)
        y_mean = 0.0
        X_centered = X_arr
        y_centered = y_arr

    row_scale = np.sqrt(w * (float(n) / w_sum))
    return (
        X_centered * row_scale[:, None],
        y_centered * row_scale,
        X_mean,
        y_mean,
    )


def _weighted_working_backend(X, y, weight, *, fit_intercept: bool, backend):
    w_sum_native = backend.sum(weight)
    w_sum = _weight_sum(w_sum_native)
    n = int(X.shape[0])

    if fit_intercept:
        X_mean = backend.sum(X * weight[:, None], axis=0) / w_sum
        y_mean = backend.sum(y * weight) / w_sum
        X_centered = X - X_mean
        y_centered = y - y_mean
    else:
        X_mean = backend.zeros((X.shape[1],), dtype=X.dtype)
        y_mean = backend.asarray(0.0, dtype=X.dtype)
        X_centered = X
        y_centered = y

    row_scale = backend.sqrt(weight * (float(n) / w_sum))
    return (
        X_centered * row_scale[:, None],
        y_centered * row_scale,
        X_mean,
        y_mean,
    )


def _weighted_alpha_grid_numpy(
    X,
    y,
    weight,
    *,
    fit_intercept: bool,
    n_alphas: int,
    alpha_min_ratio: float,
):
    X_work, y_work, _, _ = _weighted_working_numpy(
        X,
        y,
        weight,
        fit_intercept=fit_intercept,
    )
    return _lasso_impl._default_lasso_alpha_grid(
        X_work,
        y_work,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
    )


def _weighted_alpha_grid_backend(
    X,
    y,
    weight,
    *,
    fit_intercept: bool,
    backend,
    n_alphas: int,
    alpha_min_ratio: float,
):
    X_work, y_work, _, _ = _weighted_working_backend(
        X,
        y,
        weight,
        fit_intercept=fit_intercept,
        backend=backend,
    )
    return _lasso_impl._default_lasso_alpha_grid_backend(
        X_work,
        y_work,
        backend,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
    )


def _weighted_select_lasso_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state=None,
    sample_weight=None,
    fit_intercept: bool = False,
    device="cpu",
    max_iter: int = 3000,
    tol: float = 1e-4,
    cpu_solver: str = "coordinate_descent",
    method: str = "standard",
    cd_kkt_check_every=None,
    gpu_cv_mixed_precision: bool = True,
    return_details: bool = False,
    cache_key=None,
):
    if sample_weight is None:
        return _ORIGINAL_SELECT(
            X,
            y,
            alphas=alphas,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            cv_folds=cv_folds,
            cv_splits=cv_splits,
            random_state=random_state,
            sample_weight=None,
            fit_intercept=fit_intercept,
            device=device,
            max_iter=max_iter,
            tol=tol,
            cpu_solver=cpu_solver,
            method=method,
            cd_kkt_check_every=cd_kkt_check_every,
            gpu_cv_mixed_precision=gpu_cv_mixed_precision,
            return_details=return_details,
            cache_key=cache_key,
        )

    (
        device_name,
        backend_name,
        backend,
        use_gpu,
        _gpu_input_cupy,
        _gpu_input_torch,
    ) = resolve_cv_backend(device, X)

    if use_gpu:
        cv_dtype = backend.float32 if bool(gpu_cv_mixed_precision) else backend.float64
        X_full = backend.asarray(X, dtype=cv_dtype)
        y_full = backend.asarray(y, dtype=cv_dtype).reshape(-1)
        n_samples = int(X_full.shape[0])
        weight_valid = validate_cv_sample_weight(sample_weight, n_samples)
        sw_full = backend.asarray(weight_valid, dtype=cv_dtype).reshape(-1)
        if int(y_full.shape[0]) != n_samples:
            raise ValueError("y must have the same number of rows as X")
    else:
        X_full = np.asarray(X, dtype=np.float64)
        y_full = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_full.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X_full.shape[0])
        if int(y_full.shape[0]) != n_samples:
            raise ValueError("y must have the same number of rows as X")
        weight_valid = validate_cv_sample_weight(sample_weight, n_samples)
        sw_full = np.asarray(weight_valid, dtype=np.float64).reshape(-1)

    cv_method = _lasso_impl._normalize_lassocv_method(method)
    requested_cd_kkt = _lasso_impl._normalize_cd_kkt_check_every(cd_kkt_check_every)

    if alphas is None:
        if use_gpu:
            alpha_grid = _weighted_alpha_grid_backend(
                X_full,
                y_full,
                sw_full,
                fit_intercept=bool(fit_intercept),
                backend=backend,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
        else:
            alpha_grid = _weighted_alpha_grid_numpy(
                X_full,
                y_full,
                sw_full,
                fit_intercept=bool(fit_intercept),
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
    else:
        alpha_grid = np.asarray(alphas, dtype=np.float64).reshape(-1)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
        alpha_grid = alpha_grid[alpha_grid > 0.0]
        if alpha_grid.size == 0:
            if use_gpu:
                alpha_grid = _weighted_alpha_grid_backend(
                    X_full,
                    y_full,
                    sw_full,
                    fit_intercept=bool(fit_intercept),
                    backend=backend,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )
            else:
                alpha_grid = _weighted_alpha_grid_numpy(
                    X_full,
                    y_full,
                    sw_full,
                    fit_intercept=bool(fit_intercept),
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )

    user_folds = _lasso_impl._normalize_cv_splits(cv_splits, n_samples=n_samples)
    effective_n_folds = len(user_folds) if user_folds is not None else int(cv_folds)
    if n_samples < 4 or int(alpha_grid.size) == 1 or effective_n_folds < 2:
        alpha0 = float(alpha_grid[0])
        if not return_details:
            return alpha0
        return {
            "alpha": alpha0,
            "alphas": alpha_grid.astype(np.float64, copy=False),
            "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
        }

    folds = (
        user_folds
        if user_folds is not None
        else _lasso_impl._kfold_indices(
            n_samples=n_samples,
            n_splits=int(cv_folds),
            random_state=random_state,
        )
    )
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    n_alpha = int(alpha_grid.size)
    n_folds = int(len(folds))

    cache_key_eff = cache_key
    if cache_key_eff is None and _lasso_impl._LASSO_CV_ALPHA_CACHE_MAXSIZE > 0:
        cache_key_eff = _lasso_impl._make_lasso_cv_auto_cache_key(
            X=X,
            y=y,
            sample_weight=sample_weight,
            alpha_grid=alpha_grid,
            folds=folds,
            fit_intercept=bool(fit_intercept),
            use_gpu=bool(use_gpu),
            max_iter=int(max_iter),
            tol=float(tol),
            cpu_solver=str(cpu_solver),
            cv_method=str(cv_method),
            cd_kkt_check_every=requested_cd_kkt,
            gpu_cv_mixed_precision=bool(gpu_cv_mixed_precision),
        )
    cached = _lasso_impl._lasso_cv_cache_get(cache_key_eff)
    if cached is not None:
        return cached if return_details else float(cached["alpha"])

    alpha_order_desc = np.argsort(-alpha_grid)
    alpha_desc = alpha_grid[alpha_order_desc]
    mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

    if not use_gpu:
        cpu_solver_name = str(cpu_solver).lower()
        if cv_method == "glmnet":
            cpu_solver_name = "coordinate_descent"
        cd_kkt = (
            (4 if cv_method == "glmnet" else 1)
            if requested_cd_kkt is None
            else int(requested_cd_kkt)
        )
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = X_full[train_idx]
            y_train = y_full[train_idx]
            sw_train = sw_full[train_idx]
            X_work, y_work, X_mean, y_mean = _weighted_working_numpy(
                X_train,
                y_train,
                sw_train,
                fit_intercept=bool(fit_intercept),
            )
            XtX = X_work.T @ X_work
            Xty = X_work.T @ y_work
            n_train_rows = int(X_train.shape[0])
            coefs_desc, _ = _lasso_impl._solve_lasso_path_cpu_from_gram(
                XtX,
                Xty,
                n_samples=n_train_rows,
                alphas_desc=alpha_desc,
                max_iter=int(max_iter),
                tol=float(tol),
                stopping="coef_delta",
                cpu_solver=cpu_solver_name,
                lipschitz_L=None,
                cd_kkt_check_every=cd_kkt,
            )
            intercepts_desc = (
                y_mean - X_mean @ coefs_desc.T
                if fit_intercept
                else np.zeros(coefs_desc.shape[0], dtype=np.float64)
            )
            mse_desc = _lasso_impl._batch_mse_cv(
                X_full[val_idx],
                y_full[val_idx],
                coefs_desc,
                intercepts_desc,
                sample_weight=sw_full[val_idx],
            )
            mse_path[alpha_order_desc, fold_idx] = np.asarray(mse_desc, dtype=np.float64)
    else:
        XtX_folds = []
        Xty_folds = []
        X_mean_folds = []
        y_mean_folds = []
        n_train_rows = []
        eval_payload = []
        for train_idx, val_idx in folds:
            train_idx_b = backend.asarray(train_idx)
            val_idx_b = backend.asarray(val_idx)
            X_train = X_full[train_idx_b]
            y_train = y_full[train_idx_b]
            sw_train = sw_full[train_idx_b]
            X_work, y_work, X_mean, y_mean = _weighted_working_backend(
                X_train,
                y_train,
                sw_train,
                fit_intercept=bool(fit_intercept),
                backend=backend,
            )
            XtX_folds.append(X_work.T @ X_work)
            Xty_folds.append(X_work.T @ y_work)
            X_mean_folds.append(X_mean)
            y_mean_folds.append(y_mean)
            n_train_rows.append(int(X_train.shape[0]))
            eval_payload.append(
                (X_full[val_idx_b], y_full[val_idx_b], sw_full[val_idx_b])
            )

        XtX_batch = backend.stack(XtX_folds, axis=0)
        Xty_batch = backend.stack(Xty_folds, axis=0)
        if backend_name == "torch":
            import torch

            n_vec = torch.as_tensor(
                n_train_rows,
                dtype=XtX_batch.dtype,
                device=XtX_batch.device,
            )
            coefs_batch, _ = _lasso_impl._solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
                XtX_batch,
                Xty_batch,
                n_samples_vec=n_vec,
                alphas_desc=alpha_desc,
                max_iter=int(max_iter),
                tol=float(tol),
                stopping="coef_delta",
                lipschitz_L=None,
                check_every=8,
            )
            for fold_idx in range(n_folds):
                coefs_np = np.asarray(_to_numpy(coefs_batch[fold_idx]), dtype=np.float64)
                if fit_intercept:
                    X_mean_np = np.asarray(_to_numpy(X_mean_folds[fold_idx]), dtype=np.float64)
                    y_mean_np = float(np.asarray(_to_numpy(y_mean_folds[fold_idx]), dtype=np.float64))
                    intercepts_np = y_mean_np - X_mean_np @ coefs_np.T
                else:
                    intercepts_np = np.zeros(coefs_np.shape[0], dtype=np.float64)
                X_val, y_val, sw_val = eval_payload[fold_idx]
                # batch_mse is the maintained CPU scoring/reporting boundary for
                # LassoCV. Keep the already-snapshotted path on NumPy here rather
                # than uploading it only for batch_mse to copy it back to host.
                mse = _lasso_impl._batch_mse_cv(
                    X_val,
                    y_val,
                    coefs_np,
                    intercepts_np,
                    sample_weight=sw_val,
                )
                mse_path[alpha_order_desc, fold_idx] = np.asarray(mse, dtype=np.float64)
        else:
            import cupy as cp

            n_vec = cp.asarray(n_train_rows, dtype=XtX_batch.dtype)
            coefs_batch, _ = _lasso_impl._solve_lasso_path_gpu_fista_multi_fold_from_gram(
                XtX_batch,
                Xty_batch,
                n_samples_vec=n_vec,
                alphas_desc=alpha_desc,
                max_iter=int(max_iter),
                tol=float(tol),
                stopping="coef_delta",
                lipschitz_L=None,
                check_every=8,
            )
            for fold_idx in range(n_folds):
                coefs = coefs_batch[fold_idx]
                intercepts = (
                    y_mean_folds[fold_idx] - X_mean_folds[fold_idx] @ coefs.T
                    if fit_intercept
                    else backend.zeros((coefs.shape[0],), dtype=coefs.dtype)
                )
                X_val, y_val, sw_val = eval_payload[fold_idx]
                mse = _lasso_impl._batch_mse_cv(
                    X_val,
                    y_val,
                    coefs,
                    intercepts,
                    sample_weight=sw_val,
                )
                mse_path[alpha_order_desc, fold_idx] = np.asarray(
                    _to_numpy(mse), dtype=np.float64
                )

    mean_mse = np.full(n_alpha, np.nan, dtype=np.float64)
    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")
    for idx in range(n_alpha):
        valid = np.isfinite(mse_path[idx])
        if not np.any(valid):
            continue
        mean_mse[idx] = float(np.mean(mse_path[idx, valid]))
        if mean_mse[idx] < best_mse:
            best_mse = float(mean_mse[idx])
            best_alpha = float(alpha_grid[idx])

    details = {
        "alpha": best_alpha,
        "alphas": alpha_grid,
        "mse_path": mse_path,
        "mean_mse": mean_mse,
    }
    _lasso_impl._lasso_cv_cache_put(cache_key_eff, details)
    return details if return_details else float(best_alpha)


def install_weighted_lassocv_review_contract():
    current = _lasso_impl._select_lasso_alpha_cv
    if getattr(current, _WEIGHTED_LASSOCV_MARKER, False):
        return
    setattr(_weighted_select_lasso_alpha_cv, _WEIGHTED_LASSOCV_MARKER, True)
    _lasso_impl._select_lasso_alpha_cv = _weighted_select_lasso_alpha_cv


__all__ = [
    "install_weighted_lassocv_review_contract",
    "_weighted_working_numpy",
]
