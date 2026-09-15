"""Inverse-power Gamma explicit Newton/L-BFGS consumer closure.

Shared domain mechanics live in ``GammaLoss`` and the Newton/L-BFGS solvers.
The ordinary GLM boundary is owned by ``_glm_weighted_explicit_solver_contract``;
this installer reconciles only penalized fitting and smooth-L2 CV behavior that
would otherwise inject a log-link start or score Gamma under the wrong link.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model.penalized._fit_mixin import (
    _PenalizedFitMixin,
    _resolve_loss_name,
)
from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV


_PENALIZED_MARKER = "_statgpu_inverse_gamma_domain_penalized"
_CV_MARKER = "_statgpu_inverse_gamma_domain_cv"
_EVAL_MARKER = "_statgpu_inverse_gamma_domain_eval"


def _is_inverse_gamma(loss) -> bool:
    return (
        getattr(loss, "name", "") == "gamma"
        and getattr(loss, "link", None) == "inverse_power"
    )


def _install_penalized_contract() -> None:
    current = _PenalizedFitMixin._fit_loss_backend
    if getattr(current, _PENALIZED_MARKER, False):
        return

    @wraps(current)
    def wrapped(self, X, y, sample_weight, solver_name, backend_name):
        if solver_name not in ("newton", "lbfgs") or not _is_inverse_gamma(self._loss):
            return current(self, X, y, sample_weight, solver_name, backend_name)

        from statgpu.backends._array_ops import _xp_asarray
        from statgpu.backends._utils import _get_xp
        from statgpu.solvers import lbfgs_solver, newton_solver
        from statgpu.solvers._smooth_domain import _prepare_analytic_sample_weight

        xp = _get_xp(backend_name)
        ref = X if not isinstance(X, np.ndarray) else xp.zeros(1, dtype=xp.float64)
        X_arr = _xp_asarray(X, xp.float64, ref)
        y_arr = _xp_asarray(y, xp.float64, X_arr)

        if self._effective_intercept:
            p = X_arr.shape[1]
            X_work = self._column_stack(
                [X_arr, self._ones(X_arr.shape[0], backend_name, X_arr)],
                backend_name,
            )
            pen = self._selective_penalty(p, backend_name)
        else:
            p = X_arr.shape[1]
            X_work = X_arr
            pen = self._penalty

        # Penalized/CV warm starts are framework-owned. Reuse a valid one;
        # discard an invalid one so the loss can construct a fresh interior
        # point. Crucially, no inverse-Gamma default goes through the historical
        # log-link ``log(mean(y))`` branch.
        init = None
        init_features = getattr(self, "_init_coef", None)
        if init_features is not None:
            init_np = np.asarray(_to_numpy(init_features), dtype=np.float64).ravel()
            if self._effective_intercept:
                init_intercept = float(getattr(self, '_init_intercept', 0.0) or 0.0)
                init_np = np.concatenate([init_np, [init_intercept]])
            init_candidate = _xp_asarray(init_np, X_arr.dtype, X_arr)
            prepared_weight = _prepare_analytic_sample_weight(
                sample_weight,
                X_work.shape[0],
                backend_name,
                X_work,
            )
            if self._loss._loss_domain_is_feasible(
                X_work,
                init_candidate,
                sample_weight=prepared_weight,
            ):
                init = init_candidate

        solver = newton_solver if solver_name == "newton" else lbfgs_solver
        params, n_iter = solver(
            self._loss,
            pen,
            X_work,
            y_arr,
            max_iter=self._max_iter,
            tol=self._tol,
            init_coef=init,
            sample_weight=sample_weight,
        )

        self.n_iter_ = n_iter
        params_np = _to_numpy(params)
        if self._effective_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self._effective_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    setattr(wrapped, _PENALIZED_MARKER, True)
    wrapped._statgpu_original = current
    _PenalizedFitMixin._fit_loss_backend = wrapped


def _install_cv_loss_evaluator() -> None:
    """Teach the shared NumPy validator to score actual inverse-Gamma loss.

    Do not wrap ``PenalizedGLM_CV._evaluate_single``: doing so inserts a frame
    into every loss's warning path.  The inverse-Gamma L2 CV route below passes
    the correctly resolved loss object directly to this evaluator.
    """
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    current = cv_mod._evaluate_loss_numpy
    if getattr(current, _EVAL_MARKER, False):
        return

    @wraps(current)
    def wrapped(
        loss_name,
        loss_fn,
        X_val_np,
        y_val_np,
        coef_np,
        intercept,
        fit_intercept,
        sample_weight=None,
    ):
        if str(loss_name).lower() == "gamma" and _is_inverse_gamma(loss_fn):
            X_val_np = np.asarray(X_val_np, dtype=np.float64)
            y_val_np = np.asarray(y_val_np, dtype=np.float64).ravel()
            coef_np = np.asarray(coef_np, dtype=np.float64).ravel()
            if fit_intercept:
                X_design = np.column_stack(
                    [X_val_np, np.ones(X_val_np.shape[0], dtype=np.float64)]
                )
                params = np.concatenate([coef_np, [float(intercept)]])
            else:
                X_design = X_val_np
                params = coef_np
            sw = (
                None
                if sample_weight is None
                else np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            )
            return float(
                loss_fn.value(
                    X_design,
                    y_val_np,
                    params,
                    sample_weight=sw,
                )
            )
        return current(
            loss_name,
            loss_fn,
            X_val_np,
            y_val_np,
            coef_np,
            intercept,
            fit_intercept,
            sample_weight=sample_weight,
        )

    setattr(wrapped, _EVAL_MARKER, True)
    wrapped._statgpu_original = current
    cv_mod._evaluate_loss_numpy = wrapped


def _inverse_gamma_l2_cv_scores(
    self,
    X,
    y,
    alpha_grid,
    cv_device,
    folds,
    sample_weight=None,
    max_iter=None,
    tol=None,
):
    """Correctness-first CV path for the public inverse-Gamma smooth-L2 row."""
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod
    from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).ravel()
    sort_idx = np.argsort(-alpha_grid)
    alpha_sorted = alpha_grid[sort_idx]
    all_scores = np.full((len(folds), alpha_grid.size), np.nan, dtype=np.float64)
    max_iter = int(self._max_iter if max_iter is None else max_iter)
    tol = self._tol if tol is None else tol
    cv_solver = self._solver_for_cv(cv_device, X=X)
    loss_fn = _resolve_loss_name(
        "gamma", loss_kwargs=getattr(self, "_loss_kwargs", None)
    )
    device_name = cv_mod._device_to_name(cv_device)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train = cv_mod._slice_rows(X, train_idx)
        y_train = cv_mod._slice_rows(y, train_idx)
        X_val = cv_mod._slice_rows(X, val_idx)
        y_val = cv_mod._slice_rows(y, val_idx)
        sw_train = (
            cv_mod._slice_rows(sample_weight, train_idx)
            if sample_weight is not None else None
        )
        sw_val = (
            cv_mod._slice_rows(sample_weight, val_idx)
            if sample_weight is not None else None
        )

        if device_name in ("cuda", "torch"):
            backend = cv_mod._backend_name_for_cv_device(cv_device)
            X_fit = cv_mod._to_backend_float64(X_train, backend)
            y_fit = cv_mod._to_backend_float64(y_train, backend)
            sw_fit = (
                cv_mod._to_backend_float64(sw_train, backend)
                if sw_train is not None else None
            )
        else:
            X_fit, y_fit, sw_fit = X_train, y_train, sw_train

        model = PenalizedGeneralizedLinearModel(
            loss="gamma",
            loss_kwargs=getattr(self, "_loss_kwargs", None),
            penalty="l2",
            alpha=float(alpha_sorted[0]),
            l1_ratio=self.l1_ratio,
            penalty_kwargs=getattr(self, "_penalty_kwargs", None),
            device=cv_device,
            compute_inference=False,
            max_iter=max_iter,
            tol=tol,
            solver=cv_solver,
        )
        prev_coef = None
        prev_intercept = None

        for alpha_idx_sorted, alpha in enumerate(alpha_sorted):
            try:
                model.alpha = float(alpha)
                if getattr(model, "_penalty", None) is not None:
                    model._penalty.alpha = float(alpha)
                model._init_coef = (
                    None if prev_coef is None else np.asarray(prev_coef, dtype=np.float64)
                )
                model._init_intercept = prev_intercept
                model.fit(X_fit, y_fit, sample_weight=sw_fit)

                coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64).ravel()
                intercept = float(model.intercept_)
                Xv = np.asarray(_to_numpy(X_val), dtype=np.float64)
                yv = np.asarray(_to_numpy(y_val), dtype=np.float64).ravel()
                swv = (
                    None
                    if sw_val is None
                    else np.asarray(_to_numpy(sw_val), dtype=np.float64).ravel()
                )
                val = cv_mod._evaluate_loss_numpy(
                    "gamma",
                    loss_fn,
                    Xv,
                    yv,
                    coef,
                    intercept,
                    True,
                    sample_weight=swv,
                )
                all_scores[fold_idx, sort_idx[alpha_idx_sorted]] = val
                prev_coef = coef.copy()
                prev_intercept = intercept
            except Exception as exc:
                # Domain ValueError/RuntimeError is intentionally not in the
                # recoverable CV numerical-failure set, so it propagates.
                cv_mod._raise_unless_recoverable_cv_candidate_failure(exc)
                all_scores[fold_idx, sort_idx[alpha_idx_sorted]] = np.nan

    return all_scores


def _install_cv_contract() -> None:
    current = PenalizedGLM_CV._compute_cv_scores
    if getattr(current, _CV_MARKER, False):
        return

    @wraps(current)
    def wrapped(
        self,
        X,
        y,
        alpha_grid,
        cv_device,
        folds,
        sample_weight=None,
        max_iter=None,
        tol=None,
        strict=True,
    ):
        penalty_name = str(getattr(self.penalty, "name", self.penalty)).lower()
        inverse_gamma = (
            str(self.loss).lower() == "gamma"
            and str(getattr(self, "_loss_kwargs", {}).get("link", "log")) == "inverse_power"
        )
        if inverse_gamma and penalty_name == "l2":
            return _inverse_gamma_l2_cv_scores(
                self,
                X,
                y,
                alpha_grid,
                cv_device,
                folds,
                sample_weight=sample_weight,
                max_iter=max_iter,
                tol=tol,
            )
        return current(
            self,
            X,
            y,
            alpha_grid,
            cv_device,
            folds,
            sample_weight=sample_weight,
            max_iter=max_iter,
            tol=tol,
            strict=strict,
        )

    setattr(wrapped, _CV_MARKER, True)
    wrapped._statgpu_original = current
    PenalizedGLM_CV._compute_cv_scores = wrapped


def install_inverse_gamma_smooth_domain_contract() -> None:
    _install_penalized_contract()
    _install_cv_loss_evaluator()
    _install_cv_contract()
