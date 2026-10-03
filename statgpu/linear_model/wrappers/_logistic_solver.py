"""Backend-native L-BFGS adapter for the classifier's summed L2 objective."""

import warnings

import numpy as np

from statgpu.backends._array_ops import _copy_arr, _norm2_dev, _sum, _xp
from statgpu.glm_core._logistic import LogisticLoss
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.solvers._lbfgs import lbfgs_solver


class _LogisticObjective:
    """Keep the classifier's weight mass and intercept policy explicit.

    The shared GLM smooth-solvers normalize analytic weights. The classifier
    instead minimizes sum(w * Bernoulli NLL) + ||beta||**2 / (2*C), excluding
    the intercept. Store the original validated weights here and pass no
    weights/penalty to the generic solver so neither term is rescaled.
    """

    name = "logistic_l2_sum"

    def __init__(self, sample_weight, C, fit_intercept, normalizer=1.0):
        self.normalizer = normalizer
        self.sample_weight = sample_weight
        self.alpha = 1.0 / C if C > 0 else 0.0
        self.fit_intercept = fit_intercept
        self.loss = LogisticLoss()

    def preprocess(self, X, y):
        return X, y

    def fused_value_and_gradient(self, X, y, params):
        eta = X @ params
        values = self.loss.per_sample_value(eta, y)
        residual = self.loss.per_sample_gradient(eta, y)
        if self.sample_weight is not None:
            values = values * self.sample_weight
            residual = residual * self.sample_weight
        value = _sum(values)
        gradient = X.T @ residual
        if self.alpha:
            penalized = _copy_arr(params)
            if self.fit_intercept:
                penalized[0] = 0.0
            value = value + (0.5 * self.alpha) * _sum(penalized * penalized)
            gradient = gradient + self.alpha * penalized
        return value / self.normalizer, gradient / self.normalizer


def _fit_logistic_lbfgs(
    X_design, y, sample_weight, C, fit_intercept, max_iter, tol
):
    """Fit on the input backend, retaining generic L-BFGS stopping semantics."""
    # Scale the complete objective, including ridge, for stable line searches.
    # Scaling only the likelihood would change the meaning of C.
    normalizer = (
        float(X_design.shape[0])
        if sample_weight is None
        else float(_sum(sample_weight))
    )
    objective = _LogisticObjective(sample_weight, C, fit_intercept, normalizer)
    with warnings.catch_warnings(record=True) as caught:
        # The wrapper publishes one solver-specific convergence warning. Other
        # warnings, including line-search failure, remain visible to callers.
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", RuntimeWarning)
        params, n_iter = lbfgs_solver(
            objective, None, X_design, y, max_iter=max_iter, tol=tol
        )
    line_search_failed = False
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            continue
        if (
            issubclass(warning.category, RuntimeWarning)
            and "lbfgs_solver: line search failed" in str(warning.message)
        ):
            line_search_failed = True
        warnings.warn(warning.message, warning.category, stacklevel=3)

    value, gradient = objective.fused_value_and_gradient(X_design, y, params)
    xp = _xp(params)
    finite = bool(xp.all(xp.isfinite(params))) and bool(xp.isfinite(value))
    gradient_norm = float(_norm2_dev(gradient))
    if not finite or not np.isfinite(gradient_norm):
        raise FloatingPointError("LogisticRegression L-BFGS produced non-finite results")
    # The shared solver can stop on gradient norm or accepted parameter-step
    # norm. Its return value alone cannot distinguish budget exhaustion from
    # a small final step, so the boundary case requires gradient confirmation.
    converged = not line_search_failed and (
        gradient_norm < tol or n_iter < max_iter
    )
    return params, n_iter, converged
