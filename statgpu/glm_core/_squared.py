"""
Squared error loss: (1/(2n)) * ||y - Xw||^2

Convention: loss = (1/(2n)) * sum(resid^2).
All penalties use alpha*n in the normal equations / CD updates,
matching the PenalizedGeneralizedLinearModel convention and sklearn.

sklearn compatibility mapping:
  - Ridge:   sklearn alpha = statgpu alpha * n  (statgpu alpha = sklearn alpha / n)
  - Lasso:   statgpu alpha = sklearn alpha
  - ElasticNet: statgpu alpha = sklearn alpha

Internal consistency: Ridge(alpha=a) == PGLM(alpha=a, penalty='l2')
for all alpha values (verified to machine precision).

Supports numpy / cupy / torch backends via _backend helpers.
"""
from statgpu.backends._array_ops import _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('squared_error')
class SquaredErrorLoss(GLMLoss):
    name = "squared_error"
    y_type = "continuous"
    smooth_gradient = True
    has_hessian = True

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        resid = y - eta
        return 0.5 * resid * resid

    def per_sample_gradient(self, eta, y):
        return eta - y

    # ── Hessian / Lipschitz (override for weighted support) ───────────

    def hessian(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            return X.T @ (X * sample_weight[:, None]) / sample_weight.sum()
        return X.T @ X / X.shape[0]

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        if sample_weight is not None:
            sw = sample_weight[:, None] if hasattr(sample_weight, '__len__') else sample_weight
            XtWX = X.T @ (X * sw)
            return _max_eigval_power(XtWX) / sample_weight.sum()
        XtX = X.T @ X
        return _max_eigval_power(XtX) / X.shape[0]
