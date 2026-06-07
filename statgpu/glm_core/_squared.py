"""
Squared error loss: (1/(2n)) * ||y - Xw||^2

Convention: loss = (1/(2n)) * sum(resid^2).
All penalties use alpha*n in the normal equations / CD updates,
matching the PenalizedGeneralizedLinearModel convention and sklearn.

sklearn compatibility mapping:
  - Ridge:   statgpu alpha = sklearn alpha * n
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

    def value(self, X, y, coef):
        resid = y - X @ coef
        return 0.5 * (resid * resid).sum() / X.shape[0]

    def gradient(self, X, y, coef):
        return X.T @ (X @ coef - y) / X.shape[0]

    def hessian(self, X, y, coef):
        return X.T @ X / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        XtX = X.T @ X
        return _max_eigval_power(XtX) / X.shape[0]
