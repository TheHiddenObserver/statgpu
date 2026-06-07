"""
Squared error loss: (1/(2n)) * ||y - Xw||^2

Convention: loss = (1/(2n)) * sum(resid^2)
Objective = (1/n)*loss + alpha*penalty, where alpha is the per-sample
penalty strength. This convention is consistent across all penalties
(L1, ElasticNet, L2, SCAD, MCP) and all loss families.

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
