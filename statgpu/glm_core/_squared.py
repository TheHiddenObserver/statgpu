"""
Squared error loss: (1/(2n)) * ||y - Xw||^2

Convention: loss = (1/(2n)) * sum(resid^2)
This matches sklearn's Lasso convention (1/(2n)) and ensures
FISTA effective penalty = alpha (not alpha*n).

For sklearn Ridge compatibility: sklearn Ridge uses
min ||y-Xw||^2 + alpha * ||w||^2, while statgpu's L2 penalty is
(alpha/2) * ||w||^2 with loss (1/(2n)) * ||y-Xw||^2.
These are equivalent when sklearn_alpha = n * statgpu_alpha.

Supports numpy / cupy / torch backends via _backend helpers.
"""
from statgpu.backends._array_ops import _eigvalsh
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

    def lipschitz(self, X, coef):
        XtX = X.T @ X
        return float(_eigvalsh(XtX)[-1]) / X.shape[0]
