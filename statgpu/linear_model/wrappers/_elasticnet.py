"""Elastic Net regression with GPU support.

``ElasticNet`` is a thin public wrapper over
:class:`~statgpu.linear_model.penalized.PenalizedLinearRegression` with
``penalty="elasticnet"``. It preserves the shared NumPy/CuPy/Torch solver and
post-fit inference contracts; the public default solver is FISTA.

The legacy standalone implementation has been moved to
``_elasticnet_legacy.py``.
"""

from __future__ import annotations

__all__ = ["ElasticNet"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression as _PenalizedLinearRegression


class ElasticNet(_PenalizedLinearRegression):
    """Elastic Net regression with normalized squared loss and an L1/L2 penalty.

    Parameters
    ----------
    alpha : float, default=1.0
        Nonnegative overall regularization strength.
    l1_ratio : float, default=0.5
        L1 proportion in [0, 1]; zero is the L2 objective, one the L1 objective.
    fit_intercept : bool, default=True
        Fit an unpenalized intercept; formula syntax controls formula fits.
    max_iter : int, default=1000
        Positive solver iteration budget.
    tol : float, default=1e-4
        Positive convergence tolerance.
    stopping : {'coef_delta', 'kkt'}, default='coef_delta'
        Stored request; currently ignored by direct Gaussian stopping checks.
        FISTA/coordinate descent use coefficient movement and ADMM uses
        primal/dual residuals. Selecting kkt does not certify optimality.
    device : str or Device, default='auto'
        'cpu', 'cuda' (CuPy), 'torch' (Torch CUDA), or automatic selection.
    n_jobs : int or None, default=None
        Shared CPU worker setting; not a solver selector or parallel-fit promise.
    solver : str, default='fista'
        Backend-neutral optimization algorithm; supported combinations depend
        on the penalty. See the solver-penalty compatibility reference.
    cpu_solver : str, default='fista'
        Deprecated compatibility argument; use solver. Explicit legacy values
        can warn or conflict with solver; see the migration reference.
    lipschitz_L : float or None, default=None
        Optional positive upper bound for smooth-loss gradient curvature.
    gpu_memory_cleanup : bool, default=False
        Attempt GPU memory-pool cleanup after fitting.
    compute_inference : bool, default=False
        Compute post-fit coefficient uncertainty without changing prediction.
    inference_method : str, default='debiased'
        'debiased', 'post_selection_ols', or 'bootstrap'; 'auto' resolves to
        debiased. Deprecated cpu_ols/gpu_ols aliases mean post_selection_ols.
    cov_type : str, default='nonrobust'
        Covariance choice for post-selection OLS. Debiased currently ignores
        HC/HAC choices; bootstrap only accepts nonrobust.
    hac_maxlags : int or None, default=None
        Nonnegative HAC lag for supported post-selection OLS inference.
    nodewise_alpha : float or None, keyword-only, default=None
        Positive design-side precision tuning for debiased inference, or an
        automatic rule. Does not change penalized prediction coefficients.

    Notes
    -----
    Weighted loss divides by the sum of analytic weights. Inference conditions
    on the chosen tuning and does not generally correct selection uncertainty.
    ``coef_`` and ``intercept_`` describe prediction; ``_params`` and inference
    arrays can describe corrected or selected-model reporting parameters.
    Fitted coefficient/inference arrays are NumPy; ``predict`` returns NumPy by
    default even after GPU fitting. Use ``return_cpu=False`` for native output.
    After weighted debiased inference, ``rsquared`` and ``rsquared_adj`` use a
    re-centered working response and can misstate raw weighted training R².
    Evaluate ``score(X, y, sample_weight=weights)`` on original observations
    after validating finite nonnegative evaluation weights with positive sum.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "fista",
        cpu_solver: str = "fista",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.stopping = str(stopping).lower()
        super().__init__(
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            stopping=stopping,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
        )

    def fit(self, X=None, y=None, sample_weight=None, initial_coef=None, **kwargs):
        """Fit Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        initial_coef : array-like of shape (n_features,), optional
            Starting coefficients, not the fitted solution from a previous call.
            A supplied vector currently remains stored when a later fit omits
            it. Use a fresh estimator for default initialization or a new width.
        **kwargs
            ``formula`` and ``data`` select the optional Patsy/DataFrame input.
            Do not supply arrays at the same time: formula parsing replaces them.

        Returns
        -------
        self
            Fitted estimator. Prediction coefficients remain penalized.
        """
        if initial_coef is not None:
            self._init_coef = np.asarray(initial_coef, dtype=np.float64)
        return super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
