"""MCP-penalized regression (Zhang, Annals of Statistics 2010)."""

from __future__ import annotations

from typing import Union

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


class MCPRegression(PenalizedLinearRegression):
    """MCP-penalized regression.

    Non-convex penalty with an oracle property under asymptotic regularity
    and tuning conditions. This is not a finite-sample unbiasedness guarantee.
    Uses LLA+FISTA for optimization.

    Parameters
    ----------
    alpha : float, default=1.0
        Finite positive regularization strength for average squared loss.
    gamma : float, default=3.0
        Finite concavity parameter greater than 1 (conventional value 3.0).
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device.
    compute_inference : bool, default=False
        Keep False on this specialized wrapper: it does not expose
        inference_method, and fitting with inference enabled raises. For an explicit
        oracle or bootstrap request, use PenalizedLinearRegression with
        penalty="mcp" and the desired inference_method. Ordinary active-set
        intervals do not correct selection uncertainty.
    solver : str, default='auto'
        Optimization algorithm for this non-convex penalty; consult the solver
        compatibility guide for supported choices.
    gpu_memory_cleanup : bool, default=False
        Request best-effort release of reclaimable GPU cache memory.

    Methods
    -------
    fit(X=None, y=None, sample_weight=None, formula=None, data=None)
        Fit one response and return self. Analytic weights are finite,
        nonnegative, have positive total and normalize the average squared loss.
        Formula syntax controls the unpenalized intercept; supply formula/data
        instead of simultaneous array arguments.
    predict(X, return_cpu=True)
        Return (n_samples,) predictions; NumPy by default even after GPU fits.
    score(X, y, sample_weight=None)
        Return evaluation R-squared. Use a flat host response and independently
        validated evaluation weights; shared weight validation is incomplete.
    summary()
        Requires inference and raises for the supported prediction-only wrapper.
        Use a generic penalized-linear model for an explicit inference request.
    get_params(deep=True), set_params(**params)
        Read/update constructor configuration; valid updates require refitting.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Inherited helpers; they do not automatically repeat fitting or tuning.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,)
        Penalized prediction slopes.
    intercept_ : float
        Unpenalized intercept, or zero when omitted.
    n_iter_ : int
        Iteration count, not a certificate of global optimality.

    Notes
    -----
    Import from statgpu.linear_model; top-level statgpu does not export this
    class. With compute_inference=False, coefficient inference and inherited
    rsquared, rsquared_adj, fvalue, f_pvalue, llf, aic and bic are unavailable
    (normally None). Evaluate held-out predictions with score or a chosen loss.
    The continuation path serves optimization, not cross-validation. Non-convex
    fitting can return different local solutions; selecting a variable does not
    establish a causal effect or selection-adjusted significance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        compute_inference: bool = False,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        self.gamma = gamma
        super().__init__(
            penalty="mcp",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            compute_inference=compute_inference,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
            penalty_kwargs={"gamma": gamma},
        )
