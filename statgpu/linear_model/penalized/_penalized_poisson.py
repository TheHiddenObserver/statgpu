"""Penalized Poisson regression wrapper."""

from __future__ import annotations

from typing import Optional, Union
from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


class PenalizedPoissonRegression(PenalizedGeneralizedLinearModel):
    """PenalizedPoissonRegression with fixed poisson loss.

    Parameters
    ----------
    penalty : str or Penalty, default='l2'
        none, l1, l2, elasticnet, scad, mcp, adaptive_l1, supported group penalties, or a Penalty object.
    alpha : float, default=1.0
        Penalty strength on the average-loss scale. A supplied Penalty object owns its own configuration.
    l1_ratio : float, default=0.5
        L1 fraction for elasticnet.
    penalty_kwargs : dict or None, default=None
        Additional penalty constructor settings, e.g. groups or shape controls.
    fit_intercept : bool, default=True
        Unpenalized intercept; formula syntax takes precedence.
    max_iter : int, default=1000
        Per-solve iteration budget.
    tol : float, default=0.0001
        Numerical tolerance.
    device : str or Device, default='auto'
        cpu, cuda, torch, auto; see the backend guide.
    n_jobs : int or None, default=None
        Shared CPU-worker setting where used.
    cpu_solver : str, deprecated
        Historical compatibility parameter, default='fista'. It no longer selects the direct-fit algorithm;
        use solver instead. Explicit non-None user-supplied values emit FutureWarning,
        including the historical default; omitted defaults and internal clone
        replay do not. See the solver migration guide.
    solver : str, default='auto'
        Choices include 'auto', 'fista', 'fista_bb', 'admm', 'irls', 'newton',
        'lbfgs', and 'exact'. Support depends on the loss and penalty;
        unsupported explicit combinations raise an error.
    lipschitz_L : float or None, default=None
        Optional Lipschitz bound for compatible proximal paths.
    gpu_memory_cleanup : bool, default=False
        Best-effort GPU memory-pool cleanup.
    compute_inference : bool, default=False
        Run supported post-fit inference only when True.
    inference_method : str, default='auto'
        Resolve a supported method from loss/penalty; consult the inference matrix.
    cov_type : str, default='nonrobust'
        Method-specific covariance; non-Gaussian smooth inference supports nonrobust/hc0/hc1.
    hac_maxlags : int or None, default=None
        HAC lag control only on paths supporting HAC.
    stopping : str, default='coef_delta'
        Stored convergence request; direct sparse Gaussian fits currently ignore the kkt choice.
    lla : bool, default=True
        Enable local linear approximation for supported nonconvex penalties.
    max_lla_iters : int, default=50
        Maximum outer LLA iterations.
    lla_tol : float, default=1e-06
        Outer LLA convergence tolerance.
    loss_kwargs : dict or None, default=None
        Use None or an empty dictionary for this fixed loss. It exposes no
        extra loss-constructor options; other GLM families' link/dispersion/
        power arguments do not apply.

    Methods
    -------
    fit(X=None, y=None, sample_weight=None, formula=None, data=None)
        Return self; numeric X has shape (n, p), scalar-response y has shape (n,).
    predict(X, return_cpu=True)
        Return (m,) predictions; default NumPy, native fitted backend when False.
        Return positive Poisson response means, not count-class labels.
    score(X, y, sample_weight=None)
        Response-scale R-squared, not deviance pseudo-R-squared.
        Pass one-dimensional y and validate finite,
        nonnegative weights with positive sum; shared score checks are incomplete.
    get_params(deep=True), set_params(**params)
        Shared estimator configuration methods.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Shared statistical helpers, not automatic GLM resampling/refitting.

    Notes
    -----
    This typed class fixes loss="poisson"; loss is not a constructor argument.
    Alpha multiplies the slope penalty on the average-loss scale, with an
    unpenalized intercept. Formula input and analytic weights follow the shared
    fit interface. Read supported inference through _inference_result (params,
    bse, statistic, pvalues, conf_int, method, distribution and metadata).
    Prediction coef_ is a NumPy (p,) array and intercept_ is scalar. Inference
    vectors are (k,) and intervals (k, 2), with an intercept first when present;
    reporting parameters may differ from penalized prediction coefficients.
    get_params/set_params expose only this typed constructor's accepted controls.

    Default penalty is L2. There is no C or nodewise_alpha constructor control.
    The generic score is response-scale R-squared, not deviance or accuracy.
    There is no summary method; use the inference result container. Non-Gaussian
    L1/ElasticNet coefficient inference is unavailable. L2/no-penalty auto resolves
    to fixed-penalty M-estimation where supported; uncertainty does not adjust for
    selecting alpha. Non-Gaussian oracle refits have the documented reconstruction
    limitation and should not be used for inferential conclusions.
    No predict_proba or offset/exposure fit argument is supplied.

    """

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
        loss_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            loss="poisson",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
            loss_kwargs=loss_kwargs,
        )
