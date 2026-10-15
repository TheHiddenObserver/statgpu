"""PenalizedLogisticRegression — thin wrapper over PenalizedGeneralizedLinearModel."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._predict_mixin import _ETA_CLIP


class PenalizedLogisticRegression(PenalizedGeneralizedLinearModel):
    """PenalizedLogisticRegression with fixed logistic loss.

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
        Logistic returns 0/1 labels (probability strictly above 0.5 selects 1),
        while other GLMs return response means.
    score(X, y, sample_weight=None)
        Response-scale R-squared, including on logistic labels; not accuracy or
        deviance pseudo-R-squared. Pass one-dimensional y and validate finite,
        nonnegative weights with positive sum; shared score checks are incomplete.
    get_params(deep=True), set_params(**params)
        Shared estimator configuration methods.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Shared statistical helpers, not automatic GLM resampling/refitting.

    predict_proba(X)
        Return (m, 2) probabilities for classes 0/1 on the fitted NumPy/CuPy/Torch
        backend. No return_cpu argument. Formula DataFrames are accepted after a
        formula fit. predict uses probability strictly above 0.5 for class 1;
        score is R-squared on labels, not classification accuracy.

    Notes
    -----
    This typed class fixes loss="logistic"; loss is not a constructor argument.
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
    No threshold, classification-metric or plotting methods are supplied.

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
            loss="logistic",
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

    def predict_proba(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_predict_X(X)
        backend_name = self._prediction_backend_name()
        if backend_name == "cupy":
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self._effective_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            p1 = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
            return cp.column_stack([1.0 - p1, p1])
        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            p1 = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -_ETA_CLIP, _ETA_CLIP)))
            return torch.column_stack([1.0 - p1, p1])
        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
        return np.column_stack([1.0 - p1, p1])



