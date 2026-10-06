"""Inverse Gaussian regression (GLM, log link)."""

from typing import Optional

from statgpu._config import Device
from statgpu.glm_core._family import InverseGaussian
from statgpu.linear_model._glm_base import GeneralizedLinearModel


class InverseGaussianRegression(GeneralizedLinearModel):
    """Inverse Gaussian regression with a fixed log link for positive responses.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Fit an unpenalized intercept; formula syntax overrides this choice.
    max_iter : int, default=100
        Solver iteration budget.
    tol : float, default=0.0001
        Numerical convergence tolerance.
    C : float, default=1.0
        Ordinary IRLS adds ||beta||²/(4C) for positive C; C=0 removes it. Explicit newton/lbfgs/fista ignore C.
    device : str or Device, default='auto'
        cpu, cuda (CuPy), torch (Torch CUDA), or auto. Explicit GPU requests require the corresponding CUDA backend.
    n_jobs : int or None, default=None
        Shared CPU-worker configuration; no parallel-fit guarantee.
    solver : str, default='auto'
        auto, irls, fista, newton, lbfgs; ordinary auto selects IRLS. Solver changes can change the C-penalized objective.
    gpu_memory_cleanup : bool, default=False
        Best-effort GPU memory-pool cleanup.
    compute_inference : bool, default=False
        Enable supported M-estimation coefficient inference.
    cov_type : str, default='nonrobust'
        nonrobust, hc0, hc1 for ordinary GLM inference; no hac_maxlags constructor argument.

    Methods
    -------
    fit(X=None, y=None, sample_weight=None, formula=None, data=None)
        Return self after array or formula fitting; analytic weights are normalized
        by their sum. Pass arrays or formula/data, not both.
    predict(X)
        Response means, shape (n_samples,) for complete prediction rows.
    summary()
        Return a string, without printing. Works without coefficient inference.
    family_to_loss()
        Return the loss-name string for the selected family.
    get_params(deep=True), set_params(**params)
        Shared estimator configuration methods.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Shared statistical helpers; resampling does not automatically refit GLMs.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,)
        Prediction slopes, including after GPU fitting.
    intercept_ : float
        Prediction intercept; zero when omitted.
    n_iter_ : int
        Solver iteration count; not a convergence certificate.
    _bse, _zvalues, _pvalues, _conf_int : numpy.ndarray or None
        Enabled supported inference; intercept first when fitted. Vector shape
        (k,), confidence intervals (k, 2), where k includes any intercept.
    loglikelihood, llf, aic, bic : float
        Pseudo-likelihood diagnostics omit parameter-independent constants.
        Weighted loglikelihood is -n times weighted-average per-row loss.

    Notes
    -----
    Shared ordinary GLM methods and fitted fields follow GeneralizedLinearModel.
    There is no score or predict_proba method. summary() returns a string rather
    than printing it. Formula syntax owns the intercept; pass formula/data or
    arrays, not both. Parameters family, formula and data are not constructor
    controls. Enabled inference uses a normal reference, with nonrobust/HC0/HC1
    covariance; unsupported covariance choices raise.

    Default positive-C auto/IRLS fitting is ridge-penalized, not unpenalized.
    The penalty is sum(beta**2)/(4*C) on the average-loss scale and excludes the
    intercept. C=0 removes it; explicit newton/lbfgs/fista ignore C. Analytic
    weights are normalized by their sum. Failed auto/IRLS/FISTA refits can leave
    mixed old estimates and new metadata: construct a fresh model after failure.
    Explicit newton/lbfgs currently restore the prior fit after a rejected refit.
    Formula prediction currently drops rows with missing predictors and returns
    a shorter unlabelled array. Resolve missing data first and check prediction
    length against input rows before associating outputs with observations.

    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(
            family="inverse_gaussian",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            compute_inference=compute_inference,
            cov_type=cov_type,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )

    def _get_family(self):
        return InverseGaussian()
