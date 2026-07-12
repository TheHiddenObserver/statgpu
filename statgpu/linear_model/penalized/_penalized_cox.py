"""Penalized Cox proportional hazards model.

CoxPH is NOT a GLM — it inherits from LossBase, not GLMLoss.
This class provides a clean API with survival-specific parameters and prediction.
"""

__all__ = ["PenalizedCoxPHModel"]

import numpy as np
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy

from ._base import PenalizedGeneralizedLinearModel


class PenalizedCoxPHModel(PenalizedGeneralizedLinearModel):
    """Penalized Cox proportional hazards model.

    Minimizes: -partial_likelihood(X, time, event) + penalty(coef)

    The Cox PH model estimates log-hazard ratios:
        h(t|X) = h0(t) * exp(X @ coef)

    Supports L1, L2, ElasticNet, SCAD, and MCP penalties.

    Parameters
    ----------
    penalty : str or Penalty, default='l2'
        Penalty type.
    alpha : float, default=1.0
        Regularization strength.
    ties : str, default='breslow'
        Method for handling tied event times: 'breslow' or 'efron'.
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'fista_bb', 'newton'.
        'auto' selects Newton for smooth penalties.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    fit_intercept : bool, default=False
        Must be ``False``.  The Cox partial likelihood is invariant to an
        additive constant in the linear predictor, so an intercept is not
        identifiable and is never fitted.
    compute_inference : bool, default=False
        Penalized Cox inference is not yet implemented.  Passing ``True``
        raises ``NotImplementedError`` during ``fit``; use unpenalized
        :class:`statgpu.survival.CoxPH` when inference is required.
    device : str, default='auto'
        Device: 'auto', 'cpu', 'cuda', 'torch'.

    Examples
    --------
    >>> from statgpu.linear_model import PenalizedCoxPHModel
    >>> # y must be (n, 2) array with columns [time, event]
    >>> model = PenalizedCoxPHModel(penalty='l2', alpha=0.01)
    >>> model.fit(X, y_surv)
    >>> hazard_ratio = model.predict_hazard_ratio(X_test)

    >>> # Sparse Cox model with L1 penalty
    >>> model = PenalizedCoxPHModel(penalty='l1', alpha=0.05)
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        """Expose modern sklearn tags for a two-column survival target."""
        try:
            from sklearn.utils._tags import RegressorTags, Tags, TargetTags
        except ImportError:  # scikit-learn < 1.6
            return {"requires_y": True, "multioutput": True}

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(
                required=True,
                one_d_labels=False,
                two_d_labels=True,
                multi_output=True,
                single_output=False,
            ),
            regressor_tags=RegressorTags(),
        )

    def __init__(
        self,
        penalty="l2",
        alpha=1.0,
        *,
        ties="breslow",
        solver="auto",
        max_iter=1000,
        tol=1e-4,
        fit_intercept=False,
        l1_ratio=0.5,
        penalty_kwargs=None,
        device="auto",
        n_jobs=None,
        cpu_solver="fista",
        lipschitz_L=None,
        gpu_memory_cleanup=False,
        loss_kwargs=None,
        compute_inference=False,
        inference_method="debiased",
        cov_type="nonrobust",
        hac_maxlags=None,
        stopping="coef_delta",
        lla=True,
        max_lla_iters=50,
        lla_tol=1e-6,
    ):
        if fit_intercept:
            raise ValueError(
                "PenalizedCoxPHModel does not fit an intercept because the "
                "Cox partial likelihood cannot identify one; set "
                "fit_intercept=False."
            )
        ties_normalized = str(ties).lower()
        if ties_normalized not in {"breslow", "efron"}:
            raise ValueError("ties must be 'breslow' or 'efron'")
        if loss_kwargs is not None and "ties" in loss_kwargs:
            loss_ties = str(loss_kwargs["ties"]).lower()
            if loss_ties != ties_normalized:
                raise ValueError(
                    "ties and loss_kwargs['ties'] specify different tie methods"
                )

        super().__init__(
            loss="cox_ph",
            penalty=penalty,
            alpha=alpha,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=False,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            loss_kwargs=loss_kwargs,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )
        self.ties = ties if ties == ties_normalized else ties_normalized

    def _resolve_loss(self):
        """Resolve Cox loss while keeping public ``loss_kwargs`` clone-safe."""
        from statgpu.losses import get_loss

        kwargs = dict(self.loss_kwargs)
        if "ties" in kwargs:
            supplied = str(kwargs["ties"]).lower()
            if supplied != str(self.ties).lower():
                raise ValueError(
                    "ties and loss_kwargs['ties'] specify different tie methods"
                )
        kwargs["ties"] = str(self.ties).lower()
        return get_loss("cox_ph", **kwargs)

    @property
    def _effective_intercept(self):
        """Cox partial likelihood never has an identifiable intercept."""
        return False

    def _validate_inference_request(self):
        """Declare the current penalized Cox estimator estimation-only.

        Generic penalized-GLM sandwich/bootstrap inference assumes a
        one-dimensional response and is not valid for ``(time, event)`` Cox
        data.  Failing here gives callers a stable public contract instead of
        allowing a later shape-dependent ``ValueError``.
        """
        if self.compute_inference:
            raise NotImplementedError(
                "PenalizedCoxPHModel is currently estimation-only: "
                "compute_inference=True is not supported for penalized Cox "
                "models. Set compute_inference=False, or use "
                "statgpu.survival.CoxPH for unpenalized Cox inference."
            )

    def set_params(self, **params):
        """Set estimator parameters while preserving the no-intercept contract."""
        if params.get("fit_intercept", False):
            raise ValueError(
                "PenalizedCoxPHModel does not fit an intercept because the "
                "Cox partial likelihood cannot identify one; set "
                "fit_intercept=False."
            )
        if "ties" in params:
            ties = str(params["ties"]).lower()
            if ties not in {"breslow", "efron"}:
                raise ValueError("ties must be 'breslow' or 'efron'")
            params["ties"] = ties
        prospective_ties = str(params.get("ties", self.ties)).lower()
        prospective_loss_kwargs = params.get("loss_kwargs", self.loss_kwargs)
        if (
            prospective_loss_kwargs is not None
            and "ties" in prospective_loss_kwargs
            and str(prospective_loss_kwargs["ties"]).lower() != prospective_ties
        ):
            raise ValueError(
                "ties and loss_kwargs['ties'] specify different tie methods"
            )
        return super().set_params(**params)

    def _reset_fit_state(self):
        """Clear fitted state before every fit attempt."""
        self._fitted = False
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self._params = None
        self._selected_solver = None
        self._selected_backend_name = None
        self._penalty = None
        self._loss = None
        self._clear_inference_state()

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit without allowing a failed refit to expose stale coefficients."""
        self._reset_fit_state()
        try:
            if y is not None:
                if isinstance(y, dict):
                    event = np.asarray(_to_numpy(y["event"]), dtype=np.float64)
                else:
                    y_array = np.asarray(_to_numpy(y), dtype=np.float64)
                    if y_array.ndim != 2 or y_array.shape[1] != 2:
                        raise ValueError(
                            "y must be (n, 2) array with columns [time, event]"
                        )
                    event = y_array[:, 1]
                if not np.any(event == 1):
                    raise ValueError("at least one observed event is required")
            return super().fit(
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        except Exception:
            self._reset_fit_state()
            raise

    def predict(self, X, return_cpu=True):
        """Predict hazard ratio: ``exp(X @ coef)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_cpu : bool, default=True

        Returns
        -------
        hazard_ratio : ndarray of shape (n_samples,)
            ``exp(X @ coef)``, the hazard ratio relative to baseline.
        """
        return self.predict_hazard_ratio(X, return_cpu=return_cpu)

    def predict_hazard_ratio(self, X, return_cpu=True):
        """Predict hazard ratio: exp(X @ coef). Excludes intercept.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_cpu : bool, default=True

        Returns
        -------
        hr : ndarray of shape (n_samples,)
            exp(X @ coef), the hazard ratio (without baseline hazard).
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        backend_name = self._prediction_backend_name()

        if backend_name == "cupy":
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            if bool(cp.any(~cp.isfinite(Xb)).item()):
                raise ValueError("X must contain only finite values")
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            result = cp.exp(cp.clip(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            if bool(torch.any(~torch.isfinite(Xb)).item()):
                raise ValueError("X must contain only finite values")
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            result = torch.exp(torch.clamp(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        raw = X @ self.coef_
        return np.exp(np.clip(raw, -500.0, 500.0))

    def score(self, X, y, sample_weight=None):
        """Concordance index (C-index) for survival data.

        Returns the C-index measuring discrimination ability.
        Higher is better (0.5 = random, 1.0 = perfect).

        Note: C-index is a ranking metric and does not support sample_weight.
        The sample_weight parameter is accepted for sklearn API compatibility
        but is ignored during computation.
        """
        if sample_weight is not None:
            import warnings
            warnings.warn("sample_weight is not supported for C-index (ranking metric), ignoring.",
                          UserWarning, stacklevel=2)

        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        from statgpu.survival._risk_sets import counting_process_concordance

        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        y = np.asarray(_to_numpy(y), dtype=np.float64)
        if y.ndim == 2 and y.shape[1] == 2:
            time = y[:, 0]
            event = y[:, 1]
        else:
            raise ValueError("y must be (n, 2) array with columns [time, event]")
        if X_np.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        return float(
            counting_process_concordance(
                np.asarray(self.coef_, dtype=np.float64),
                X_np,
                time,
                event,
            )
        )
