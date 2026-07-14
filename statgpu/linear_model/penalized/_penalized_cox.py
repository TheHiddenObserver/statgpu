"""Penalized Cox proportional hazards model.

CoxPH is NOT a GLM — it inherits from LossBase, not GLMLoss.
This class provides a clean API with survival-specific parameters and prediction.
"""

__all__ = ["PenalizedCoxPHModel"]

import numbers
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
        if "max_iter" in params:
            self._validate_positive_integer(params["max_iter"], "max_iter")
        if "max_lla_iters" in params:
            self._validate_positive_integer(
                params["max_lla_iters"], "max_lla_iters"
            )
        for name in ("tol", "lla_tol"):
            if name in params:
                self._validate_finite_positive(params[name], name)
        if params.get("lipschitz_L", self.lipschitz_L) is not None:
            self._validate_finite_positive(
                params.get("lipschitz_L", self.lipschitz_L), "lipschitz_L"
            )
        if "alpha" in params:
            alpha = float(params["alpha"])
            if not np.isfinite(alpha) or alpha < 0:
                raise ValueError("alpha must be a finite non-negative number")
        if "l1_ratio" in params:
            l1_ratio = float(params["l1_ratio"])
            if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
                raise ValueError("l1_ratio must be between 0 and 1")

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
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._use_intercept = None
        self._clear_inference_state()

    @staticmethod
    def _validate_positive_integer(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_finite_positive(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive number") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")

    def _validate_cox_hyperparameters(self):
        try:
            alpha = float(self.alpha)
            l1_ratio = float(self.l1_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha and l1_ratio must be finite numbers") from exc
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite non-negative number")
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        self._validate_positive_integer(self.max_iter, "max_iter")
        self._validate_finite_positive(self.tol, "tol")
        self._validate_positive_integer(self.max_lla_iters, "max_lla_iters")
        self._validate_finite_positive(self.lla_tol, "lla_tol")
        if self.lipschitz_L is not None:
            self._validate_finite_positive(self.lipschitz_L, "lipschitz_L")

    @staticmethod
    def _parse_survival_formula(formula, data):
        if data is None:
            raise ValueError(
                "formula was provided but data is None. "
                "Pass data=your_dataframe when using formula."
            )
        try:
            import pandas as pd
            import patsy
            from patsy import EvalEnvironment
        except ImportError as exc:
            raise ImportError(
                "pandas and patsy are required for the penalized Cox formula interface"
            ) from exc
        if not isinstance(data, pd.DataFrame):
            raise TypeError("formula data must be a pandas DataFrame")
        from statgpu.core.formula import make_surv_env

        formula_data = data.copy(deep=False)
        formula_data.index = np.arange(len(data), dtype=np.int64)
        y_patsy, X_patsy = patsy.dmatrices(
            formula,
            formula_data,
            eval_env=EvalEnvironment([make_surv_env()]),
            return_type="dataframe",
        )
        y_array = np.asarray(y_patsy, dtype=np.float64)
        if y_array.ndim != 2 or y_array.shape[1] not in (2, 3):
            raise ValueError(
                "Formula response must be Surv(time, event) or "
                "Surv(start, stop, event)"
            )
        if y_array.shape[1] == 3:
            raise NotImplementedError(
                "PenalizedCoxPHModel currently supports right-censored "
                "Surv(time, event) formulas only; use statgpu.survival.CoxPH "
                "for start-stop data."
            )
        design_info = X_patsy.design_info
        column_names = list(design_info.column_names)
        has_intercept = "Intercept" in column_names
        X_array = np.asarray(X_patsy, dtype=np.float64)
        if has_intercept:
            X_array = np.delete(X_array, column_names.index("Intercept"), axis=1)
        feature_names = [name for name in column_names if name != "Intercept"]
        return X_array, y_array, design_info, has_intercept, feature_names

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit without allowing a failed refit to expose stale coefficients."""
        self._reset_fit_state()
        try:
            self._validate_cox_hyperparameters()
            if sample_weight is not None:
                raise NotImplementedError(
                    "PenalizedCoxPHModel does not support sample_weight"
                )

            formula_state = None
            if formula is not None:
                if X is not None or y is not None:
                    raise ValueError("pass either formula+data or X+y, not both")
                X, y, design_info, has_intercept, feature_names = (
                    self._parse_survival_formula(formula, data)
                )
                formula_state = (
                    design_info,
                    has_intercept,
                    feature_names,
                )
                formula = None
                data = None

            if y is not None:
                if isinstance(y, dict):
                    if "time" not in y or "event" not in y:
                        raise ValueError("survival y dict must contain time and event")
                    event = np.asarray(_to_numpy(y["event"]), dtype=np.float64)
                else:
                    y_array = np.asarray(_to_numpy(y), dtype=np.float64)
                    if y_array.ndim != 2 or y_array.shape[1] != 2:
                        raise ValueError(
                            "y must be (n, 2) array with columns [time, event]"
                        )
                    event = y_array[:, 1]
                if not np.all(np.isfinite(event)) or np.any(
                    (event != 0) & (event != 1)
                ):
                    raise ValueError("event must contain only 0/1 finite values")
                if not np.any(event == 1):
                    raise ValueError("at least one observed event is required")

            result = super().fit(
                X=X,
                y=y,
                sample_weight=None,
                formula=formula,
                data=data,
            )
            if formula_state is not None:
                self._design_info, self._formula_has_intercept, self._feature_names = (
                    formula_state
                )
                self._use_intercept = False
            return result
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
