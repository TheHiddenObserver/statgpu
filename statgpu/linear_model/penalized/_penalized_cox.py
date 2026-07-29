"""Penalized Cox proportional hazards model.

CoxPH is NOT a GLM — it inherits from LossBase, not GLMLoss.
This class provides a clean API with survival-specific parameters and prediction.
"""

__all__ = ["PenalizedCoxPHModel"]

import numbers
import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp
from statgpu.backends._utils import (
    _is_complex_array,
    _require_real_array,
    _to_float_scalar,
    _to_numpy,
)
from statgpu.survival._cox_fit_adapter import _normalize_boolean_control
from statgpu.survival._numeric import _safe_exp_linear_predictor

from ._base import PenalizedGeneralizedLinearModel

class PenalizedCoxPHModel(PenalizedGeneralizedLinearModel):
    _SUPPORTED_PENALTY_NAMES = frozenset(
        {
            "",
            "none",
            "null",
            "l1",
            "l2",
            "l2_squared",
            "ridge",
            "elasticnet",
            "en",
            "scad",
            "mcp",
        }
    )
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
        for name, value in (
            ("fit_intercept", fit_intercept),
            ("gpu_memory_cleanup", gpu_memory_cleanup),
            ("compute_inference", compute_inference),
            ("lla", lla),
        ):
            _normalize_boolean_control(value, name)
        if bool(fit_intercept):
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
        for name in (
            "fit_intercept",
            "gpu_memory_cleanup",
            "compute_inference",
            "lla",
        ):
            if name in params:
                _normalize_boolean_control(params[name], name)
        if bool(params.get("fit_intercept", False)):
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
        if "penalty" in params:
            self._validate_supported_penalty(params["penalty"])
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
        self._release_loss_fit_cache()
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

    def _release_loss_fit_cache(self):
        """Drop large sorted training arrays retained by the Cox loss."""
        loss = getattr(self, "_loss", None)
        release = getattr(loss, "release_fit_cache", None)
        if release is not None:
            release()

    def _cleanup_backend_memory(self, backend_name):
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _cleanup_selected_backend_memory(self):
        self._cleanup_backend_memory(
            getattr(self, "_selected_backend_name", None)
        )

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

    @classmethod
    def _validate_supported_penalty(cls, penalty):
        from statgpu.penalties import (
            ElasticNetPenalty,
            L1Penalty,
            L2Penalty,
            MCPPenalty,
            Penalty,
            SCADPenalty,
        )

        name = str(getattr(penalty, "name", penalty)).lower().strip()
        if name not in cls._SUPPORTED_PENALTY_NAMES:
            raise ValueError(
                "PenalizedCoxPHModel supports only L1, L2/Ridge, "
                "ElasticNet, SCAD, MCP, or no penalty; "
                f"got penalty={name!r}"
            )
        if not isinstance(penalty, Penalty):
            return

        supported_types = (
            L1Penalty,
            L2Penalty,
            ElasticNetPenalty,
            SCADPenalty,
            MCPPenalty,
        )
        if not isinstance(penalty, supported_types):
            raise ValueError(
                "PenalizedCoxPHModel accepts only built-in validated penalty "
                "objects for L1, L2, ElasticNet, SCAD, or MCP"
            )

        try:
            alpha = float(penalty.alpha)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("penalty object alpha must be finite") from exc
        minimum = 0.0 if isinstance(
            penalty, (L1Penalty, L2Penalty, ElasticNetPenalty)
        ) else np.nextafter(0.0, 1.0)
        if not np.isfinite(alpha) or alpha < minimum:
            qualifier = "non-negative" if minimum == 0.0 else "positive"
            raise ValueError(f"penalty object alpha must be finite and {qualifier}")
        if isinstance(penalty, ElasticNetPenalty):
            l1_ratio = float(penalty.l1_ratio)
            if not np.isfinite(l1_ratio) or not 0.0 <= l1_ratio <= 1.0:
                raise ValueError("penalty object l1_ratio must be between 0 and 1")
        if isinstance(penalty, SCADPenalty):
            a = float(penalty.a)
            if not np.isfinite(a) or a <= 2.0:
                raise ValueError("SCAD penalty object a must be greater than 2")
        if isinstance(penalty, MCPPenalty):
            gamma = float(penalty.gamma)
            if not np.isfinite(gamma) or gamma <= 1.0:
                raise ValueError("MCP penalty object gamma must be greater than 1")

    def _validate_cox_hyperparameters(self):
        self._validate_supported_penalty(self.penalty)
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

    @staticmethod
    def _validate_event_target(y):
        """Validate event values while transferring only two status scalars."""
        if isinstance(y, dict):
            if "time" not in y or "event" not in y:
                raise ValueError("survival y dict must contain time and event")
            if _is_complex_array(y["time"]):
                raise ValueError("time must be real-valued")
            event_raw = y["event"]
        else:
            if _is_complex_array(y):
                raise ValueError("y must be real-valued")
            target_xp = _get_xp(y)
            target = (
                y
                if target_xp.__name__ == "torch"
                else target_xp.asarray(y)
            )
            if target.ndim != 2 or int(target.shape[1]) != 2:
                raise ValueError(
                    "y must be (n, 2) array with columns [time, event]"
                )
            event_raw = target[:, 1]

        if _is_complex_array(event_raw):
            raise ValueError("event must be real-valued")
        xp = _get_xp(event_raw)
        if xp.__name__ == "torch":
            event = event_raw.to(dtype=xp.float64)
        else:
            event = xp.asarray(event_raw, dtype=xp.float64)
        invalid = xp.any(
            ~xp.isfinite(event) | ((event != 0) & (event != 1))
        )
        has_event = xp.any(event == 1)
        status = xp.stack((invalid, has_event))
        invalid_host, has_event_host = np.asarray(
            _to_numpy(status), dtype=bool
        )
        if bool(invalid_host):
            raise ValueError("event must contain only 0/1 finite values")
        if not bool(has_event_host):
            raise ValueError("at least one observed event is required")

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

            if X is not None and _is_complex_array(X):
                raise ValueError("X must be real-valued")
            if self._init_coef is not None and _is_complex_array(self._init_coef):
                raise ValueError("coef must be real-valued")
            if y is not None:
                self._validate_event_target(y)

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
            backend_name = getattr(self, "_selected_backend_name", None)
            self._reset_fit_state()
            self._cleanup_backend_memory(backend_name)
            raise
        finally:
            self._release_loss_fit_cache()
            self._cleanup_selected_backend_memory()

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
        """Predict hazard ratios and release unused backend cache blocks."""
        try:
            return self._predict_hazard_ratio_impl(X, return_cpu=return_cpu)
        finally:
            self._cleanup_selected_backend_memory()

    def predict_risk_score(self, X, return_cpu=True):
        """Predict unexponentiated log-risk on the selected backend."""
        try:
            return self._predict_risk_score_impl(X, return_cpu=return_cpu)
        finally:
            self._cleanup_selected_backend_memory()

    def _penalized_cox_prediction_backend(self):
        """Resolve the fitted prediction backend through BaseEstimator."""
        return self._get_backend(backend=self._prediction_backend_name())

    def _prepare_penalized_cox_prediction(self, X):
        """Normalize a real finite prediction matrix on the fitted backend."""
        _require_real_array(X, "X")
        backend = self._penalized_cox_prediction_backend()
        Xb = backend.asarray(X, dtype=backend.float64)
        if Xb.ndim == 1:
            Xb = Xb.reshape(-1, 1)
        if bool(_to_float_scalar(backend.xp.any(~backend.xp.isfinite(Xb)))):
            raise ValueError("X must contain only finite values")
        return backend, Xb

    @staticmethod
    def _prepare_penalized_cox_target(y, backend):
        """Normalize a right-censored target without backend-specific code."""
        if isinstance(y, dict):
            if "time" not in y or "event" not in y:
                raise ValueError("survival y dict must contain time and event")
            _require_real_array(y["time"], "time")
            _require_real_array(y["event"], "event")
            time = backend.asarray(
                y["time"], dtype=backend.float64
            ).reshape(-1)
            event = backend.asarray(
                y["event"], dtype=backend.float64
            ).reshape(-1)
            return time, event

        _require_real_array(y, "y")
        yb = backend.asarray(y, dtype=backend.float64)
        if yb.ndim != 2 or int(yb.shape[1]) != 2:
            raise ValueError(
                "y must be (n, 2) array with columns [time, event]"
            )
        return yb[:, 0], yb[:, 1]

    def _predict_risk_score_impl(self, X, return_cpu=True):
        """Return ``X @ coef`` without hazard-ratio range restrictions."""
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        backend, Xb = self._prepare_penalized_cox_prediction(X)
        coef = backend.asarray(self.coef_, dtype=backend.float64)
        result = Xb @ coef
        return backend.to_numpy(result) if return_cpu else result

    def _predict_hazard_ratio_impl(self, X, return_cpu=True):
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
        raw = self._predict_risk_score_impl(X, return_cpu=False)
        result = _safe_exp_linear_predictor(raw)
        return _to_numpy(result) if return_cpu else result

    def score(self, X, y, sample_weight=None):
        """Return Harrell concordance and release unused backend cache blocks."""
        try:
            return self._score_impl(X, y, sample_weight=sample_weight)
        finally:
            self._cleanup_selected_backend_memory()

    def _score_impl(self, X, y, sample_weight=None):
        """Return the backend-native Harrell concordance index.

        ``sample_weight`` is accepted for sklearn compatibility but is ignored
        because the shared concordance definition is pair-based.
        """
        if sample_weight is not None:
            import warnings

            warnings.warn(
                "sample_weight is not supported for C-index (ranking metric), "
                "ignoring.",
                UserWarning,
                stacklevel=2,
            )
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        from statgpu.survival._risk_sets import counting_process_concordance

        X = self._prepare_predict_X(X)
        backend, Xb = self._prepare_penalized_cox_prediction(X)
        time, event = self._prepare_penalized_cox_target(y, backend)
        coef = backend.asarray(self.coef_, dtype=backend.float64)
        if (
            int(time.shape[0]) != int(event.shape[0])
            or int(Xb.shape[0]) != int(time.shape[0])
        ):
            raise ValueError(
                "X, time, and event must contain the same number of rows"
            )

        return _to_float_scalar(
            counting_process_concordance(coef, Xb, time, event)
        )

    def __del__(self):
        try:
            self._release_loss_fit_cache()
            self._cleanup_selected_backend_memory()
        except Exception:
            pass
