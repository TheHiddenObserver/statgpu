"""
Cox Proportional Hazards regression with GPU acceleration.

Implements Cox PH models with Breslow, Efron, and Exact tie handling,
counting-process risk sets, and Newton-Raphson optimization.
"""

from typing import Optional, Union
from functools import wraps
import numbers
import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _is_cupy_array, _is_torch_array, _to_float_scalar
from statgpu.backends._utils import _require_real_array
from statgpu.inference._distributions_backend import chi2, norm
from statgpu.inference._results import ParameterInferenceResult
from statgpu.survival._cox_fit_adapter import (
    _is_native_backend_array,
    _normalize_boolean_control,
    _normalize_mutable_fit_controls,
    _PreencodedCoxLabels,
)
from statgpu.survival._cox_errors import CoxFitNumericalError
from statgpu.survival._cox_counting import _score_test_statistic
from statgpu.survival._cox_inference import (
    _invert_information_cupy,
    _invert_information_numpy,
    _invert_information_torch,
)
from statgpu.survival._numeric import _safe_exp_linear_predictor


def _cleanup_after_public_gpu_work(method):
    """Run both estimator cleanup hooks after public prediction/scoring work."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()

    return wrapped


def _is_device_resident_array(value):
    """Return whether an input already occupies accelerator memory."""
    if value is None:
        return False
    if _is_cupy_array(value):
        return True
    if _is_torch_array(value):
        device = getattr(value, "device", None)
        return str(getattr(device, "type", device)).lower() != "cpu"
    return False


def _align_cox_side_array(values, retained_rows, original_n, name="array"):
    """Filter a side array to match rows retained by Patsy after NA drops.

    Parameters
    ----------
    values : array-like or None
        Side array (entry, cluster, etc.).  May be NumPy, CuPy, or Torch.
    retained_rows : ndarray of int64
        Zero-based row positions kept by Patsy.
    original_n : int
        Number of rows in the original DataFrame.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    array-like or None
        Filtered array matching the retained rows, or None.
    """
    if values is None:
        return None

    # Detect backend BEFORE any np.asarray() to avoid CuPy 13.x implicit
    # conversion errors and unnecessary GPU→CPU transfers.
    module = type(values).__module__

    if module.startswith("cupy"):
        import cupy as cp
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        n_values = int(values.shape[0])
        n_retained = len(retained_rows)
        if n_values == n_retained:
            return values
        if n_values != original_n:
            raise ValueError(
                f"{name} length {n_values} does not match "
                f"original data length {original_n}"
            )
        idx = cp.asarray(retained_rows, dtype=cp.int64)
        return values[idx]

    if module.startswith("torch"):
        import torch
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        n_values = int(values.shape[0])
        n_retained = len(retained_rows)
        if n_values == n_retained:
            return values
        if n_values != original_n:
            raise ValueError(
                f"{name} length {n_values} does not match "
                f"original data length {original_n}"
            )
        idx = torch.as_tensor(retained_rows, dtype=torch.long, device=values.device)
        return values.index_select(0, idx)

    # NumPy / list / pandas path
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    n_values = arr.shape[0]
    if n_values == len(retained_rows):
        return values
    if n_values != original_n:
        raise ValueError(
            f"{name} length {n_values} does not match "
            f"original data length {original_n}"
        )
    return arr[retained_rows]


class CoxPH(BaseEstimator):
    """
    Cox Proportional Hazards regression with GPU acceleration.
    
    Parameters
    ----------
    ties : str, default='breslow'
        Method for handling ties: 'breslow', 'efron', or 'exact'.
    tol : float, default=1e-9
        Convergence tolerance for Newton-Raphson.
    max_iter : int, default=100
        Maximum number of iterations.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', 'torch', or 'auto'.
    compute_inference : bool, default=True
        If True, compute standard errors, tests, and baseline hazards on the
        active backend. Set to False to skip these outputs and reduce work.
    compute_cindex : bool, default=True
        If True, compute training-set C-index during fit. Disabling this can
        significantly reduce fit time, especially on CUDA/Torch for moderate n.
    cov_type : {'nonrobust', 'hc0', 'hc1', 'cluster'}, default='nonrobust'
        Covariance estimator. Cluster covariance requires ``cluster`` in fit.
    penalty : float, default=0.0
        Non-negative L2 penalty.
    inference_mode : {'strict', 'approx'}, default='strict'
        Robust-inference compatibility control. Both values currently use the
        exact counting-process score sandwich; ``'approx'`` remains accepted
        for backward compatibility and does not select an approximate path.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (log hazard ratios).
    hazard_ratios_ : ndarray of shape (n_features,)
        exp(coef) = hazard ratios.
    converged_ : bool
        Whether the final normalized KKT condition met its tolerance.
    termination_reason_ : str
        One of ``kkt_converged``, ``line_search_failed``,
        ``stalled_with_large_kkt``, or ``max_iter``.
    """

    _estimator_type = "regressor"
    _canonical_fit_path = "counting_process"

    def __sklearn_tags__(self):
        """Expose sklearn tags for packed two/three-column survival targets."""
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
        ties: str = 'breslow',
        tol: float = 1e-9,
        max_iter: int = 100,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        compute_cindex: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        penalty: float = 0.0,
        inference_mode: str = 'strict',
    ):
        for name, value in (
            ("compute_inference", compute_inference),
            ("compute_cindex", compute_cindex),
            ("gpu_memory_cleanup", gpu_memory_cleanup),
        ):
            _normalize_boolean_control(value, name)
        super().__init__(device=device, n_jobs=n_jobs)
        ties_normalized = str(ties).lower()
        cov_type_normalized = str(cov_type).lower()
        inference_mode_normalized = str(inference_mode).lower()
        # Preserve canonical constructor objects so sklearn.clone can verify
        # that __init__ does not mutate public parameters.
        self.ties = ties if ties == ties_normalized else ties_normalized
        self.tol = tol
        self.max_iter = max_iter
        self.compute_inference = compute_inference
        self.compute_cindex = bool(compute_cindex)
        self.cov_type = (
            cov_type if cov_type == cov_type_normalized else cov_type_normalized
        )
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.penalty = float(penalty)
        self.inference_mode = (
            inference_mode
            if inference_mode == inference_mode_normalized
            else inference_mode_normalized
        )

        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, numbers.Integral
        ) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        try:
            tol_value = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("tol must be a finite positive number") from exc
        if not np.isfinite(tol_value) or tol_value <= 0:
            raise ValueError("tol must be a finite positive number")
        if not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("penalty must be a finite non-negative number")
        if self.ties not in ('breslow', 'efron', 'exact'):
            raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "cluster"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'")
        if self.inference_mode not in ('strict', 'approx'):
            raise ValueError('inference_mode must be strict or approx')
        if self.penalty < 0:
            raise ValueError("penalty must be non-negative")
        
        # Keep fitted-state initialization and failed-refit cleanup identical.
        self._reset_fit_state()

    def _reset_fit_state(self):
        """Clear data-dependent state before every fit attempt.

        A failed or deliberately zero-iteration refit must not expose
        coefficients, convergence, inference, or baseline-hazard results from
        an earlier successful fit on the same estimator instance.
        """
        self._fitted = False
        self.coef_ = None
        self.hazard_ratios_ = None
        self._time = None
        self._event = None
        self._X = None
        self._entry = None
        self._nobs = None
        self._nevents = None
        self._bse = None
        self._zvalues = None
        self._tvalues = None
        self._pvalues = None
        self._conf_int = None
        self._params = None
        self._inference_result = None
        self._log_likelihood = None
        self._log_likelihood_null = None
        self._iterations = 0
        self._converged = False
        self._termination_reason = None
        self._final_kkt_inf = None
        self._final_kkt_normalized = None
        self._penalized_objective = None
        self._objective_history = []
        self.converged_ = False
        self.termination_reason_ = None
        self.n_iter_ = 0
        self.final_kkt_inf_ = None
        self.final_kkt_normalized_ = None
        self.inference_method_ = None
        self.inference_backend_ = None
        self.inference_approximate_ = False
        self.inference_fallback_reason_ = None
        self.full_host_transfer_performed_ = False
        self.concordance_ = None
        self._var_matrix = None
        self._score_test_stat = None
        self._score_test_pvalue = None
        self.score_test_available_ = False
        self.score_test_failure_reason_ = None
        self._wald_test_stat = None
        self._wald_test_pvalue = None
        self._lr_test_stat = None
        self._lr_test_pvalue = None
        self._baseline_hazard = None
        self._baseline_cumulative_hazard = None
        self._baseline_log_hazard = None
        self._baseline_log_cumulative_hazard = None
        self._baseline_log_cumulative_hazard_centered = None
        self._baseline_x_reference = None
        self._unique_times = None
        self._cindex = None
        self._feature_names = None
        self._design_info = None
        self._baseline_by_stratum = None
        self._strata = None
        self._strata_labels = None
        self._subject_id = None
        self._is_counting_process = False
        self._fit_call = None
        self._stop_reason = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA cache cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
        except Exception:
            pass

    def _validate_optimization_controls(self):
        """Validate mutable optimization controls before every fit attempt."""
        if isinstance(self.max_iter, (bool, np.bool_)) or not isinstance(
            self.max_iter, numbers.Integral
        ) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        try:
            tol = float(self.tol)
            penalty = float(self.penalty)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tol and penalty must be finite numeric values"
            ) from exc
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite positive number")
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be a finite non-negative number")

    def fit(
        self,
        X=None,
        time=None,
        event=None,
        entry=None,
        cluster=None,
        init_coef=None,
        formula=None,
        data=None,
        *,
        start=None,
        strata=None,
        subject_id=None,
        _right_censored_prepared=None,
    ):
        """Fit and clear all state if validation or inference fails."""
        self._reset_fit_state()
        try:
            _normalize_mutable_fit_controls(self)
            if formula is None and X is not None:
                x_shape = getattr(X, "shape", None)
                if x_shape is None:
                    x_shape = np.asarray(X).shape
                if len(x_shape) not in (1, 2):
                    raise ValueError("X must be a one- or two-dimensional array")
                if len(x_shape) == 2 and int(x_shape[1]) < 1:
                    raise ValueError("X must contain at least one feature")
            _require_real_array(X, "X")
            if formula is None and event is None and time is not None:
                _require_real_array(time, "packed survival target")
            else:
                _require_real_array(time, "time")
                _require_real_array(event, "event")
            _require_real_array(entry, "entry")
            _require_real_array(start, "start")
            _require_real_array(init_coef, "init_coef")
            if formula is None and event is None and time is not None:
                target = time
                if not _is_native_backend_array(target):
                    target = np.asarray(target)
                if target.ndim != 2 or target.shape[1] not in (2, 3):
                    raise ValueError(
                        "When event is omitted, time must be a survival target "
                        "with columns [time, event] or [start, stop, event]"
                    )
                if target.shape[1] == 2:
                    time, event = target[:, 0], target[:, 1]
                else:
                    if entry is not None or start is not None:
                        raise ValueError(
                            "Do not pass entry/start separately when the target "
                            "already has [start, stop, event] columns"
                        )
                    start, time, event = (
                        target[:, 0],
                        target[:, 1],
                        target[:, 2],
                    )
            if _right_censored_prepared is not None:
                if formula is not None or not _right_censored_prepared.matches_sources(
                    X, time, event, self.ties
                ):
                    raise ValueError(
                        "prepared right-censored metadata does not match the "
                        "current matrix fit inputs"
                    )
            result = self._fit_impl(
                X=X,
                time=time,
                event=event,
                entry=entry,
                cluster=cluster,
                init_coef=init_coef,
                formula=formula,
                data=data,
                start=start,
                strata=strata,
                subject_id=subject_id,
                _right_censored_prepared=_right_censored_prepared,
            )
            if not self._is_counting_process:
                self._entry = None
            coef = np.asarray(self.coef_, dtype=np.float64)
            if not np.all(np.isfinite(coef)) or not np.isfinite(
                self._log_likelihood
            ):
                raise CoxFitNumericalError(
                    "CoxPH fit produced non-finite coefficients or log-likelihood"
                )
            if self.compute_inference and any(
                value is None or not np.all(np.isfinite(value))
                for value in (self._bse, self._pvalues, self._conf_int)
            ):
                raise FloatingPointError(
                    "CoxPH inference produced non-finite standard errors, "
                    "p-values, or confidence intervals"
                )
            return result
        except Exception:
            self._reset_fit_state()
            raise

    def _fit_impl(
        self,
        X=None,
        time=None,
        event=None,
        entry=None,
        cluster=None,
        init_coef=None,
        formula=None,
        data=None,
        *,
        start=None,
        strata=None,
        subject_id=None,
        _right_censored_prepared=None,
    ):
        """
        Fit Cox Proportional Hazards model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix. Required if ``formula`` is None.
        time : array-like of shape (n_samples,)
            Time to event or censoring. Required if ``formula`` is None.
        event : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored). Required if ``formula`` is None.
        entry : array-like of shape (n_samples,), optional
            Entry time for delayed entry (left truncation).
        start : array-like of shape (n_samples,), optional
            Alias for ``entry`` used by counting-process data.  Rows are at
            risk on ``(start, time]``.  Pass only one of ``entry`` and ``start``.
        strata : array-like of shape (n_samples,), optional
            Stratum labels.  Coefficients are shared while each stratum gets an
            independent risk set and baseline hazard.
        subject_id : array-like of shape (n_samples,), optional
            Subject identifiers for time-varying data.  Used to exclude
            within-subject pairs from concordance calculations.
        cluster : array-like of shape (n_samples,), optional
            Cluster ids for cluster-robust covariance when `cov_type='cluster'`.
        init_coef : array-like of shape (n_features,), optional
            Initial coefficient guess for warm-start optimization.
        formula : str or None
            R-style formula with Surv() response, e.g.
            ``"Surv(time, event) ~ x1 + x2 + C(sex)"``.
        data : pd.DataFrame or None
            DataFrame used with ``formula`` for column lookup.

        Returns
        -------
        self : CoxPH
            Fitted estimator.
        """
        formula_entry_was_explicit = entry is not None or start is not None
        if entry is not None and start is not None:
            raise ValueError("pass only one of entry and start")
        if start is not None:
            entry = start

        # Handle formula interface
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            from statgpu.core.formula import make_surv_env
            import patsy
            from patsy import EvalEnvironment

            env = make_surv_env()
            custom_env = EvalEnvironment([env])
            if not hasattr(data, "copy") or not hasattr(data, "index"):
                raise TypeError("formula data must be a pandas DataFrame")
            # Use a positional RangeIndex so patsy's retained index is an
            # unambiguous row selector even when the caller's DataFrame index
            # contains duplicate labels.
            formula_data = data.copy(deep=False)
            formula_data.index = np.arange(len(data), dtype=np.int64)
            y_patsy, X_patsy = patsy.dmatrices(
                formula,
                formula_data,
                eval_env=custom_env,
                return_type="dataframe",
            )
            retained_rows = np.asarray(X_patsy.index, dtype=np.int64)

            n_original = len(data)
            entry = _align_cox_side_array(
                entry, retained_rows, n_original, "entry/start"
            )
            cluster = _align_cox_side_array(
                cluster, retained_rows, n_original, "cluster"
            )
            strata = _align_cox_side_array(
                strata, retained_rows, n_original, "strata"
            )
            subject_id = _align_cox_side_array(
                subject_id, retained_rows, n_original, "subject_id"
            )
            design_info = X_patsy.design_info
            # Surv(time, event) -> (n, 2); Surv(start, stop, event) -> (n, 3).
            y_arr = np.asarray(y_patsy)
            if y_arr.ndim == 1:
                raise ValueError(
                    "Formula response must be Surv(time, event), not a single variable. "
                    "Use: formula='Surv(time, event) ~ x1 + x2'"
                )
            if y_arr.shape[1] == 2:
                time = y_arr[:, 0]
                event = y_arr[:, 1]
            elif y_arr.shape[1] == 3:
                if formula_entry_was_explicit:
                    raise ValueError(
                        "Surv(start, stop, event) already defines entry times; "
                        "do not also pass entry= or start="
                    )
                entry = y_arr[:, 0]
                time = y_arr[:, 1]
                event = y_arr[:, 2]
            else:
                raise ValueError(
                    "Formula response must be Surv(time, event) or "
                    "Surv(start, stop, event)"
                )
            X_arr = np.asarray(X_patsy)

            # Drop intercept column from design matrix (CoxPH doesn't use intercept)
            self._feature_names = list(design_info.column_names)
            if "Intercept" in self._feature_names:
                intercept_index = self._feature_names.index("Intercept")
                X_arr = np.delete(X_arr, intercept_index, axis=1)
                self._feature_names = [
                    name
                    for index, name in enumerate(self._feature_names)
                    if index != intercept_index
                ]
            self._design_info = design_info
            X = X_arr
        else:
            if X is None or time is None or event is None:
                raise ValueError(
                    "Either formula+data or X+time+event must be provided."
                )
            self._design_info = None
        _require_real_array(X, "X")
        _require_real_array(time, "time")
        _require_real_array(event, "event")
        _require_real_array(entry, "entry/start")
        _require_real_array(init_coef, "init_coef")
        self._fit_call = {
            "interface": "formula" if formula is not None else "matrix",
            "formula": None if formula is None else str(formula),
            "counting_process": entry is not None or subject_id is not None,
            "stratified": strata is not None,
            "subject_grouped": subject_id is not None,
            "clustered": cluster is not None,
            "ties": self.ties,
        }
        device = self._get_compute_device()

        # The shared counting-process objective is the canonical implementation
        # for every Cox fit, including ordinary right-censored Breslow/Efron.
        # It uses risk-set-local scaling and therefore cannot overflow merely
        # because a finite initial coefficient gives a large predictor range.
        return self._fit_counting_process_dispatch(
            X,
            time,
            event,
            entry=entry,
            strata=strata,
            cluster=cluster,
            subject_id=subject_id,
            init_coef=init_coef,
            device=device,
            right_censored_prepared=_right_censored_prepared,
        )

    def set_params(self, **params):
        """Set sklearn-style parameters with Cox-specific validation."""
        if "ties" in params:
            ties = str(params["ties"]).lower()
            if ties not in {"breslow", "efron", "exact"}:
                raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
            params["ties"] = ties
        if "cov_type" in params:
            cov_type = str(params["cov_type"]).lower()
            if cov_type not in {"nonrobust", "hc0", "hc1", "cluster"}:
                raise ValueError(
                    "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'"
                )
            params["cov_type"] = cov_type
        if "max_iter" in params:
            max_iter = params["max_iter"]
            if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
                max_iter, numbers.Integral
            ) or int(max_iter) < 1:
                raise ValueError("max_iter must be a positive integer")
        if "tol" in params:
            try:
                tol = float(params["tol"])
            except (TypeError, ValueError) as exc:
                raise ValueError("tol must be a finite positive number") from exc
            if not np.isfinite(tol) or tol <= 0:
                raise ValueError("tol must be a finite positive number")
        if "penalty" in params:
            penalty = float(params["penalty"])
            if not np.isfinite(penalty) or penalty < 0:
                raise ValueError("penalty must be a finite non-negative number")
            params["penalty"] = penalty
        if "inference_mode" in params:
            mode = str(params["inference_mode"]).lower()
            if mode not in {"strict", "approx"}:
                raise ValueError("inference_mode must be strict or approx")
            params["inference_mode"] = mode
        return super().set_params(**params)

    @staticmethod
    def _encode_group_labels(
        values, n_samples, name, *, return_labels=True
    ):
        """Encode arbitrary labels without collapsing non-integral device values."""
        if values is None:
            return None, None
        if isinstance(values, _PreencodedCoxLabels):
            codes = values.codes
            if getattr(codes, "ndim", None) != 1 or int(codes.shape[0]) != n_samples:
                raise ValueError(f"{name} must have shape (n_samples,)")
            labels = values.labels.copy() if return_labels else None
            return codes, labels
        module = type(values).__module__
        if module.startswith("cupy"):
            import cupy as cp

            if getattr(values, "ndim", None) != 1 or int(values.shape[0]) != n_samples:
                raise ValueError(f"{name} must have shape (n_samples,)")
            if values.dtype.kind in "fc" and bool(cp.any(~cp.isfinite(values)).item()):
                raise ValueError(f"{name} must contain only finite labels")
            labels, encoded = cp.unique(values, return_inverse=True)
            labels_host = cp.asnumpy(labels) if return_labels else None
            return encoded.astype(cp.int64, copy=False), labels_host
        if module.startswith("torch"):
            import torch

            if getattr(values, "ndim", None) != 1 or int(values.shape[0]) != n_samples:
                raise ValueError(f"{name} must have shape (n_samples,)")
            if (values.is_floating_point() or values.is_complex()) and bool(
                torch.any(~torch.isfinite(values)).item()
            ):
                raise ValueError(f"{name} must contain only finite labels")
            labels, encoded = torch.unique(
                values, sorted=True, return_inverse=True
            )
            labels_host = (
                labels.detach().cpu().numpy() if return_labels else None
            )
            return encoded.to(dtype=torch.int64), labels_host
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.shape[0] != n_samples:
            raise ValueError(f"{name} must have shape (n_samples,)")
        if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite labels")
        labels, encoded = np.unique(arr, return_inverse=True)
        return encoded.astype(np.int64, copy=False), (
            labels if return_labels else None
        )

    def _fit_counting_process_dispatch(
        self,
        X,
        time,
        event,
        *,
        entry,
        strata,
        cluster,
        subject_id,
        init_coef,
        device,
        right_censored_prepared=None,
    ):
        """Fit entry/start-stop, stratified, or exact-ties Cox natively."""
        from statgpu.survival._cox_counting import fit_counting_process_cox
        from statgpu.survival._risk_sets import (
            counting_process_concordance,
            prepare_counting_process_inputs,
        )

        input_shape = getattr(X, "shape", None)
        if input_shape is None:
            input_shape = np.asarray(X).shape
        n_samples = int(input_shape[0])
        input_full_host_transfer = any(
            _is_device_resident_array(value)
            for value in (
                X,
                time,
                event,
                entry,
                strata,
                cluster,
                subject_id,
            )
        ) and device == Device.CPU
        strata_encoded, strata_labels = self._encode_group_labels(
            strata, n_samples, "strata"
        )
        cluster_encoded, _ = self._encode_group_labels(
            cluster, n_samples, "cluster", return_labels=False
        )
        subject_encoded, _ = self._encode_group_labels(
            subject_id, n_samples, "subject_id", return_labels=False
        )

        if (
            self.ties == "exact"
            and self.compute_inference
            and self.cov_type != "nonrobust"
        ):
            raise NotImplementedError(
                "robust covariance is not yet defined for ties='exact'; "
                "use cov_type='nonrobust'"
            )

        backend_name = {
            Device.CPU: "numpy",
            Device.CUDA: "cupy",
            Device.TORCH: "torch",
        }[device]
        compute_backend = self._get_backend(backend=backend_name)
        backend = compute_backend.name
        xp = compute_backend.xp
        Xb = compute_backend.asarray(X, dtype=compute_backend.float64)
        stopb = compute_backend.asarray(time, dtype=compute_backend.float64)
        eventb = compute_backend.asarray(event, dtype=compute_backend.float64)
        startb = (
            compute_backend.zeros(stopb.shape, dtype=compute_backend.float64)
            if entry is None
            else compute_backend.asarray(entry, dtype=compute_backend.float64)
        )
        stratab = (
            compute_backend.zeros((n_samples,), dtype=compute_backend.int64)
            if strata_encoded is None
            else compute_backend.asarray(
                strata_encoded, dtype=compute_backend.int64
            )
        )
        clusterb = (
            None
            if cluster_encoded is None
            else compute_backend.asarray(
                cluster_encoded, dtype=compute_backend.int64
            )
        )
        subjectb = (
            None
            if subject_encoded is None
            else compute_backend.asarray(
                subject_encoded, dtype=compute_backend.int64
            )
        )

        if Xb.ndim == 1:
            Xb = Xb.reshape(-1, 1)
        if entry is None and bool(_to_float_scalar(xp.any(stopb <= 0))):
            raise ValueError("time must contain only positive values")
        Xb, stopb, eventb, startb, stratab = prepare_counting_process_inputs(
            Xb,
            stopb,
            eventb,
            start=startb,
            strata=stratab,
        )
        right_censored_fast_path = (
            entry is None
            and strata is None
            and subject_id is None
            and self.cov_type == "nonrobust"
            and self.ties in {"breslow", "efron"}
        )
        if right_censored_prepared is not None and not right_censored_fast_path:
            raise ValueError(
                "prepared right-censored metadata is incompatible with this fit"
            )
        result = fit_counting_process_cox(
            Xb,
            stopb,
            eventb,
            start=startb,
            strata=stratab,
            ties=self.ties,
            penalty=self.penalty,
            tol=self.tol,
            max_iter=self.max_iter,
            init_coef=init_coef,
            compute_baseline=self.compute_inference,
            compute_score_residuals=(
                self.compute_inference and self.cov_type != "nonrobust"
            ),
            right_censored_fast_path=right_censored_fast_path,
            right_censored_prepared=right_censored_prepared,
            _inputs_prepared=True,
        )

        to_numpy = compute_backend.to_numpy
        scalar = _to_float_scalar

        self.coef_ = to_numpy(result["coef"]).astype(np.float64, copy=False)
        self.hazard_ratios_ = _safe_exp_linear_predictor(
            self.coef_,
            error_type=CoxFitNumericalError,
            name="fitted Cox coefficients",
        )
        self._log_likelihood = scalar(result["log_likelihood"])
        self._log_likelihood_null = scalar(result["null_log_likelihood"])
        self._iterations = int(result["iterations"])
        self._converged = bool(result["converged"])
        self._stop_reason = result["stop_reason"]
        self._objective_history = np.asarray(
            [scalar(value) for value in result["objective_history"]], dtype=np.float64
        )
        self._nobs = n_samples
        self._nevents = int(scalar(eventb.sum()))
        self._entry = None if entry is None else to_numpy(startb)
        self._strata = (
            None
            if strata is None
            else to_numpy(stratab).astype(np.int64, copy=False)
        )
        self._strata_labels = strata_labels
        self._subject_id = None if subjectb is None else to_numpy(subjectb)
        self._is_counting_process = entry is not None or subject_id is not None
        if self._feature_names is None:
            self._feature_names = [f"x{i + 1}" for i in range(int(Xb.shape[1]))]

        if backend == "numpy":
            self._X = np.asarray(Xb).copy()
            self._time = np.asarray(stopb).copy()
            self._event = np.asarray(eventb).copy()
        else:
            # Model outputs cross the device boundary explicitly; training
            # arrays remain on the selected backend and are not cached on host.
            self._X = None
            self._time = None
            self._event = None

        information = result["information"]
        if self.penalty > 0:
            identity = compute_backend.eye(
                information.shape[0], dtype=information.dtype
            )
            information = information + 2.0 * self.penalty * identity
        if self.compute_inference:
            if backend == "torch":
                bread = _invert_information_torch(information)
            elif backend == "cupy":
                bread = _invert_information_cupy(information)
            else:
                bread = _invert_information_numpy(information)
            if self.cov_type == "nonrobust":
                variance = bread
            else:
                residuals = result["score_residuals"]
                if self.cov_type == "cluster":
                    if clusterb is None:
                        raise ValueError(
                            "cluster ids are required when cov_type='cluster'"
                        )
                    unit_codes = clusterb
                else:
                    # Repeated start-stop rows from one subject are not
                    # independent sandwich units.  Aggregate them before the
                    # outer product whenever subject_id is available.
                    unit_codes = subjectb

                if unit_codes is None:
                    unit_scores = residuals
                    n_units = n_samples
                else:
                    _, inverse = xp.unique(unit_codes, return_inverse=True)
                    n_units = int(xp.max(inverse).item()) + 1
                    if backend == "torch":
                        unit_scores = xp.zeros(
                            (n_units, residuals.shape[1]),
                            dtype=residuals.dtype,
                            device=residuals.device,
                        )
                        unit_scores.index_add_(0, inverse, residuals)
                    else:
                        unit_scores = xp.zeros(
                            (n_units, residuals.shape[1]),
                            dtype=residuals.dtype,
                        )
                        xp.add.at(unit_scores, inverse, residuals)
                meat = unit_scores.T @ unit_scores
                if self.cov_type == "hc1":
                    meat = meat * n_units / max(
                        n_units - int(Xb.shape[1]), 1
                    )
                variance = bread @ meat @ bread
            variance = 0.5 * (variance + variance.T)
            self._var_matrix = to_numpy(variance)
            self._bse = np.sqrt(np.maximum(np.diag(self._var_matrix), 0.0))
            self._zvalues = self.coef_ / (self._bse + 1e-30)
            self._pvalues = 2.0 * norm.sf(np.abs(self._zvalues))
            ci_quantile = float(norm.ppf(0.975))
            self._conf_int = np.column_stack(
                [
                    self.coef_ - ci_quantile * self._bse,
                    self.coef_ + ci_quantile * self._bse,
                ]
            )
            self._lr_test_stat = 2.0 * (
                self._log_likelihood - self._log_likelihood_null
            )
            self._lr_test_pvalue = chi2.sf(
                self._lr_test_stat, df=int(Xb.shape[1])
            )
            try:
                self._wald_test_stat = float(
                    self.coef_ @ np.linalg.solve(self._var_matrix, self.coef_)
                )
            except np.linalg.LinAlgError:
                self._wald_test_stat = np.nan
            self._wald_test_pvalue = chi2.sf(
                self._wald_test_stat, df=int(Xb.shape[1])
            )
            # The solver already evaluates the null objective (and starts there
            # for the default zero initialization), so reuse its score test terms.
            score0 = result["null_score"]
            score_stat, score_failure = _score_test_statistic(
                score0, result["null_information"], backend, xp
            )
            if score_failure is None:
                self._score_test_stat = scalar(score_stat)
                self.score_test_available_ = True
                self.score_test_failure_reason_ = None
            else:
                self._score_test_stat = np.nan
                self.score_test_available_ = False
                self.score_test_failure_reason_ = score_failure
            self._score_test_pvalue = chi2.sf(
                self._score_test_stat, df=int(Xb.shape[1])
            )
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._tvalues = None
            self._pvalues = None
            self._conf_int = None
            self._inference_result = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self.score_test_available_ = False
            self.score_test_failure_reason_ = "compute_inference=False"

        if result["baseline"] is None:
            self._baseline_by_stratum = None
            self._unique_times = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._baseline_log_hazard = None
            self._baseline_log_cumulative_hazard = None
            self._baseline_log_cumulative_hazard_centered = None
            self._baseline_x_reference = None
        else:
            baseline_by_stratum = {
                int(key): {
                    name: to_numpy(value).astype(np.float64, copy=False)
                    for name, value in baseline.items()
                }
                for key, baseline in result["baseline"].items()
            }
            if len(baseline_by_stratum) == 1:
                baseline = next(iter(baseline_by_stratum.values()))
                self._unique_times = baseline["time"]
                self._baseline_hazard = baseline["hazard"]
                self._baseline_cumulative_hazard = baseline["cumulative_hazard"]
                self._baseline_log_hazard = baseline.get("log_hazard")
                self._baseline_log_cumulative_hazard = baseline.get(
                    "log_cumulative_hazard"
                )
                self._baseline_log_cumulative_hazard_centered = baseline.get(
                    "log_cumulative_hazard_centered"
                )
                self._baseline_x_reference = baseline.get("x_reference")
                self._baseline_by_stratum = (
                    None
                    if strata is None and entry is None and subject_id is None
                    else baseline_by_stratum
                )
            else:
                self._baseline_by_stratum = baseline_by_stratum
                self._unique_times = None
                self._baseline_hazard = None
                self._baseline_cumulative_hazard = None
                self._baseline_log_hazard = None
                self._baseline_log_cumulative_hazard = None
                self._baseline_log_cumulative_hazard_centered = None
                self._baseline_x_reference = None

        if self.compute_cindex:
            self._cindex = scalar(
                counting_process_concordance(
                    result["coef"],
                    Xb,
                    stopb,
                    eventb,
                    start=startb,
                    strata=stratab,
                    subject_id=subjectb,
                )
            )
        else:
            self._cindex = None

        score_inf = scalar(xp.max(xp.abs(result["penalized_score"])))
        raw_score_inf = scalar(xp.max(xp.abs(result["score"])))
        beta_inf = scalar(xp.max(xp.abs(result["coef"])))
        self._final_kkt_inf = score_inf
        self._final_kkt_normalized = score_inf / (
            1.0 + raw_score_inf + 2.0 * float(self.penalty) * beta_inf
        )
        self._penalized_objective = scalar(result["penalized_log_likelihood"])
        if self._converged:
            self._termination_reason = "kkt_converged"
        elif self._stop_reason == "line_search_failed":
            self._termination_reason = "line_search_failed"
        else:
            self._termination_reason = "stalled_with_large_kkt"
        self.concordance_ = self._cindex
        self.full_host_transfer_performed_ = bool(
            input_full_host_transfer
            or result.get("full_target_host_transfer_performed", False)
            or (
                backend != "numpy"
                and any(
                    value is not None
                    for value in (entry, strata, subject_id)
                )
            )
        )
        if self.compute_inference:
            self.inference_method_ = (
                "penalized_observed_information"
                if self.cov_type == "nonrobust" and self.penalty > 0
                else "observed_information"
                if self.cov_type == "nonrobust"
                else "counting_process_score_sandwich"
            )
            self.inference_backend_ = backend
            self.inference_approximate_ = False
            self.inference_fallback_reason_ = None
            inference_result = ParameterInferenceResult(
                method=self.inference_method_,
                feature_names=list(self._feature_names),
                params=self.coef_,
                bse=self._bse,
                statistic=self._zvalues,
                statistic_name="z",
                pvalues=self._pvalues,
                conf_int=self._conf_int,
                cov_type=self.cov_type,
                distribution="normal",
                metadata={
                    "inference_backend": backend,
                    "approximate": False,
                    "ties": self.ties,
                },
            )
            inference_result.apply_to(self)
        else:
            self._params = self.coef_.copy()
            self._inference_result = None
        if not self._converged:
            import warnings

            warnings.warn(
                f"CoxPH did not converge after {self._iterations} iterations "
                f"(stop_reason={self._stop_reason})",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.penalty > 0:
            self._lr_test_stat = None
            self._lr_test_pvalue = None
        self._fitted = True
        self._sync_public_fit_state()
        return self

    def _sync_public_fit_state(self):
        '''Publish the backend-neutral fitted-state contract.'''
        self.converged_ = bool(self._converged)
        self.termination_reason_ = self._termination_reason
        self.n_iter_ = int(self._iterations)
        self.final_kkt_inf_ = self._final_kkt_inf
        self.final_kkt_normalized_ = self._final_kkt_normalized
        self.concordance_ = self._cindex
    
    @property
    def log_likelihood(self):
        """Fitted (unpenalized) Cox partial log-likelihood."""
        self._check_is_fitted()
        return float(self._log_likelihood)

    @property
    def concordance_index(self):
        """Training concordance, or ``None`` when computation was disabled."""
        self._check_is_fitted()
        return None if self._cindex is None else float(self._cindex)

    def _require_classical_information_criterion(self, name):
        self._check_is_fitted()
        if self.penalty > 0:
            raise RuntimeError(
                f"{name} is only defined here for an unpenalized CoxPH fit; "
                "the penalized estimate is not the partial-likelihood MLE"
            )

    @property
    def aic(self):
        """Partial-likelihood AIC for an unpenalized fitted model."""
        self._require_classical_information_criterion("AIC")
        return float(-2.0 * self._log_likelihood + 2.0 * len(self.coef_))

    @property
    def bic(self):
        """Event-count partial-likelihood BIC for an unpenalized fit."""
        self._require_classical_information_criterion("BIC")
        return float(
            -2.0 * self._log_likelihood
            + np.log(max(int(self._nevents), 1)) * len(self.coef_)
        )

    def _format_fit_call(self):
        """Return only fitted-call details that the estimator can guarantee."""
        call = self._fit_call or {
            "interface": "matrix",
            "formula": None,
            "counting_process": bool(self._is_counting_process),
            "stratified": self._strata is not None,
            "subject_grouped": self._subject_id is not None,
            "clustered": self.cov_type == "cluster",
            "ties": self.ties,
        }
        parts = []
        if call["interface"] == "formula":
            parts.append(f"formula={call['formula']!r}")
        else:
            parts.append("interface='matrix'")
        parts.extend(
            [
                f"ties={call['ties']!r}",
                f"counting_process={bool(call['counting_process'])}",
                f"stratified={bool(call['stratified'])}",
                f"subject_grouped={bool(call['subject_grouped'])}",
                f"clustered={bool(call['clustered'])}",
            ]
        )
        return f"CoxPH({', '.join(parts)})"

    def summary(self):
        """Print a fitted CoxPH summary with truthful call metadata."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 80)
        print("                     Cox Proportional Hazards Model")
        print("=" * 80)
        print("Call:")
        print(f"  {self._format_fit_call()}")
        print()
        print(f"  n= {self._nobs}, number of events= {int(self._nevents)}")
        print(f"  covariance type= {self.cov_type}")
        print()
        if self.compute_inference and self._bse is not None:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12} {'se(coef)':>10} {'z':>10} {'Pr(>|z|)':>10}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f} "
                      f"{self._bse[i]:>10.4f} {self._zvalues[i]:>10.3f} {self._pvalues[i]:>10.4f}")
            
            print("-" * 80)
            print(f"{'':<15} {'exp(coef)':>12} {'exp(-coef)':>12} {'lower .95':>12} {'upper .95':>12}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                hr = self.hazard_ratios_[i]
                inverse_hr = _safe_exp_linear_predictor(
                    np.asarray([-self.coef_[i]]),
                    name="inverse Cox coefficient",
                )[0]
                interval_hr = _safe_exp_linear_predictor(
                    self._conf_int[i], name="Cox confidence interval"
                )
                print(f"{name:<15} {hr:>12.4f} {inverse_hr:>12.4f} "
                      f"{interval_hr[0]:>12.4f} {interval_hr[1]:>12.4f}")
        else:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12}")
            print("-" * 80)
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f}")
            print("-" * 80)
            print("Inference statistics disabled (compute_inference=False).")
        
        print("=" * 80)
        if self._cindex is None:
            print("Concordance: skipped (compute_cindex=False)")
        else:
            print(f"Concordance: {self._cindex:.3f} (if 0.5-0.7: moderate, 0.7-0.9: strong)")
        if self.compute_inference and self._lr_test_stat is not None:
            print(f"Likelihood ratio test: {self._lr_test_stat:.2f} on {len(self.coef_)} df, p={self._lr_test_pvalue:.4e}")
            print(f"Wald test:            {self._wald_test_stat:.2f} on {len(self.coef_)} df, p={self._wald_test_pvalue:.4e}")
            if self.score_test_available_:
                print(f"Score (logrank) test: {self._score_test_stat:.2f} on {len(self.coef_)} df, p={self._score_test_pvalue:.4e}")
            else:
                print(
                    "Score (logrank) test unavailable: "
                    f"{self.score_test_failure_reason_ or 'null information is singular'}"
                )
        elif self.compute_inference and self.penalty > 0:
            print(
                "Classical LR/AIC/BIC diagnostics suppressed for the penalized "
                "fit; coefficient inference is conditional on the chosen penalty."
            )
        else:
            print("Likelihood/Wald/Score tests skipped (compute_inference=False).")
        print(f"Number of Newton-Raphson iterations: {self._iterations}")
        print(f"Converged: {self._converged}")
        print("=" * 80)
    
    def _prepare_prediction_X(self, X):
        """Normalize prediction input on the estimator's active backend."""
        _require_real_array(X, "X")
        if self._design_info is not None:
            try:
                import pandas as pd
            except ImportError:  # pragma: no cover
                pd = None
            if pd is not None and isinstance(X, pd.DataFrame):
                from statgpu.core.formula import FormulaParser
                n_rows = len(X)
                parser = FormulaParser.__new__(FormulaParser)
                parser._design_info = self._design_info
                parser.formula = None
                X = parser.transform(X)
                if X.shape[0] != n_rows:
                    raise ValueError("formula prediction data contains missing values; rows cannot be dropped silently")
                names = list(self._design_info.column_names)
                if "Intercept" in names:
                    X = np.delete(X, names.index("Intercept"), axis=1)
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = backend.asarray(X, dtype=backend.float64)
        n_features = int(len(self.coef_))
        if X_arr.ndim == 1:
            if n_features == 1:
                X_arr = X_arr.reshape(-1, 1)
            elif int(X_arr.shape[0]) == n_features:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise ValueError("One-dimensional X must contain one complete feature row or observations for a one-feature model.")
        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if int(X_arr.shape[1]) != n_features:
            raise ValueError(
                f"X has {int(X_arr.shape[1])} features; expected {n_features}"
            )
        if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
            raise ValueError("X contains NaN or infinite values")
        return X_arr, backend, backend.asarray(self.coef_, dtype=backend.float64)

    @_cleanup_after_public_gpu_work
    def predict_hazard_ratio(self, X):
        """Predict backend-native hazard ratios ``exp(X @ coef_)``."""
        self._check_is_fitted()
        X_arr, _, coef = self._prepare_prediction_X(X)
        return _safe_exp_linear_predictor(X_arr @ coef)

    @_cleanup_after_public_gpu_work
    def predict_risk_score(self, X):
        """Predict backend-native linear risk scores ``X @ coef_``."""
        self._check_is_fitted()
        X_arr, _, coef = self._prepare_prediction_X(X)
        return X_arr @ coef

    @_cleanup_after_public_gpu_work
    def predict_survival(self, X, times=None, strata=None):
        """Predict backend-native survival curves for each requested stratum."""
        _require_real_array(times, "times")
        self._check_is_fitted()
        X_arr, backend, coef = self._prepare_prediction_X(X)
        xp = backend.xp
        n_samples = int(X_arr.shape[0])
        baselines = self._baseline_by_stratum
        if baselines is None and self._unique_times is not None and self._baseline_cumulative_hazard is not None:
            ordinary_baseline = {
                "time": self._unique_times,
                "cumulative_hazard": self._baseline_cumulative_hazard,
            }
            if (
                self._baseline_log_cumulative_hazard_centered is not None
                and self._baseline_x_reference is not None
            ):
                ordinary_baseline.update(
                    {
                        "log_cumulative_hazard_centered": (
                            self._baseline_log_cumulative_hazard_centered
                        ),
                        "x_reference": self._baseline_x_reference,
                    }
                )
            baselines = {0: ordinary_baseline}
        if not baselines:
            raise RuntimeError("Baseline cumulative hazard is unavailable. Refit with compute_inference=True before calling predict_survival().")
        if len(baselines) == 1:
            codes = backend.zeros((n_samples,), dtype=backend.int64)
            only_code = int(next(iter(baselines)))
            if only_code:
                codes = codes + only_code
        else:
            if strata is None:
                raise ValueError("strata is required when predicting from a stratified CoxPH fit")
            labels = np.asarray(self._to_numpy(strata))
            if labels.ndim != 1 or labels.shape[0] != n_samples:
                raise ValueError("strata must have shape (n_samples,)")
            if self._strata_labels is not None:
                mapping = {value: idx for idx, value in enumerate(self._strata_labels.tolist())}
                try:
                    codes_host = np.asarray([mapping[value] for value in labels.tolist()], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError(f"unknown prediction stratum: {exc.args[0]!r}") from exc
            else:
                codes_host = labels.astype(np.int64, copy=False)
            unknown = set(np.unique(codes_host)) - set(baselines)
            if unknown:
                raise ValueError(f"unknown prediction strata: {sorted(unknown)}")
            codes = backend.asarray(codes_host, dtype=backend.int64)
        if times is None:
            union = np.unique(np.concatenate([np.asarray(item["time"], dtype=np.float64).reshape(-1) for item in baselines.values()]))
            eval_times = backend.asarray(union, dtype=backend.float64)
        else:
            eval_times = backend.asarray(times, dtype=backend.float64)
            if eval_times.ndim == 0:
                eval_times = eval_times.reshape(1)
            elif eval_times.ndim != 1:
                raise ValueError("times must be a scalar or one-dimensional array")
            if not bool(_to_float_scalar(xp.all(xp.isfinite(eval_times)))):
                raise ValueError("times must contain only finite values")
        result = backend.ones((n_samples, int(eval_times.shape[0])), dtype=backend.float64)
        if int(eval_times.shape[0]) == 0:
            return result, eval_times
        for code, baseline in baselines.items():
            rows = codes == int(code)
            if not bool(_to_float_scalar(xp.any(rows))):
                continue
            knots = backend.asarray(baseline["time"], dtype=backend.float64)
            values = backend.asarray(baseline["cumulative_hazard"], dtype=backend.float64)
            if knots.ndim != 1 or values.shape != knots.shape or int(knots.shape[0]) == 0:
                raise RuntimeError("Stored baseline hazard state is inconsistent.")
            positions = xp.searchsorted(knots, eval_times, side="right") - 1
            safe = backend.clip(positions, 0, int(knots.shape[0]) - 1)
            cumulative = xp.where(positions >= 0, values[safe], xp.zeros_like(eval_times))
            if "log_cumulative_hazard_centered" in baseline and "x_reference" in baseline:
                log_values = backend.asarray(baseline["log_cumulative_hazard_centered"], dtype=backend.float64)
                reference = backend.asarray(baseline["x_reference"], dtype=backend.float64)
                if log_values.shape != knots.shape:
                    raise RuntimeError("Stored log-baseline state is inconsistent.")
                log_base = xp.where(positions >= 0, log_values[safe], xp.full_like(eval_times, -float("inf")))
                log_risk = log_base[None, :] + ((X_arr[rows] - reference) @ coef)[:, None]
                risk = xp.exp(
                    backend.minimum(
                        log_risk, float(np.log(np.finfo(np.float64).max))
                    )
                )
            else:
                positive = cumulative > 0
                safe_cumulative = xp.where(
                    positive, cumulative, xp.ones_like(cumulative)
                )
                log_base = xp.where(
                    positive,
                    xp.log(safe_cumulative),
                    xp.full_like(cumulative, -float("inf")),
                )
                log_risk = (
                    log_base[None, :]
                    + (X_arr[rows] @ coef)[:, None]
                )
                risk = xp.exp(
                    backend.minimum(
                        log_risk,
                        float(np.log(np.finfo(np.float64).max)),
                    )
                )
            result[rows] = xp.exp(-risk)
        return result, eval_times

    def predict(self, X):
        """Alias for predict_hazard_ratio."""
        return self.predict_hazard_ratio(X)
    
    @_cleanup_after_public_gpu_work
    def score(self, X, time, event=None, start=None, strata=None, subject_id=None):
        """Compute a backend-native Harrell-style concordance index."""
        from statgpu.survival._cox_score import score as _score_impl

        return _score_impl(
            self,
            X,
            time,
            event=event,
            start=start,
            strata=strata,
            subject_id=subject_id,
        )
