"""Public contract repair for penalized GLM coefficient inference.

This module deliberately reuses the maintained numerical implementations. It
repairs the public method resolver, resampling scope, backend provenance, and
PenalizedGLM_CV final-refit boundary after the earlier sparse-Gaussian contract
installers have run.

Existing specialized contracts (post-selection aliases, group penalties, Cox,
node-wise sparse Gaussian inference) remain authoritative for their own rows.
"""

from __future__ import annotations

import copy
import functools
import inspect
import warnings

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model._penalized_inference_api import (
    POST_SELECTION_OLS,
    normalize_penalized_inference_method,
)
from statgpu.penalties._categories import GROUP as _GROUP_PENALTY_NAMES
from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.linear_model.penalized._penalized_logistic import PenalizedLogisticRegression
from statgpu.linear_model.penalized._penalized_poisson import PenalizedPoissonRegression
from statgpu.linear_model.penalized._penalized_gamma import PenalizedGammaRegression
from statgpu.linear_model.penalized._penalized_inverse_gaussian import (
    PenalizedInverseGaussianRegression,
)
from statgpu.linear_model.penalized._penalized_negative_binomial import (
    PenalizedNegativeBinomialRegression,
)
from statgpu.linear_model.penalized._penalized_tweedie import PenalizedTweedieRegression
from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV


_INSTALL_MARKER = "_statgpu_penalized_glm_inference_contract"
_AUTO_DEFAULT_CLASSES = (
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
    PenalizedGammaRegression,
    PenalizedInverseGaussianRegression,
    PenalizedNegativeBinomialRegression,
    PenalizedTweedieRegression,
)
_POST_SELECTION_REQUESTS = frozenset({"cpu_ols", "gpu_ols", POST_SELECTION_OLS})
_ORACLE_LOSSES = frozenset(
    {
        "squared_error",
        "logistic",
        "poisson",
        "gamma",
        "inverse_gaussian",
        "negative_binomial",
        "tweedie",
    }
)
_M_ESTIMATION_COV_TYPES = frozenset({"nonrobust", "hc0", "hc1"})
_PROVENANCE_FIELDS = (
    "inference_requested_method_",
    "inference_resolved_method_",
    "inference_method_",
    "inference_target_",
    "penalty_conditioning_",
    "penalty_selection_adjusted_",
)


def _penalty_name(estimator) -> str:
    return str(
        getattr(getattr(estimator, "_penalty", None), "name", estimator.penalty)
    ).strip().lower()


def _loss_name(estimator) -> str:
    return str(getattr(estimator, "loss", "")).strip().lower()


def _raw_public_request(estimator) -> str:
    return str(
        getattr(
            estimator,
            "inference_method",
            getattr(estimator, "_inference_method", "auto"),
        )
    ).strip().lower()


def _normalized_request(estimator) -> str:
    return normalize_penalized_inference_method(_raw_public_request(estimator))


def _positive_penalty(estimator) -> bool:
    resolved = getattr(estimator, "_penalty", None)
    alpha = getattr(resolved, "alpha", getattr(estimator, "alpha", 0.0))
    try:
        return float(alpha) > 0.0
    except (TypeError, ValueError):
        return True


def _target_for(estimator, resolved: str) -> str:
    if resolved == "debiased":
        return "unpenalized_population_coefficient"
    if resolved in (POST_SELECTION_OLS, "oracle"):
        return "active_set_refit_coefficient"
    if resolved == "residual_bootstrap":
        return "penalized_coefficient_distribution"
    if resolved in ("classical", "sandwich", "m_estimation"):
        return (
            "penalized_estimating_equation"
            if _positive_penalty(estimator)
            else "unpenalized_population_coefficient"
        )
    raise RuntimeError(f"Unknown resolved inference method {resolved!r}")


def _unsupported(estimator, request: str, *, reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"Inference is not supported for loss={_loss_name(estimator)!r}, "
        f"penalty={_penalty_name(estimator)!r}, inference_method={request!r}: "
        f"{reason} Set compute_inference=False or choose a supported combination."
    )


def _resolve_contract(estimator):
    """Resolve one public request to one concrete reporting method."""
    request = _normalized_request(estimator)
    loss = _loss_name(estimator)
    penalty = _penalty_name(estimator)
    cov_type = str(getattr(estimator, "cov_type", "nonrobust")).strip().lower()
    has_hessian = bool(getattr(getattr(estimator, "_loss", None), "has_hessian", False))
    legacy_alias = None

    if penalty in _GROUP_PENALTY_NAMES:
        raise _unsupported(
            estimator,
            request,
            reason="this group-penalty row is estimation-only under its existing contract.",
        )
    if loss == "cox_ph":
        raise _unsupported(
            estimator,
            request,
            reason="the penalized Cox estimator is currently estimation-only.",
        )

    if request == "debiased" and penalty == "l2":
        warnings.warn(
            "inference_method='debiased' for L2/no-penalty models is a "
            "deprecated compatibility spelling. L2 inference was not "
            "debiased-Lasso inference. Use inference_method='auto' (or "
            "'m_estimation' for a supported non-Gaussian L2 model). The "
            "compatibility spelling will be removed in a future breaking release.",
            FutureWarning,
            stacklevel=4,
        )
        legacy_alias = "debiased"
        request_for_resolution = "auto"
    else:
        request_for_resolution = request

    if request_for_resolution == "auto":
        if loss == "squared_error" and penalty == "l2":
            resolved = "classical" if cov_type == "nonrobust" else "sandwich"
            dispatch = "auto"
        elif loss == "squared_error" and penalty in ("l1", "elasticnet", "en"):
            resolved = "debiased"
            dispatch = "debiased"
        elif penalty in ("scad", "mcp"):
            raise _unsupported(
                estimator,
                request,
                reason=(
                    "auto does not silently select oracle/selection-conditional "
                    "inference; request inference_method='oracle' explicitly"
                    + (
                        " or 'bootstrap' for an unweighted CPU Gaussian fit"
                        if loss == "squared_error"
                        else ""
                    )
                    + "."
                ),
            )
        elif loss != "squared_error" and has_hessian and penalty == "l2":
            if cov_type not in _M_ESTIMATION_COV_TYPES:
                raise _unsupported(
                    estimator,
                    request,
                    reason=(
                        "penalized M-estimation currently supports cov_type "
                        "'nonrobust', 'hc0', or 'hc1' only."
                    ),
                )
            resolved = "m_estimation"
            dispatch = "m_estimation"
        else:
            raise _unsupported(
                estimator,
                request,
                reason="no statistically defined automatic inference path exists for this row.",
            )
    elif request_for_resolution == "m_estimation":
        if loss == "squared_error" or not has_hessian or penalty != "l2":
            raise _unsupported(
                estimator,
                request,
                reason=(
                    "m_estimation is limited to supported non-Gaussian smooth "
                    "L2/no-penalty objectives."
                ),
            )
        if cov_type not in _M_ESTIMATION_COV_TYPES:
            raise _unsupported(
                estimator,
                request,
                reason="m_estimation currently supports nonrobust/HC0/HC1 covariance only.",
            )
        resolved = "m_estimation"
        dispatch = "m_estimation"
    elif request_for_resolution == "debiased":
        if loss != "squared_error" or penalty not in ("l1", "elasticnet", "en"):
            raise _unsupported(
                estimator,
                request,
                reason="debiased inference is supported only for sparse Gaussian L1/ElasticNet.",
            )
        resolved = "debiased"
        dispatch = "debiased"
    elif request_for_resolution == POST_SELECTION_OLS:
        if loss != "squared_error" or penalty not in ("l1", "elasticnet", "en"):
            raise _unsupported(
                estimator,
                request,
                reason="post_selection_ols is supported only for sparse Gaussian L1/ElasticNet.",
            )
        resolved = POST_SELECTION_OLS
        dispatch = POST_SELECTION_OLS
    elif request_for_resolution == "bootstrap":
        if loss != "squared_error" or penalty not in (
            "l1",
            "elasticnet",
            "en",
            "scad",
            "mcp",
        ):
            raise _unsupported(
                estimator,
                request,
                reason="bootstrap in this contract is Gaussian residual bootstrap only.",
            )
        if cov_type != "nonrobust":
            raise _unsupported(
                estimator,
                request,
                reason=(
                    "Gaussian residual bootstrap currently requires cov_type='nonrobust'; "
                    "robust/HAC bootstrap semantics are not inferred from cov_type."
                ),
            )
        resolved = "residual_bootstrap"
        dispatch = "bootstrap"
    elif request_for_resolution == "oracle":
        if penalty not in ("scad", "mcp") or loss not in _ORACLE_LOSSES:
            raise _unsupported(
                estimator,
                request,
                reason="oracle active-set refit is supported only for SCAD/MCP scalar GLM families.",
            )
        resolved = "oracle"
        dispatch = "oracle"
    else:
        raise _unsupported(
            estimator,
            request,
            reason=(
                "valid inference requests are auto, debiased, post_selection_ols, "
                "m_estimation, oracle, or bootstrap (subject to loss/penalty support)."
            ),
        )

    return {
        "requested": request,
        "resolved": resolved,
        "dispatch": dispatch,
        "target": _target_for(estimator, resolved),
        "legacy_alias": legacy_alias,
    }


def _replace_source_default(callable_obj, name: str, value):
    """Change one real constructor default without adding a runtime frame."""
    source = inspect.unwrap(callable_obj)
    signature = inspect.signature(source, follow_wrapped=False)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return

    if parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        positional = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        defaults = list(source.__defaults__ or ())
        first_default = len(positional) - len(defaults)
        index = positional.index(parameter) - first_default
        if index < 0:
            raise RuntimeError(
                f"Cannot replace non-default positional parameter {name!r} on {source!r}."
            )
        defaults[index] = value
        source.__defaults__ = tuple(defaults)
    elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        kwdefaults = dict(source.__kwdefaults__ or {})
        kwdefaults[name] = value
        source.__kwdefaults__ = kwdefaults
    else:
        raise RuntimeError(f"Unsupported constructor parameter kind for {name!r}.")

    cursor = callable_obj
    seen = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        explicit = getattr(cursor, "__signature__", None)
        if explicit is not None and name in explicit.parameters:
            cursor.__signature__ = explicit.replace(
                parameters=[
                    p.replace(default=value) if p.name == name else p
                    for p in explicit.parameters.values()
                ]
            )
        cursor = getattr(cursor, "__wrapped__", None)


def _install_auto_constructor_default(cls):
    # Do not wrap __init__: existing solver/post-selection deprecation policies
    # inspect the call stack to distinguish user calls from statgpu-internal
    # reconstruction. An extra contract-module frame would incorrectly suppress
    # those established warnings. Mutating the true source default preserves the
    # existing wrapper stack and its caller-intent semantics.
    _replace_source_default(cls.__init__, "inference_method", "auto")


def _install_state_cleanup():
    current = PenalizedGeneralizedLinearModel._clear_inference_state
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        result = current(self)
        for field in _PROVENANCE_FIELDS:
            setattr(self, field, None)
        return result

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGeneralizedLinearModel._clear_inference_state = wrapped


def _install_validator():
    current = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        if not self._compute_inference_enabled:
            self._statgpu_pending_inference_contract = None
            return current(self)

        raw_request = _raw_public_request(self)
        penalty = _penalty_name(self)
        loss = _loss_name(self)

        # Preserve the already-reviewed contracts for historical
        # post-selection aliases, group penalties, and penalized Cox. They own
        # exact warning/error wording and result provenance for those consumers.
        if (
            raw_request in _POST_SELECTION_REQUESTS
            or penalty in _GROUP_PENALTY_NAMES
            or loss == "cox_ph"
        ):
            self._statgpu_pending_inference_contract = None
            return current(self)

        contract = _resolve_contract(self)
        self._statgpu_pending_inference_contract = contract
        return None

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGeneralizedLinearModel._validate_inference_request = wrapped


def _selected_backend(estimator) -> str:
    backend = str(getattr(estimator, "_selected_backend_name", "") or "").lower()
    if backend not in ("numpy", "cupy", "torch"):
        raise RuntimeError(
            "Inference requires fit-recorded backend provenance; "
            f"got _selected_backend_name={backend!r}."
        )
    return backend


def _selected_device(estimator, backend: str) -> str:
    device = str(getattr(estimator, "_selected_backend_device", "") or "")
    if backend == "numpy":
        return "cpu"
    if not device:
        raise RuntimeError(
            f"Inference is missing concrete device provenance for backend={backend!r}."
        )
    return device


def _run_sandwich_on_fit_backend(current, self, X, y, sample_weight=None):
    backend = _selected_backend(self)
    device = _selected_device(self, backend)

    if backend == "numpy":
        X_native = self._to_array(X, backend="numpy")
        y_native = self._to_array(y, backend="numpy")
        sw_native = (
            None if sample_weight is None else self._to_array(sample_weight, backend="numpy")
        )
        result = current(self, X_native, y_native, sample_weight=sw_native)
    elif backend == "cupy":
        import cupy as cp

        if not device.startswith("cuda:"):
            raise RuntimeError(f"Invalid CuPy fit device provenance: {device!r}")
        device_id = int(device.split(":", 1)[1])
        with cp.cuda.Device(device_id):
            X_native = cp.asarray(X, dtype=cp.float64)
            y_native = cp.asarray(y, dtype=cp.float64)
            sw_native = (
                None
                if sample_weight is None
                else cp.asarray(sample_weight, dtype=cp.float64)
            )
            result = current(self, X_native, y_native, sample_weight=sw_native)
    else:
        import torch

        target = torch.device(device)
        X_native = torch.as_tensor(X, dtype=torch.float64, device=target)
        y_native = torch.as_tensor(y, dtype=torch.float64, device=target)
        sw_native = (
            None
            if sample_weight is None
            else torch.as_tensor(sample_weight, dtype=torch.float64, device=target)
        )
        result = current(self, X_native, y_native, sample_weight=sw_native)

    if self._inference_result is not None:
        metadata = self._inference_result.metadata
        metadata.pop("backend", None)
        metadata["numerical_backend"] = backend
        metadata["numerical_device"] = device
        metadata["reporting_backend"] = "numpy"
        metadata["reporting_boundary"] = "post_numerical_inference"
    return result


def _install_sandwich_backend_contract():
    current = _PenalizedInferenceMixin._compute_penalized_sandwich_inference
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None):
        return _run_sandwich_on_fit_backend(
            current, self, X, y, sample_weight=sample_weight
        )

    setattr(wrapped, _INSTALL_MARKER, True)
    _PenalizedInferenceMixin._compute_penalized_sandwich_inference = wrapped


def _bootstrap_penalty_template(estimator):
    resolved = getattr(estimator, "_penalty", None)
    if resolved is not None:
        return copy.deepcopy(resolved)
    return estimator.penalty


def _gaussian_residual_bootstrap(self, X, y):
    """Unweighted CPU Gaussian residual bootstrap preserving the penalty."""
    backend = _selected_backend(self)
    if backend != "numpy":
        raise NotImplementedError(
            "Gaussian residual-bootstrap inference is currently CPU-only and "
            "must not silently refit on CPU after a GPU/Torch fit. Use "
            "device='cpu' or choose another supported inference method."
        )
    if str(getattr(self, "cov_type", "nonrobust")).lower() != "nonrobust":
        raise NotImplementedError(
            "Gaussian residual-bootstrap inference currently requires "
            "cov_type='nonrobust'."
        )

    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
    n = int(X_np.shape[0])
    coef = np.asarray(self.coef_, dtype=np.float64)
    y_pred = X_np @ coef + (float(self.intercept_) if self._effective_intercept else 0.0)
    resid = y_np - y_pred

    self._X_design = (
        np.column_stack([np.ones(n), X_np])
        if self._effective_intercept
        else X_np.copy()
    )
    self._y = y_np
    self._resid = resid
    self._nobs = n

    B = int(getattr(self, "n_bootstrap", 200))
    rng = np.random.default_rng(getattr(self, "bootstrap_random_state", None))
    params_dim = int(len(self._params))
    boot_params = np.zeros((B, params_dim), dtype=float)
    penalty_template = _bootstrap_penalty_template(self)

    for b in range(B):
        eps_star = rng.choice(resid, size=n, replace=True)
        y_star = y_pred + eps_star
        refit = PenalizedLinearRegression(
            penalty=copy.deepcopy(penalty_template),
            alpha=float(self.alpha),
            l1_ratio=float(getattr(self, "l1_ratio", 0.5)),
            penalty_kwargs=copy.deepcopy(getattr(self, "penalty_kwargs", None) or {}),
            fit_intercept=self._effective_intercept,
            max_iter=int(getattr(self, "max_iter", 1000)),
            tol=float(getattr(self, "tol", 1e-4)),
            device="cpu",
            cpu_solver=getattr(self, "cpu_solver", "fista"),
            solver=getattr(self, "solver", "auto"),
            stopping=getattr(self, "stopping", "coef_delta"),
            lla=bool(getattr(self, "lla", True)),
            max_lla_iters=int(getattr(self, "max_lla_iters", 50)),
            lla_tol=float(getattr(self, "lla_tol", 1e-6)),
            compute_inference=False,
            inference_method="auto",
        )
        refit.fit(X_np, y_star)
        boot_params[b, :] = np.asarray(refit._params, dtype=float)

    bse = np.std(boot_params, axis=0, ddof=1)
    pvalues = np.zeros(params_dim, dtype=float)
    for i in range(params_dim):
        coef_b = boot_params[:, i]
        pvalues[i] = min(
            1.0,
            2.0 * min(float(np.mean(coef_b <= 0.0)), float(np.mean(coef_b >= 0.0))),
        )
    conf_int = np.column_stack(
        [
            np.quantile(boot_params, 0.025, axis=0),
            np.quantile(boot_params, 0.975, axis=0),
        ]
    )
    tvalues = np.asarray(self._params, dtype=float) / (bse + 1e-30)

    self._bse = bse
    self._pvalues = pvalues
    self._conf_int = conf_int
    self._tvalues = tvalues
    self._inference_result = ParameterInferenceResult(
        method="residual_bootstrap",
        params=np.asarray(self._params, dtype=float).copy(),
        bse=bse.copy(),
        statistic=tvalues.copy(),
        statistic_name="z",
        pvalues=pvalues.copy(),
        conf_int=conf_int.copy(),
        distribution="bootstrap_percentile",
        metadata={
            "n_bootstrap": B,
            "random_state": getattr(self, "bootstrap_random_state", None),
            "resampling_scope": "unweighted_gaussian_residual",
            "refit_penalty": _penalty_name(self),
            "numerical_backend": "numpy",
            "numerical_device": "cpu",
            "reporting_backend": "numpy",
        },
    )
    self._inference_result.apply_to(self)


def _install_bootstrap_contract():
    current = _PenalizedInferenceMixin._compute_post_fit_bootstrap_inference
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y):
        return _gaussian_residual_bootstrap(self, X, y)

    setattr(wrapped, _INSTALL_MARKER, True)
    _PenalizedInferenceMixin._compute_post_fit_bootstrap_inference = wrapped


def _publish_contract(estimator):
    result = getattr(estimator, "_inference_result", None)
    contract = getattr(estimator, "_statgpu_pending_inference_contract", None)
    if result is None or contract is None:
        return
    actual = str(result.method).strip().lower()
    expected = str(contract["resolved"]).strip().lower()
    if actual != expected:
        raise RuntimeError(
            "Penalized inference method identity mismatch: requested "
            f"{contract['requested']!r}, resolved {expected!r}, but the result "
            f"reported {actual!r}."
        )

    estimator.inference_requested_method_ = contract["requested"]
    estimator.inference_resolved_method_ = expected
    estimator.inference_method_ = actual
    estimator.inference_target_ = contract["target"]
    estimator.penalty_conditioning_ = "fixed_penalty"
    estimator.penalty_selection_adjusted_ = None

    metadata = result.metadata
    metadata["inference_requested_method"] = contract["requested"]
    metadata["inference_resolved_method"] = expected
    metadata["inference_target"] = contract["target"]
    metadata["penalty_conditioning"] = "fixed_penalty"
    metadata["penalty_selection_adjusted"] = None
    if contract.get("legacy_alias") is not None:
        metadata["legacy_inference_alias"] = contract["legacy_alias"]

    backend = str(getattr(estimator, "_selected_backend_name", "") or "").lower()
    if backend in ("numpy", "cupy", "torch"):
        metadata.setdefault("numerical_backend", backend)
        metadata.setdefault("numerical_device", _selected_device(estimator, backend))
        metadata.setdefault("reporting_backend", "numpy")
        metadata.setdefault("reporting_boundary", "post_numerical_inference")


def _install_post_fit_contract():
    current = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None):
        if not self._compute_inference_enabled:
            return current(self, X, y, sample_weight=sample_weight)

        raw_request = _raw_public_request(self)
        penalty = _penalty_name(self)
        loss = _loss_name(self)
        contract = getattr(self, "_statgpu_pending_inference_contract", None)

        # Direct internal-helper compatibility and specialized contracts stay
        # with the already-reviewed previous method. A real fit owned by this
        # contract always records both a pending contract and fit backend.
        if contract is None and (
            raw_request in _POST_SELECTION_REQUESTS
            or penalty in _GROUP_PENALTY_NAMES
            or loss == "cox_ph"
            or str(getattr(self, "_selected_backend_name", "") or "").lower()
            not in ("numpy", "cupy", "torch")
        ):
            return current(self, X, y, sample_weight=sample_weight)

        if contract is None:
            contract = _resolve_contract(self)
            self._statgpu_pending_inference_contract = contract

        if contract["resolved"] == "residual_bootstrap":
            if sample_weight is not None:
                raise NotImplementedError(
                    "Weighted Gaussian residual-bootstrap inference is not "
                    "implemented. Set sample_weight=None or choose another "
                    "supported inference method."
                )
            if _selected_backend(self) != "numpy":
                raise NotImplementedError(
                    "Gaussian residual-bootstrap inference is currently CPU-only "
                    "and will not silently refit on CPU after a GPU/Torch fit."
                )
        if contract["resolved"] == "oracle" and _selected_backend(self) != "numpy":
            raise NotImplementedError(
                "SCAD/MCP oracle inference currently performs a CPU active-set "
                "refit and therefore is unsupported after an executed GPU/Torch "
                "fit. Use device='cpu' or set compute_inference=False."
            )

        public_request = getattr(self, "inference_method", None)
        internal_request = getattr(self, "_inference_method", None)
        self.inference_method = contract["dispatch"]
        self._inference_method = contract["dispatch"]
        try:
            value = current(self, X, y, sample_weight=sample_weight)
        finally:
            if public_request is not None:
                self.inference_method = public_request
            if internal_request is not None:
                self._inference_method = internal_request
            else:
                self.__dict__.pop("_inference_method", None)

        _publish_contract(self)
        return value

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = wrapped


def _install_cv_constructor():
    current = PenalizedGLM_CV.__init__
    if getattr(current, _INSTALL_MARKER, False):
        return
    signature = inspect.signature(current)
    additions = (
        inspect.Parameter("compute_inference", inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter("inference_method", inspect.Parameter.KEYWORD_ONLY, default="auto"),
        inspect.Parameter("cov_type", inspect.Parameter.KEYWORD_ONLY, default="nonrobust"),
        inspect.Parameter("hac_maxlags", inspect.Parameter.KEYWORD_ONLY, default=None),
    )
    public_signature = signature.replace(
        parameters=[*signature.parameters.values(), *additions]
    )

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        compute_inference = bool(kwargs.pop("compute_inference", False))
        inference_method = str(kwargs.pop("inference_method", "auto")).strip().lower()
        cov_type = str(kwargs.pop("cov_type", "nonrobust")).strip().lower()
        hac_maxlags = kwargs.pop("hac_maxlags", None)
        value = current(self, *args, **kwargs)
        self.compute_inference = compute_inference
        self.inference_method = inference_method
        self.cov_type = cov_type
        self.hac_maxlags = hac_maxlags
        return value

    wrapped.__signature__ = public_signature
    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGLM_CV.__init__ = wrapped


def _install_cv_reset():
    current = PenalizedGLM_CV._reset_cv_fit_state
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        value = current(self)
        self._inference_result = None
        self._bse = None
        self._tvalues = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        for field in _PROVENANCE_FIELDS:
            setattr(self, field, None)
        return value

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGLM_CV._reset_cv_fit_state = wrapped


def _install_cv_refit():
    current = PenalizedGLM_CV._refit_best
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, best_alpha, sample_weight=None):
        if not bool(getattr(self, "compute_inference", False)):
            return current(self, X, y, best_alpha, sample_weight=sample_weight)
        if str(self.loss).lower() == "cox_ph":
            raise NotImplementedError(
                "PenalizedGLM_CV(loss='cox_ph') is estimation-only; coefficient "
                "inference is not implemented for the penalized Cox final refit."
            )

        refit_device = self._device
        if str(getattr(self._device, "value", self._device)).lower() == "auto":
            refit_device = getattr(self, "cv_selected_device_", self._device) or self._device
        refit_solver = self._solver_for_cv(refit_device, X=X)
        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=copy.deepcopy(self.penalty),
            alpha=float(best_alpha),
            l1_ratio=float(self.l1_ratio),
            penalty_kwargs=copy.deepcopy(getattr(self, "_penalty_kwargs", None) or {}),
            fit_intercept=bool(getattr(self, "_fit_intercept", True)),
            device=refit_device,
            n_jobs=getattr(self, "_n_jobs", None),
            compute_inference=True,
            inference_method=getattr(self, "inference_method", "auto"),
            cov_type=getattr(self, "cov_type", "nonrobust"),
            hac_maxlags=getattr(self, "hac_maxlags", None),
            max_iter=int(self._max_iter),
            tol=float(self._tol),
            solver=refit_solver,
            gpu_memory_cleanup=bool(getattr(self, "_gpu_memory_cleanup", False)),
            stopping=getattr(self, "_stopping", "coef_delta"),
            lla=bool(getattr(self, "_lla", True)),
            max_lla_iters=int(getattr(self, "_max_lla_iters", 50)),
            lla_tol=float(getattr(self, "_lla_tol", 1e-6)),
            loss_kwargs=copy.deepcopy(getattr(self, "_loss_kwargs", None) or {}),
        )
        model.fit(X, y, sample_weight=sample_weight)
        return model

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGLM_CV._refit_best = wrapped


def _copy_cv_inference_state(cv):
    estimator = getattr(cv, "estimator_", None)
    result = getattr(estimator, "_inference_result", None)
    if result is None:
        return

    estimator.penalty_conditioning_ = "cv_selected_penalty"
    estimator.penalty_selection_adjusted_ = False
    result.metadata["penalty_conditioning"] = "cv_selected_penalty"
    result.metadata["penalty_selection_adjusted"] = False
    result.metadata["selected_alpha"] = float(cv.alpha_)

    cv._inference_result = result
    for name in ("_bse", "_tvalues", "_zvalues", "_pvalues", "_conf_int"):
        setattr(cv, name, getattr(estimator, name, None))
    for field in _PROVENANCE_FIELDS:
        setattr(cv, field, getattr(estimator, field, None))
    cv.penalty_conditioning_ = "cv_selected_penalty"
    cv.penalty_selection_adjusted_ = False


def _install_cv_fit():
    current = PenalizedGLM_CV.fit
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None):
        if bool(getattr(self, "compute_inference", False)) and str(self.loss).lower() == "cox_ph":
            raise NotImplementedError(
                "PenalizedGLM_CV(loss='cox_ph') is estimation-only; set "
                "compute_inference=False."
            )
        value = current(self, X, y, sample_weight=sample_weight)
        if bool(getattr(self, "compute_inference", False)):
            _copy_cv_inference_state(self)
        return value

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGLM_CV.fit = wrapped


def _install_cv_summary():
    if "summary" in PenalizedGLM_CV.__dict__:
        return

    def summary(self, *args, **kwargs):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("PenalizedGLM_CV is not fitted yet. Call fit() first.")
        if getattr(self, "_inference_result", None) is None:
            raise RuntimeError(
                "Inference is not available. Fit with compute_inference=True "
                "and a supported final-refit loss/penalty combination."
            )
        return self.estimator_.summary(*args, **kwargs)

    PenalizedGLM_CV.summary = summary


def install_penalized_glm_inference_contract():
    for cls in _AUTO_DEFAULT_CLASSES:
        _install_auto_constructor_default(cls)
    _install_state_cleanup()
    _install_validator()
    _install_sandwich_backend_contract()
    _install_bootstrap_contract()
    _install_post_fit_contract()
    _install_cv_constructor()
    _install_cv_reset()
    _install_cv_refit()
    _install_cv_fit()
    _install_cv_summary()


__all__ = ["install_penalized_glm_inference_contract"]
