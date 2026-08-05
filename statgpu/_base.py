"""
Base classes for statgpu estimators.
"""

from __future__ import annotations

__all__ = ["BaseEstimator"]

from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import copy
import functools
import inspect
import numpy as np

from statgpu._config import Device, get_device
from statgpu.backends import (
    get_backend,
    BackendBase,
    _get_torch_device_str,
    _cupy_to_torch_dlpack,
    _torch_to_cupy_dlpack,
    _numpy_to_torch_tensor,
    _move_torch_tensor,
)


class BaseEstimator(ABC):
    """
    Base class for all statgpu estimators.
    
    Provides common functionality for device management and input validation.
    """
    
    _FINITE_PUBLIC_METHODS = frozenset({
        "fit",
        "partial_fit",
        "fit_predict",
        "fit_transform",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
        "survfit",
        "predict_survival",
        "predict_survival_function",
        "predict_cumulative_hazard",
        "inverse_transform",
        "score_samples",
        "bic",
        "aic",
        "predict_with_threshold",
        "confusion_matrix",
        "classification_table",
        "roc_curve",
        "roc_auc_score",
        "precision_recall_curve",
        "average_precision_score",
    })
    _NORMALIZED_PRIVATE_NAMES = {
        "compute_inference": "_compute_inference_enabled",
    }

    @classmethod
    def _normalized_private_name(cls, name):
        return cls._NORMALIZED_PRIVATE_NAMES.get(name, f"_{name}")

    _NORMALIZED_CONSTRUCTOR_PARAMS = frozenset({
        "device",
        "cov_type",
        "hac_maxlags",
        "gpu_memory_cleanup",
        "solver",
        "cpu_solver",
        "stopping",
        "inference_method",
        "simultaneous_method",
        "n_bootstrap",
        "enable_simultaneous_inference",
        "simultaneous_alpha",
        "simultaneous_n_bootstrap",
        "simultaneous_include_intercept",
        "method",
        "admm_rho",
        "alpha_min_ratio",
        "cd_kkt_check_every",
        "compute_inference",
        "cv",
        "fit_intercept",
        "gpu_cv_mixed_precision",
        "max_iter",
        "n_alphas",
        "tol",
        "n_Cs",
        "C_min_ratio",
        "penalty_kwargs",
        "loss_kwargs",
        "epsilon",
        "ties",
        "acknowledge_approx",
        "refine_top_k",
        "batch_size",
        "min_effective_weight",
        "quantile",
        "cv_strategy",
    })

    _FINITE_PARAMETER_NAMES = frozenset({
        "X",
        "X_new",
        "x",
        "y",
        "sample_weight",
        "weights",
        "offset",
        "exposure",
        "entry",
        "start",
        "stop",
        "time",
        "event",
        "times",
        "cluster",
        "clusters",
        "strata",
        "subject",
        "subjects",
        "groups",
        "init",
        "init_coef",
        "initial_coef",
        "time_index",
        "entity_ids",
        "time_ids",
        "pvalues",
        "arrays",
        "scores",
        "thresholds",
        "Xk",
        "mu",
        "Sigma",
    })

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("_estimator_type", ...)
        if declared is ...:
            inherited_type = next(
                (
                    base.__dict__.get("_estimator_type")
                    for base in cls.__mro__[1:]
                    if base.__dict__.get("_estimator_type")
                    in {"classifier", "regressor"}
                ),
                None,
            )
            name = cls.__name__.lower()
            module = cls.__module__
            internal_module = module.startswith("statgpu.")
            nonpredictive_module = module.startswith(
                (
                    "statgpu.covariance",
                    "statgpu.unsupervised",
                    "statgpu.preprocessing",
                    "statgpu.feature_selection",
                )
            )
            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.nonparametric",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
            if not internal_module:
                inferred_type = inherited_type
            elif nonpredictive_module:
                inferred_type = None
            elif (
                "classifier" in name
                or "logistic" in name
                or "logit" in name
                or "probit" in name
                or "orderedgeneralizedlinearmodel" in name
            ) and classifier_module:
                inferred_type = "classifier"
            elif (
                "regressor" in name
                or "kernelridge" in name
                or (
                    regression_module
                    and any(
                        token in name
                        for token in (
                            "regression",
                            "generalizedlinearmodel",
                            "glm",
                            "ridge",
                            "lasso",
                            "elasticnet",
                            "quantile",
                            "cox",
                            "panel",
                            "ols",
                            "effects",
                            "fama",
                            "gam",
                        )
                    )
                )
            ):
                inferred_type = "regressor"
            else:
                inferred_type = inherited_type
            cls._estimator_type = inferred_type
        elif declared not in {None, "classifier", "regressor"}:
            raise ValueError(
                "_estimator_type must be None, 'classifier', or 'regressor'"
            )
        cls._install_constructor_capture()
        cls._install_public_finite_validation()

    @classmethod
    def _install_constructor_capture(cls):
        original_init = cls.__dict__.get("__init__")
        if original_init is None or getattr(
            original_init, "__statgpu_constructor_capture__", False
        ):
            return
        try:
            signature = inspect.signature(original_init)
        except (TypeError, ValueError):
            return

        @functools.wraps(original_init)
        def wrapped(self, *args, **kwargs):
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                return original_init(self, *args, **kwargs)
            raw_params = {
                name: value
                for name, value in bound.arguments.items()
                if name != "self"
                and signature.parameters[name].kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }

            depth = int(getattr(self, "_statgpu_constructor_depth", 0))
            if depth == 0:
                self._statgpu_constructor_raw_pending = {}
            self._statgpu_constructor_depth = depth + 1

            try:
                result = original_init(self, *args, **kwargs)
            except BaseException:
                if depth == 0:
                    self.__dict__.pop("_statgpu_constructor_depth", None)
                    self.__dict__.pop("_statgpu_constructor_raw_pending", None)
                else:
                    self._statgpu_constructor_depth = depth
                raise

            pending = self._statgpu_constructor_raw_pending
            # Inner wrappers finish first; the most-derived constructor finishes
            # last and therefore overrides shared defaults with its actual call.
            pending.update(raw_params)
            normalized_names = type(self)._NORMALIZED_CONSTRUCTOR_PARAMS

            # Make runtime values available to any outer constructor code without
            # restoring public raw values until the complete chain has returned.
            for name, raw_value in raw_params.items():
                private_name = type(self)._normalized_private_name(name)
                if name in normalized_names:
                    if name == "device" and hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
                    setattr(self, private_name, runtime_value)
                elif not hasattr(self, name):
                    setattr(self, name, raw_value)

            remaining = depth
            if remaining > 0:
                self._statgpu_constructor_depth = remaining
                return result

            self.__dict__.pop("_statgpu_constructor_depth", None)
            merged_raw = dict(pending)
            self.__dict__.pop("_statgpu_constructor_raw_pending", None)

            for name, raw_value in merged_raw.items():
                private_name = type(self)._normalized_private_name(name)
                if name in normalized_names:
                    if name == "device" and hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
                    setattr(self, private_name, runtime_value)
                # sklearn <=1.2 requires the public attribute to be the exact
                # object supplied to the outermost constructor.
                setattr(self, name, raw_value)
            self._constructor_params_raw = merged_raw
            return result

        wrapped.__statgpu_constructor_capture__ = True
        cls.__init__ = wrapped

    @classmethod
    def _install_public_finite_validation(cls):
        from statgpu.backends._validation import check_finite

        def wrap_method(original):
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                return original

            @functools.wraps(original)
            def guarded(self, *args, **kwargs):
                try:
                    bound = signature.bind(self, *args, **kwargs)
                except TypeError:
                    return original(self, *args, **kwargs)
                loss_value = getattr(self, "loss", "")
                loss_name = str(getattr(loss_value, "name", loss_value)).lower()
                formula_active = (
                    bound.arguments.get("formula") is not None
                    or bound.arguments.get("data") is not None
                )
                for name, value in bound.arguments.items():
                    if name == "y" and loss_name in {"cox", "coxph", "cox_ph"}:
                        # Cox response matrices have stronger joint time/event
                        # contracts. Preserve model-specific errors and validate
                        # them before device selection inside the Cox estimator.
                        continue
                    if formula_active and type(value).__module__.startswith("pandas"):
                        # Formula/model-matrix code owns row dropping, categorical
                        # encoding, and aligned side-array error semantics.
                        continue
                    if name in self._FINITE_PARAMETER_NAMES and value is not None:
                        check_finite(value, name=name)
                return original(self, *args, **kwargs)

            guarded.__statgpu_finite_validation__ = True
            return guarded

        candidate_names = set(cls._FINITE_PUBLIC_METHODS)
        candidate_names.update(
            name for name in dir(cls) if not name.startswith("_")
        )
        for method_name in sorted(candidate_names):
            original = getattr(cls, method_name, None)
            if not callable(original):
                continue
            if getattr(original, "__isabstractmethod__", False):
                continue
            if getattr(original, "__statgpu_finite_validation__", False):
                continue
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                continue
            numerical_parameters = set(signature.parameters) & cls._FINITE_PARAMETER_NAMES
            if method_name not in cls._FINITE_PUBLIC_METHODS and not numerical_parameters:
                continue
            setattr(cls, method_name, wrap_method(original))

    def __init__(
        self,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize base estimator.
        
        Parameters
        ----------
        device : str or Device, default='auto'
            Computation device: 'cpu', 'cuda', or 'auto'.
        n_jobs : int, optional
            Number of parallel jobs for CPU computation.
            -1 means using all processors.
        """
        self.device = device
        self._device = device if isinstance(device, Device) else Device(device)
        self.n_jobs = n_jobs
        self._fitted = False
    
    def _get_compute_device(self) -> Device:
        """Resolve device for actual computation."""
        if self._device == Device.AUTO:
            return get_device()
        return self._device

    def _get_backend(self, backend: str = "auto") -> BackendBase:
        """
        Return the compute backend appropriate for this estimator's device.

        Parameters
        ----------
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Override which array library to use.  When ``'auto'``, the backend
            is chosen based on :attr:`device` and GPU availability.

        Returns
        -------
        BackendBase
            A backend instance whose :attr:`~BackendBase.xp` attribute is the
            underlying array module (NumPy, CuPy, or PyTorch).
        """
        compute_device = self._get_compute_device()
        device_str = compute_device.value  # 'cpu', 'cuda', or 'torch'

        if (
            self._device != Device.AUTO
            and compute_device == Device.CUDA
            and backend == "auto"
        ):
            cupy_backend = get_backend(backend="cupy", device="cuda")
            if not cupy_backend.is_available():
                raise RuntimeError(
                    "device='cuda' requires a working CuPy CUDA backend. "
                    "Use device='auto' to allow automatic backend selection."
                )
            return cupy_backend

        # Handle Device.TORCH explicitly - use torch backend
        if compute_device == Device.TORCH:
            try:
                import torch
            except Exception as exc:
                raise RuntimeError(
                    "device='torch' requires PyTorch with CUDA support."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='torch' requires torch.cuda.is_available() to be True. "
                    "Use device='auto' or device='cpu' if CUDA is unavailable."
                )
            return get_backend(backend="torch", device="cuda")

        return get_backend(backend=backend, device=device_str)
    
    def _to_array(self, X, device: Optional[Device] = None, backend: Optional[str] = None) -> Any:
        """
        Convert input to appropriate array type for device.

        Parameters
        ----------
        X : array-like
            Input data.
        device : Device, optional
            Target device. If None, uses self._get_compute_device().
        backend : {'numpy', 'cupy', 'torch'}, optional
            Explicit backend selection. If None, uses device-based default.

        Returns
        -------
        array
            NumPy array (CPU), CuPy array (GPU), or Torch tensor.
        """
        target_device = device or self._get_compute_device()

        # If backend is explicitly specified, use it
        if backend == "torch":
            torch_target = "cpu" if target_device == Device.CPU else "cuda"
            return self._to_torch(X, device=torch_target)
        elif backend == "cupy":
            return self._to_cupy(X)
        elif backend == "numpy":
            if hasattr(X, "get"):
                return X.get()
            # Handle torch tensors that may be on CUDA — must move to CPU first
            if hasattr(X, 'cpu') and hasattr(X, 'numpy'):
                return X.detach().cpu().numpy()
            return np.asarray(X)

        # Otherwise, use device-based default
        if target_device == Device.TORCH:
            return self._to_torch(X, device="cuda")

        if target_device == Device.CUDA:
            # Strict CUDA means CuPy-backed arrays. Torch has its own explicit
            # device mode and is not used as an implicit CUDA fallback.
            try:
                import cupy as cp
                if isinstance(X, cp.ndarray):
                    return X
            except Exception:
                pass

        # Convert to numpy first for CPU paths or for non-CuPy inputs.
        if hasattr(X, "get"):  # CuPy-like array
            X_np = X.get()
        elif hasattr(X, "cpu"):  # PyTorch tensor
            X_cpu = X.detach().cpu() if hasattr(X, "detach") else X.cpu()
            X_np = X_cpu.numpy() if hasattr(X_cpu, "numpy") else np.asarray(X_cpu)
        else:
            X_np = np.asarray(X)

        if target_device == Device.CUDA:
            try:
                import cupy as cp
                return cp.asarray(X_np)
            except Exception as exc:
                raise RuntimeError(
                    "device='cuda' requires a working CuPy CUDA backend; "
                    "no CPU/Torch fallback is performed for explicit CUDA."
                ) from exc

        return X_np

    def _to_torch(self, X, device=None):
        """Convert input to Torch tensor."""
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='torch' requires torch.cuda.is_available() to be True; "
                "no Torch CPU fallback is performed for explicit Torch GPU."
            )

        if isinstance(X, torch.Tensor):
            return _move_torch_tensor(X, device=device) if device else X

        if hasattr(X, "get"):  # CuPy
            tensor = _cupy_to_torch_dlpack(X, device=device)
            if tensor is not None:
                return tensor
            X_np = X.get()
        elif hasattr(X, "cpu"):  # Tensor-like
            X_cpu = X.detach().cpu() if hasattr(X, "detach") else X.cpu()
            X_np = X_cpu.numpy() if hasattr(X_cpu, "numpy") else np.asarray(X_cpu)
        else:
            target_device = device or _get_torch_device_str()
            return _numpy_to_torch_tensor(
                X,
                device=target_device,
                pin_memory=str(target_device).startswith("cuda"),
            )

        target_device = device or _get_torch_device_str()
        return _numpy_to_torch_tensor(
            X_np,
            device=target_device,
            pin_memory=str(target_device).startswith("cuda"),
        )

    def _to_cupy(self, X):
        """Convert input to CuPy array."""
        import cupy as cp

        if isinstance(X, cp.ndarray):
            return X

        if hasattr(X, "cpu"):  # PyTorch
            arr = _torch_to_cupy_dlpack(X)
            if arr is not None:
                return arr
            X_np = X.detach().cpu().numpy()
        elif hasattr(X, "get"):  # CuPy (shouldn't happen, but handle it)
            X_np = X.get()
        else:
            X_np = np.asarray(X)

        return cp.asarray(X_np)
    
    def _to_numpy(self, X) -> np.ndarray:
        """Convert array back to numpy."""
        if hasattr(X, 'get'):  # CuPy
            return X.get()
        elif hasattr(X, 'cpu'):  # PyTorch
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _resolve_inference_backend(self, backend: str) -> str:
        """Resolve model-context inference backend from the estimator device."""
        backend_name = str(backend).strip().lower()
        if backend_name == "auto":
            compute_device = self._get_compute_device()
            if compute_device == Device.CUDA:
                return "cupy"
            if compute_device == Device.TORCH:
                return "torch"
        return backend_name

    def _cast_inference_array(self, value, backend_name: str):
        """Cast an inference input to the explicitly resolved backend."""
        if backend_name == "cupy":
            return self._to_array(value, Device.CUDA, backend="cupy")
        if backend_name == "torch":
            return self._to_array(value, Device.TORCH, backend="torch")
        if backend_name == "numpy":
            return self._to_numpy(value)
        return value

    def adjust_pvalues(
        self,
        pvalues=None,
        method: str = "bh",
        alpha: float = 0.05,
        axis: Optional[int] = 0,
        backend: str = "auto",
    ):
        """
        Adjust p-values for multiple testing (FDR/FWER controls).

        Parameters
        ----------
        pvalues : array-like, optional
            Raw p-values. If omitted, uses this estimator's ``_pvalues``.
        method : str, default='bh'
            Adjustment method: ``bh``, ``by``, ``holm``, ``bonferroni``
            (aliases accepted).
        alpha : float, default=0.05
            Rejection threshold in (0, 1).
        axis : int or None, default=0
            Axis along which to adjust. ``None`` flattens all entries.
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Compute backend. ``'auto'`` follows the estimator's resolved device.

        Returns
        -------
        dict
            Contains ``pvalues``, ``pvalues_adjusted``, ``reject``,
            ``method``, ``alpha``, and ``axis``.
        """
        from statgpu.inference import adjust_pvalues as _adjust_pvalues

        source = pvalues
        if source is None:
            source = getattr(self, "_pvalues", None)
        if source is None:
            raise RuntimeError(
                "No p-values available. Fit with inference enabled or pass pvalues explicitly."
            )

        backend_name = self._resolve_inference_backend(backend)

        pvals = self._cast_inference_array(source, backend_name)

        reject, pvals_adj = _adjust_pvalues(
            pvals,
            method=method,
            alpha=alpha,
            axis=axis,
            backend=backend_name,
        )

        return {
            "method": method,
            "alpha": float(alpha),
            "axis": axis,
            "backend": backend_name,
            "pvalues": pvals,
            "pvalues_adjusted": pvals_adj,
            "reject": reject,
        }

    def combine_pvalues(
        self,
        pvalues=None,
        method: str = "fisher",
        weights=None,
        axis: Optional[int] = None,
        backend: str = "auto",
    ):
        """
        Combine p-values into a global p-value.

        Parameters
        ----------
        pvalues : array-like, optional
            Raw p-values. If omitted, uses this estimator's ``_pvalues``.
        method : str, default='fisher'
            Combination method: ``fisher`` or ``cauchy`` (aliases accepted).
        weights : array-like, optional
            Optional non-negative weights for cauchy combination.
        axis : int or None, default=None
            Axis along which to combine p-values. ``None`` flattens input.
        backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
            Compute backend. ``'auto'`` follows the estimator's resolved device.

        Returns
        -------
        dict
            Contains ``pvalues``, ``statistic``, ``pvalue``,
            ``method``, ``axis``, and ``backend``.
        """
        from statgpu.inference import combine_pvalues as _combine_pvalues

        source = pvalues
        if source is None:
            source = getattr(self, "_pvalues", None)
        if source is None:
            raise RuntimeError(
                "No p-values available. Fit with inference enabled or pass pvalues explicitly."
            )

        backend_name = self._resolve_inference_backend(backend)

        pvals = self._cast_inference_array(source, backend_name)
        w_cast = (
            None
            if weights is None
            else self._cast_inference_array(weights, backend_name)
        )

        statistic, pvalue = _combine_pvalues(
            pvals,
            method=method,
            weights=w_cast,
            axis=axis,
            backend=backend_name,
        )

        return {
            "method": method,
            "axis": axis,
            "backend": backend_name,
            "pvalues": pvals,
            "weights": w_cast,
            "statistic": statistic,
            "pvalue": pvalue,
        }

    def bootstrap_statistic(
        self,
        statistic,
        *arrays,
        n_resamples: int = 200,
        strategy: str = "iid",
        strata=None,
        clusters=None,
        block_size: Optional[int] = None,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        statistic_name: str = "statistic",
        backend: str = "auto",
    ):
        """
        Run unified bootstrap engine from model context.

        This is a thin wrapper over ``statgpu.inference.bootstrap_statistic``.
        """
        from statgpu.inference import bootstrap_statistic as _bootstrap_statistic

        arrays_use = arrays
        if len(arrays_use) == 0:
            X_cache = getattr(self, "_X_design", None)
            y_cache = getattr(self, "_y", None)
            if X_cache is None or y_cache is None:
                raise RuntimeError(
                    "No cached training arrays available. Pass arrays explicitly or fit first."
                )
            arrays_use = (X_cache, y_cache)

        backend_name = self._resolve_inference_backend(backend)

        arrays_cast = tuple(
            self._cast_inference_array(a, backend_name) for a in arrays_use
        )
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        clusters_cast = (
            None
            if clusters is None
            else self._cast_inference_array(clusters, backend_name)
        )

        return _bootstrap_statistic(
            statistic,
            *arrays_cast,
            n_resamples=n_resamples,
            strategy=strategy,
            strata=strata_cast,
            clusters=clusters_cast,
            block_size=block_size,
            confidence_level=confidence_level,
            random_state=random_state,
            statistic_name=statistic_name,
            backend=backend_name,
        )

    def permutation_test(
        self,
        statistic,
        X,
        y,
        n_resamples: int = 1000,
        strategy: str = "iid",
        strata=None,
        groups=None,
        alternative: str = "two-sided",
        random_state: Optional[int] = None,
        statistic_name: str = "statistic",
        backend: str = "auto",
    ):
        """
        Run unified permutation test engine from model context.

        This is a thin wrapper over ``statgpu.inference.permutation_test``.
        """
        from statgpu.inference import permutation_test as _permutation_test

        backend_name = self._resolve_inference_backend(backend)

        X_cast = self._cast_inference_array(X, backend_name)
        y_cast = self._cast_inference_array(y, backend_name)
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        groups_cast = (
            None
            if groups is None
            else self._cast_inference_array(groups, backend_name)
        )

        return _permutation_test(
            statistic,
            X_cast,
            y_cast,
            n_resamples=n_resamples,
            strategy=strategy,
            strata=strata_cast,
            groups=groups_cast,
            alternative=alternative,
            random_state=random_state,
            statistic_name=statistic_name,
            backend=backend_name,
        )
    
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the estimator."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def _check_is_fitted(self):
        """Check if estimator has been fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
    
    def _statgpu_estimator_type(self):
        """Return the class-level sklearn estimator classification."""
        estimator_type = getattr(type(self), "_estimator_type", None)
        if estimator_type in {"classifier", "regressor"}:
            return estimator_type
        return None

    def __sklearn_tags__(self):
        """Return public estimator tags when sklearn >= 1.6 is installed."""
        estimator_type = self._statgpu_estimator_type()
        try:
            from sklearn.utils import (
                ClassifierTags,
                RegressorTags,
                Tags,
                TargetTags,
                TransformerTags,
            )
        except ImportError:
            return self._more_tags()

        has_transform = callable(getattr(self, "transform", None))
        return Tags(
            estimator_type=estimator_type,
            target_tags=TargetTags(required=estimator_type is not None),
            transformer_tags=TransformerTags() if has_transform else None,
            classifier_tags=(
                ClassifierTags() if estimator_type == "classifier" else None
            ),
            regressor_tags=(
                RegressorTags() if estimator_type == "regressor" else None
            ),
            requires_fit=True,
        )

    def _more_tags(self):
        """Return the legacy sklearn tag dictionary."""
        estimator_type = self._statgpu_estimator_type()
        return {"requires_y": estimator_type in {"classifier", "regressor"}}

    def __sklearn_is_fitted__(self):
        """Expose statgpu fitted state to sklearn meta-estimators."""
        return bool(getattr(self, "_fitted", False))

    def __sklearn_clone__(self):
        """Return an unfitted recursive clone for scikit-learn >= 1.3.

        Constructor values are preserved by the public raw-parameter contract,
        while estimator-valued parameters must be cloned recursively so fitted
        state is never copied into the new estimator.
        """
        from copy import deepcopy

        params = self.get_params(deep=False)
        try:
            from sklearn.base import clone as sklearn_clone
        except ImportError:
            cloned_params = deepcopy(params)
        else:
            cloned_params = {
                name: sklearn_clone(value, safe=False)
                for name, value in params.items()
            }
        return type(self)(**cloned_params)

    def get_params(self, deep=True):
        """Get constructor parameters for this estimator.

        Nested estimator parameters are exposed as ``name__param`` when
        ``deep=True``, matching the scikit-learn estimator contract.
        """
        import inspect

        params = {}
        raw_params = getattr(self, "_constructor_params_raw", {})
        try:
            sig = inspect.signature(type(self).__init__)
        except (ValueError, TypeError):
            return params

        for name, parameter in sig.parameters.items():
            if name == "self" or parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in raw_params:
                params[name] = raw_params[name]
            elif hasattr(self, name):
                params[name] = getattr(self, name)
            elif hasattr(self, f"_{name}"):
                params[name] = getattr(self, f"_{name}")

        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params"):
                    for sub_name, sub_value in value.get_params(deep=True).items():
                        params[f"{name}__{sub_name}"] = sub_value
        return params


    def set_params(self, **params):
        """Set parameters transactionally and refresh normalized state."""
        if not params:
            return self

        import copy
        from collections.abc import Iterator

        valid_deep = self.get_params(deep=True)
        direct = self.get_params(deep=False)
        direct_updates = {}
        nested = {}
        for key, value in params.items():
            root, delimiter, sub_key = key.partition("__")
            if key not in valid_deep and root not in direct:
                valid_names = sorted(
                    name for name in valid_deep if "__" not in name
                )
                raise ValueError(
                    f"Invalid parameter {root!r} for estimator "
                    f"{type(self).__name__}. Valid parameters are: {valid_names}."
                )
            if delimiter:
                nested.setdefault(root, {})[sub_key] = value
            else:
                direct[root] = value
                direct_updates[root] = value

        explicitly_updated = set(direct_updates)
        for key, value in tuple(direct.items()):
            if isinstance(value, Iterator) and key not in explicitly_updated:
                snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                if snapshot is None:
                    snapshot = list(value)
                direct[key] = copy.deepcopy(snapshot)

        if nested:
            try:
                from sklearn.base import clone as sklearn_clone
            except ImportError:
                sklearn_clone = None
            for root in nested:
                nested_value = direct[root]
                if sklearn_clone is not None and hasattr(nested_value, "get_params"):
                    direct[root] = sklearn_clone(nested_value)
                else:
                    direct[root] = copy.deepcopy(nested_value)

        try:
            fresh = type(self)(**direct)
        except (TypeError, ValueError):
            deferred = set(getattr(type(self), "_DEFERRED_SET_PARAMS", ()))
            if nested or not direct_updates or not set(direct_updates).issubset(deferred):
                raise
            # A small number of estimators intentionally validate selected
            # controls at fit time. Apply only those explicitly declared values
            # after the complete update has been classified as deferred.
            for key, value in direct_updates.items():
                setattr(self, key, value)
                if key in self._NORMALIZED_CONSTRUCTOR_PARAMS:
                    setattr(self, self._normalized_private_name(key), value)
                raw_params = getattr(self, "_constructor_params_raw", None)
                if raw_params is None:
                    raw_params = {}
                    self._constructor_params_raw = raw_params
                raw_params[key] = value
            reset = getattr(self, "_reset_fit_state", None)
            if callable(reset):
                reset()
            else:
                self._fitted = False
            return self

        for root, sub_params in nested.items():
            nested_estimator = getattr(fresh, root, None)
            if nested_estimator is None:
                nested_estimator = getattr(fresh, f"_{root}", None)
            if not hasattr(nested_estimator, "set_params"):
                raise ValueError(
                    f"Parameter {root!r} of {type(self).__name__} does not "
                    "support nested parameters."
                )
            nested_estimator.set_params(**sub_params)

        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self


def refresh_public_finite_validation_contracts():
    """Install finite guards after all estimator mixins and aliases are bound."""
    BaseEstimator._install_public_finite_validation()
    seen = set()
    stack = list(BaseEstimator.__subclasses__())
    while stack:
        estimator_cls = stack.pop()
        if estimator_cls in seen:
            continue
        seen.add(estimator_cls)
        stack.extend(estimator_cls.__subclasses__())
        estimator_cls._install_public_finite_validation()


BaseEstimator._install_public_finite_validation()
