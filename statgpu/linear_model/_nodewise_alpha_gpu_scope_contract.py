"""Scope the node-wise GPU context strictly to sparse Gaussian debiased inference.

The initial node-wise API integration wrapped ``_fit_gpu_backend`` at both the
base and typed-linear levels. This installer collapses those wrappers to one
scope-aware layer and delegates unrelated GPU/fake-backend paths directly to the
pre-nodewise implementation. Precision caching itself is a normal helper, not a
second installer.
"""

from __future__ import annotations

import functools

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.linear_model.penalized import _nodewise_precision_cache as _cache
from statgpu.linear_model import _nodewise_alpha_inference_contract as _nodewise
from statgpu.linear_model import _post_selection_ols_fifth_review_contract as _fifth

_MARKER = "__statgpu_nodewise_alpha_gpu_scope__"
_NODEWISE_MARKER = "__statgpu_nodewise_alpha_contract__"


def _runtime_method(model) -> str:
    return str(
        getattr(model, "_inference_method", getattr(model, "inference_method", ""))
    ).strip().lower()


def _applies(model) -> bool:
    return (
        bool(getattr(model, "_compute_inference_enabled", False))
        and "debiased" in _runtime_method(model)
        and bool(_fifth._supports_sparse_gaussian_migration(model))
    )


def _unwrap_nodewise_fit(current):
    base = current
    while getattr(base, _NODEWISE_MARKER, False) and hasattr(base, "__wrapped__"):
        base = base.__wrapped__
    return base


def _install_for(cls) -> None:
    current = cls._fit_gpu_backend
    if getattr(current, _MARKER, False):
        return
    base = _unwrap_nodewise_fit(current)

    @functools.wraps(base)
    def wrapped(self, X, y, sample_weight=None, backend_name="cupy"):
        if not _applies(self):
            return base(self, X, y, sample_weight, backend_name=backend_name)

        n = int(X.shape[0])
        effective_n = _nodewise._gpu_effective_n(
            n, sample_weight, backend_name, X
        )
        token = _nodewise._NODEWISE_CONTEXT.set(
            (id(self), effective_n, sample_weight is not None)
        )
        try:
            result = base(
                self,
                X,
                y,
                sample_weight,
                backend_name=backend_name,
            )
            _nodewise._promote_candidate(self)
            return result
        except Exception:
            self.nodewise_alpha_ = None
            self.__dict__.pop("_nodewise_alpha_candidate", None)
            self.__dict__.pop("_nodewise_metadata_candidate", None)
            raise
        finally:
            _nodewise._NODEWISE_CONTEXT.reset(token)

    setattr(wrapped, _MARKER, True)
    cls._fit_gpu_backend = wrapped


def install_nodewise_alpha_gpu_scope_contract() -> None:
    _install_for(PenalizedGeneralizedLinearModel)
    _install_for(PenalizedLinearRegression)

    # Replace only the local builder references captured by the main node-wise
    # integration module. Cache hits/misses therefore do not alter estimator
    # signatures, warning stacks, or unrelated inference routing.
    _nodewise.build_nodewise_precision_numpy = _cache.build_nodewise_precision_numpy
    _nodewise.build_nodewise_precision_cupy = _cache.build_nodewise_precision_cupy
    _nodewise.build_nodewise_precision_torch = _cache.build_nodewise_precision_torch


__all__ = ["install_nodewise_alpha_gpu_scope_contract"]
