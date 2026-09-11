"""Public-boundary and fit-transaction binding for penalized-GLM inference.

The public constructor parameter remains the caller's request (for example
``auto``), while GPU sparse inference needs the concrete dispatch method during
``_fit_gpu`` / ``_fit_torch``. Bind the resolved method immediately after the
pre-fit inference validator and restore the public request after the complete
fit transaction.

Two compatibility details live here because they require the *outermost* public
boundary after the existing installer stack has been assembled:

- omitted generic/typed ``inference_method`` must really enter as ``auto`` even
  when older wrappers captured the historical default in a closure;
- PenalizedGLM_CV must store newly added constructor parameters by identity so
  sklearn<=1.2 clone reconstruction remains valid.

The tiny constructor boundary functions deliberately identify themselves as the
existing penalized-inference API contract. That contract is already treated as
transparent by the staged deprecation-warning policies, so caller identity is
still discovered correctly: explicit user legacy arguments warn, while sklearn
clone/set_params framework replay stays silent.

A failed refit must also stop advertising any prior successful fit. This matters
because inference compatibility is validated before the core fit clears fitted
state; a newly unsupported method/loss/penalty row can therefore fail before the
usual reset point.
"""

from __future__ import annotations

import functools
import inspect

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
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


_MARKER = "_statgpu_penalized_glm_inference_fit_transaction"
_TRANSPARENT_MODULE = "statgpu.linear_model._penalized_inference_api_contract"
_PUBLIC_DEFAULT_CLASSES = (
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
    PenalizedGammaRegression,
    PenalizedInverseGaussianRegression,
    PenalizedNegativeBinomialRegression,
    PenalizedTweedieRegression,
)


def _make_public_auto_boundary(current, public_signature):
    namespace = {
        "__name__": _TRANSPARENT_MODULE,
        "_current": current,
        "_signature": public_signature,
    }
    exec(
        "def wrapped(self, *args, **kwargs):\n"
        "    bound = _signature.bind_partial(self, *args, **kwargs)\n"
        "    if 'inference_method' not in bound.arguments:\n"
        "        kwargs['inference_method'] = 'auto'\n"
        "    return _current(self, *args, **kwargs)\n",
        namespace,
    )
    wrapped = namespace["wrapped"]
    functools.update_wrapper(wrapped, current)
    wrapped.__signature__ = public_signature
    setattr(wrapped, _MARKER, True)
    return wrapped


def _install_public_auto_boundary(cls):
    current = cls.__init__
    if getattr(current, _MARKER, False):
        return
    signature = inspect.signature(current)
    parameter = signature.parameters.get("inference_method")
    if parameter is None:
        return
    public_signature = signature.replace(
        parameters=[
            p.replace(default="auto") if p.name == "inference_method" else p
            for p in signature.parameters.values()
        ]
    )
    cls.__init__ = _make_public_auto_boundary(current, public_signature)


def _install_clone_safe_cv_constructor():
    current = PenalizedGLM_CV.__init__
    if getattr(current, _MARKER, False):
        return

    # The preceding contract wrapper added these keyword-only parameters. Peel
    # that one wrapper so we can preserve exact caller objects rather than
    # lowercasing/casting them before sklearn clone verifies identity.
    underlying = getattr(current, "__wrapped__", current)
    public_signature = inspect.signature(current)
    namespace = {
        "__name__": _TRANSPARENT_MODULE,
        "_underlying": underlying,
    }
    exec(
        "def wrapped(self, *args, **kwargs):\n"
        "    compute_inference = kwargs.pop('compute_inference', False)\n"
        "    inference_method = kwargs.pop('inference_method', 'auto')\n"
        "    cov_type = kwargs.pop('cov_type', 'nonrobust')\n"
        "    hac_maxlags = kwargs.pop('hac_maxlags', None)\n"
        "    value = _underlying(self, *args, **kwargs)\n"
        "    self.compute_inference = compute_inference\n"
        "    self.inference_method = inference_method\n"
        "    self.cov_type = cov_type\n"
        "    self.hac_maxlags = hac_maxlags\n"
        "    return value\n",
        namespace,
    )
    wrapped = namespace["wrapped"]
    functools.update_wrapper(wrapped, underlying)
    wrapped.__signature__ = public_signature
    setattr(wrapped, _MARKER, True)
    PenalizedGLM_CV.__init__ = wrapped


def _install_validator_binding():
    current = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(current, _MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        value = current(self)
        contract = getattr(self, "_statgpu_pending_inference_contract", None)
        if contract is not None:
            self.inference_method = contract["dispatch"]
            self._inference_method = contract["dispatch"]
        return value

    setattr(wrapped, _MARKER, True)
    PenalizedGeneralizedLinearModel._validate_inference_request = wrapped


def _invalidate_failed_refit(self):
    """Clear result-bearing state after any failed inference-enabled refit."""
    try:
        self._clear_inference_state()
    except Exception:
        # Failure invalidation is best effort but must never replace the
        # original statistical/validation exception.
        pass
    for name, value in (
        ("coef_", None),
        ("intercept_", None),
        ("n_iter_", 0),
        ("_selected_solver", None),
        ("_selected_backend_name", None),
        ("_selected_backend_device", None),
        ("_native_fit_coef", None),
        ("_native_fit_intercept", None),
        ("_inference_precomputed", False),
        ("_precomputed_gaussian_state", None),
    ):
        try:
            setattr(self, name, value)
        except Exception:
            pass
    self._fitted = False


def _install_fit_restore():
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, _MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        public_request = getattr(self, "inference_method", None)
        internal_request = getattr(self, "_inference_method", None)
        try:
            return current(self, *args, **kwargs)
        except Exception:
            if bool(getattr(self, "compute_inference", False)):
                _invalidate_failed_refit(self)
            raise
        finally:
            if public_request is not None:
                self.inference_method = public_request
            if internal_request is not None:
                self._inference_method = internal_request
            else:
                self.__dict__.pop("_inference_method", None)

    setattr(wrapped, _MARKER, True)
    PenalizedGeneralizedLinearModel.fit = wrapped


def install_penalized_glm_inference_fit_transaction():
    for cls in _PUBLIC_DEFAULT_CLASSES:
        _install_public_auto_boundary(cls)
    _install_clone_safe_cv_constructor()
    _install_validator_binding()
    _install_fit_restore()


__all__ = ["install_penalized_glm_inference_fit_transaction"]
