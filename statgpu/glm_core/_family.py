"""
Link and Family abstractions for GLM.

Extracted from the duplicated IRLS loops in _logistic.py across CPU/GPU/Torch backends.
Each Family defines: link function, variance function, and IRLS weights/working response.

All operations are backend-aware: numpy/cupy/torch via _xp dispatch.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from statgpu.backends._utils import _get_xp
from statgpu.inference._distributions_backend import get_distribution


def _backend_name(arr):
    """Infer backend name from array type."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        return "cupy"
    if mod.startswith("torch"):
        return "torch"
    return "numpy"


def _xp(arr):
    """Get the array module (numpy/cupy/torch) from array type."""
    return _get_xp(_backend_name(arr))


def _clip(arr, lo, hi):
    xp = _xp(arr)
    if xp.__name__ == "torch":
        import torch
        result = arr.clone()
        if lo is not None:
            result = torch.clamp(result, min=lo)
        if hi is not None:
            result = torch.clamp(result, max=hi)
        return result
    return xp.clip(arr, lo, hi)


def _exp(arr):
    xp = _xp(arr)
    # Clip to prevent overflow in exp (matching backend conventions)
    if xp.__name__ == "torch":
        import torch
        return torch.exp(torch.clamp(arr, min=-500, max=500))
    return xp.exp(xp.clip(arr, -500, 500))


def _log(arr):
    return _xp(arr).log(arr)


def _ones_like(arr):
    return _xp(arr).ones_like(arr)


def _cdf(arr):
    """Standard normal CDF (Phi)."""
    backend = _backend_name(arr)
    return get_distribution("norm", backend=backend).cdf(arr)


def _ppd(arr):
    """Standard normal PPF (inverse CDF, Phi^{-1})."""
    backend = _backend_name(arr)
    return get_distribution("norm", backend=backend).ppf(arr)


def _pdf(arr):
    """Standard normal PDF (phi)."""
    backend = _backend_name(arr)
    return get_distribution("norm", backend=backend).pdf(arr)


# ─── Link Functions ────────────────────────────────────────────────────────


class Link(ABC):
    """Link function abstract base class.

    Maps between mean (mu) and linear predictor (eta):
        eta = link(mu)
        mu  = inverse(eta)
    """

    name: str

    @abstractmethod
    def link(self, mu):
        """eta = g(mu)."""
        pass

    @abstractmethod
    def inverse(self, eta):
        """mu = g^{-1}(eta)."""
        pass

    @abstractmethod
    def derivative(self, mu):
        """g'(mu) = d eta / d mu."""
        pass


class LogitLink(Link):
    name = "logit"

    def link(self, mu):
        return _log(mu / (1 - mu))

    def inverse(self, eta):
        return 1.0 / (1.0 + _exp(-_clip(eta, -500, 500)))

    def derivative(self, mu):
        return 1.0 / (mu * (1 - mu))


class ProbitLink(Link):
    """Probit link: inverse of standard normal CDF (Phi)."""

    name = "probit"

    def link(self, mu):
        return _ppd(_clip(mu, 1e-10, 1 - 1e-10))

    def inverse(self, eta):
        return _cdf(eta)

    def derivative(self, mu):
        return 1.0 / _pdf(
            _ppd(_clip(mu, 1e-10, 1 - 1e-10))
        )


class LogLink(Link):
    name = "log"

    def link(self, mu):
        return _log(_clip(mu, 1e-10, None))

    def inverse(self, eta):
        return _exp(eta)

    def derivative(self, mu):
        return 1.0 / mu


class IdentityLink(Link):
    name = "identity"

    def link(self, mu):
        return mu

    def inverse(self, eta):
        return eta

    def derivative(self, mu):
        return _ones_like(mu)


# ─── Families ──────────────────────────────────────────────────────────────


class GLMFamily(ABC):
    """GLM distribution family.

    Each family defines:
    - link function: eta <-> mu mapping
    - variance function: Var(Y) = phi * V(mu)
    - IRLS weights and working response computation
    """

    name: str
    link: Link

    @abstractmethod
    def variance(self, mu):
        """Variance function V(mu)."""
        pass

    def irls_weights(self, mu, y):
        """IRLS working weights.

        W = V(mu) * (g'(mu))^2

        Default uses W = V(mu) * (link'(mu))^2.
        Subclasses can override for more efficient implementations.
        """
        return self.variance(mu) * self.link.derivative(mu) ** 2

    def irls_working_response(self, mu, y, eta):
        """Working response z = eta + (y - mu) * link'(mu)."""
        return eta + (y - mu) * self.link.derivative(mu)


class Gaussian(GLMFamily):
    """Gaussian family with identity link (standard linear regression)."""

    name = "gaussian"
    link = IdentityLink()

    def variance(self, mu):
        return _ones_like(mu)


class Binomial(GLMFamily):
    """Binomial family with configurable link (logistic/probit regression)."""

    name = "binomial"

    def __init__(self, link=None):
        self.link = link if link is not None else LogitLink()

    def variance(self, mu):
        return mu * (1 - mu)

    def irls_weights(self, mu, y):
        mu_c = _clip(mu, 1e-10, 1 - 1e-10)
        return mu_c * (1 - mu_c)

    def irls_working_response(self, mu, y, eta):
        mu_c = _clip(mu, 1e-10, 1 - 1e-10)
        var = mu_c * (1 - mu_c)
        return eta + (y - mu_c) / var


class Poisson(GLMFamily):
    """Poisson family with log link (Poisson regression)."""

    name = "poisson"
    link = LogLink()

    def variance(self, mu):
        return mu

    def irls_weights(self, mu, y):
        return _clip(mu, 1e-10, None)

    def irls_working_response(self, mu, y, eta):
        return eta + (y - mu) / _clip(mu, 1e-10, None)
