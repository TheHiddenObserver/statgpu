"""Cox partial-likelihood loss for survival analysis.

The public loss API uses failure-time-local normalization so Breslow and Efron
likelihoods remain stable after the observation attaining the global maximum
linear predictor has left a later risk set. First-order and Hessian evaluations
use the specialized right-censored kernels in this module; the shared
counting-process engine remains the independent compatibility baseline.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends._array_ops import (
    _max_eigval_power,
    _xp as _get_xp,
    _xp_asarray,
    _xp_zeros,
)
from statgpu.backends._utils import (
    _is_complex_array,
    _to_float_scalar,
    _to_numpy,
)
from statgpu.survival._risk_sets import cox_counting_process_objective

from ._base import LossBase
from ._registry import register_loss


def _build_efron_pre_numpy(time_np, event_np):
    """Build deterministic Efron failure groups for compatibility helpers."""
    event_mask = event_np == 1
    event_idx = np.where(event_mask)[0]
    event_times = time_np[event_idx]
    uft, inv = np.unique(event_times, return_inverse=True)
    nuft = len(uft)
    uft_ix = [event_idx[inv == group].astype(np.int64) for group in range(nuft)]
    first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    risk_enter = [[np.int64(index)] for index in first_idx]
    risk_exit = [
        [np.int64(np.searchsorted(time_np, value, side="right"))] for value in uft
    ]
    return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx


def _build_breslow_pre_numpy(time_np, event_np):
    """Build the historical two-array Breslow preprocessing tuple."""
    event_times = time_np[event_np == 1]
    uft, counts = np.unique(event_times, return_counts=True)
    first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    return first_idx, counts.astype(np.float64)


def _build_breslow_event_indices_numpy(time_np, event_np):
    """Return event-row indices grouped in Breslow failure-time order."""
    event_idx = np.flatnonzero(event_np == 1)
    if event_idx.size == 0:
        return []
    event_times = time_np[event_idx]
    _, inverse = np.unique(event_times, return_inverse=True)
    return [
        event_idx[inverse == group].astype(np.int64)
        for group in range(int(inverse.max()) + 1)
    ]


def _backend_index(values, xp, reference):
    """Create integer indices on the backend/device of ``reference``."""
    if xp.__name__ == "torch":
        return xp.as_tensor(values, dtype=xp.long, device=reference.device)
    return xp.asarray(values, dtype=xp.int64)


def _backend_zeros(shape, xp, reference):
    if xp.__name__ == "torch":
        return xp.zeros(shape, dtype=reference.dtype, device=reference.device)
    return xp.zeros(shape, dtype=reference.dtype)


def _sum(value, xp, axis=None):
    if xp.__name__ == "torch":
        return xp.sum(value) if axis is None else xp.sum(value, dim=axis)
    return xp.sum(value, axis=axis)


class _CoxPreprocessedTarget:
    """Opaque proof that ``X`` belongs to the active Cox fit cache."""

    __slots__ = ("generation", "n_samples")

    def __init__(self, generation: int, n_samples: int):
        self.generation = int(generation)
        self.n_samples = int(n_samples)


@register_loss("cox_ph")
class CoxPartialLikelihoodLoss(LossBase):
    """Negative Cox partial likelihood with Breslow or Efron ties.

    The response is either a ``{"time": ..., "event": ...}`` dictionary or an
    ``(n, 2)`` array.  ``sample_weight`` is intentionally unsupported because
    case weights require a separate, explicitly documented survival contract.
    An all-censored response is a valid loss boundary with zero value and zero
    derivatives, even though estimators reject it because no coefficient can be
    identified from such data.
    """

    name = "cox_ph"
    y_type = "survival"
    smooth_gradient = True
    has_hessian = True

    _lipschitz_safety = 1.0
    _lipschitz_uses_y = True
    _has_constant_hessian = False

    def __init__(self, ties: str = "breslow"):
        ties = str(ties).lower()
        if ties not in {"breslow", "efron"}:
            raise ValueError("ties must be 'breslow' or 'efron'")
        self.ties = ties
        self._sorted = False
        self._X_sorted = None
        self._time_sorted = None
        self._event_sorted = None
        self._order = None
        self._time_np = None
        self._event_np = None
        self._efron_pre_np = None
        self._breslow_pre_np = None
        self._breslow_event_indices_np = None
        self._efron_csr = None
        self._efron_backend_index_cache = {}
        self._n_events = 0
        self._x_reference = None
        self._group_first_indices_np = None
        self._group_counts_np = None
        self._group_event_indices_np = None
        self._event_group_codes_np = None
        self._efron_fractions_np = None
        self._group_first_indices_backend = None
        self._group_counts_backend = None
        self._group_event_indices_backend = None
        self._event_group_codes_backend = None
        self._efron_fractions_backend = None
        self._cache_generation = 0
        self._preprocessed_target = None

    def is_preprocessed(self, X, y) -> bool:
        """Return whether ``(X, y)`` is the active sorted fit-cache pair."""
        return bool(
            self._sorted
            and X is self._X_sorted
            and y is self._preprocessed_target
            and getattr(y, "generation", None) == self._cache_generation
        )

    def _ensure_sorted(self, X, y):
        if self.is_preprocessed(X, y):
            return
        self.preprocess(X, y)

    def preprocess(self, X, y):
        """Validate, center, and stably sort right-censored survival data."""
        self.release_fit_cache()
        self._reject_complex(X, "X")
        xp = _get_xp(X)
        if isinstance(y, dict):
            if "time" not in y or "event" not in y:
                raise ValueError("survival y dict must contain time and event")
            self._reject_complex(y["time"], "time")
            self._reject_complex(y["event"], "event")
            time = _xp_asarray(y["time"], dtype=xp.float64, ref_arr=X)
            event = _xp_asarray(y["event"], dtype=xp.float64, ref_arr=X)
        else:
            self._reject_complex(y, "y")
            y_arr = _xp_asarray(y, dtype=xp.float64, ref_arr=X)
            if y_arr.ndim != 2 or int(y_arr.shape[1]) != 2:
                raise ValueError("y must be dict or (n, 2) array")
            time, event = y_arr[:, 0], y_arr[:, 1]

        X_arr = _xp_asarray(X, dtype=xp.float64, ref_arr=X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if time.ndim != 1 or event.ndim != 1:
            raise ValueError("time and event must have shape (n_samples,)")
        if int(time.shape[0]) != int(X_arr.shape[0]) or int(event.shape[0]) != int(
            X_arr.shape[0]
        ):
            raise ValueError("X, time, and event must contain the same number of rows")
        if _to_float_scalar(xp.sum(~xp.isfinite(X_arr))) > 0 or _to_float_scalar(
            xp.sum(~xp.isfinite(time))
        ) > 0:
            raise ValueError("X and time must contain only finite values")
        if _to_float_scalar(xp.sum(~xp.isfinite(event))) > 0 or _to_float_scalar(
            xp.sum((event != 0) & (event != 1))
        ) > 0:
            raise ValueError("event must contain only 0/1 finite values")
        if _to_float_scalar(xp.sum(time <= 0)) > 0:
            raise ValueError("time must contain only positive values")

        self._x_reference = (
            xp.mean(X_arr, dim=0)
            if xp.__name__ == "torch"
            else xp.mean(X_arr, axis=0)
        )
        X_arr = X_arr - self._x_reference.reshape(1, -1)
        order = (
            xp.argsort(time, stable=True)
            if xp.__name__ == "torch"
            else xp.argsort(time, kind="stable")
            if xp.__name__ == "numpy"
            else xp.argsort(time)
        )
        self._X_sorted = X_arr[order]
        self._time_sorted = time[order]
        self._event_sorted = event[order]
        self._order = order
        self._sorted = True
        self._n_events = int(_to_float_scalar(xp.sum(self._event_sorted)))
        self._efron_backend_index_cache = {}

        self._time_np = np.asarray(_to_numpy(self._time_sorted), dtype=np.float64)
        self._event_np = np.asarray(_to_numpy(self._event_sorted), dtype=np.float64)
        if self.ties == "efron":
            self._efron_pre_np = _build_efron_pre_numpy(
                self._time_np, self._event_np
            )
            self._breslow_pre_np = None
            self._breslow_event_indices_np = None
        else:
            self._breslow_pre_np = _build_breslow_pre_numpy(
                self._time_np, self._event_np
            )
            self._breslow_event_indices_np = _build_breslow_event_indices_numpy(
                self._time_np, self._event_np
            )
            self._efron_pre_np = None
        self._efron_csr = None

        if self.ties == "efron":
            _, grouped, _, _, _, first_indices = self._efron_pre_np
            counts = np.asarray(
                [len(indices) for indices in grouped], dtype=np.int64
            )
        else:
            first_indices, counts = self._breslow_pre_np
            grouped = self._breslow_event_indices_np
            counts = np.asarray(counts, dtype=np.int64)
        self._group_first_indices_np = np.asarray(
            first_indices, dtype=np.int64
        )
        self._group_counts_np = counts
        self._group_event_indices_np = (
            np.concatenate(grouped).astype(np.int64, copy=False)
            if grouped
            else np.empty(0, dtype=np.int64)
        )
        self._event_group_codes_np = np.repeat(
            np.arange(len(grouped), dtype=np.int64), counts
        )
        self._efron_fractions_np = (
            np.concatenate(
                [np.arange(count, dtype=np.float64) / count for count in counts]
            )
            if counts.size
            else np.empty(0, dtype=np.float64)
        )
        self._backend_group_metadata(xp, self._X_sorted)
        self._preprocessed_target = _CoxPreprocessedTarget(
            self._cache_generation, int(X_arr.shape[0])
        )
        return self._X_sorted, self._preprocessed_target

    def release_fit_cache(self):
        """Release all training-data and backend metadata references."""
        self._cache_generation += 1
        self._sorted = False
        self._X_sorted = None
        self._time_sorted = None
        self._event_sorted = None
        self._order = None
        self._time_np = None
        self._event_np = None
        self._efron_pre_np = None
        self._breslow_pre_np = None
        self._breslow_event_indices_np = None
        self._efron_csr = None
        self._efron_backend_index_cache = {}
        self._n_events = 0
        self._x_reference = None
        self._group_first_indices_np = None
        self._group_counts_np = None
        self._group_event_indices_np = None
        self._event_group_codes_np = None
        self._efron_fractions_np = None
        self._group_first_indices_backend = None
        self._group_counts_backend = None
        self._group_event_indices_backend = None
        self._event_group_codes_backend = None
        self._efron_fractions_backend = None
        self._preprocessed_target = None

    def _backend_group_metadata(self, xp, reference):
        """Return failure-group metadata cached by backend and device."""
        device = str(getattr(reference, "device", "cpu"))
        key = (xp.__name__, device, str(getattr(reference, "dtype", "")))
        cached = self._efron_backend_index_cache.get(key)
        if cached is None:
            cached = (
                _backend_index(self._group_first_indices_np, xp, reference),
                _xp_asarray(
                    self._group_counts_np, dtype=xp.float64, ref_arr=reference
                ),
                _backend_index(self._group_event_indices_np, xp, reference),
                _backend_index(self._event_group_codes_np, xp, reference),
                _xp_asarray(
                    self._efron_fractions_np,
                    dtype=xp.float64,
                    ref_arr=reference,
                ),
            )
            self._efron_backend_index_cache[key] = cached
        if reference is self._X_sorted:
            (
                self._group_first_indices_backend,
                self._group_counts_backend,
                self._group_event_indices_backend,
                self._event_group_codes_backend,
                self._efron_fractions_backend,
            ) = cached
        return cached

    @staticmethod
    def _reject_sample_weight(sample_weight):
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )

    @staticmethod
    def _reject_complex(value, name):
        if _is_complex_array(value):
            raise ValueError(f"{name} must be real-valued")

    def _coerce_coef(self, coef, xp):
        self._reject_complex(coef, "coef")
        return _xp_asarray(
            coef, dtype=xp.float64, ref_arr=self._X_sorted
        ).reshape(-1)

    def _zero_objective(self, *, compute_derivatives: bool):
        xp = _get_xp(self._X_sorted)
        result = {
            "log_likelihood": _backend_zeros((), xp, self._X_sorted),
        }
        if compute_derivatives:
            n_features = int(self._X_sorted.shape[1])
            result["score"] = _backend_zeros(
                (n_features,), xp, self._X_sorted
            )
            result["information"] = _backend_zeros(
                (n_features, n_features), xp, self._X_sorted
            )
        return result

    def _validate_coef(self, coef_dev, *, finite=True):
        self._reject_complex(coef_dev, "coef")
        xp = _get_xp(self._X_sorted)
        n_features = int(self._X_sorted.shape[1])
        if int(coef_dev.ndim) != 1 or int(coef_dev.shape[0]) != n_features:
            raise ValueError("coef must have shape (n_features,)")
        if finite and _to_float_scalar(xp.sum(~xp.isfinite(coef_dev))) > 0:
            raise ValueError("coef must contain only finite values")

    def _shared_objective(self, coef_dev, *, compute_derivatives: bool):
        """Use the audited three-backend risk-set implementation."""
        self._validate_coef(coef_dev)
        if self._n_events == 0:
            return self._zero_objective(compute_derivatives=compute_derivatives)
        return cox_counting_process_objective(
            coef_dev,
            self._X_sorted,
            self._time_sorted,
            self._event_sorted,
            ties=self.ties,
            compute_derivatives=compute_derivatives,
        )

    def value(self, X, y, coef, sample_weight=None) -> float:
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = self._coerce_coef(coef, xp)
        self._validate_coef(coef_dev)
        eta = self._X_sorted @ coef_dev
        loglik, _, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        return -_to_float_scalar(loglik) / self._X_sorted.shape[0]

    def gradient(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = self._coerce_coef(coef, xp)
        self._validate_coef(coef_dev)
        eta = self._X_sorted @ coef_dev
        _, score, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        return -score / self._X_sorted.shape[0]

    def gradient_preprocessed(self, coef):
        """Return a stable gradient from the active solver-owned fit cache.

        The trusted solver path skips duplicate scalar validity checks, but it
        retains the scaled direct-moment risk-set calculation used by the
        public gradient. Predictor-range checks may synchronize a device
        scalar; this is required until an associative signed-moment scan is
        available.
        """
        if not self._sorted or self._preprocessed_target is None:
            raise RuntimeError("Cox fit cache is not active")
        xp = _get_xp(self._X_sorted)
        coef_dev = self._coerce_coef(coef, xp)
        self._validate_coef(coef_dev, finite=False)
        eta = self._X_sorted @ coef_dev
        _, score, _ = self._objective_from_eta_backend(
            eta,
            self._X_sorted,
            xp,
            self.ties,
            compute_information=False,
            validate_finite_state=False,
        )
        return -score / self._X_sorted.shape[0]

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = self._coerce_coef(coef, xp)
        self._validate_coef(coef_dev)
        eta = self._X_sorted @ coef_dev
        loglik, score, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        n = self._X_sorted.shape[0]
        return -loglik / n, -score / n

    def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = self._coerce_coef(coef, xp)
        self._validate_coef(coef_dev)
        eta = self._X_sorted @ coef_dev
        _, score, loglik_hessian = self._objective_from_eta_backend(
            eta,
            self._X_sorted,
            xp,
            self.ties,
            compute_information=True,
        )
        n = self._X_sorted.shape[0]
        return -score / n, -loglik_hessian / n

    def hessian(self, X, y, coef, sample_weight=None):
        return self.fused_gradient_and_hessian(
            X, y, coef, sample_weight=sample_weight
        )[1]

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = (
            self._coerce_coef(coef, xp)
            if coef is not None
            else _xp_zeros(
                self._X_sorted.shape[1],
                dtype=xp.float64,
                ref_arr=self._X_sorted,
            )
        )
        self._validate_coef(coef_dev)
        eta = self._X_sorted @ coef_dev
        _, _, loglik_hessian = self._objective_from_eta_backend(
            eta,
            self._X_sorted,
            xp,
            self.ties,
            compute_information=True,
        )
        information = -loglik_hessian / self._X_sorted.shape[0]
        return _max_eigval_power(information)

    @staticmethod
    def _reverse_cumsum(values, xp):
        """Return an axis-zero reverse cumulative sum on every backend."""
        if xp.__name__ == "torch":
            return xp.cumsum(values.flip(0), dim=0).flip(0)
        return xp.cumsum(values[::-1], axis=0)[::-1]

    @staticmethod
    def _stable_segment_boundaries(eta, xp, max_block_rows):
        """Split predictor blocks until every block spans at most 500 logs.

        CuPy does not implement ``maximum.accumulate``. A bounded recursive
        range check provides the same numerical guarantee using only scalar
        device reductions, without copying the predictor to the host.
        """
        n = int(eta.shape[0])
        pending = [
            (lo, min(lo + max_block_rows, n))
            for lo in range(0, n, max_block_rows)
        ]
        boundaries = {0, n}
        while pending:
            lo, hi = pending.pop()
            if hi - lo > 1:
                block = eta[lo:hi]
                block_range = _to_float_scalar(
                    xp.max(block) - xp.min(block)
                )
                if block_range > 500.0:
                    midpoint = lo + (hi - lo) // 2
                    pending.append((lo, midpoint))
                    pending.append((midpoint, hi))
                    continue
            boundaries.add(lo)
            boundaries.add(hi)
        return np.asarray(sorted(boundaries), dtype=np.int64)

    def _suffix_group_moments(
        self, eta, X, xp, first_indices, *, validate_finite_state=True
    ):
        """Compute stable suffix log-sums and means at failure-group starts.

        A single global shift is fast but can underflow after the observation
        attaining that shift leaves a later risk set. Re-scanning every risk
        set avoids the underflow at quadratic cost. This routine instead
        uses reverse cumulative sums in adaptively bounded segments. Trusted
        solver calls use the same signed direct-moment calculation so large
        positive and negative first moments are never reconstructed by
        subtracting exponentiated log-sums. ``validate_finite_state`` controls
        redundant scalar error checks, never the stable risk-set calculation.
        """
        n, p = int(X.shape[0]), int(X.shape[1])
        n_groups = int(len(first_indices))
        risk_log_sum = _backend_zeros((n_groups,), xp, X)
        risk_mean = _backend_zeros((n_groups, p), xp, X)
        if n_groups == 0:
            return risk_log_sum, risk_mean
        if validate_finite_state and not bool(
            _to_float_scalar(xp.all(xp.isfinite(eta)))
        ):
            raise FloatingPointError("Cox linear predictor contains non-finite values")

        # Keep temporary moment buffers bounded for high-dimensional inputs.
        max_block_rows = max(
            1,
            min(n, 65_536, 2_000_000 // max(p, 1)),
        )
        boundaries = self._stable_segment_boundaries(eta, xp, max_block_rows)
        first_indices_backend = self._backend_group_metadata(xp, X)[0]

        tail_shift = None
        tail_sum = None
        tail_first = None
        first_indices = np.asarray(first_indices, dtype=np.int64)
        for boundary_index in range(len(boundaries) - 2, -1, -1):
            lo = int(boundaries[boundary_index])
            hi = int(boundaries[boundary_index + 1])
            block_shift = xp.max(eta[lo:hi])
            shift = (
                block_shift
                if tail_shift is None
                else xp.maximum(block_shift, tail_shift)
            )
            weights = xp.exp(eta[lo:hi] - shift)
            block_sum = self._reverse_cumsum(weights, xp)
            block_first = self._reverse_cumsum(
                X[lo:hi] * weights.reshape(-1, 1), xp
            )
            if tail_shift is not None:
                tail_scale = xp.exp(tail_shift - shift)
                block_sum = block_sum + tail_sum * tail_scale
                block_first = block_first + tail_first * tail_scale

            group_lo = int(np.searchsorted(first_indices, lo, side="left"))
            group_hi = int(np.searchsorted(first_indices, hi, side="left"))
            if group_hi > group_lo:
                local_indices = (
                    first_indices_backend[group_lo:group_hi] - lo
                )
                selected_sum = block_sum[local_indices]
                if validate_finite_state and bool(
                    _to_float_scalar(xp.any(selected_sum <= 0))
                ):
                    raise FloatingPointError(
                        "non-positive Cox risk-set denominator"
                    )
                risk_log_sum[group_lo:group_hi] = (
                    xp.log(selected_sum) + shift
                )
                risk_mean[group_lo:group_hi] = (
                    block_first[local_indices]
                    / selected_sum.reshape(-1, 1)
                )

            tail_shift = shift
            tail_sum = block_sum[0]
            tail_first = block_first[0]

        return risk_log_sum, risk_mean

    def _first_order_objective_from_eta_backend(
        self, eta, X, xp, ties, *, validate_finite_state=True
    ):
        """Evaluate log likelihood and score in near-linear time."""
        p = int(X.shape[1])
        first_indices = self._group_first_indices_np
        (
            _,
            counts_backend,
            event_indices,
            event_groups,
            fractions,
        ) = self._backend_group_metadata(xp, X)
        if len(first_indices) == 0:
            return (
                _backend_zeros((), xp, X),
                _backend_zeros((p,), xp, X),
                None,
            )

        risk_log_sum, risk_mean = self._suffix_group_moments(
            eta,
            X,
            xp,
            first_indices,
            validate_finite_state=validate_finite_state,
        )
        event_X = X[event_indices]
        event_eta = eta[event_indices]

        if ties == "breslow":
            loglik = _sum(event_eta, xp) - _sum(
                counts_backend * risk_log_sum, xp
            )
            score = _sum(event_X, xp, axis=0) - _sum(
                risk_mean * counts_backend.reshape(-1, 1),
                xp,
                axis=0,
            )
            return loglik, score, None

        # Efron correction in denominator-ratio space. Scaling each event
        # weight by its complete risk denominator avoids both overflow and the
        # global-shift underflow that motivated this implementation.
        event_weight_ratio = xp.exp(
            event_eta - risk_log_sum[event_groups]
        )
        event_ratio_sum = _backend_zeros(
            (len(first_indices),), xp, X
        )
        event_first_ratio = _backend_zeros(
            (len(first_indices), p), xp, X
        )
        if xp.__name__ == "torch":
            event_ratio_sum.index_add_(
                0, event_groups, event_weight_ratio
            )
            event_first_ratio.index_add_(
                0,
                event_groups,
                event_X * event_weight_ratio.reshape(-1, 1),
            )
        else:
            xp.add.at(event_ratio_sum, event_groups, event_weight_ratio)
            xp.add.at(
                event_first_ratio,
                event_groups,
                event_X * event_weight_ratio.reshape(-1, 1),
            )

        denominator_ratio = (
            1.0 - fractions * event_ratio_sum[event_groups]
        )
        if validate_finite_state and bool(
            _to_float_scalar(xp.any(denominator_ratio <= 0))
        ):
            raise FloatingPointError("non-positive Cox risk-set denominator")
        adjusted_mean = (
            risk_mean[event_groups]
            - fractions.reshape(-1, 1)
            * event_first_ratio[event_groups]
        ) / denominator_ratio.reshape(-1, 1)
        loglik = _sum(event_eta, xp) - _sum(
            risk_log_sum[event_groups] + xp.log(denominator_ratio), xp
        )
        score = _sum(event_X, xp, axis=0) - _sum(
            adjusted_mean, xp, axis=0
        )
        return loglik, score, None

    def _full_objective_from_eta_backend(self, eta, X, xp, ties):
        """Evaluate likelihood, score, and information in stable blocks."""
        n, p = int(X.shape[0]), int(X.shape[1])
        first_indices = self._group_first_indices_np
        counts = self._group_counts_np
        loglik = _backend_zeros((), xp, X)
        score = _backend_zeros((p,), xp, X)
        information = _backend_zeros((p, p), xp, X)
        if len(first_indices) == 0:
            return loglik, score, -information
        if not bool(_to_float_scalar(xp.all(xp.isfinite(eta)))):
            raise FloatingPointError("Cox linear predictor contains non-finite values")

        # Bound the n-by-p-by-p temporary used for suffix second moments.
        moment_width = max(p * p, 1)
        max_block_rows = max(
            1,
            min(n, 16_384, 2_000_000 // moment_width),
        )
        boundaries = self._stable_segment_boundaries(
            eta, xp, max_block_rows
        )
        event_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(counts)]
        )
        (
            first_indices_backend,
            counts_backend_all,
            event_indices_backend,
            event_groups_backend,
            fractions_backend,
        ) = self._backend_group_metadata(xp, X)

        tail_shift = None
        tail_sum = None
        tail_first = None
        tail_second = None
        tail_feature_shift = None
        for boundary_index in range(len(boundaries) - 2, -1, -1):
            lo = int(boundaries[boundary_index])
            hi = int(boundaries[boundary_index + 1])
            block_shift = xp.max(eta[lo:hi])
            shift = (
                block_shift
                if tail_shift is None
                else xp.maximum(block_shift, tail_shift)
            )
            feature_shift = X[hi - 1]
            centered_X = X[lo:hi] - feature_shift
            weights = xp.exp(eta[lo:hi] - shift)
            weighted_first = centered_X * weights.reshape(-1, 1)
            weighted_second = (
                weighted_first[:, :, None] * centered_X[:, None, :]
            )
            block_sum = self._reverse_cumsum(weights, xp)
            block_first = self._reverse_cumsum(weighted_first, xp)
            block_second = self._reverse_cumsum(weighted_second, xp)
            if tail_shift is not None:
                feature_delta = tail_feature_shift - feature_shift
                transformed_tail_first = (
                    tail_first + tail_sum * feature_delta
                )
                transformed_tail_second = (
                    tail_second
                    + feature_delta[:, None] * tail_first[None, :]
                    + tail_first[:, None] * feature_delta[None, :]
                    + tail_sum
                    * feature_delta[:, None]
                    * feature_delta[None, :]
                )
                tail_scale = xp.exp(tail_shift - shift)
                block_sum = block_sum + tail_sum * tail_scale
                block_first = block_first + transformed_tail_first * tail_scale
                block_second = block_second + transformed_tail_second * tail_scale

            group_lo = int(np.searchsorted(first_indices, lo, side="left"))
            group_hi = int(np.searchsorted(first_indices, hi, side="left"))
            if group_hi > group_lo:
                local_indices = (
                    first_indices_backend[group_lo:group_hi] - lo
                )
                selected_sum = block_sum[local_indices]
                if bool(_to_float_scalar(xp.any(selected_sum <= 0))):
                    raise FloatingPointError(
                        "non-positive Cox risk-set denominator"
                    )
                risk_log_sum = xp.log(selected_sum) + shift
                selected_first = block_first[local_indices]
                selected_second = block_second[local_indices]
                risk_mean = (
                    selected_first
                    / selected_sum.reshape(-1, 1)
                )
                risk_second = (
                    selected_second
                    / selected_sum.reshape(-1, 1, 1)
                )

                event_lo = int(event_offsets[group_lo])
                event_hi = int(event_offsets[group_hi])
                event_indices = event_indices_backend[
                    event_lo:event_hi
                ]
                event_groups = (
                    event_groups_backend[event_lo:event_hi] - group_lo
                )
                event_X = X[event_indices]
                centered_event_X = event_X - feature_shift
                event_eta = eta[event_indices]

                if ties == "breslow":
                    counts_backend = counts_backend_all[
                        group_lo:group_hi
                    ]
                    loglik = loglik + _sum(event_eta, xp) - _sum(
                        counts_backend * risk_log_sum, xp
                    )
                    score = score + _sum(centered_event_X, xp, axis=0) - _sum(
                        risk_mean * counts_backend.reshape(-1, 1),
                        xp,
                        axis=0,
                    )
                    covariance = (
                        risk_second
                        - risk_mean[:, :, None] * risk_mean[:, None, :]
                    )
                    information = information + _sum(
                        covariance
                        * counts_backend.reshape(-1, 1, 1),
                        xp,
                        axis=0,
                    )
                else:
                    n_groups = group_hi - group_lo
                    event_weight_ratio = xp.exp(event_eta - shift)
                    event_ratio_sum = _backend_zeros(
                        (n_groups,), xp, X
                    )
                    event_first_ratio = _backend_zeros(
                        (n_groups, p), xp, X
                    )
                    event_second_ratio = _backend_zeros(
                        (n_groups, p, p), xp, X
                    )
                    weighted_event_first = (
                        centered_event_X * event_weight_ratio.reshape(-1, 1)
                    )
                    weighted_event_second = (
                        weighted_event_first[:, :, None]
                        * centered_event_X[:, None, :]
                    )
                    if xp.__name__ == "torch":
                        event_ratio_sum.index_add_(
                            0, event_groups, event_weight_ratio
                        )
                        event_first_ratio.index_add_(
                            0, event_groups, weighted_event_first
                        )
                        event_second_ratio.index_add_(
                            0, event_groups, weighted_event_second
                        )
                    else:
                        xp.add.at(
                            event_ratio_sum, event_groups, event_weight_ratio
                        )
                        xp.add.at(
                            event_first_ratio,
                            event_groups,
                            weighted_event_first,
                        )
                        xp.add.at(
                            event_second_ratio,
                            event_groups,
                            weighted_event_second,
                        )

                    fractions = fractions_backend[
                        event_lo:event_hi
                    ]
                    denominator_ratio = (
                        selected_sum[event_groups]
                        - fractions * event_ratio_sum[event_groups]
                    )
                    if bool(
                        _to_float_scalar(xp.any(denominator_ratio <= 0))
                    ):
                        raise FloatingPointError(
                            "non-positive Cox risk-set denominator"
                        )
                    fractions_first = fractions.reshape(-1, 1)
                    fractions_second = fractions.reshape(-1, 1, 1)
                    adjusted_mean = (
                        selected_first[event_groups]
                        - fractions_first
                        * event_first_ratio[event_groups]
                    ) / denominator_ratio.reshape(-1, 1)
                    adjusted_second = (
                        selected_second[event_groups]
                        - fractions_second
                        * event_second_ratio[event_groups]
                    ) / denominator_ratio.reshape(-1, 1, 1)
                    loglik = loglik + _sum(event_eta, xp) - _sum(
                        xp.log(denominator_ratio) + shift,
                        xp,
                    )
                    score = score + _sum(centered_event_X, xp, axis=0) - _sum(
                        adjusted_mean, xp, axis=0
                    )
                    information = information + _sum(
                        adjusted_second
                        - adjusted_mean[:, :, None]
                        * adjusted_mean[:, None, :],
                        xp,
                        axis=0,
                    )

            tail_shift = shift
            tail_sum = block_sum[0]
            tail_first = block_first[0]
            tail_second = block_second[0]
            tail_feature_shift = feature_shift

        return loglik, score, -information

    def _objective_from_eta_backend(
        self,
        eta,
        X,
        xp,
        ties,
        *,
        compute_information=True,
        validate_finite_state=True,
    ):
        """Evaluate log likelihood and score from a precomputed predictor."""
        if not compute_information:
            return self._first_order_objective_from_eta_backend(
                eta,
                X,
                xp,
                ties,
                validate_finite_state=validate_finite_state,
            )
        return self._full_objective_from_eta_backend(eta, X, xp, ties)

    def _is_gpu(self, arr):
        xp = _get_xp(arr)
        return xp.__name__ == "cupy" or (
            xp.__name__ == "torch" and bool(arr.is_cuda)
        )

    def _compute_grad_hess(self, coef_dev, X_s):
        xp = _get_xp(X_s)
        self._validate_coef(coef_dev)
        eta = X_s @ coef_dev
        _, score, hessian = self._objective_from_eta_backend(
            eta, X_s, xp, self.ties, compute_information=True
        )
        return score, hessian

    def _gpu_loglik(self, coef_dev, X_s):
        xp = _get_xp(X_s)
        self._validate_coef(coef_dev)
        eta = X_s @ coef_dev
        return self._objective_from_eta_backend(
            eta, X_s, xp, self.ties, compute_information=False
        )[0]

    def _loglik_from_eta(self, eta, X_s):
        xp = _get_xp(X_s)
        return self._objective_from_eta_backend(eta, X_s, xp, self.ties)[0]

    def _gpu_loglik_from_eta(self, eta, X_s):
        return self._loglik_from_eta(eta, X_s)

    def _grad_from_eta(self, eta, X_s):
        xp = _get_xp(X_s)
        return self._objective_from_eta_backend(eta, X_s, xp, self.ties)[1]

    def _cupy_grad_hess(self, coef_dev, X_s):
        return self._compute_grad_hess(coef_dev, X_s)

    def _triton_grad_hess(self, coef_dev, X_s):
        return None

    def _torch_breslow_grad_hess(self, coef_dev, X_s):
        return None

    def _efron_loglik_backend(self, eta, X, xp):
        return self._objective_from_eta_backend(eta, X, xp, "efron")[0]

    def _efron_grad_hess_backend(self, eta, X, xp):
        _, score, hessian = self._objective_from_eta_backend(eta, X, xp, "efron")
        return score, hessian

    def _cpu_loglik_cached(self, eta_np, X_np):
        return self._cpu_loglik(eta_np, self._time_np, self._event_np)

    def _cpu_loglik(self, eta_np, time_np, event_np):
        X_np = np.asarray(_to_numpy(self._X_sorted), dtype=np.float64)
        eta_np = np.asarray(eta_np, dtype=np.float64)
        return float(
            self._objective_from_eta_backend(
                eta_np, X_np, np, self.ties
            )[0]
        )

    def _cpu_grad_hess(self, eta_np, time_np, event_np):
        X_np = np.asarray(_to_numpy(self._X_sorted), dtype=np.float64)
        eta_np = np.asarray(eta_np, dtype=np.float64)
        _, score, hessian = self._objective_from_eta_backend(
            eta_np, X_np, np, self.ties
        )
        return np.asarray(score, dtype=np.float64), np.asarray(
            hessian, dtype=np.float64
        )

    @staticmethod
    def _efron_grad_hess_np(eta, X, efron_pre):
        temp = CoxPartialLikelihoodLoss(ties="efron")
        temp._X_sorted = np.asarray(X, dtype=np.float64)
        temp._time_np = np.asarray(efron_pre[0], dtype=np.float64)
        temp._event_np = np.zeros(X.shape[0], dtype=np.float64)
        temp._efron_pre_np = efron_pre
        _, score, hessian = temp._objective_from_eta_backend(
            np.asarray(eta, dtype=np.float64), temp._X_sorted, np, "efron"
        )
        return np.asarray(score), np.asarray(hessian)

    def _cpu_fused_loglik_grad(self, eta_np, X_np, time_np, event_np):
        loglik = self._cpu_loglik(eta_np, time_np, event_np)
        score, _ = self._cpu_grad_hess(eta_np, time_np, event_np)
        return loglik, score, None

    def _cpu_loglik_grad(self, eta_np, X_np):
        loglik = self._cpu_loglik(eta_np, self._time_np, self._event_np)
        score, _ = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
        return loglik, score

    def _cpu_fused_loglik_grad_hess(self, eta_np, X_np, time_np, event_np):
        loglik = self._cpu_loglik(eta_np, time_np, event_np)
        score, hessian = self._cpu_grad_hess(eta_np, time_np, event_np)
        return loglik, score, hessian
