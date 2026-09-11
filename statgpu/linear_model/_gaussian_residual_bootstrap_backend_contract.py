"""Backend-native execution for the Gaussian residual-bootstrap contract.

This follow-up keeps the statistical scope intentionally unchanged:

* squared-error penalized models only;
* no analytic sample weights;
* ``cov_type='nonrobust'``;
* residual resampling with replacement;
* the same penalty/tuning/solver semantics for every child refit.

The expensive numerical refits remain on the fit-recorded NumPy/CuPy/Torch
backend and concrete device. A deterministic NumPy-generated integer index
schedule is control-plane state only; each backend consumes the same draws.
Child estimators retain their established NumPy reporting snapshot after each
backend-native fit, and the final bootstrap summary is published through the
existing NumPy reporting boundary.
"""

from __future__ import annotations

import copy
import functools
import hashlib
from contextlib import contextmanager

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.inference._results import ParameterInferenceResult
from statgpu.linear_model._gaussian_inference import _as_backend_array
from statgpu.linear_model._penalized_glm_inference_contract import (
    _bootstrap_penalty_template,
    _penalty_name,
    _publish_contract,
    _resolve_contract,
    _selected_backend,
    _selected_device,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


_MARKER = "_statgpu_backend_native_gaussian_residual_bootstrap"


def _draw_resample_indices(n: int, B: int, random_state) -> np.ndarray:
    """Return one deterministic backend-neutral residual-index schedule."""
    if B < 2:
        raise ValueError("n_bootstrap must be an integer >= 2.")
    if n <= 0:
        raise ValueError("Residual bootstrap requires at least one observation.")
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(B, n), dtype=np.int64)


def _schedule_sha256(schedule: np.ndarray) -> str:
    """Return a stable hash for the exact backend-neutral index schedule."""
    normalized = np.ascontiguousarray(schedule, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _backend_device_request(backend: str) -> str:
    if backend == "numpy":
        return "cpu"
    if backend == "cupy":
        return "cuda"
    if backend == "torch":
        return "torch"
    raise RuntimeError(f"Unsupported bootstrap backend {backend!r}.")


def _bootstrap_owner_value(owner, public_name: str, private_name: str, default):
    if hasattr(owner, public_name):
        return getattr(owner, public_name)
    return getattr(owner, private_name, default)


def _make_child_refit(owner, *, backend: str) -> PenalizedLinearRegression:
    """Create one inference-disabled refit that preserves the fitted contract."""
    kwargs = dict(
        penalty=copy.deepcopy(_bootstrap_penalty_template(owner)),
        alpha=float(owner.alpha),
        l1_ratio=float(getattr(owner, "l1_ratio", 0.5)),
        penalty_kwargs=copy.deepcopy(getattr(owner, "penalty_kwargs", None) or {}),
        fit_intercept=bool(getattr(owner, "_effective_intercept", True)),
        max_iter=int(_bootstrap_owner_value(owner, "max_iter", "_max_iter", 1000)),
        tol=float(_bootstrap_owner_value(owner, "tol", "_tol", 1e-4)),
        device=_backend_device_request(backend),
        n_jobs=getattr(owner, "n_jobs", getattr(owner, "_n_jobs", None)),
        solver=_bootstrap_owner_value(owner, "solver", "_solver", "auto"),
        lipschitz_L=_bootstrap_owner_value(
            owner, "lipschitz_L", "_lipschitz_L", None
        ),
        gpu_memory_cleanup=bool(
            getattr(owner, "gpu_memory_cleanup", getattr(owner, "_gpu_memory_cleanup", False))
        ),
        stopping=_bootstrap_owner_value(owner, "stopping", "_stopping", "coef_delta"),
        lla=bool(_bootstrap_owner_value(owner, "lla", "_lla", True)),
        max_lla_iters=int(
            _bootstrap_owner_value(owner, "max_lla_iters", "_max_lla_iters", 50)
        ),
        lla_tol=float(_bootstrap_owner_value(owner, "lla_tol", "_lla_tol", 1e-6)),
        loss_kwargs=copy.deepcopy(
            _bootstrap_owner_value(owner, "loss_kwargs", "_loss_kwargs", None)
        ),
        compute_inference=False,
        inference_method="auto",
    )
    # cpu_solver is a legacy CPU-stage control. Preserve it only where it can
    # actually own execution; passing it explicitly on GPU would create a
    # misleading deprecated request without affecting the canonical solver.
    if backend == "numpy":
        kwargs["cpu_solver"] = getattr(owner, "cpu_solver", "fista")
    return PenalizedLinearRegression(**kwargs)


def _backend_index(schedule_row: np.ndarray, backend: str, resid):
    if backend == "torch":
        import torch

        return torch.as_tensor(schedule_row, dtype=torch.long, device=resid.device)
    if backend == "cupy":
        import cupy as cp

        with cp.cuda.Device(int(resid.device.id)):
            return cp.asarray(schedule_row, dtype=cp.int64)
    return schedule_row


def _take_residuals(resid, index, backend: str):
    if backend == "torch":
        return resid.index_select(0, index)
    return resid[index]


@contextmanager
def _child_device_context(backend: str, device: str):
    """Enter the exact parent device before constructing/running a child fit."""
    if backend == "numpy":
        if device != "cpu":
            raise RuntimeError(
                f"Invalid NumPy residual-bootstrap device provenance: {device!r}."
            )
        yield
        return

    if backend == "cupy":
        import cupy as cp

        if not str(device).startswith("cuda:"):
            raise RuntimeError(
                f"Invalid CuPy residual-bootstrap device provenance: {device!r}."
            )
        device_id = int(str(device).split(":", 1)[1])
        with cp.cuda.Device(device_id):
            yield
        return

    if backend == "torch":
        # ``cpu`` is accepted only for host-side contract doubles. A real
        # maintained Torch fit records a concrete CUDA device.
        if device == "cpu":
            yield
            return
        if not str(device).startswith("cuda:"):
            raise RuntimeError(
                f"Invalid Torch residual-bootstrap device provenance: {device!r}."
            )
        import torch

        target = torch.device(device)
        with torch.cuda.device(target):
            yield
        return

    raise RuntimeError(f"Unsupported bootstrap backend {backend!r}.")


def _assert_child_provenance(child, *, backend: str, device: str) -> None:
    actual_backend = str(getattr(child, "_selected_backend_name", "") or "").lower()
    actual_device = str(getattr(child, "_selected_backend_device", "") or "")
    expected_device = "cpu" if backend == "numpy" else device
    if actual_backend != backend or actual_device != expected_device:
        raise RuntimeError(
            "Residual-bootstrap child refit changed execution provenance: "
            f"expected {backend}/{expected_device}, got "
            f"{actual_backend or '<missing>'}/{actual_device or '<missing>'}."
        )


def _record_successful_diagnostics(self, X_native, y_native, y_pred, n: int) -> None:
    """Preserve the established Gaussian diagnostic/reporting snapshot."""
    X_np = np.asarray(_to_numpy(X_native), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y_native), dtype=np.float64).reshape(-1)
    y_pred_np = np.asarray(_to_numpy(y_pred), dtype=np.float64).reshape(-1)
    self._X_design = (
        np.column_stack([np.ones(n), X_np])
        if bool(getattr(self, "_effective_intercept", True))
        else X_np.copy()
    )
    self._y = y_np
    self._resid = y_np - y_pred_np
    self._nobs = n


def _backend_native_gaussian_residual_bootstrap(self, X, y):
    """Execute the established unweighted Gaussian residual bootstrap."""
    backend = _selected_backend(self)
    device = _selected_device(self, backend)
    if str(getattr(self, "cov_type", "nonrobust")).lower() != "nonrobust":
        raise NotImplementedError(
            "Gaussian residual-bootstrap inference requires cov_type='nonrobust'."
        )

    try:
        B = int(getattr(self, "n_bootstrap", 200))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_bootstrap must be an integer >= 2.") from exc
    if B < 2:
        raise ValueError("n_bootstrap must be an integer >= 2.")

    X_native = _as_backend_array(X, backend, device=device)
    y_native = _as_backend_array(y, backend, like=X_native, device=device).reshape(-1)
    n = int(X_native.shape[0])
    coef_native = _as_backend_array(
        np.asarray(self.coef_, dtype=np.float64),
        backend,
        like=X_native,
        device=device,
    ).reshape(-1)
    y_pred = X_native @ coef_native
    if bool(getattr(self, "_effective_intercept", True)):
        y_pred = y_pred + float(self.intercept_)
    resid = y_native - y_pred

    random_state = getattr(self, "bootstrap_random_state", None)
    schedule = _draw_resample_indices(n, B, random_state)
    schedule_hash = _schedule_sha256(schedule)
    params_dim = int(len(self._params))
    boot_params = np.empty((B, params_dim), dtype=np.float64)
    selected_solvers = []

    for b in range(B):
        with _child_device_context(backend, device):
            index = _backend_index(schedule[b], backend, resid)
            y_star = y_pred + _take_residuals(resid, index, backend)
            child = _make_child_refit(self, backend=backend)
            child.fit(X_native, y_star)
        _assert_child_provenance(child, backend=backend, device=device)
        child_params = np.asarray(_to_numpy(child._params), dtype=np.float64).reshape(-1)
        if child_params.shape != (params_dim,):
            raise RuntimeError(
                "Residual-bootstrap child parameter shape changed across refits: "
                f"expected {(params_dim,)}, got {child_params.shape}."
            )
        boot_params[b] = child_params
        selected_solver = str(getattr(child, "_selected_solver", "") or "")
        if selected_solver:
            selected_solvers.append(selected_solver)

    bse = np.std(boot_params, axis=0, ddof=1)
    pvalues = np.empty(params_dim, dtype=np.float64)
    for i in range(params_dim):
        coef_b = boot_params[:, i]
        pvalues[i] = min(
            1.0,
            2.0
            * min(
                float(np.mean(coef_b <= 0.0)),
                float(np.mean(coef_b >= 0.0)),
            ),
        )
    conf_int = np.column_stack(
        [
            np.quantile(boot_params, 0.025, axis=0),
            np.quantile(boot_params, 0.975, axis=0),
        ]
    )
    params = np.asarray(self._params, dtype=np.float64)
    statistic = params / (bse + 1e-30)

    # Host transfer happens only after all numerical child refits complete.
    _record_successful_diagnostics(self, X_native, y_native, y_pred, n)

    self._bse = bse
    self._pvalues = pvalues
    self._conf_int = conf_int
    self._tvalues = statistic
    metadata = {
        "n_bootstrap": B,
        "random_state": random_state,
        "resampling_scope": "unweighted_gaussian_residual",
        "resampling_schedule": "numpy_generator_control_plane",
        "resampling_schedule_sha256": schedule_hash,
        "refit_penalty": _penalty_name(self),
        "numerical_backend": backend,
        "numerical_device": device,
        "reporting_backend": "numpy",
        "reporting_boundary": "post_numerical_inference",
    }
    if selected_solvers:
        metadata["child_selected_solvers"] = sorted(set(selected_solvers))

    self._inference_result = ParameterInferenceResult(
        method="residual_bootstrap",
        params=params.copy(),
        bse=bse.copy(),
        statistic=statistic.copy(),
        statistic_name="z",
        pvalues=pvalues.copy(),
        conf_int=conf_int.copy(),
        distribution="bootstrap_percentile",
        metadata=metadata,
    )
    self._inference_result.apply_to(self)


def _install_post_fit_bootstrap_backend_contract() -> None:
    current = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    if getattr(current, _MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None):
        if not bool(getattr(self, "_compute_inference_enabled", False)):
            return current(self, X, y, sample_weight=sample_weight)

        contract = getattr(self, "_statgpu_pending_inference_contract", None)
        if contract is None:
            try:
                contract = _resolve_contract(self)
            except NotImplementedError:
                return current(self, X, y, sample_weight=sample_weight)

        if str(contract.get("resolved", "")).lower() != "residual_bootstrap":
            return current(self, X, y, sample_weight=sample_weight)
        if sample_weight is not None:
            raise NotImplementedError(
                "Weighted Gaussian residual-bootstrap inference is not implemented. "
                "Set sample_weight=None or choose another supported inference method."
            )

        self._statgpu_pending_inference_contract = contract
        _backend_native_gaussian_residual_bootstrap(self, X, y)
        _publish_contract(self)
        return None

    setattr(wrapped, _MARKER, True)
    PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = wrapped


def install_backend_native_gaussian_residual_bootstrap() -> None:
    """Install the backend execution extension after the PR #142 contracts."""
    _install_post_fit_bootstrap_backend_contract()


__all__ = [
    "install_backend_native_gaussian_residual_bootstrap",
]
