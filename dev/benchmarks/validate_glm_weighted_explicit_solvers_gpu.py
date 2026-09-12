#!/usr/bin/env python3
"""Physical CUDA acceptance for issue #150 weighted smooth GLM solvers.

Schema v3 freezes the final pre-P100 contract. A successful artifact proves:

- clean exact source;
- every claimed ordinary GLM family/link for explicit Newton and L-BFGS;
- NumPy/CuPy/Torch numerical parity and concrete-device provenance;
- positive global analytic-weight rescaling invariance;
- no convergence or L-BFGS line-search warning on accepted rows;
- representative heterogeneous-container routing; and
- weighted Negative-Binomial/Gamma/Inverse-Gaussian penalized and CV L-BFGS
  consumer parity, including selected-alpha identity.

The numerical tolerances below were fixed before the first physical run. A
failed run must not be made green by loosening them without a reviewed schema
change and a fresh physical artifact.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from statgpu.linear_model import (
    GammaRegression,
    GeneralizedLinearModel,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    PenalizedGLM_CV,
    PenalizedGeneralizedLinearModel,
    TweedieRegression,
)
from statgpu.solvers._convergence import ConvergenceWarning


SCHEMA_VERSION = 3
DATA_SEED = 150001
N_SAMPLES = 128
N_FEATURES = 3
SOLVER_TOL = 1.0e-8
ATOL_COEF = 2.0e-5
ATOL_INTERCEPT = 2.0e-5
ATOL_WEIGHT_RESCALE = 2.0e-6

_CASES = (
    "gaussian",
    "binomial",
    "poisson",
    "gamma_log",
    "gamma_inverse",
    "inverse_gaussian",
    "negative_binomial",
    "tweedie",
)
_SOLVERS = ("newton", "lbfgs")
_CONSUMER_CASES = ("negative_binomial", "gamma", "inverse_gaussian")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("physical validation requires a clean worktree")
    return sha


def _require_gpu_backends():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("CuPy is required for physical validation") from exc
    try:
        import torch
    except Exception as exc:  # pragma: no cover - physical runner
        raise RuntimeError("Torch is required for physical validation") from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")
    return cp, torch, 0, torch.device("cuda:0")


@contextmanager
def _solver_warning_gate():
    """Turn accepted-path solver failure/stagnation warnings into hard failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings(
            "error",
            message="lbfgs_solver: line search failed.*",
            category=RuntimeWarning,
        )
        yield


def _device_name(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _to_numpy(value):
    if hasattr(value, "get"):
        return np.asarray(value.get())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _backend_of(value) -> str:
    return type(value).__module__.split(".", 1)[0]


def _case_data(case: str):
    rng = np.random.default_rng(DATA_SEED + _CASES.index(case))
    X = rng.normal(scale=0.25, size=(N_SAMPLES, N_FEATURES)).astype(np.float64)
    beta = np.array([0.18, -0.12, 0.08], dtype=np.float64)
    eta = 0.20 + X @ beta

    if case == "gaussian":
        y = eta + rng.normal(scale=0.08, size=N_SAMPLES)
    elif case == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p).astype(np.float64)
        y[0], y[1] = 0.0, 1.0
    elif case == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    elif case == "gamma_log":
        y = np.exp(eta) * rng.lognormal(0.0, 0.08, size=N_SAMPLES)
    elif case == "gamma_inverse":
        eta_i = np.clip(
            1.0 + X @ np.array([0.08, -0.05, 0.04]), 0.6, 1.4
        )
        y = (1.0 / eta_i) * rng.lognormal(0.0, 0.04, size=N_SAMPLES)
    elif case == "inverse_gaussian":
        y = np.exp(eta) * rng.lognormal(0.0, 0.06, size=N_SAMPLES)
    elif case == "negative_binomial":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    elif case == "tweedie":
        y = np.exp(eta) * rng.lognormal(0.0, 0.08, size=N_SAMPLES)
    else:  # pragma: no cover - validator table bug
        raise AssertionError(case)

    weights = np.linspace(0.55, 1.65, N_SAMPLES, dtype=np.float64)
    weights[::17] = 0.0
    return X, np.asarray(y, dtype=np.float64), weights


def _make_model(case: str, solver: str, device: str):
    kwargs = dict(
        solver=solver,
        device=device,
        max_iter=1000,
        tol=SOLVER_TOL,
        compute_inference=False,
    )
    if case == "gamma_log":
        return GammaRegression(link="log", **kwargs)
    if case == "gamma_inverse":
        return GammaRegression(link="inverse_power", **kwargs)
    if case == "inverse_gaussian":
        return InverseGaussianRegression(**kwargs)
    if case == "negative_binomial":
        return NegativeBinomialRegression(alpha=0.7, **kwargs)
    if case == "tweedie":
        return TweedieRegression(power=1.5, **kwargs)
    return GeneralizedLinearModel(family=case, **kwargs)


def _fit_ordinary(case, solver, X, y, weights, *, device):
    model = _make_model(case, solver, device)
    with _solver_warning_gate():
        model.fit(X, y, sample_weight=weights)
    return {
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64),
        "intercept": float(model.intercept_),
        "n_iter": int(model.n_iter_),
        "selected_solver": str(getattr(model, "_selected_solver", "")),
        "backend": str(getattr(model, "_selected_backend_name", "")),
        "device": str(getattr(model, "_selected_backend_device", "")),
        "design_backend": _backend_of(model._X_design),
        "weight_backend": _backend_of(model._sample_weight_inf),
    }


def _assert_provenance(name, snap, *, solver, backend, device):
    expected = {
        "selected_solver": solver,
        "backend": backend,
        "device": device,
        "design_backend": backend,
        "weight_backend": backend,
    }
    for key, value in expected.items():
        if snap.get(key) != value:
            raise AssertionError(
                f"{name}: {key}={snap.get(key)!r}, expected {value!r}"
            )


def _errors(reference, candidate):
    return {
        "coef": float(np.max(np.abs(reference["coef"] - candidate["coef"]))),
        "intercept": float(abs(reference["intercept"] - candidate["intercept"])),
    }


def _assert_parity(name, reference, candidate):
    errors = _errors(reference, candidate)
    if errors["coef"] > ATOL_COEF:
        raise AssertionError(
            f"{name}: coefficient error {errors['coef']:.3e} exceeds {ATOL_COEF:.3e}"
        )
    if errors["intercept"] > ATOL_INTERCEPT:
        raise AssertionError(
            f"{name}: intercept error {errors['intercept']:.3e} exceeds "
            f"{ATOL_INTERCEPT:.3e}"
        )
    return errors


def _assert_rescaling(name, base, scaled):
    errors = _errors(base, scaled)
    if max(errors.values()) > ATOL_WEIGHT_RESCALE:
        raise AssertionError(
            f"{name}: weight-rescaling drift {errors} exceeds "
            f"{ATOL_WEIGHT_RESCALE:.3e}"
        )
    return errors


def _json_snap(snap):
    return {
        **{k: v for k, v in snap.items() if k != "coef"},
        "coef": snap["coef"].tolist(),
    }


def _container_arrays(route, X_np, y_np, w_np, cp, torch, torch_device):
    if route == "cupy":
        return cp.asarray(X_np), cp.asarray(y_np), cp.asarray(w_np)
    if route == "torch":
        return (
            torch.as_tensor(X_np, dtype=torch.float64, device=torch_device),
            torch.as_tensor(y_np, dtype=torch.float64, device=torch_device),
            torch.as_tensor(w_np, dtype=torch.float64, device=torch_device),
        )
    return X_np, y_np, w_np


def _consumer_source_case(case: str) -> str:
    return {
        "negative_binomial": "negative_binomial",
        "gamma": "gamma_log",
        "inverse_gaussian": "inverse_gaussian",
    }[case]


def _consumer_loss_kwargs(case: str):
    if case == "negative_binomial":
        return {"alpha": 0.7}
    if case == "gamma":
        return {"link": "log"}
    return None


def _fit_penalized_consumer(case, X, y, weights, *, device):
    with _solver_warning_gate():
        model = PenalizedGeneralizedLinearModel(
            loss=case,
            loss_kwargs=_consumer_loss_kwargs(case),
            penalty="l2",
            alpha=0.03,
            solver="lbfgs",
            device=device,
            max_iter=800,
            tol=SOLVER_TOL,
        ).fit(X, y, sample_weight=weights)
    return {
        "coef": np.asarray(_to_numpy(model.coef_), dtype=np.float64),
        "intercept": float(model.intercept_),
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
    }


def _fit_cv_consumer(case, X, y, weights, *, device):
    with _solver_warning_gate():
        cv = PenalizedGLM_CV(
            loss=case,
            loss_kwargs=_consumer_loss_kwargs(case),
            penalty="l2",
            alpha_grid=np.array([0.08, 0.03]),
            cv=2,
            random_state=150,
            solver="auto",
            device=device,
            max_iter=400,
            tol=SOLVER_TOL,
        ).fit(X, y, sample_weight=weights)
    return {
        "coef": np.asarray(_to_numpy(cv.estimator_.coef_), dtype=np.float64),
        "intercept": float(cv.estimator_.intercept_),
        "selected_alpha": float(cv.alpha_),
        "selected_solver": str(cv.estimator_._selected_solver),
        "backend": str(cv.estimator_._selected_backend_name),
        "device": str(cv.estimator_._selected_backend_device),
    }


def _assert_consumer_provenance(name, snap, *, backend, device):
    expected = {
        "selected_solver": "lbfgs",
        "backend": backend,
        "device": device,
    }
    for key, value in expected.items():
        if snap.get(key) != value:
            raise AssertionError(
                f"{name}: {key}={snap.get(key)!r}, expected {value!r}"
            )


def run(output: Path):
    source_sha = _require_clean_source()
    cp, torch, device_id, torch_device = _require_gpu_backends()
    cupy_properties = cp.cuda.runtime.getDeviceProperties(device_id)

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "torch": torch.__version__,
        "cuda_device_ordinal": device_id,
        "cupy_device_name": _device_name(cupy_properties["name"]),
        "torch_device_name": torch.cuda.get_device_name(torch_device),
    }

    ordinary = {}
    for case in _CASES:
        X_np, y_np, w_np = _case_data(case)
        for solver in _SOLVERS:
            reference = _fit_ordinary(
                case, solver, X_np, y_np, w_np, device="cpu"
            )
            _assert_provenance(
                f"{case}/{solver}/numpy",
                reference,
                solver=solver,
                backend="numpy",
                device="cpu",
            )
            key = f"{case}/{solver}"
            ordinary[key] = {"numpy": {"snapshot": _json_snap(reference)}}

            for backend, route, device in (
                ("cupy", "cupy", "cuda"),
                ("torch", "torch", "torch"),
            ):
                Xb, yb, wb = _container_arrays(
                    route, X_np, y_np, w_np, cp, torch, torch_device
                )
                context = (
                    cp.cuda.Device(device_id)
                    if backend == "cupy"
                    else torch.cuda.device(torch_device)
                )
                with context:
                    snap = _fit_ordinary(case, solver, Xb, yb, wb, device=device)
                    scaled = _fit_ordinary(
                        case, solver, Xb, yb, 7.0 * wb, device=device
                    )
                _assert_provenance(
                    f"{case}/{solver}/{backend}",
                    snap,
                    solver=solver,
                    backend=backend,
                    device=f"cuda:{device_id}",
                )
                ordinary[key][backend] = {
                    "snapshot": _json_snap(snap),
                    "errors_vs_numpy": _assert_parity(
                        f"{case}/{solver}/{backend}", reference, snap
                    ),
                    "errors_weight_rescale": _assert_rescaling(
                        f"{case}/{solver}/{backend}/weight_rescale", snap, scaled
                    ),
                }

    crossings = {}
    X_np, y_np, w_np = _case_data("binomial")
    X_cp, y_cp, w_cp = _container_arrays(
        "cupy", X_np, y_np, w_np, cp, torch, torch_device
    )
    X_t, y_t, w_t = _container_arrays(
        "torch", X_np, y_np, w_np, cp, torch, torch_device
    )
    for solver in _SOLVERS:
        reference = _fit_ordinary(
            "binomial", solver, X_np, y_np, w_np, device="cpu"
        )
        with cp.cuda.Device(device_id):
            t_to_c = _fit_ordinary(
                "binomial", solver, X_t, y_t, w_t, device="cuda"
            )
        _assert_provenance(
            f"{solver}/torch_to_cupy",
            t_to_c,
            solver=solver,
            backend="cupy",
            device=f"cuda:{device_id}",
        )
        with torch.cuda.device(torch_device):
            c_to_t = _fit_ordinary(
                "binomial", solver, X_cp, y_cp, w_cp, device="torch"
            )
        _assert_provenance(
            f"{solver}/cupy_to_torch",
            c_to_t,
            solver=solver,
            backend="torch",
            device=f"cuda:{device_id}",
        )
        crossings[f"{solver}/torch_to_cupy"] = {
            "snapshot": _json_snap(t_to_c),
            "errors_vs_numpy": _assert_parity(
                f"{solver}/torch_to_cupy", reference, t_to_c
            ),
        }
        crossings[f"{solver}/cupy_to_torch"] = {
            "snapshot": _json_snap(c_to_t),
            "errors_vs_numpy": _assert_parity(
                f"{solver}/cupy_to_torch", reference, c_to_t
            ),
        }

    consumers = {}
    for case in _CONSUMER_CASES:
        X_np, y_np, w_np = _case_data(_consumer_source_case(case))
        ref_pen = _fit_penalized_consumer(case, X_np, y_np, w_np, device="cpu")
        ref_cv = _fit_cv_consumer(case, X_np, y_np, w_np, device="cpu")
        _assert_consumer_provenance(
            f"{case}/penalized/numpy",
            ref_pen,
            backend="numpy",
            device="cpu",
        )
        _assert_consumer_provenance(
            f"{case}/cv/numpy",
            ref_cv,
            backend="numpy",
            device="cpu",
        )
        consumers[case] = {
            "numpy": {
                "penalized": _json_snap(ref_pen),
                "cv": _json_snap(ref_cv),
            }
        }

        for backend, route, device in (
            ("cupy", "cupy", "cuda"),
            ("torch", "torch", "torch"),
        ):
            Xb, yb, wb = _container_arrays(
                route, X_np, y_np, w_np, cp, torch, torch_device
            )
            context = (
                cp.cuda.Device(device_id)
                if backend == "cupy"
                else torch.cuda.device(torch_device)
            )
            with context:
                pen = _fit_penalized_consumer(case, Xb, yb, wb, device=device)
                cv = _fit_cv_consumer(case, Xb, yb, wb, device=device)
            expected_device = f"cuda:{device_id}"
            _assert_consumer_provenance(
                f"{case}/penalized/{backend}",
                pen,
                backend=backend,
                device=expected_device,
            )
            _assert_consumer_provenance(
                f"{case}/cv/{backend}",
                cv,
                backend=backend,
                device=expected_device,
            )
            if cv["selected_alpha"] != ref_cv["selected_alpha"]:
                raise AssertionError(
                    f"{case}/cv/{backend}: selected_alpha={cv['selected_alpha']}, "
                    f"numpy={ref_cv['selected_alpha']}"
                )
            consumers[case][backend] = {
                "penalized": {
                    **_json_snap(pen),
                    "errors_vs_numpy": _assert_parity(
                        f"{case}/penalized/{backend}", ref_pen, pen
                    ),
                },
                "cv": {
                    **_json_snap(cv),
                    "errors_vs_numpy": _assert_parity(
                        f"{case}/cv/{backend}", ref_cv, cv
                    ),
                    "selected_alpha_matches_numpy": True,
                },
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": True,
        "environment": environment,
        "data_contract": {
            "seed": DATA_SEED,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "solver_tol": SOLVER_TOL,
            "weight_pattern": "linspace(0.55,1.65), every 17th row zero",
        },
        "tolerances": {
            "coef_max_abs": ATOL_COEF,
            "intercept_abs": ATOL_INTERCEPT,
            "weight_rescale_max_abs": ATOL_WEIGHT_RESCALE,
        },
        "warning_gate": {
            "convergence_warning": "error",
            "lbfgs_line_search_failure": "error",
        },
        "ordinary_cases": ordinary,
        "cross_container_cases": crossings,
        "shared_consumers": consumers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "success",
                "schema_version": SCHEMA_VERSION,
                "source_sha": source_sha,
                "ordinary_route_count": len(_CASES) * len(_SOLVERS) * 3,
                "cross_container_case_count": len(crossings),
                "consumer_case_count": len(_CONSUMER_CASES),
                "consumer_backend_route_count": len(_CONSUMER_CASES) * 3,
                "output": str(output),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pr151_glm_weighted_explicit_solvers_gpu/"
            "pr151_glm_weighted_explicit_solvers_gpu.json"
        ),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
