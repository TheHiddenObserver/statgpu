#!/usr/bin/env python3
"""PR151 physical CUDA schema v4: inverse-power Gamma domain closure.

Schema v4 is a strict extension of the already accepted PR151 schema-v3
validator.  It first runs the complete v3 matrix unchanged, then adds the
inverse-power Gamma domain/initialization rows introduced after the historical
``c6781cb6`` physical run.

The v4 acceptance proves, on one clean exact source revision:

- every historical schema-v3 ordinary/cross-container/shared-consumer gate;
- feasible no-intercept inverse-Gamma Newton/L-BFGS on NumPy/CuPy/Torch;
- active training predictors stay inside the maintained smooth numerical band;
- positive global analytic-weight rescaling invariance;
- executed backend/device provenance, including heterogeneous containers;
- fail-hard contradictory active designs and zero-weight-row rescue on GPU;
- penalized L2 inverse-Gamma no-intercept closure; and
- intercept-bearing smooth-L2 inverse-Gamma CV parity with exact selected-alpha
  identity on NumPy/CuPy/Torch.

The numerical tolerances below are frozen before the first schema-v4 physical
run.  A physical failure must not be made green by loosening them without a
reviewed schema change and a fresh exact-source artifact.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dev.benchmarks import validate_glm_weighted_explicit_solvers_gpu as v3
from statgpu.backends import _to_numpy
from statgpu.glm_core import get_glm_loss
from statgpu.linear_model import GammaRegression, PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


SCHEMA_VERSION = 4
DATA_SEED = 151004
N_SAMPLES = 128
N_FEATURES = 3
SOLVER_TOL = v3.SOLVER_TOL
ATOL_COEF = v3.ATOL_COEF
ATOL_INTERCEPT = v3.ATOL_INTERCEPT
ATOL_WEIGHT_RESCALE = v3.ATOL_WEIGHT_RESCALE
ALPHA_GRID = np.array([0.08, 0.03], dtype=np.float64)
_SOLVERS = ("newton", "lbfgs")


def _domain_data(seed=DATA_SEED, n=N_SAMPLES, p=N_FEATURES):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.06, size=(n, p)).astype(np.float64)
    X[:, 0] = rng.uniform(0.82, 1.18, size=n)
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = 0.92
    if p > 1:
        beta[1] = 0.05
    if p > 2:
        beta[2] = -0.035
    eta = X @ beta
    if not np.all(eta > 0):
        raise AssertionError("validator fixture must have a positive no-intercept predictor")
    y = (1.0 / eta) * rng.lognormal(0.0, 0.025, size=n)
    weights = np.linspace(0.55, 1.65, n, dtype=np.float64)
    weights[::17] = 0.0
    return X, y.astype(np.float64), weights


def _container_arrays(route, X, y, w, cp, torch, torch_device):
    return v3._container_arrays(route, X, y, w, cp, torch, torch_device)


def _snapshot_ordinary(model, X, weights):
    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64)
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    w_np = np.asarray(_to_numpy(weights), dtype=np.float64)
    eta = X_np @ coef + float(model.intercept_)
    active = w_np > 0
    return {
        "coef": coef,
        "intercept": float(model.intercept_),
        "n_iter": int(model.n_iter_),
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
        "eta_min_active": float(np.min(eta[active])),
        "eta_max_active": float(np.max(eta[active])),
    }


def _fit_no_intercept(solver, X, y, weights, *, device):
    with v3._solver_warning_gate():
        model = GammaRegression(
            link="inverse_power",
            fit_intercept=False,
            solver=solver,
            device=device,
            max_iter=1000,
            tol=SOLVER_TOL,
            compute_inference=False,
        ).fit(X, y, sample_weight=weights)
    return model, _snapshot_ordinary(model, X, weights)


def _assert_domain_snapshot(name, snap, X_ref):
    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(X_ref)
    if not snap["eta_min_active"] > lo:
        raise AssertionError(
            f"{name}: active eta minimum {snap['eta_min_active']:.6e} <= {lo:.6e}"
        )
    if not snap["eta_max_active"] < hi:
        raise AssertionError(
            f"{name}: active eta maximum {snap['eta_max_active']:.6e} >= {hi:.6e}"
        )


def _assert_ordinary_provenance(name, snap, *, solver, backend, device):
    expected = {
        "selected_solver": solver,
        "backend": backend,
        "device": device,
    }
    for key, value in expected.items():
        if snap.get(key) != value:
            raise AssertionError(
                f"{name}: {key}={snap.get(key)!r}, expected {value!r}"
            )


def _errors(ref, candidate):
    return {
        "coef": float(np.max(np.abs(ref["coef"] - candidate["coef"]))),
        "intercept": float(abs(ref["intercept"] - candidate["intercept"])),
    }


def _assert_parity(name, ref, candidate):
    errors = _errors(ref, candidate)
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


def _json_snap(snap):
    return {
        **{k: v for k, v in snap.items() if k != "coef"},
        "coef": np.asarray(snap["coef"], dtype=np.float64).tolist(),
    }


def _fit_penalized_no_intercept(solver, X, y, weights, *, device):
    with v3._solver_warning_gate():
        model = PenalizedGeneralizedLinearModel(
            loss="gamma",
            loss_kwargs={"link": "inverse_power"},
            penalty="l2",
            alpha=0.03,
            fit_intercept=False,
            solver=solver,
            device=device,
            max_iter=1000,
            tol=SOLVER_TOL,
            compute_inference=False,
        ).fit(X, y, sample_weight=weights)
    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64)
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    w_np = np.asarray(_to_numpy(weights), dtype=np.float64)
    eta = X_np @ coef
    active = w_np > 0
    return {
        "coef": coef,
        "intercept": float(model.intercept_),
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
        "eta_min_active": float(np.min(eta[active])),
        "eta_max_active": float(np.max(eta[active])),
    }


def _fit_inverse_gamma_cv(X, y, weights, *, device):
    with v3._solver_warning_gate():
        cv = PenalizedGLM_CV(
            loss="gamma",
            loss_kwargs={"link": "inverse_power"},
            penalty="l2",
            alpha_grid=ALPHA_GRID.copy(),
            cv=2,
            random_state=151,
            solver="auto",
            device=device,
            max_iter=700,
            tol=SOLVER_TOL,
        ).fit(X, y, sample_weight=weights)
    est = cv.estimator_
    return {
        "coef": np.asarray(_to_numpy(est.coef_), dtype=np.float64),
        "intercept": float(est.intercept_),
        "selected_alpha": float(cv.alpha_),
        "selected_solver": str(est._selected_solver),
        "backend": str(est._selected_backend_name),
        "device": str(est._selected_backend_device),
        "resolved_link": str(getattr(est._loss, "link", "")),
    }


def _negative_domain_pair(backend, X_factory, *, device):
    X_bad_np = np.array([[1.0], [-1.0]], dtype=np.float64)
    y_bad_np = np.array([1.0, 1.0], dtype=np.float64)
    w_bad_np = np.array([1.0, 1.0], dtype=np.float64)
    X_bad, y_bad, w_bad = X_factory(X_bad_np, y_bad_np, w_bad_np)

    failures = {}
    for solver in _SOLVERS:
        try:
            GammaRegression(
                link="inverse_power",
                fit_intercept=False,
                solver=solver,
                device=device,
                max_iter=100,
                tol=SOLVER_TOL,
                compute_inference=False,
            ).fit(X_bad, y_bad, sample_weight=w_bad)
        except RuntimeError as exc:
            text = str(exc)
            if "smooth-domain start" not in text:
                raise AssertionError(
                    f"{backend}/{solver}: wrong domain failure message: {text}"
                ) from exc
            failures[solver] = text
        else:
            raise AssertionError(
                f"{backend}/{solver}: active contradictory design unexpectedly fitted"
            )

    X_rescue_np = np.array([[1.0], [1.4], [-1.0]], dtype=np.float64)
    y_rescue_np = np.array([1.0, 0.8, 1.2], dtype=np.float64)
    w_rescue_np = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    X_rescue, y_rescue, w_rescue = X_factory(
        X_rescue_np, y_rescue_np, w_rescue_np
    )
    X_drop, y_drop, w_drop = X_factory(
        X_rescue_np[:2], y_rescue_np[:2], w_rescue_np[:2]
    )

    rescue = {}
    for solver in _SOLVERS:
        _, full = _fit_no_intercept(
            solver, X_rescue, y_rescue, w_rescue, device=device
        )
        _, dropped = _fit_no_intercept(
            solver, X_drop, y_drop, w_drop, device=device
        )
        errors = _assert_parity(
            f"{backend}/{solver}/zero_weight_rescue", dropped, full
        )
        rescue[solver] = {
            "full": _json_snap(full),
            "dropped": _json_snap(dropped),
            "errors_vs_row_deletion": errors,
        }
    return {"active_contradiction_failures": failures, "zero_weight_rescue": rescue}


def run(output: Path):
    # Run the complete historical validator unchanged first.  It performs the
    # clean-source gate and records the canonical environment/source identity.
    with tempfile.TemporaryDirectory(prefix="pr151-v4-") as tmpdir:
        v3_path = Path(tmpdir) / "schema_v3.json"
        v3.run(v3_path)
        legacy = json.loads(v3_path.read_text(encoding="utf-8"))

    if legacy.get("status") != "success" or not legacy.get("source_clean"):
        raise AssertionError("embedded schema-v3 acceptance did not succeed cleanly")

    source_sha = str(legacy["source_sha"])
    cp, torch, device_id, torch_device = v3._require_gpu_backends()
    expected_cuda = f"cuda:{device_id}"
    X_np, y_np, w_np = _domain_data()

    ordinary = {}
    for solver in _SOLVERS:
        _, ref = _fit_no_intercept(
            solver, X_np, y_np, w_np, device="cpu"
        )
        _assert_ordinary_provenance(
            f"{solver}/numpy", ref,
            solver=solver, backend="numpy", device="cpu",
        )
        _assert_domain_snapshot(f"{solver}/numpy", ref, X_np)
        ordinary[solver] = {"numpy": _json_snap(ref)}

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
                _, snap = _fit_no_intercept(
                    solver, Xb, yb, wb, device=device
                )
                _, scaled = _fit_no_intercept(
                    solver, Xb, yb, 7.0 * wb, device=device
                )
            _assert_ordinary_provenance(
                f"{solver}/{backend}", snap,
                solver=solver, backend=backend, device=expected_cuda,
            )
            _assert_domain_snapshot(f"{solver}/{backend}", snap, X_np)
            rescale = _errors(snap, scaled)
            if max(rescale.values()) > ATOL_WEIGHT_RESCALE:
                raise AssertionError(
                    f"{solver}/{backend}: weight rescaling drift {rescale}"
                )
            ordinary[solver][backend] = {
                **_json_snap(snap),
                "errors_vs_numpy": _assert_parity(
                    f"{solver}/{backend}", ref, snap
                ),
                "errors_weight_rescale": rescale,
            }

    # Heterogeneous containers prove the domain hooks follow the executed
    # explicit device/backend rather than the input array class.
    crossings = {}
    X_cp, y_cp, w_cp = _container_arrays(
        "cupy", X_np, y_np, w_np, cp, torch, torch_device
    )
    X_t, y_t, w_t = _container_arrays(
        "torch", X_np, y_np, w_np, cp, torch, torch_device
    )
    for solver in _SOLVERS:
        ref = ordinary[solver]["numpy"]
        ref_native = {
            **ref,
            "coef": np.asarray(ref["coef"], dtype=np.float64),
        }
        with cp.cuda.Device(device_id):
            _, t_to_c = _fit_no_intercept(
                solver, X_t, y_t, w_t, device="cuda"
            )
        _assert_ordinary_provenance(
            f"{solver}/torch_to_cupy", t_to_c,
            solver=solver, backend="cupy", device=expected_cuda,
        )
        with torch.cuda.device(torch_device):
            _, c_to_t = _fit_no_intercept(
                solver, X_cp, y_cp, w_cp, device="torch"
            )
        _assert_ordinary_provenance(
            f"{solver}/cupy_to_torch", c_to_t,
            solver=solver, backend="torch", device=expected_cuda,
        )
        crossings[solver] = {
            "torch_to_cupy": {
                **_json_snap(t_to_c),
                "errors_vs_numpy": _assert_parity(
                    f"{solver}/torch_to_cupy", ref_native, t_to_c
                ),
            },
            "cupy_to_torch": {
                **_json_snap(c_to_t),
                "errors_vs_numpy": _assert_parity(
                    f"{solver}/cupy_to_torch", ref_native, c_to_t
                ),
            },
        }

    penalized = {}
    for solver in _SOLVERS:
        ref = _fit_penalized_no_intercept(
            solver, X_np, y_np, w_np, device="cpu"
        )
        _assert_ordinary_provenance(
            f"penalized/{solver}/numpy", ref,
            solver=solver, backend="numpy", device="cpu",
        )
        _assert_domain_snapshot(f"penalized/{solver}/numpy", ref, X_np)
        penalized[solver] = {"numpy": _json_snap(ref)}
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
                snap = _fit_penalized_no_intercept(
                    solver, Xb, yb, wb, device=device
                )
            _assert_ordinary_provenance(
                f"penalized/{solver}/{backend}", snap,
                solver=solver, backend=backend, device=expected_cuda,
            )
            _assert_domain_snapshot(
                f"penalized/{solver}/{backend}", snap, X_np
            )
            penalized[solver][backend] = {
                **_json_snap(snap),
                "errors_vs_numpy": _assert_parity(
                    f"penalized/{solver}/{backend}", ref, snap
                ),
            }

    cv_ref = _fit_inverse_gamma_cv(X_np, y_np, w_np, device="cpu")
    if cv_ref["resolved_link"] != "inverse_power":
        raise AssertionError("numpy inverse-Gamma CV final refit lost its link")
    if cv_ref["selected_solver"] != "lbfgs":
        raise AssertionError(
            f"numpy inverse-Gamma CV selected {cv_ref['selected_solver']!r}, expected 'lbfgs'"
        )
    if cv_ref["backend"] != "numpy" or cv_ref["device"] != "cpu":
        raise AssertionError(f"numpy inverse-Gamma CV provenance invalid: {cv_ref}")

    cv_routes = {"numpy": _json_snap(cv_ref)}
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
            snap = _fit_inverse_gamma_cv(Xb, yb, wb, device=device)
        if snap["resolved_link"] != "inverse_power":
            raise AssertionError(f"{backend} inverse-Gamma CV lost its link")
        if snap["selected_alpha"] != cv_ref["selected_alpha"]:
            raise AssertionError(
                f"{backend} inverse-Gamma CV selected_alpha={snap['selected_alpha']}, "
                f"numpy={cv_ref['selected_alpha']}"
            )
        _assert_ordinary_provenance(
            f"cv/{backend}", snap,
            solver="lbfgs", backend=backend, device=expected_cuda,
        )
        cv_routes[backend] = {
            **_json_snap(snap),
            "errors_vs_numpy": _assert_parity(
                f"cv/{backend}", cv_ref, snap
            ),
            "selected_alpha_matches_numpy": True,
        }

    def cupy_factory(X, y, w):
        return cp.asarray(X), cp.asarray(y), cp.asarray(w)

    def torch_factory(X, y, w):
        return (
            torch.as_tensor(X, dtype=torch.float64, device=torch_device),
            torch.as_tensor(y, dtype=torch.float64, device=torch_device),
            torch.as_tensor(w, dtype=torch.float64, device=torch_device),
        )

    with cp.cuda.Device(device_id):
        negative_cupy = _negative_domain_pair(
            "cupy", cupy_factory, device="cuda"
        )
    with torch.cuda.device(torch_device):
        negative_torch = _negative_domain_pair(
            "torch", torch_factory, device="torch"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "source_sha": source_sha,
        "source_clean": True,
        "environment": legacy["environment"],
        "frozen_tolerances": {
            "coef_max_abs": ATOL_COEF,
            "intercept_abs": ATOL_INTERCEPT,
            "weight_rescale_max_abs": ATOL_WEIGHT_RESCALE,
            "solver_tol": SOLVER_TOL,
        },
        "legacy_schema_v3": legacy,
        "inverse_gamma_domain": {
            "data_contract": {
                "seed": DATA_SEED,
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "weight_pattern": "linspace(0.55,1.65), every 17th row zero",
            },
            "ordinary_no_intercept": ordinary,
            "cross_container": crossings,
            "penalized_l2_no_intercept": penalized,
            "smooth_l2_cv_intercept": cv_routes,
            "gpu_negative_domain": {
                "cupy": negative_cupy,
                "torch": negative_torch,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "v3_ordinary_route_count": len(v3._CASES) * len(v3._SOLVERS) * 3,
        "inverse_gamma_ordinary_route_count": len(_SOLVERS) * 3,
        "inverse_gamma_cross_container_count": len(_SOLVERS) * 2,
        "inverse_gamma_penalized_route_count": len(_SOLVERS) * 3,
        "inverse_gamma_cv_route_count": 3,
        "gpu_negative_backend_count": 2,
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/pr151_inverse_gamma_domain_gpu_v4.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
