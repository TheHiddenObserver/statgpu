#!/usr/bin/env python3
"""PR151 physical CUDA schema v5: final review-closure gate.

Schema v5 preserves the accepted historical schema-v4 validator unchanged and
runs it first on the same exact clean source. It then adds the numerical edges
found by the final ``code-review`` / fix loop:

- analytic-weight classification is invariant to extreme positive global
  rescaling on CuPy and Torch for Newton and L-BFGS;
- integral public designs with fractional analytic weights execute the same
  ordinary binomial objective as their float64 design counterpart and retain
  floating post-fit diagnostic/inference state; and
- inverse-power Gamma fails closed on CuPy/Torch when the mathematical optimum
  lies beyond the maintained smooth-training domain instead of publishing a
  tiny domain-capped step as convergence.

A successful v5 artifact is the required final-source physical CUDA evidence
for PR151. Earlier schema-v3/v4 artifacts remain historical exact-source
records and are intentionally not rewritten.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dev.benchmarks import validate_pr151_inverse_gamma_domain_gpu_v4 as v4
from statgpu.backends import _to_numpy
from statgpu.linear_model import GammaRegression, GeneralizedLinearModel


SCHEMA_VERSION = 5
SOLVER_TOL = v4.SOLVER_TOL
ATOL_COEF = v4.ATOL_COEF
ATOL_INTERCEPT = v4.ATOL_INTERCEPT
ATOL_WEIGHT_RESCALE = v4.ATOL_WEIGHT_RESCALE
_SOLVERS = v4._SOLVERS
_EXTREME_WEIGHT_SCALES = (1.0e-200, 1.0e200)


def _ordinary_snapshot(model):
    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64)
    params = np.asarray(_to_numpy(model._params), dtype=np.float64)
    design = np.asarray(_to_numpy(model._X_design))
    return {
        "coef": coef,
        "intercept": float(model.intercept_),
        "selected_solver": str(model._selected_solver),
        "backend": str(model._selected_backend_name),
        "device": str(model._selected_backend_device),
        "postfit_design_is_floating": bool(
            np.issubdtype(design.dtype, np.floating)
        ),
        "postfit_params_error_vs_coef": float(np.max(np.abs(params - coef))),
        "loglikelihood": float(model.loglikelihood),
    }


def _fit_binomial_no_intercept(solver, X, y, weights, *, device):
    with v4.v3._solver_warning_gate():
        model = GeneralizedLinearModel(
            family="binomial",
            fit_intercept=False,
            solver=solver,
            device=device,
            max_iter=1000,
            tol=SOLVER_TOL,
            compute_inference=False,
        ).fit(X, y, sample_weight=weights)
    return _ordinary_snapshot(model)


def _integer_design_data(seed=151005, n=128, p=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(-2, 3, size=(n, p), dtype=np.int64)
    beta = np.array([0.34, -0.22, 0.13], dtype=np.float64)[:p]
    prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    weights = np.linspace(0.35, 1.75, n, dtype=np.float64)
    return X, y, weights


def _assert_provenance(name, snap, *, solver, backend, device):
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


def _json_snap(snap):
    return {
        **{key: value for key, value in snap.items() if key != "coef"},
        "coef": np.asarray(snap["coef"], dtype=np.float64).tolist(),
    }


def _weight_scale_gate(cp, torch, device_id, torch_device):
    X_np, y_np, w_np = v4._domain_data(seed=151006)
    expected_cuda = f"cuda:{device_id}"
    results = {}

    for solver in _SOLVERS:
        _, ref = v4._fit_no_intercept(
            solver, X_np, y_np, w_np, device="cpu"
        )
        results[solver] = {}
        for backend, route, device in (
            ("cupy", "cupy", "cuda"),
            ("torch", "torch", "torch"),
        ):
            Xb, yb, wb = v4._container_arrays(
                route, X_np, y_np, w_np, cp, torch, torch_device
            )
            context = (
                cp.cuda.Device(device_id)
                if backend == "cupy"
                else torch.cuda.device(torch_device)
            )
            with context:
                _, base = v4._fit_no_intercept(
                    solver, Xb, yb, wb, device=device
                )
                scaled = {}
                for scale in _EXTREME_WEIGHT_SCALES:
                    _, scaled_snap = v4._fit_no_intercept(
                        solver, Xb, yb, scale * wb, device=device
                    )
                    scaled[str(scale)] = scaled_snap

            _assert_provenance(
                f"weight_scale/{solver}/{backend}",
                base,
                solver=solver,
                backend=backend,
                device=expected_cuda,
            )
            v4._assert_parity(
                f"weight_scale/{solver}/{backend}/base", ref, base
            )

            scale_payload = {}
            for scale_text, snap in scaled.items():
                errors = v4._errors(base, snap)
                if max(errors.values()) > ATOL_WEIGHT_RESCALE:
                    raise AssertionError(
                        f"weight_scale/{solver}/{backend}/{scale_text}: "
                        f"global-rescaling drift {errors}"
                    )
                scale_payload[scale_text] = {
                    **v4._json_snap(snap),
                    "errors_vs_base": errors,
                }
            results[solver][backend] = {
                "base": v4._json_snap(base),
                "extreme_scales": scale_payload,
            }
    return results


def _integer_design_gate(cp, torch, device_id, torch_device):
    X_int, y_np, w_np = _integer_design_data()
    X_float = X_int.astype(np.float64)
    expected_cuda = f"cuda:{device_id}"
    results = {}

    for solver in _SOLVERS:
        ref = _fit_binomial_no_intercept(
            solver, X_float, y_np, w_np, device="cpu"
        )
        results[solver] = {"numpy_float64": _json_snap(ref)}

        for backend, device in (("cupy", "cuda"), ("torch", "torch")):
            if backend == "cupy":
                Xb = cp.asarray(X_int)
                yb = cp.asarray(y_np)
                wb = cp.asarray(w_np)
                context = cp.cuda.Device(device_id)
            else:
                Xb = torch.as_tensor(X_int, dtype=torch.int64, device=torch_device)
                yb = torch.as_tensor(y_np, dtype=torch.float64, device=torch_device)
                wb = torch.as_tensor(w_np, dtype=torch.float64, device=torch_device)
                context = torch.cuda.device(torch_device)

            with context:
                snap = _fit_binomial_no_intercept(
                    solver, Xb, yb, wb, device=device
                )
            _assert_provenance(
                f"integer_design/{solver}/{backend}",
                snap,
                solver=solver,
                backend=backend,
                device=expected_cuda,
            )
            errors = v4._errors(ref, snap)
            if errors["coef"] > ATOL_COEF or errors["intercept"] > ATOL_INTERCEPT:
                raise AssertionError(
                    f"integer_design/{solver}/{backend}: parity error {errors}"
                )
            if not snap["postfit_design_is_floating"]:
                raise AssertionError(
                    f"integer_design/{solver}/{backend}: retained inference "
                    "design was not floating"
                )
            if snap["postfit_params_error_vs_coef"] > 1.0e-12:
                raise AssertionError(
                    f"integer_design/{solver}/{backend}: retained post-fit "
                    f"parameters were truncated ({snap['postfit_params_error_vs_coef']})"
                )
            if not np.isfinite(snap["loglikelihood"]):
                raise AssertionError(
                    f"integer_design/{solver}/{backend}: non-finite loglikelihood"
                )
            results[solver][backend] = {
                **_json_snap(snap),
                "errors_vs_numpy_float64": errors,
            }
    return results


def _domain_pinned_gate(cp, torch, device_id, torch_device):
    X_np = np.ones((8, 1), dtype=np.float64)
    y_np = np.full(8, 1.0e-8, dtype=np.float64)
    w_np = np.linspace(0.5, 1.5, 8, dtype=np.float64)
    results = {}

    for backend, device in (("cupy", "cuda"), ("torch", "torch")):
        if backend == "cupy":
            Xb = cp.asarray(X_np)
            yb = cp.asarray(y_np)
            wb = cp.asarray(w_np)
            context = cp.cuda.Device(device_id)
        else:
            Xb = torch.as_tensor(X_np, dtype=torch.float64, device=torch_device)
            yb = torch.as_tensor(y_np, dtype=torch.float64, device=torch_device)
            wb = torch.as_tensor(w_np, dtype=torch.float64, device=torch_device)
            context = torch.cuda.device(torch_device)

        backend_failures = {}
        for solver in _SOLVERS:
            with context:
                try:
                    GammaRegression(
                        link="inverse_power",
                        fit_intercept=False,
                        solver=solver,
                        device=device,
                        max_iter=100,
                        tol=1.0e-8,
                        compute_inference=False,
                    ).fit(Xb, yb, sample_weight=wb)
                except RuntimeError as exc:
                    text = str(exc)
                    if not (
                        "pinned to the maintained smooth-domain boundary" in text
                        or "no positive interior line-search step" in text
                    ):
                        raise AssertionError(
                            f"domain_pinned/{backend}/{solver}: wrong failure: {text}"
                        ) from exc
                    backend_failures[solver] = text
                else:
                    raise AssertionError(
                        f"domain_pinned/{backend}/{solver}: boundary surrogate "
                        "was published instead of failing closed"
                    )
        results[backend] = backend_failures
    return results


def run(output: Path):
    # Preserve v4 byte-for-byte as historical contract and execute it first on
    # this exact source. Its internal v3 run performs the clean-source gate.
    with tempfile.TemporaryDirectory(prefix="pr151-v5-") as tmpdir:
        v4_path = Path(tmpdir) / "schema_v4.json"
        v4.run(v4_path)
        legacy = json.loads(v4_path.read_text(encoding="utf-8"))

    if legacy.get("status") != "success" or not legacy.get("source_clean"):
        raise AssertionError("embedded schema-v4 acceptance did not succeed cleanly")

    source_sha = str(legacy["source_sha"])
    cp, torch, device_id, torch_device = v4.v3._require_gpu_backends()

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
            "extreme_weight_scales": list(_EXTREME_WEIGHT_SCALES),
        },
        "legacy_schema_v4": legacy,
        "review_closure": {
            "extreme_global_weight_rescaling": _weight_scale_gate(
                cp, torch, device_id, torch_device
            ),
            "integer_design_fractional_weights": _integer_design_gate(
                cp, torch, device_id, torch_device
            ),
            "gpu_domain_pinned_fail_closed": _domain_pinned_gate(
                cp, torch, device_id, torch_device
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "extreme_weight_backend_solver_rows": len(_SOLVERS) * 2,
        "integer_design_backend_solver_rows": len(_SOLVERS) * 2,
        "domain_pinned_backend_solver_rows": len(_SOLVERS) * 2,
        "output": str(output),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/reviews/pr151_final_gpu_v5.json"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
