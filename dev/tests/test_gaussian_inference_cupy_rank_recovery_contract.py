"""Hosted locks for PR #129 CuPy rank-recovery semantics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import types

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._fit_mixin import _PenalizedFitMixin
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin


ROOT = Path(__file__).parents[2]


def test_cupy_solver_status_errors_are_scoped_at_maintained_rank_recovery_sites():
    inference_source = (
        ROOT / "statgpu" / "backends" / "_gpu_inference_cupy.py"
    ).read_text()
    linear_source = (
        ROOT / "statgpu" / "linear_model" / "wrappers" / "_linear.py"
    ).read_text()
    base_source = (
        ROOT / "statgpu" / "linear_model" / "penalized" / "_base.py"
    ).read_text()

    assert "import cupyx" in inference_source
    assert 'with cupyx.errstate(linalg="raise"):' in inference_source

    assert "import cupyx" in linear_source
    assert linear_source.count('with cupyx.errstate(linalg="raise"):') >= 3
    assert "cp.linalg.cholesky(XtX)" in linear_source
    assert "cp.linalg.lstsq(X_design, y, rcond=None)" in linear_source
    assert "cp.linalg.inv(XtX_cov)" in linear_source

    assert "def _solve_exact_cupy" in base_source
    assert "def _precompute_exact_l2_inference_cupy" in base_source
    assert base_source.count('with cupyx.errstate(linalg="raise"):') >= 2
    assert "_PenalizedFitMixin._solve_exact_cupy" in base_source
    assert "_PenalizedInferenceMixin._precompute_exact_l2_inference_cupy" in base_source


def test_penalized_exact_l2_overrides_enable_and_restore_cupy_errstate(monkeypatch):
    state = {"depth": 0, "entries": 0}

    @contextmanager
    def fake_errstate(**kwargs):
        assert kwargs == {"linalg": "raise"}
        state["depth"] += 1
        state["entries"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    monkeypatch.setitem(sys.modules, "cupyx", types.SimpleNamespace(errstate=fake_errstate))

    def fake_fit(self, XtX, Xty, normalization):
        assert state["depth"] == 1
        return (XtX, Xty, normalization)

    def fake_inference(self, *args, **kwargs):
        assert state["depth"] == 1
        return (args, kwargs)

    monkeypatch.setattr(_PenalizedFitMixin, "_solve_exact_cupy", fake_fit)
    monkeypatch.setattr(
        _PenalizedInferenceMixin,
        "_precompute_exact_l2_inference_cupy",
        fake_inference,
    )

    model = object.__new__(PenalizedGeneralizedLinearModel)
    assert model._solve_exact_cupy("XtX", "Xty", 8.0) == ("XtX", "Xty", 8.0)
    assert state["depth"] == 0
    args, kwargs = model._precompute_exact_l2_inference_cupy("X", "y", marker=True)
    assert args == ("X", "y")
    assert kwargs == {"marker": True}
    assert state == {"depth": 0, "entries": 2}


def test_focused_physical_runner_locks_all_defect3_recovery_paths():
    source = (
        ROOT
        / "dev"
        / "benchmarks"
        / "validate_gaussian_inference_cupy_rank_recovery_gpu.py"
    ).read_text()

    for required in (
        'parser.add_argument("--out", required=True)',
        'parser.add_argument("--expected-sha", required=True)',
        '"--validation-tier", required=True',
        "physical acceptance requires a clean working tree",
        "direct_inference_rank_recovery",
        '"nonrobust"',
        '"hc3"',
        "exact_l2_alpha0_rank_recovery",
        'alpha=0.0',
        'solver="exact"',
        'device="cuda"',
        '"working_tree_clean_after_checks"',
        'artifact["status"] = "failure"',
    ):
        assert required in source
