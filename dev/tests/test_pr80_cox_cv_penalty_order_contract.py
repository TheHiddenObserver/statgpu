"""Independent contracts for Cox CV custom regularization grids."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedCoxPHModel
from statgpu.linear_model.penalized import _penalized_cox_cv as penalized_cv
from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv
from statgpu.survival import _cox_cv_penalty_order_contract as order_contract


def _survival_data():
    rng = np.random.default_rng(12001)
    X = rng.normal(size=(24, 2))
    time = np.linspace(1.0, 24.0, 24)
    event = np.tile(np.array([1.0, 0.0, 1.0]), 8)
    return X, time, event


def test_custom_penalty_grid_runs_by_rank_and_restores_public_order(monkeypatch):
    supplied = np.array([0.1, 1.0, 0.5])
    sorted_grid = np.array([1.0, 0.5, 0.1])
    sorted_mean = np.array([5.0, 5.0 + 1e-11, 4.0])
    sorted_matrix = np.arange(6, dtype=np.float64).reshape(3, 2)
    calls = []

    def fake_selector(*args, **kwargs):
        calls.append(np.asarray(kwargs["penalties"]).copy())
        np.testing.assert_array_equal(kwargs["penalties"], sorted_grid)
        details = {
            "penalty": 0.5,
            "penalties": sorted_grid.copy(),
            "pl_path": sorted_matrix.copy(),
            "mean_pl": sorted_mean.copy(),
            "best_pl": float(sorted_mean[1]),
            "converged_path": np.ones((3, 2), dtype=bool),
            "convergence": np.ones((3, 2), dtype=bool),
            "attempted_path": np.ones((3, 2), dtype=bool),
            "iterations_path": np.arange(6).reshape(3, 2),
            "failure_path": np.full((3, 2), None, dtype=object),
            "effective_fold_counts": np.array([2, 2, 2]),
            "candidate_complete": np.array([True, True, True]),
        }
        return 0.5, details

    monkeypatch.setattr(
        order_contract,
        "_ORIGINAL_SELECT_COXPH_PENALTY_CV",
        fake_selector,
    )
    best, details = cox_cv._select_coxph_penalty_cv(
        np.zeros((4, 1)),
        np.arange(1.0, 5.0),
        np.array([1.0, 0.0, 1.0, 0.0]),
        penalties=supplied,
        return_details=True,
    )

    assert len(calls) == 1
    assert best == pytest.approx(1.0)
    assert details["penalty"] == pytest.approx(1.0)
    np.testing.assert_array_equal(details["penalties"], supplied)
    np.testing.assert_array_equal(
        details["penalty_evaluation_order"], sorted_grid
    )
    assert details["penalty_input_order_preserved"] is True
    np.testing.assert_array_equal(details["mean_pl"], [4.0, 5.0, 5.0 + 1e-11])
    np.testing.assert_array_equal(
        details["pl_path"], sorted_matrix[[2, 0, 1]]
    )
    np.testing.assert_array_equal(
        details["iterations_path"], np.arange(6).reshape(3, 2)[[2, 0, 1]]
    )


def test_descending_penalty_order_is_stable_for_duplicate_strengths():
    grid = np.array([0.1, 0.5, 0.5, 1.0])
    order = order_contract._descending_penalty_order(grid)
    np.testing.assert_array_equal(order, [3, 1, 2, 0])


@pytest.mark.parametrize(
    "grid, message",
    [
        ([True, 0.1], "booleans"),
        (["0.2", "0.1"], "strings or bytes"),
        (np.array([0.2 + 0.0j, 0.1 + 0.0j]), "real numeric"),
    ],
)
def test_coxphcv_rejects_lossy_grid_before_numerical_work(
    monkeypatch, grid, message
):
    X, time, event = _survival_data()
    work_started = False

    def forbidden(self, *args, **kwargs):
        nonlocal work_started
        work_started = True
        raise AssertionError("CV numerical work must not start")

    monkeypatch.setattr(
        order_contract,
        "_ORIGINAL_COXPHCV_FIT_CV",
        forbidden,
    )
    model = CoxPHCV(
        penalties=grid,
        cv=2,
        compute_inference=False,
        device="cpu",
    )
    with pytest.raises(ValueError, match=message):
        model.fit(X, time, event)
    assert work_started is False
    assert model.penalties is grid
    assert model.penalty_ is None
    assert model.estimator_ is None


@pytest.mark.parametrize(
    "grid, message",
    [
        ([True, 0.1], "booleans"),
        (["0.2", "0.1"], "strings or bytes"),
        (np.array([0.2 + 0.0j, 0.1 + 0.0j]), "real numeric"),
    ],
)
def test_penalized_cox_cv_rejects_lossy_alpha_grid(grid, message):
    with pytest.raises(ValueError, match=message):
        penalized_cv._validate_alpha_grid(grid, "l1")


def test_penalized_cox_public_class_has_real_docstring():
    documentation = inspect.getdoc(PenalizedCoxPHModel)
    assert documentation is not None
    assert "Penalized Cox proportional hazards model" in documentation
    assert "estimation-only" in documentation


def test_penalized_glm_cv_dispatch_sees_strict_cox_alpha_validator():
    # Importing the public CV class must install the Cox-specific validator;
    # this guards against direct-module import order regressions.
    assert PenalizedGLM_CV is not None
    assert penalized_cv._validate_alpha_grid.__name__ == "_validate_alpha_grid"


def test_physical_gpu_runner_manifest_covers_order_contract_sources():
    runner = Path("dev/benchmarks/benchmark_cox_cv_penalty_order_gpu.py")
    tree = ast.parse(runner.read_text())
    source_files = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SOURCE_FILES"
                for target in node.targets
            )
        ):
            source_files = ast.literal_eval(node.value)
            break

    assert source_files is not None
    required = {
        runner.as_posix(),
        "dev/tests/test_pr80_cox_cv_penalty_order_contract.py",
        "statgpu/cross_validation/_grid_validation.py",
        "statgpu/survival/__init__.py",
        "statgpu/survival/_cox_cv.py",
        "statgpu/survival/_cox_cv_penalty_order_contract.py",
        "statgpu/survival/_risk_sets.py",
    }
    assert required.issubset(set(source_files))
    assert all(Path(path).is_file() for path in source_files)
