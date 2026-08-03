"""Regression gates for public Group Lasso layout semantics in PR #80."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import GroupLassoPenalty, get_penalty


def _backend_inputs(backend_name, X, y):
    if backend_name == "numpy":
        return "cpu", X, y
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception:
            pytest.skip("CuPy CUDA runtime unavailable")
        return "cuda", cp.asarray(X), cp.asarray(y)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return (
        "torch",
        torch.as_tensor(X, dtype=torch.float64, device="cuda"),
        torch.as_tensor(y, dtype=torch.float64, device="cuda"),
    )


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample(seed, p):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, p))
    beta = np.linspace(0.8, -0.35, p)
    y = 0.4 + X @ beta + rng.normal(scale=0.04, size=X.shape[0])
    return X, y


def _fit_group_lasso(X, y, groups, *, device="cpu", fit_intercept=True):
    return PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        alpha=0.035,
        penalty_kwargs={"groups": groups},
        solver="auto",
        device=device,
        fit_intercept=fit_intercept,
        compute_inference=False,
        max_iter=3000,
        tol=1e-10,
    ).fit(X, y)


def _objective(model, X, y, groups, alpha=0.035):
    coef = np.asarray(model.coef_, dtype=np.float64)
    pred = np.asarray(model.predict(X), dtype=np.float64)
    loss = 0.5 * float(np.mean((np.asarray(y) - pred) ** 2))
    penalty = sum(
        np.sqrt(len(group)) * np.linalg.norm(coef[np.asarray(group, dtype=int)])
        for group in groups
    )
    return loss + alpha * float(penalty)


def test_public_group_lasso_canonicalizes_within_group_order_and_preserves_identity():
    from statgpu.penalties._group_lasso import GroupLassoPenalty as direct_class

    raw_groups = [[0, 3], [2, 1]]
    assert all(raw_groups[g][0] == g * 2 for g in range(2))

    penalty = get_penalty("group_lasso", alpha=0.1, groups=raw_groups)

    assert direct_class is GroupLassoPenalty
    assert type(penalty) is GroupLassoPenalty
    np.testing.assert_array_equal(penalty._group_indices[0], np.array([0, 3]))
    np.testing.assert_array_equal(penalty._group_indices[1], np.array([1, 2]))
    np.testing.assert_array_equal(penalty._flat_indices, np.array([0, 3, 1, 2]))
    assert penalty._all_equal_size is True
    assert penalty._is_contiguous is False


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_group_lasso_within_group_permutation_is_direct_fit_invariant(fit_intercept):
    X, y = _sample(seed=9201, p=4)
    canonical = [[0, 3], [1, 2]]
    permuted = [[3, 0], [2, 1]]

    reference = _fit_group_lasso(
        X, y, canonical, fit_intercept=fit_intercept
    )
    actual = _fit_group_lasso(
        X, y, permuted, fit_intercept=fit_intercept
    )

    np.testing.assert_allclose(actual.coef_, reference.coef_, rtol=0.0, atol=0.0)
    assert actual.intercept_ == pytest.approx(reference.intercept_, rel=0.0, abs=0.0)
    np.testing.assert_allclose(actual.predict(X), reference.predict(X), rtol=0.0, atol=0.0)


def test_group_lasso_within_group_permutation_is_cv_and_refit_invariant():
    X, y = _sample(seed=9202, p=4)
    kwargs = dict(
        loss="squared_error",
        penalty="group_lasso",
        alpha_grid=[0.2, 0.05, 0.01],
        cv=3,
        random_state=19,
        device="cpu",
        max_iter=2000,
        tol=1e-9,
    )

    reference = PenalizedGLM_CV(
        penalty_kwargs={"groups": [[0, 3], [1, 2]]},
        **kwargs,
    ).fit(X, y)
    actual = PenalizedGLM_CV(
        penalty_kwargs={"groups": [[3, 0], [2, 1]]},
        **kwargs,
    ).fit(X, y)

    np.testing.assert_array_equal(actual.alpha_grid_, reference.alpha_grid_)
    np.testing.assert_allclose(
        actual.cv_results_["all_scores"],
        reference.cv_results_["all_scores"],
        rtol=0.0,
        atol=0.0,
    )
    assert actual.alpha_ == pytest.approx(reference.alpha_, rel=0.0, abs=0.0)
    np.testing.assert_allclose(actual.coef_, reference.coef_, rtol=0.0, atol=0.0)
    assert actual.estimator_.alpha == pytest.approx(actual.alpha_)


_LAYOUTS = [
    pytest.param([[0, 2], [1, 3]], id="equal-noncontiguous"),
    pytest.param([[3, 0], [2, 1]], id="misleading-first-index"),
    pytest.param([[0, 3, 4], [1, 2]], id="unequal-serial"),
]


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize("groups", _LAYOUTS)
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_group_lasso_gpu_layouts_match_cpu_objective_and_coefficients(
    backend_name, groups, fit_intercept
):
    p = max(max(group) for group in groups) + 1
    X, y = _sample(seed=9203 + p, p=p)
    reference = _fit_group_lasso(
        X, y, groups, device="cpu", fit_intercept=fit_intercept
    )
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    actual = _fit_group_lasso(
        Xb, yb, groups, device=device, fit_intercept=fit_intercept
    )

    actual_coef = _as_numpy(actual.coef_)
    np.testing.assert_allclose(actual_coef, reference.coef_, rtol=2e-5, atol=2e-6)
    assert actual.intercept_ == pytest.approx(
        reference.intercept_, rel=2e-5, abs=2e-6
    )
    np.testing.assert_allclose(
        _as_numpy(actual.predict(Xb)),
        reference.predict(X),
        rtol=2e-5,
        atol=2e-6,
    )
    assert _objective(actual, X, y, groups) == pytest.approx(
        _objective(reference, X, y, groups), rel=2e-6, abs=2e-7
    )


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize(
    "groups",
    [
        pytest.param([[3, 0], [2, 1]], id="misleading-first-index"),
        pytest.param([[0, 3, 4], [1, 2]], id="unequal-serial"),
    ],
)
def test_group_lasso_gpu_cv_selection_and_refit_match_cpu(backend_name, groups):
    p = max(max(group) for group in groups) + 1
    X, y = _sample(seed=9210 + p, p=p)
    kwargs = dict(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": groups},
        alpha_grid=[0.12, 0.035],
        cv=2,
        random_state=23,
        max_iter=2500,
        tol=1e-9,
    )
    reference = PenalizedGLM_CV(device="cpu", **kwargs).fit(X, y)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    actual = PenalizedGLM_CV(device=device, **kwargs).fit(Xb, yb)

    np.testing.assert_allclose(
        actual.cv_results_["all_scores"],
        reference.cv_results_["all_scores"],
        rtol=2e-5,
        atol=2e-7,
    )
    assert actual.alpha_ == pytest.approx(reference.alpha_)
    assert actual.estimator_.alpha == pytest.approx(actual.alpha_)
    np.testing.assert_allclose(
        _as_numpy(actual.coef_), reference.coef_, rtol=2e-5, atol=2e-6
    )
