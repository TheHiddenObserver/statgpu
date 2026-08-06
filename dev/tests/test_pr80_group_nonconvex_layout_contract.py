"""Group MCP/SCAD non-contiguous LLA and estimator contracts for PR #80."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import (
    GroupMCPPenalty,
    GroupSCADPenalty,
    get_penalty,
)


_INTERLEAVED = [[0, 3], [1, 2]]
_GROUPED = [[0, 1], [2, 3]]
_PERM = np.array([0, 3, 1, 2], dtype=np.int64)
_INVERSE_PERM = np.argsort(_PERM)


def _as_numpy(value):
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(value)
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def _sample(seed=9601):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    beta = np.array([0.9, -0.55, 0.25, 0.7])
    y = 0.35 + X @ beta + rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _penalty(kind, groups, alpha=0.18):
    if kind == "group_mcp":
        return GroupMCPPenalty(alpha=alpha, gamma=3.0, groups=groups)
    return GroupSCADPenalty(alpha=alpha, a=3.7, groups=groups)


def _expected_group_derivatives(kind, coef, alpha=0.18):
    derivatives = np.zeros_like(coef, dtype=np.float64)
    for group in _INTERLEAVED:
        idx = np.asarray(group, dtype=np.int64)
        norm = np.linalg.norm(coef[idx])
        alpha_g = alpha * np.sqrt(idx.size)
        if kind == "group_mcp":
            derivative = max(alpha_g - norm / 3.0, 0.0)
        elif norm <= alpha_g:
            derivative = alpha_g
        elif norm <= 3.7 * alpha_g:
            derivative = (3.7 * alpha_g - norm) / 2.7
        else:
            derivative = 0.0
        derivatives[idx] = derivative
    return derivatives


def _fit(kind, X, y, groups, *, device="cpu"):
    kwargs = {"groups": groups}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=kwargs,
        alpha=0.18,
        solver="auto",
        device=device,
        fit_intercept=True,
        compute_inference=False,
        max_iter=500,
        tol=1e-8,
        max_lla_iters=20,
        lla_tol=1e-8,
    ).fit(X, y)


def _unpermute_grouped_coef(coef):
    return np.asarray(coef)[_INVERSE_PERM]


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_public_group_nonconvex_registry_import_clone_and_pickle(kind):
    if kind == "group_mcp":
        from statgpu.penalties._group_mcp import GroupMCPPenalty as direct_class

        public_class = GroupMCPPenalty
        kwargs = {"gamma": 3.0}
    else:
        from statgpu.penalties._group_scad import GroupSCADPenalty as direct_class

        public_class = GroupSCADPenalty
        kwargs = {"a": 3.7}

    penalty = get_penalty(
        kind,
        alpha=0.18,
        groups=[[3, 0], [2, 1]],
        **kwargs,
    )
    cloned = clone(penalty)
    restored = pickle.loads(pickle.dumps(penalty))

    assert direct_class is public_class
    assert type(penalty) is public_class
    assert type(cloned) is public_class
    assert type(restored) is public_class
    assert penalty.groups == ((0, 3), (1, 2))
    assert penalty.get_params(deep=False)["groups"] is penalty.groups
    assert "n_groups" not in penalty.get_params(deep=False)
    np.testing.assert_array_equal(penalty._flat_indices, _PERM)
    np.testing.assert_array_equal(cloned._flat_indices, _PERM)
    np.testing.assert_array_equal(restored._flat_indices, _PERM)
    assert penalty._is_contiguous is False


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_noncontiguous_lla_weights_match_original_feature_coordinates(kind):
    coef = np.array([0.15, 0.9, -0.7, 0.05])
    penalty = _penalty(kind, _INTERLEAVED)

    actual = penalty.lla_weights(coef)
    expected = _expected_group_derivatives(kind, coef)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert actual[0] == pytest.approx(actual[3])
    assert actual[1] == pytest.approx(actual[2])
    assert not np.isclose(actual[0], actual[1])


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_noncontiguous_lla_weights_match_cpu_on_gpu(backend_name, kind):
    coef = np.array([0.15, 0.9, -0.7, 0.05])
    penalty = _penalty(kind, _INTERLEAVED)
    expected = penalty.lla_weights(coef)
    _, coef_backend, _ = _backend_inputs(backend_name, coef, coef)

    actual = penalty.lla_weights(coef_backend)

    np.testing.assert_allclose(
        _as_numpy(actual), expected, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_huber_group_nonconvex_interleaved_layout_matches_grouped_design(kind):
    X, y = _sample()
    interleaved = _fit(kind, X, y, _INTERLEAVED)
    grouped = _fit(kind, X[:, _PERM], y, _GROUPED)

    grouped_coef_original_order = _unpermute_grouped_coef(grouped.coef_)
    np.testing.assert_allclose(
        interleaved.coef_, grouped_coef_original_order, rtol=2e-6, atol=2e-7
    )
    assert interleaved.intercept_ == pytest.approx(
        grouped.intercept_, rel=2e-6, abs=2e-7
    )
    np.testing.assert_allclose(
        interleaved.predict(X),
        grouped.predict(X[:, _PERM]),
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_huber_group_nonconvex_cv_interleaved_layout_matches_grouped_design(kind):
    X, y = _sample(seed=9602)
    penalty_kwargs = {"groups": _INTERLEAVED}
    grouped_kwargs = {"groups": _GROUPED}
    if kind == "group_mcp":
        penalty_kwargs["gamma"] = grouped_kwargs["gamma"] = 3.0
    else:
        penalty_kwargs["a"] = grouped_kwargs["a"] = 3.7
    common = dict(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        alpha_grid=[0.24, 0.12],
        cv=2,
        random_state=17,
        device="cpu",
        max_iter=400,
        tol=1e-7,
    )

    interleaved = PenalizedGLM_CV(
        penalty_kwargs=penalty_kwargs, **common
    ).fit(X, y)
    grouped = PenalizedGLM_CV(
        penalty_kwargs=grouped_kwargs, **common
    ).fit(X[:, _PERM], y)

    np.testing.assert_allclose(
        interleaved.cv_results_["all_scores"],
        grouped.cv_results_["all_scores"],
        rtol=2e-5,
        atol=2e-7,
    )
    assert interleaved.alpha_ == pytest.approx(grouped.alpha_)
    assert interleaved.estimator_.alpha == pytest.approx(interleaved.alpha_)
    np.testing.assert_allclose(
        interleaved.coef_,
        _unpermute_grouped_coef(grouped.coef_),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_huber_group_nonconvex_gpu_interleaved_matches_cpu(backend_name, kind):
    X, y = _sample(seed=9603)
    reference = _fit(kind, X, y, _INTERLEAVED)
    device, Xb, yb = _backend_inputs(backend_name, X, y)
    actual = _fit(kind, Xb, yb, _INTERLEAVED, device=device)

    np.testing.assert_allclose(
        _as_numpy(actual.coef_), reference.coef_, rtol=3e-5, atol=3e-6
    )
    assert actual.intercept_ == pytest.approx(
        reference.intercept_, rel=3e-5, abs=3e-6
    )
    np.testing.assert_allclose(
        _as_numpy(actual.predict(Xb)),
        reference.predict(X),
        rtol=3e-5,
        atol=3e-6,
    )
