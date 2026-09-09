from types import SimpleNamespace

import numpy as np
import pytest

import statgpu.linear_model._debiased_simultaneous_reporting_finite_contract as finite_contract


def test_cpu_feature_only_simultaneous_rejects_nonfinite_publication(monkeypatch):
    n = 6
    p = 2
    B = 32
    X_feat = np.ones((n, p), dtype=np.float64)
    model = SimpleNamespace(
        simultaneous_include_intercept=False,
        _effective_intercept=True,
        _debiased_M_cpu=np.eye(p, dtype=np.float64),
        _y=np.zeros(n, dtype=np.float64),
        _resid=np.ones(n, dtype=np.float64),
        _bse=np.ones(p + 1, dtype=np.float64),
        _params=np.zeros(p + 1, dtype=np.float64),
        _conf_int=np.zeros((p + 1, 2), dtype=np.float64),
        _X_design=np.column_stack([np.ones(n, dtype=np.float64), X_feat]),
        _nobs=n,
        _simultaneous_enabled=False,
        simultaneous_alpha=0.05,
        simultaneous_n_bootstrap=B,
        simultaneous_random_state=20260909,
    )
    xi = np.zeros((B, n), dtype=np.float64)
    xi[0, :] = np.finfo(np.float64).max
    assert np.all(np.isfinite(xi))

    class _ExtremeFiniteRng:
        def standard_normal(self, size):
            assert tuple(size) == xi.shape
            return xi

    monkeypatch.setattr(
        finite_contract.np.random,
        "default_rng",
        lambda seed: _ExtremeFiniteRng(),
    )

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(
            FloatingPointError,
            match="non-finite or negative critical value",
        ):
            finite_contract._compute_simultaneous_ci_maxz_bootstrap(model)

    assert model._simultaneous_enabled is False
    assert model._conf_int_simultaneous is None
    assert not hasattr(model, "_simultaneous_critical_value")


def test_simultaneous_publication_guard_accepts_finite_feature_only_state():
    model = SimpleNamespace(
        _simultaneous_enabled=True,
        _simultaneous_critical_value=2.25,
        _conf_int_simultaneous=np.asarray(
            [[-1.0, 1.0], [-0.5, 0.75]],
            dtype=np.float64,
        ),
    )

    finite_contract._validate_simultaneous_publication(model)

    assert model._simultaneous_enabled is True
    assert model._simultaneous_critical_value == 2.25


@pytest.mark.parametrize(
    "conf_int",
    [
        np.asarray([[np.nan, 1.0]], dtype=np.float64),
        np.asarray([[2.0, 1.0]], dtype=np.float64),
        np.asarray([1.0, 2.0], dtype=np.float64),
    ],
)
def test_simultaneous_publication_guard_rejects_invalid_joint_intervals(conf_int):
    model = SimpleNamespace(
        _simultaneous_enabled=True,
        _simultaneous_critical_value=2.0,
        _conf_int_simultaneous=conf_int,
    )

    with pytest.raises(FloatingPointError, match="confidence intervals"):
        finite_contract._validate_simultaneous_publication(model)

    assert model._simultaneous_enabled is False
    assert model._conf_int_simultaneous is None
    assert not hasattr(model, "_simultaneous_critical_value")
