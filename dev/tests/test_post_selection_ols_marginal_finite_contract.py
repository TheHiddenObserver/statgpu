from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.linear_model import _debiased_marginal_finite_contract as finite_contract


def _result():
    return SimpleNamespace(
        params=np.asarray([0.4, -0.2, 0.1], dtype=np.float64),
        bse=np.asarray([0.1, 0.2, 0.15], dtype=np.float64),
        statistic=np.asarray([4.0, -1.0, np.inf], dtype=np.float64),
        pvalues=np.asarray([0.01, 0.3, 0.5], dtype=np.float64),
        conf_int=np.asarray(
            [[0.2, 0.6], [-0.6, 0.2], [-0.2, 0.4]],
            dtype=np.float64,
        ),
    )


def test_centered_marginal_guard_allows_signed_infinite_statistic():
    result = _result()
    finite_contract._validate_centered_marginal_result(result)
    assert np.isposinf(result.statistic[-1])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("params", np.inf, "non-finite parameter estimates"),
        ("bse", np.inf, "non-finite parameter estimates"),
        ("pvalues", np.nan, "non-finite parameter estimates"),
        ("conf_int", np.inf, "non-finite parameter estimates"),
    ],
)
def test_centered_marginal_guard_rejects_nonfinite_reporting_arrays(
    field,
    value,
    match,
):
    result = _result()
    array = getattr(result, field).copy()
    array.flat[0] = value
    setattr(result, field, array)

    with pytest.raises(FloatingPointError, match=match):
        finite_contract._validate_centered_marginal_result(result)


def test_centered_marginal_guard_rejects_invalid_bse_pvalue_and_ci_order():
    result = _result()
    result.bse[0] = -0.1
    with pytest.raises(FloatingPointError, match="negative standard error"):
        finite_contract._validate_centered_marginal_result(result)

    result = _result()
    result.pvalues[1] = 1.01
    with pytest.raises(FloatingPointError, match=r"outside \[0, 1\]"):
        finite_contract._validate_centered_marginal_result(result)

    result = _result()
    result.conf_int[2] = [0.5, -0.5]
    with pytest.raises(FloatingPointError, match="reversed confidence interval"):
        finite_contract._validate_centered_marginal_result(result)


def test_centered_marginal_finalizer_guard_is_scoped_to_intercept_path(monkeypatch):
    invalid = _result()
    invalid.params[0] = np.inf
    calls = []

    def fake_finalizer(model, **kwargs):
        calls.append(bool(kwargs["original_intercept"]))
        return invalid

    monkeypatch.setattr(finite_contract, "_ORIGINAL_FINALIZER", fake_finalizer)

    common = dict(
        model=object(),
        X_arr=object(),
        y_work=object(),
        X_work=object(),
        row_scale=object(),
        coef_native=object(),
        backend_name="numpy",
        simultaneous_requested=False,
        sample_weighted=False,
    )

    # Historical no-intercept behavior remains outside this PR's new guard.
    output = finite_contract._finalize_weighted_debiased_result(
        **common,
        original_intercept=False,
    )
    assert output is invalid

    with pytest.raises(FloatingPointError, match="non-finite parameter estimates"):
        finite_contract._finalize_weighted_debiased_result(
            **common,
            original_intercept=True,
        )

    assert calls == [False, True]
