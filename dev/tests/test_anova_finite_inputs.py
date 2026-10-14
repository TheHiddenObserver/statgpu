"""Finite-input validation and valid-data controls for every public ANOVA path."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import statgpu
from statgpu.anova import (
    bonferroni,
    cohens_f,
    f_oneway,
    f_twoway,
    f_welch,
    partial_eta_squared,
    tukey_hsd,
)


@pytest.fixture(params=["numpy", "torch_cpu", "torch_cuda", "cupy"])
def native_backend(request):
    """Use real arrays/devices; unavailable optional runtimes skip explicitly."""
    name = request.param
    if name == "numpy":
        xp, backend, device = np, "numpy", None
    elif name.startswith("torch"):
        xp = pytest.importorskip("torch")
        backend = "torch"
        device = "cuda:0" if name == "torch_cuda" else "cpu"
        if name == "torch_cuda" and not xp.cuda.is_available():
            pytest.skip("Torch CUDA is unavailable; physical GPU evidence required")
    else:
        xp = pytest.importorskip("cupy")
        backend, device = "cupy", None
        try:
            count = xp.cuda.runtime.getDeviceCount()
        except xp.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CuPy CUDA runtime is unavailable: {exc}")
        if count == 0:
            pytest.skip("CuPy has no CUDA device; physical GPU evidence required")

    def array(values, dtype="float64"):
        kwargs = {"dtype": getattr(xp, dtype)}
        if backend == "torch":
            kwargs["device"] = device
        return xp.asarray(values, **kwargs)

    return SimpleNamespace(xp=xp, backend=backend, array=array, device=device)


GROUP_FUNCTIONS = [f_oneway, cohens_f, f_welch, tukey_hsd, bonferroni]
NONFINITE = [pytest.param(np.nan, id="nan"), pytest.param(np.inf, id="posinf"),
             pytest.param(-np.inf, id="neginf")]


@pytest.mark.parametrize("function", GROUP_FUNCTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("bad", NONFINITE)
@pytest.mark.parametrize("bad_group", [0, 1])
@pytest.mark.parametrize("selection", ["explicit", "auto"])
def test_group_functions_reject_nonfinite(native_backend, function, bad, bad_group, selection):
    groups = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    groups[bad_group][-1] = bad
    arrays = [native_backend.array(group) for group in groups]
    backend = native_backend.backend if selection == "explicit" else "auto"
    with pytest.raises(ValueError, match="NaN or infinite"):
        function(*arrays, backend=backend)


@pytest.mark.parametrize("function", [f_oneway, cohens_f], ids=lambda f: f.__name__)
@pytest.mark.parametrize("bad", NONFINITE)
def test_python_lists_reject_nonfinite(function, bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        function([1.0, 2.0, bad], [2.0, 3.0, 4.0], backend="numpy")


@pytest.mark.parametrize("bad", NONFINITE)
@pytest.mark.parametrize("interaction", [True, False])
@pytest.mark.parametrize("bad_cell", [(0, 0), (1, 1)])
def test_twoway_rejects_nonfinite(native_backend, bad, interaction, bad_cell):
    cells = [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
             [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]]
    cells[bad_cell[0]][bad_cell[1]][-1] = bad
    arrays = [[native_backend.array(cell) for cell in row] for row in cells]
    with pytest.raises(ValueError, match="NaN or infinite"):
        f_twoway(arrays, interaction=interaction, backend=native_backend.backend)


@pytest.mark.parametrize("bad", NONFINITE)
@pytest.mark.parametrize("argument", [0, 1])
def test_partial_eta_squared_rejects_nonfinite(bad, argument):
    values = [1.0, 2.0]
    values[argument] = bad
    with pytest.raises(ValueError, match="finite"):
        partial_eta_squared(*values)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_oneway_valid_unequal_sizes_and_effect_size(native_backend, dtype):
    groups = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0, 7.0], [-1.0, 0.0]]
    arrays = [native_backend.array(group, dtype) for group in groups]
    kwargs = {"backend": native_backend.backend, "dtype": getattr(native_backend.xp, dtype)}
    actual = f_oneway(*arrays, **kwargs)
    expected = stats.f_oneway(*groups)
    assert_allclose(actual.statistic, expected.statistic, rtol=2e-6)
    assert_allclose(actual.pvalue, expected.pvalue, rtol=5e-5, atol=1e-9)
    assert actual.df_between == 2
    assert actual.df_within == 6
    expected_eta = actual.statistic * 2 / (actual.statistic * 2 + 6)
    assert_allclose(actual.eta_squared, expected_eta, rtol=2e-6)
    assert_allclose(cohens_f(*arrays, **kwargs), np.sqrt(expected_eta / (1 - expected_eta)), rtol=2e-6)
    for field in (actual.statistic, actual.pvalue, actual.eta_squared):
        assert isinstance(field, float)


@pytest.mark.parametrize("function", [f_welch, tukey_hsd, bonferroni], ids=lambda f: f.__name__)
def test_other_group_paths_valid_controls(native_backend, function):
    groups = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 5.0, 7.0])]
    arrays = [native_backend.array(group) for group in groups]
    result = function(*arrays, backend=native_backend.backend)
    equal_var = function is tukey_hsd
    expected_t = stats.ttest_ind(*groups, equal_var=equal_var)
    sizes = np.array([group.size for group in groups])
    variances = np.array([group.var(ddof=1) for group in groups])
    if equal_var:
        expected_df = int(sizes.sum() - 2)
        expected_se = np.sqrt(np.sum((sizes - 1) * variances) / expected_df * np.sum(1 / sizes))
    else:
        contributions = variances / sizes
        expected_df = contributions.sum() ** 2 / np.sum(contributions ** 2 / (sizes - 1))
        expected_se = np.sqrt(contributions.sum())
    if function is f_welch:
        assert_allclose(result.statistic, expected_t.statistic ** 2, rtol=1e-12)
        assert_allclose(result.pvalue, expected_t.pvalue, rtol=5e-5, atol=1e-9)
        assert result.df_between == 1
        assert result.df_within == pytest.approx(expected_df)
        assert np.isnan(result.eta_squared)
    else:
        assert len(result.comparisons) == 1
        comparison = result.comparisons[0]
        mean_diff = groups[0].mean() - groups[1].mean()
        margin = stats.t.isf(0.025, expected_df) * expected_se
        assert comparison.mean_diff == pytest.approx(mean_diff)
        assert_allclose(comparison.pvalue, expected_t.pvalue, rtol=5e-5, atol=1e-9)
        assert_allclose([comparison.ci_lower, comparison.ci_upper],
                        [mean_diff - margin, mean_diff + margin], rtol=5e-5, atol=1e-9)


@pytest.mark.parametrize("interaction", [True, False])
def test_twoway_valid_balanced_design(native_backend, interaction):
    arrays = [[native_backend.array(np.array([-1.0, 0.0, 1.0]) + offset)
               for offset in row] for row in [[0.0, 1.0], [2.0, 4.0]]]
    result = f_twoway(arrays, interaction=interaction, backend=native_backend.backend)
    # Orthogonal balanced-design sums of squares: A=18.75, B=6.75, AB=.75.
    expected_error = 8.0 if interaction else 8.75
    expected_df = 8 if interaction else 9
    expected_mse = expected_error / expected_df
    assert result.df_within == expected_df
    assert result.ss_within == pytest.approx(expected_error)
    for name, ss in (("factor_a", 18.75), ("factor_b", 6.75)):
        assert getattr(result, f"{name}_df") == 1
        assert getattr(result, f"{name}_statistic") == pytest.approx(ss / expected_mse)
        assert_allclose(getattr(result, f"{name}_pvalue"),
                        stats.f.sf(ss / expected_mse, 1, expected_df), rtol=5e-5, atol=1e-9)
    if interaction:
        assert result.interaction_statistic == pytest.approx(0.75 / expected_mse)
        assert result.interaction_df == 1
    else:
        assert result.interaction_statistic is None
        assert result.interaction_pvalue is None
        assert result.interaction_df is None
    assert all(value is None or isinstance(value, (int, float)) for value in asdict(result).values())


def test_finite_degenerate_groups_keep_statistical_semantics(native_backend):
    one = native_backend.array([1.0, 1.0, 1.0])
    two = native_backend.array([2.0, 2.0, 2.0])
    kwargs = {"backend": native_backend.backend}
    identical = f_oneway(one, one, **kwargs)
    assert np.isnan(identical.statistic)
    assert np.isnan(identical.pvalue)
    assert np.isnan(identical.eta_squared)
    assert np.isnan(cohens_f(one, one, **kwargs))
    separated = f_oneway(one, two, **kwargs)
    assert separated.statistic == np.inf
    assert separated.pvalue == 0.0
    assert separated.eta_squared == 1.0
    assert cohens_f(one, two, **kwargs) == np.inf
    twoway = f_twoway([[one, one], [one, one]], **kwargs)
    assert np.isnan(twoway.factor_a_statistic)
    assert np.isnan(twoway.factor_b_pvalue)
    assert np.isnan(twoway.interaction_statistic)
    assert twoway.df_within == 8
    assert twoway.ss_within == 0.0


def test_sample_size_and_dof_requirements_are_preserved(native_backend):
    single = native_backend.array([1.0])
    pair = native_backend.array([2.0, 3.0])
    kwargs = {"backend": native_backend.backend}
    with pytest.raises(ValueError, match="at least 2 groups"):
        f_oneway(pair, **kwargs)
    with pytest.raises(ValueError, match="at least 1 observation"):
        f_oneway(native_backend.array([]), pair, **kwargs)
    with pytest.raises(ValueError, match="must exceed"):
        f_oneway(single, single, **kwargs)
    result = f_oneway(single, pair, **kwargs)
    assert result.df_within == 1
    assert result.statistic == pytest.approx(stats.f_oneway([1.0], [2.0, 3.0]).statistic)
    cells = [[single, native_backend.array([2.0])],
             [native_backend.array([3.0]), native_backend.array([5.0])]]
    with pytest.raises(ValueError, match="Not enough observations"):
        f_twoway(cells, interaction=True, **kwargs)
    assert f_twoway(cells, interaction=False, **kwargs).df_within == 1


@pytest.mark.parametrize("function", [f_oneway, f_twoway], ids=lambda f: f.__name__)
def test_nonfinite_validation_uses_native_arrays(native_backend, monkeypatch, function):
    xp = native_backend.xp
    actual_isfinite = xp.isfinite
    checked = []

    def record_isfinite(values, *args, **kwargs):
        checked.append(values)
        return actual_isfinite(values, *args, **kwargs)

    monkeypatch.setattr(xp, "isfinite", record_isfinite)
    invalid = native_backend.array([1.0, np.nan, 3.0])
    valid = native_backend.array([2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        if function is f_oneway:
            function(invalid, valid, backend=native_backend.backend)
        else:
            function([[valid, valid], [valid, invalid]], backend=native_backend.backend)
    assert checked
    assert all(type(values) is type(invalid) and values.dtype == invalid.dtype for values in checked)
    if native_backend.backend != "numpy":
        assert all(values.device == invalid.device for values in checked)


def test_top_level_public_aliases_and_scalar_effect_size_controls():
    for function in GROUP_FUNCTIONS + [f_twoway, partial_eta_squared]:
        assert getattr(statgpu, function.__name__) is function
    assert partial_eta_squared(10.0, 5.0) == pytest.approx(2 / 3)
    assert partial_eta_squared(0.0, 5.0) == 0.0
    assert partial_eta_squared(5.0, 0.0) == 1.0
    assert np.isnan(partial_eta_squared(0.0, 0.0))
