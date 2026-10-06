"""Execute both distribution guides and verify their documented CPU contracts.

Reference comparisons use scalar parameters, float64 NumPy outputs and moderate
probabilities. They do not establish GPU accuracy or an extreme-tail guarantee.
"""

import linecache
import re
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from statgpu import inference
from statgpu.inference import _distributions_backend as distributions

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_NAMES = (
    "normal_probabilities",
    "count_probabilities",
    "student_t_inference",
    "validate_density_inputs",
    "fixed_numpy_backend",
    "reproducible_sampling",
    "f_sampling_workaround",
    "compare_lut_paths",
    "small_shape_quantiles",
    "explicit_scipy_fallback",
    "compatibility_migration",
)


def _examples(language):
    path = _ROOT / f"docs/{language}/guides/distribution-api.md"
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    examples = {}
    for block in blocks:
        identifier = re.search(r"^# Example: (\w+)$", block, re.MULTILINE)
        assert identifier is not None, f"{path}: every Python fence must be executed"
        name = identifier.group(1)
        assert name not in examples, f"{path}: duplicate example {name}"
        examples[name] = block
    assert tuple(examples) == _EXAMPLE_NAMES
    return examples


@pytest.fixture(autouse=True)
def _preserve_numpy_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("example", _EXAMPLE_NAMES)
def test_every_distribution_example_is_independent_and_numerically_checked(language, example):
    namespace = {}
    source = _examples(language)[example]
    filename = f"docs/{language}/guides/distribution-api.md:{example}"
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - execute reviewed documentation

    if example == "normal_probabilities":
        x = namespace["x"]
        for key, method in (("cdf", "cdf"), ("sf", "sf"), ("density", "pdf")):
            np.testing.assert_allclose(namespace[key], getattr(stats.norm, method)(x), atol=1e-15)
        np.testing.assert_allclose(namespace["quantiles"], stats.norm.ppf([0.025, 0.5, 0.975]))
        np.testing.assert_allclose(namespace["upper_cutoff"], 1.959963984540054, atol=1e-12)
        np.testing.assert_allclose(namespace["central_probability"], 0.9500042097035593)
    elif example == "count_probabilities":
        np.testing.assert_allclose(namespace["mass"], stats.poisson.pmf(namespace["k"], mu=3))
        np.testing.assert_allclose(namespace["cumulative"], stats.poisson.cdf(namespace["k"], mu=3))
        np.testing.assert_allclose(namespace["probability_at_least_three"], 0.5768099188731564)
        assert namespace["cutoff_95"] == 6
        np.testing.assert_array_equal(namespace["binomial_quantiles"], [2, 4, 6])
        assert stats.poisson.cdf(5, mu=3) < 0.95 <= stats.poisson.cdf(6, mu=3)
    elif example == "student_t_inference":
        values = namespace["values"]
        reference_test = stats.ttest_1samp(values, popmean=namespace["null_mean"])
        np.testing.assert_allclose(namespace["statistic"], reference_test.statistic)
        np.testing.assert_allclose(namespace["pvalue"], reference_test.pvalue, rtol=1e-12)
        np.testing.assert_allclose(namespace["critical"], stats.t.isf(0.025, df=10), rtol=1e-10)
        np.testing.assert_allclose(namespace["interval"], [2.25222645, 2.63868264], atol=5e-9)
        assert namespace["pvalue"] < namespace["alpha"]
        assert namespace["null_mean"] < namespace["interval"][0]
    elif example == "validate_density_inputs":
        helper = namespace["finite_gamma_density"]
        np.testing.assert_allclose(namespace["density"], stats.gamma.pdf([0.5, 1.0, 2.0], a=2))
        for invalid in ([0.5, np.nan], [np.inf], [-np.inf]):
            with pytest.raises(ValueError, match="must be finite"):
                helper(invalid)
    elif example == "fixed_numpy_backend":
        result = namespace["result"]
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, stats.norm.cdf(namespace["x"]))
        assert namespace["scalar"] == 0.5
    elif example == "reproducible_sampling":
        np.testing.assert_allclose(
            namespace["sample"],
            [[1.690525703800356, -0.465937370540833, 0.032820163678584],
             [0.407516282996508, -0.788923028625739, 0.002065572905948]],
            atol=1e-14,
        )
        np.testing.assert_array_equal(namespace["counts_integer"], [4, 1, 4, 5])
        assert namespace["counts_integer"].dtype == np.int64
        assert namespace["sample"].shape == (2, 3)
        assert namespace["counts"].dtype == np.float64
    elif example == "f_sampling_workaround":
        assert namespace["sample"].shape == (4,)
        assert np.all(namespace["sample"] > 0)
        np.testing.assert_allclose(namespace["sample"], [2.350997, 0.207108, 0.491596, 3.343483], atol=5e-7)
    elif example == "compare_lut_paths":
        reference = stats.t.ppf(namespace["q"], df=10)
        np.testing.assert_allclose(namespace["reference_values"], reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(namespace["fast_values"], reference, atol=1e-8, rtol=1e-8)
    elif example == "small_shape_quantiles":
        # Student t with df=1 is exactly Cauchy, independently of beta inversion.
        expected = np.tan(np.pi * (namespace["q"] - 0.5))
        np.testing.assert_allclose(namespace["quantiles"], expected, atol=1e-12)
        np.testing.assert_allclose(namespace["quantiles"], [-3.077684, 0.0, 3.077684], atol=5e-7)
    elif example == "explicit_scipy_fallback":
        np.testing.assert_allclose(namespace["out"], stats.gumbel_r.cdf(namespace["x"]))
        np.testing.assert_allclose(namespace["out"], [0.367879, 0.692201, 0.873423], atol=5e-7)
    elif example == "compatibility_migration":
        np.testing.assert_allclose(namespace["mass_old"], stats.poisson.pmf(3, mu=4))
        np.testing.assert_allclose(namespace["mass_new"], 0.19536681481316454)


def test_english_and_chinese_execute_the_same_examples():
    assert _examples("en") == _examples("cn")


# All 15 advertised families: test supported methods at interior support points.
_FAMILIES = (
    ("norm", {}, [-1.0, 0.0, 1.0]),
    ("t", {"df": 10}, [-1.0, 0.0, 1.0]),
    ("uniform", {}, [0.1, 0.4, 0.9]),
    ("expon", {}, [0.1, 1.0, 2.0]),
    ("cauchy", {}, [-1.0, 0.0, 1.0]),
    ("laplace", {}, [-1.0, 0.0, 1.0]),
    ("logistic", {}, [-1.0, 0.0, 1.0]),
    ("chi2", {"df": 5}, [1.0, 4.0, 9.0]),
    ("gamma", {"a": 2}, [0.4, 2.0, 6.0]),
    ("beta", {"a": 2, "b": 3}, [0.1, 0.4, 0.9]),
    ("f", {"dfn": 5, "dfd": 10}, [0.4, 1.0, 2.0]),
    ("weibull_min", {"c": 1.5}, [0.4, 1.0, 2.0]),
    ("lognorm", {"s": 0.7}, [0.4, 1.0, 2.0]),
    ("poisson", {"mu": 3}, [0, 2, 5]),
    ("binom", {"n": 20, "p": 0.2}, [0, 3, 6]),
)


@pytest.mark.parametrize("name, parameters, points", _FAMILIES, ids=[row[0] for row in _FAMILIES])
def test_native_family_methods_match_scipy_and_moderate_inverse_identities(name, parameters, points):
    fixed = inference.get_distribution(name, backend="numpy", use_lut=False)
    proxy = getattr(inference, name)
    reference = getattr(stats, name)
    discrete = name in {"poisson", "binom"}
    x = np.asarray(points)
    q = np.array([0.1, 0.5, 0.9])
    for method in ("cdf", "sf", "pmf" if discrete else "pdf"):
        expected = getattr(reference, method)(x, **parameters)
        result = getattr(fixed, method)(x, **parameters)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(
            getattr(proxy, method)(x, backend="numpy", use_lut=False, **parameters), result,
        )
        assert np.asarray(result).shape == x.shape
    for method in ("ppf", "isf"):
        result = getattr(fixed, method)(q, **parameters)
        expected = getattr(reference, method)(q, **parameters)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)
    quantiles = fixed.ppf(q, **parameters)
    if discrete:
        assert np.all(fixed.cdf(quantiles - 1, **parameters) < q)
        assert np.all(fixed.cdf(quantiles, **parameters) >= q)
    else:
        np.testing.assert_allclose(fixed.cdf(quantiles, **parameters), q, atol=1e-12)
        np.testing.assert_allclose(fixed.sf(fixed.isf(q, **parameters), **parameters), q, atol=1e-12)


class _FSamplingKeywordMismatch(Exception):
    """The F sampler passes the two known wrong NumPy keywords."""


@contextmanager
def _known_f_sampling_failure():
    try:
        yield
    except TypeError as error:
        # Require the exact NumPy error and its direct _rvs_f call boundary.
        # A fixture, backend, or unrelated sampler TypeError must fail normally.
        traceback = error.__traceback__
        caller = None
        while traceback.tb_next is not None:
            caller, traceback = traceback, traceback.tb_next
        if (
            str(error) == "f() got an unexpected keyword argument 'dfn'"
            and caller is not None
            and caller.tb_frame.f_code is distributions._rvs_f.__code__
            and linecache.getline(caller.tb_frame.f_code.co_filename, caller.tb_lineno).strip()
            == "out = np.random.f(dfn=float(dfn), dfd=float(dfd), size=size)"
            and traceback.tb_frame.f_code.co_filename == "numpy/random/mtrand.pyx"
            and traceback.tb_frame.f_code.co_name == "numpy.random.mtrand.RandomState.f"
        ):
            raise _FSamplingKeywordMismatch(str(error)) from error
        raise


# This describes an intended successful call, not a desired TypeError contract.
# Remove the expected failure and the guide limitation after the production fix.
_F_SAMPLING_BUG = pytest.mark.xfail(
    strict=True,
    raises=_FSamplingKeywordMismatch,
    reason="Existing _rvs_f bug: np.random.f expects dfnum/dfden, not dfn/dfd",
)


@pytest.mark.parametrize(
    "name, parameters",
    [pytest.param(name, parameters, id=name, marks=[_F_SAMPLING_BUG] if name == "f" else [])
     for name, parameters, _ in _FAMILIES],
)
def test_native_sampling_shapes_and_finite_observations(name, parameters):
    fixed = inference.get_distribution(name, backend="numpy")
    with _known_f_sampling_failure():
        draws = fixed.rvs(size=(2, 3), **parameters)
    assert draws.shape == (2, 3)
    assert np.all(np.isfinite(draws))


def test_native_inventory_matches_documented_families():
    names = inference.list_available_distributions()
    assert set(names) == {row[0] for row in _FAMILIES}
    assert len(names) == 15
    for language in ("en", "cn"):
        text = (_ROOT / f"docs/{language}/guides/distribution-api.md").read_text(encoding="utf-8")
        for name in names:
            assert f"`{name}`" in text


def test_scalar_parameters_location_scale_and_count_semantics():
    x = np.array([-1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        inference.norm.cdf(x, loc=2, scale=3, backend="numpy"), stats.norm.cdf(x, loc=2, scale=3),
    )
    np.testing.assert_allclose(
        inference.gamma.ppf([0.1, 0.5, 0.9], a=2, loc=1, scale=3, backend="numpy", use_lut=False),
        stats.gamma.ppf([0.1, 0.5, 0.9], a=2, loc=1, scale=3),
    )
    np.testing.assert_allclose(
        inference.poisson.cdf(x=np.array([2.1, 2.9]), mu=3, backend="numpy"),
        stats.poisson.cdf([2.1, 2.9], mu=3),
    )
    assert inference.poisson.pmf(2.5, mu=3, backend="numpy") == 0
    assert np.isnan(inference.norm.cdf(0, scale=0, backend="numpy"))
    with pytest.raises(TypeError):
        inference.norm.cdf(0, unrecognized_parameter=True, backend="numpy")


def test_proxy_auto_is_input_driven_while_factory_auto_tries_backends(monkeypatch):
    attempts = []

    def numpy_only(backend, device=None, *, use_lut=True):
        attempts.append(backend)
        if backend != "numpy":
            raise ImportError("Optional backend intentionally unavailable in this routing test")
        return distributions.ScipySpecialFunctions(use_lut=use_lut)

    monkeypatch.setattr(distributions, "_make_sf", numpy_only)
    np.testing.assert_allclose(inference.norm.cdf(np.array([0.0])), [0.5])
    assert attempts == ["numpy"]
    attempts.clear()
    fixed = inference.get_distribution("norm", backend="auto")
    assert attempts == ["cupy", "torch", "numpy"]
    assert fixed.cdf(0.0) == 0.5
    with pytest.raises(TypeError):
        fixed.cdf(0.0, backend="numpy")


def test_additional_scipy_families_require_the_explicit_fallback():
    with pytest.raises(ValueError, match="Unknown distribution"):
        inference.get_distribution("gumbel_r", backend="numpy")
    with pytest.raises(ValueError, match="allow_fallback=True"):
        inference.get_distribution_gpu("gumbel_r")
    with pytest.raises(ValueError, match="Unknown scipy.stats distribution"):
        inference.get_distribution_gpu("not_a_distribution", allow_fallback=True)


# Each retained R-style wrapper is called with supported keyword shape arguments.
_R_FAMILIES = (
    ("norm", "norm", {}),
    ("t", "t", {"df": 10}),
    ("chisq", "chi2", {"df": 5}),
    ("gamma", "gamma", {"a": 2}),
    ("beta", "beta", {"a": 2, "b": 3}),
    ("f", "f", {"dfn": 5, "dfd": 10}),
    ("pois", "poisson", {"mu": 3}),
    ("binom", "binom", {"n": 20, "p": 0.2}),
)


@pytest.mark.parametrize("suffix, name, parameters", _R_FAMILIES, ids=[row[0] for row in _R_FAMILIES])
def test_retained_r_wrappers_match_keyword_shape_object_calls(suffix, name, parameters):
    proxy = getattr(inference, name)
    density_method = "pmf" if name in {"poisson", "binom"} else "pdf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for prefix, method, point in (("d", density_method, 0.5), ("p", "cdf", 0.5), ("q", "ppf", 0.9)):
            if name in {"poisson", "binom"} and prefix in {"d", "p"}:
                point = 2
            wrapper = getattr(inference, f"{prefix}{suffix}_gpu")
            np.testing.assert_allclose(wrapper(point, **parameters), getattr(proxy, method)(point, **parameters))
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


@pytest.mark.parametrize(
    "suffix, name, parameters",
    [pytest.param(suffix, name, parameters, id=suffix, marks=[_F_SAMPLING_BUG] if name == "f" else [])
     for suffix, name, parameters in _R_FAMILIES],
)
def test_retained_r_sampling_wrapper_matches_object_call(suffix, name, parameters):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        np.random.seed(123)
        old_sampler = getattr(inference, f"r{suffix}_gpu")
        with _known_f_sampling_failure():
            old_sample = old_sampler(size=5, **parameters)
        np.random.seed(123)
        new_sampler = getattr(inference, name).rvs
        with _known_f_sampling_failure():
            new_sample = new_sampler(size=5, **parameters)
        np.testing.assert_array_equal(old_sample, new_sample)
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


_LEGACY_CALLS = (
    ("norm_cdf_gpu", "norm", "cdf", 1.0, {}),
    ("norm_sf_gpu", "norm", "sf", 1.0, {}),
    ("norm_ppf_gpu", "norm", "ppf", 0.9, {}),
    ("norm_isf_gpu", "norm", "isf", 0.1, {}),
    ("norm_two_sided_pvalue_gpu", "norm", "two_sided_pvalue", 2.0, {}),
    ("norm_two_sided_critical_value_gpu", "norm", "two_sided_critical_value", 0.05, {}),
    ("t_cdf_gpu", "t", "cdf", 1.0, {"df": 10}),
    ("t_sf_gpu", "t", "sf", 1.0, {"df": 10}),
    ("t_ppf_gpu", "t", "ppf", 0.9, {"df": 10}),
    ("t_two_sided_pvalue_gpu", "t", "two_sided_pvalue", 2.0, {"df": 10}),
    ("t_two_sided_critical_value_gpu", "t", "two_sided_critical_value", 0.05, {"df": 10}),
)


@pytest.mark.parametrize("legacy, family, method, value, parameters", _LEGACY_CALLS)
def test_non_r_migration_preserves_values_and_emits_deprecation(legacy, family, method, value, parameters):
    with pytest.warns(DeprecationWarning, match="deprecated"):
        old = getattr(inference, legacy)(value, **parameters)
    new = getattr(getattr(inference, family), method)(value, backend="numpy", **parameters)
    np.testing.assert_allclose(old, new)


def test_distribution_factory_help_matches_observable_cpu_and_lut_contracts():
    help_text = inference.get_distribution.__doc__
    assert "Torch may use CPU" in help_text
    assert "no universal speed or accuracy guarantee" in help_text
    assert "10-500x" not in help_text
    assert "dtype" in help_text and "NaN" in help_text
    fixed = inference.get_distribution("norm", backend="numpy", use_lut=False)
    np.testing.assert_allclose(fixed.cdf([0.0, 1.0]), stats.norm.cdf([0.0, 1.0]))
    with pytest.raises(TypeError):
        fixed.cdf(0.0, backend="numpy")


@pytest.mark.parametrize("name, parameters, points", _FAMILIES, ids=[row[0] for row in _FAMILIES])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_documented_scalar_empty_and_matrix_output_shapes(name, parameters, points, shape):
    fixed = inference.get_distribution(name, backend="numpy", use_lut=False)
    x = np.full(shape, points[1], dtype=np.float32)
    q = np.full(shape, 0.3, dtype=np.float32)
    for method, values in (("cdf", x), ("sf", x),
                           ("pmf" if name in {"poisson", "binom"} else "pdf", x),
                           ("ppf", q), ("isf", q)):
        result = np.asarray(getattr(fixed, method)(values, **parameters))
        assert result.shape == shape
        assert result.dtype == np.float64


def test_documented_scalar_parameter_boundary_and_method_keywords():
    with pytest.raises((TypeError, ValueError)):
        inference.norm.cdf([0.0, 1.0], loc=[0.0, 1.0], backend="numpy")
    with pytest.raises((TypeError, ValueError)):
        inference.t.cdf([0.0, 1.0], df=[5.0, 10.0], backend="numpy")
    with pytest.raises(TypeError):
        inference.poisson.cdf(k=2, mu=3, backend="numpy")
    with pytest.raises(TypeError):
        inference.t.cdf(1.0, 10, backend="numpy")
    for name, parameters in (("poisson", {"mu": 3}), ("binom", {"n": 20, "p": 0.2})):
        proxy = getattr(inference, name)
        assert proxy.ppf(0, loc=2, backend="numpy", **parameters) == 1
        assert proxy.isf(1, loc=2, backend="numpy", **parameters) == 1
        np.testing.assert_allclose(
            proxy.cdf([2.1, 2.9], loc=2, backend="numpy", **parameters),
            getattr(stats, name).cdf([2.1, 2.9], loc=2, **parameters),
        )


def test_special_function_help_has_no_unqualified_speed_or_accuracy_promises():
    for helper in (distributions.CuPySpecialFunctions, distributions.ScipySpecialFunctions,
                   distributions._get_torch_betaincinv_lut, distributions._get_torch_betainc_lut,
                   distributions.TorchSpecialFunctions._betainc_batch):
        doc = helper.__doc__
        assert not re.search(r"(?:~|<)\s*\d+(?:-\d+)?\s*(?:ms|x|1e-)", doc)
    assert "ordinary quantiles" in inference.get_distribution.__doc__
    assert "regardless of" in distributions.TorchSpecialFunctions._betainc_batch.__doc__


# Numerical regressions for the disclosed singular incomplete-beta fallback.
# These assert correct central probabilities/quantiles, not the current wrong
# outputs. Native Torch special functions bypass the affected implementation.
class _TorchSmallShapeMismatch(Exception):
    """The singular incomplete-beta fallback disagrees with SciPy."""


_TORCH_SMALL_SHAPE_BUG = pytest.mark.xfail(
    strict=True,
    raises=_TorchSmallShapeMismatch,
    reason="Torch incomplete-beta endpoint quadrature corrupts small-shape probabilities and quantiles",
)
_SMALL_SHAPE_FAMILIES = (
    ("beta", {"a": 0.5, "b": 0.5}),
    ("t", {"df": 1}),
    ("f", {"dfn": 1, "dfd": 10}),
)


# These fingerprints identify the endpoint-quadrature/clipped-Newton failure
# on the fixed three-point fixtures below. They are an xfail gate, never the
# desired result: a correct reference result returns first and becomes XPASS.
_SMALL_SHAPE_BAD_QUANTILES = {
    ("beta", False): [1e-10, 0.3844459093332461, 0.9161292824887509],
    ("beta", True): [1e-10, 1e-10, 0.9269879811408208],
    ("t", False): [-99999.999995, 0.0003556923139616064, 99999.999995],
    ("t", True): [-99999.999995, 0.000010000000414201846, 99999.999995],
    ("f", False): [1.0000000001e-9, 0.20527631797555754, 1.9446628902603664],
    ("f", True): [1.0000000001e-9, 1.0000000001e-9, 1.3816107506318904],
}


def _assert_small_shape_reference_or_known_failure(actual, expected, signature):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all(), "Nonfinite output is not the clipped small-shape failure"
    if np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        return
    np.testing.assert_allclose(
        actual, signature, atol=1e-12, rtol=1e-8,
        err_msg="A different finite corruption must not be classified as issue #201",
    )
    raise _TorchSmallShapeMismatch('Known endpoint-quadrature/clipped-Newton output')


@pytest.mark.parametrize("name, parameters", _SMALL_SHAPE_FAMILIES)
@_TORCH_SMALL_SHAPE_BUG
def test_torch_small_shape_cdf_without_lut_matches_reference(name, parameters):
    torch = pytest.importorskip("torch")
    if hasattr(torch.special, "betainc"):
        pytest.skip("Native Torch betainc bypasses the affected quadrature fallback")
    fixed = inference.get_distribution(name, backend="torch", device="cpu", use_lut=False)
    x = np.array([0.25, 0.5, 0.75])
    actual = fixed.cdf(x, **parameters).detach().cpu().numpy()
    expected = getattr(stats, name).cdf(x, **parameters)
    # Singular endpoint integration clips the incomplete beta to one; t then
    # maps that value to 0.5. Arbitrary NaN/constant outputs are different defects.
    signature = np.full(3, 0.5 if name == "t" else 1.0)
    _assert_small_shape_reference_or_known_failure(actual, expected, signature)


@pytest.mark.parametrize("use_lut", (True, False))
@pytest.mark.parametrize("name, parameters", _SMALL_SHAPE_FAMILIES)
@_TORCH_SMALL_SHAPE_BUG
def test_torch_small_shape_central_quantiles_match_reference(name, parameters, use_lut):
    torch = pytest.importorskip("torch")
    if hasattr(torch.special, "betaincinv"):
        pytest.skip("Native Torch betaincinv bypasses the affected inverse fallback")
    fixed = inference.get_distribution(name, backend="torch", device="cpu", use_lut=use_lut)
    q = np.array([0.1, 0.5, 0.9])
    actual = fixed.ppf(q, **parameters).detach().cpu().numpy()
    expected = getattr(stats, name).ppf(q, **parameters)
    _assert_small_shape_reference_or_known_failure(
        actual, expected, _SMALL_SHAPE_BAD_QUANTILES[name, use_lut],
    )


@pytest.mark.parametrize("name, parameters", _SMALL_SHAPE_FAMILIES)
@pytest.mark.parametrize("use_lut", (True, False))
def test_small_shape_guard_accepts_repair_and_rejects_unrelated_corruption(name, parameters, use_lut):
    expected = getattr(stats, name).ppf([0.1, 0.5, 0.9], **parameters)
    signature = _SMALL_SHAPE_BAD_QUANTILES[name, use_lut]
    _assert_small_shape_reference_or_known_failure(expected, expected, signature)
    for corrupted in (np.full(3, np.nan), np.full(3, 0.42), expected[:2]):
        with pytest.raises(AssertionError) as caught:
            _assert_small_shape_reference_or_known_failure(corrupted, expected, signature)
        assert not isinstance(caught.value, _TorchSmallShapeMismatch)
    with pytest.raises(_TorchSmallShapeMismatch):
        _assert_small_shape_reference_or_known_failure(signature, expected, signature)


@pytest.mark.parametrize("df", (1, 2))
@pytest.mark.parametrize("use_lut", (True, False))
def test_torch_low_df_two_sided_helpers_have_separate_stable_runtime_path(df, use_lut):
    pytest.importorskip("torch")
    fixed = inference.get_distribution("t", backend="torch", device="cpu", use_lut=use_lut)
    x = np.array([0.5, 1.0, 2.0])
    pvalues = fixed.two_sided_pvalue(x, df=df).detach().cpu().numpy()
    critical = float(fixed.two_sided_critical_value(0.05, df=df))
    np.testing.assert_allclose(pvalues, 2 * stats.t.sf(x, df=df), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(critical, stats.t.isf(0.025, df=df), atol=1e-9, rtol=1e-9)
