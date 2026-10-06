"""Fresh cycle-5 spline help, reader workflow and kernel finite-result contracts.

All numerical checks use NumPy CPU. Strict xfails identify unresolved production
behavior; documentation changes do not repair numerical kernels or GPU routing.
"""
import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline

from statgpu.nonparametric.kernel_methods import KernelRidge, Nystroem
from statgpu.nonparametric.splines import (
    SplineTransformer,
    bspline_basis,
    cyclic_cubic_spline_basis,
    natural_cubic_spline_basis,
    thin_plate_spline_basis,
)

ROOT = Path(__file__).resolve().parents[2]


class _NonfiniteKernelFittedState(Exception):
    """The known overflowing or nonfinite kernel published nonfinite results."""


@pytest.mark.parametrize("language", ["en", "cn"])
def test_spline_training_query_example_is_self_contained_and_reuses_fit(language):
    path = ROOT / f"docs/{language}/models/splines.md"
    text = path.read_text()
    examples = dict(re.findall(
        r"<!-- example: ([\w-]+) -->\s*```python\n(.*?)```", text, re.DOTALL))
    namespace = {}
    exec(compile(examples["spline-transformer-reuse-cpu"], str(path), "exec"), namespace)  # noqa: S102
    model = namespace["transformer"]
    assert namespace["B_train"].shape == (41, 7)
    assert namespace["B_query"].shape == (3, 7)
    assert_allclose(namespace["prediction"], [-.778, 0., .778], atol=5e-4)
    assert namespace["names"] == [f"time_bspline{i}" for i in range(7)]
    assert model.include_bias is False
    assert_allclose(model.knots_[0], np.linspace(-2, 2, 6))
    # Independent augmented-knot construction verifies held-out query features.
    knots = np.r_[np.repeat(-2., 4), np.linspace(-2, 2, 6)[1:-1], np.repeat(2., 4)]
    reference = BSpline(knots, np.eye(8), 3)
    assert_allclose(namespace["B_query"], reference(namespace["X_query"][:, 0])[:, :-1], atol=1e-14)
    # Querying must not relearn quantiles/boundaries or change the training basis.
    assert_allclose(model.transform(namespace["X_train"]), namespace["B_train"])


def test_installed_spline_help_covers_constructor_methods_and_learned_state():
    doc = inspect.getdoc(SplineTransformer)
    for name in inspect.signature(SplineTransformer).parameters:
        assert f"{name} :" in doc
    for name in ["knots_", "boundary_lo_", "boundary_hi_", "n_features_in_", "n_features_out_"]:
        assert name in doc
    for method in ["fit", "transform", "fit_transform", "predict", "get_feature_names_out", "get_params", "set_params"]:
        assert inspect.getdoc(getattr(SplineTransformer, method)), method
    assert "sample_weight does not implement weighted quantiles" in doc
    assert "refit" in inspect.getdoc(SplineTransformer.set_params).lower()
    assert "not response predictions" in inspect.getdoc(SplineTransformer.predict)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_spline_reference_lists_all_exported_calls_and_exact_parameters(language):
    text = (ROOT / f"docs/{language}/models/splines.md").read_text()
    objects = {
        "bspline_basis": bspline_basis,
        "natural_cubic_spline_basis": natural_cubic_spline_basis,
        "cyclic_cubic_spline_basis": cyclic_cubic_spline_basis,
        "thin_plate_spline_basis": thin_plate_spline_basis,
        "SplineTransformer": SplineTransformer,
    }
    objects.update({f"SplineTransformer.{name}": getattr(SplineTransformer, name)
                    for name in ["fit", "transform", "fit_transform", "predict", "get_feature_names_out", "get_params", "set_params"]})
    for name, obj in objects.items():
        match = re.search(r"^" + re.escape(name) + r"\((.*)\)$", text, re.MULTILINE)
        assert match, name
        parsed = ast.parse(f"def temporary({match.group(1)}): pass").body[0].args
        documented = [arg.arg for arg in parsed.args + parsed.kwonlyargs]
        if parsed.kwarg:
            documented.append(parsed.kwarg.arg)
        runtime = [name for name in inspect.signature(obj).parameters if name != "self"]
        assert documented == runtime, name
        # Evaluate simple documented defaults without source annotations.
        defaults = [ast.literal_eval(value) for value in parsed.defaults]
        expected = [parameter.default for parameter in inspect.signature(obj).parameters.values()
                    if parameter.default is not inspect.Parameter.empty]
        expected = [getattr(value, "value", value) for value in expected]
        assert defaults == expected, name


def test_custom_knots_define_boundaries_and_ignored_weights_do_not_move_knots():
    X = np.linspace(-2., 2., 41)[:, None]
    weights = np.linspace(1., 10., len(X))
    model = SplineTransformer(n_knots=6, knots="quantile", device="cpu")
    model.fit(X, sample_weight=weights)
    assert_allclose(model.knots_[0], np.quantile(X[:, 0], np.linspace(0, 1, 6)))
    assert_allclose(model.fit_transform(X, sample_weight=weights), model.transform(X))
    # Custom boundaries are not silently replaced by the training extrema.
    custom = SplineTransformer(n_knots=3, knots=[-1., 0., 1.], extrapolation="error", device="cpu")
    assert custom.fit(X) is custom
    assert_allclose(custom.boundary_lo_, [-1.])
    assert_allclose(custom.boundary_hi_, [1.])
    with pytest.raises(ValueError, match="outside"):
        custom.transform(X)
    with pytest.raises(ValueError, match="outside"):
        custom.fit_transform(X)
    with pytest.raises(ValueError, match="length"):
        model.get_feature_names_out(["a", "b"])


@pytest.mark.parametrize("language", ["en", "cn"])
def test_kernel_nonfinite_guidance_has_learned_and_output_checks(language):
    text = (ROOT / f"docs/{language}/models/kernel-methods.md").read_text()
    for phrase in ["np.isfinite(model.dual_coef_).all()", "np.isfinite(model.normalization_).all()",
                   "np.isfinite(model.eigenvalues_).all()", "NaN", "alpha"]:
        assert phrase in text
    assert "already\noverflowed" in text if language == "en" else "核已经溢出" in text
    for cls in [KernelRidge, Nystroem]:
        doc = inspect.getdoc(cls)
        assert "NaN learned arrays" in doc
        assert "Increasing regularization does not repair" in doc


def test_finite_kernel_checks_preserve_an_ordinary_polynomial_fit():
    X = np.linspace(-1., 1., 8)[:, None]
    y = np.sin(X[:, 0])
    model = KernelRidge(kernel="poly", degree=3, alpha=.2, device="cpu").fit(X, y)
    kernel = (X @ X.T + 1.) ** 3
    expected = np.linalg.solve(kernel + .2 * np.eye(len(X)), y)
    assert np.isfinite(model.dual_coef_).all()
    assert_allclose(model.dual_coef_[:, 0], expected, atol=1e-14)
    assert_allclose(model.predict(X), kernel @ expected, atol=1e-14)
    features = Nystroem(kernel="poly", degree=3, n_components=2, random_state=0, device="cpu").fit(X)
    assert np.isfinite(features.normalization_).all()
    assert np.isfinite(features.eigenvalues_).all()
    assert np.isfinite(features.transform(X)).all()


@pytest.mark.xfail(strict=True, raises=_NonfiniteKernelFittedState,
                   reason="Issue #236: finite polynomial overflow publishes NaN fitted state")
@pytest.mark.parametrize("kind", ["ridge", "nystroem"])
def test_finite_polynomial_overflow_should_not_publish_nan_fitted_state(kind):
    X = np.arange(8.)[:, None]
    y = np.sin(X[:, 0])
    if kind == "ridge":
        model = KernelRidge(kernel="poly", degree=200, device="cpu")
        fields = ["dual_coef_"]
    else:
        model = Nystroem(kernel="poly", degree=200, n_components=2, random_state=0, device="cpu")
        fields = ["normalization_", "eigenvalues_"]
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            model.fit(X, y)
            prediction = model.predict(X)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return  # A corrected clear rejection produces strict XPASS for cleanup.
    finite_fields = [np.isfinite(getattr(model, field)).all() for field in fields]
    finite_prediction = np.isfinite(prediction).all()
    if not all(finite_fields) or not finite_prediction:
        raise _NonfiniteKernelFittedState("Polynomial overflow published nonfinite fitted results")


@pytest.mark.xfail(strict=True, raises=_NonfiniteKernelFittedState,
                   reason="Issue #236: nonfinite kernel controls can publish NaN fits")
@pytest.mark.parametrize("controls", [{"gamma": np.nan}, {"kernel": "poly", "coef0": np.inf}])
def test_nonfinite_kernel_controls_should_not_publish_nan_fit(controls):
    X = np.arange(8.)[:, None]
    model = KernelRidge(device="cpu", **controls)
    try:
        model.fit(X, np.sin(X[:, 0]))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return
    prediction = model.predict(X)
    finite_coefficients = np.isfinite(model.dual_coef_).all()
    finite_prediction = np.isfinite(prediction).all()
    if not finite_coefficients or not finite_prediction:
        raise _NonfiniteKernelFittedState("Nonfinite kernel controls published nonfinite fitted results")


def test_selector_and_result_help_covers_runtime_fields_and_metadata_boundaries(monkeypatch):
    from dataclasses import fields

    from statgpu.nonparametric import (
        BandwidthSelectionResult,
        KDEBootstrapResult,
        fit_kde,
        kde_confidence_interval,
        select_bandwidth,
        select_bandwidth_factor,
    )
    from statgpu.nonparametric.kernel_smoothing import _bandwidth_selection as selectors

    doc = inspect.getdoc(select_bandwidth)
    for name in inspect.signature(select_bandwidth).parameters:
        assert f"{name} :" in doc
    assert "select_bandwidth" in inspect.getdoc(select_bandwidth_factor)
    for cls in [BandwidthSelectionResult, KDEBootstrapResult]:
        for field in fields(cls):
            assert field.name in inspect.getdoc(cls)
        assert inspect.getdoc(cls.to_dict)

    samples = np.linspace(-2, 2, 30)
    weights = np.arange(1., 31.)
    fitted = fit_kde(samples, weights=weights, bandwidth=1., backend="numpy")
    arguments = {
        "n_eff": 1 / np.sum(fitted.weights_**2), "n_features": 1,
        "samples_2d": fitted.samples_, "weights_1d": fitted.weights_,
        "data_cov": fitted.covariance_, "xp": np,
    }

    def forbidden_resampling(*args, **kwargs):
        raise AssertionError("Scott's rule must not quantile-resample")

    monkeypatch.setattr(selectors, "_weighted_quantile_resample_1d", forbidden_resampling)
    selected = select_bandwidth("scott", **arguments)
    assert selected.weighted
    assert selected.weighted_strategy == "quantile_resample"  # Configuration, not execution provenance.
    assert not selected.used_r_selector
    assert_allclose(selected.factor, arguments["n_eff"] ** (-1 / 5))
    assert select_bandwidth_factor("scott", **arguments) == selected.factor
    payload = selected.to_dict()
    assert set(payload) == {field.name for field in fields(selected)}
    assert payload["details"] == selected.details and payload["details"] is not selected.details
    assert "alone does not prove quantile resampling ran" in inspect.getdoc(BandwidthSelectionResult)
    for language in ["en", "cn"]:
        reference = (ROOT / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
        assert "1e-12" in reference and "Scott/Silverman" in reference
        assert "quantile_resample" in reference

    interval = kde_confidence_interval(samples, [0., 1.], bandwidth=.4, backend="numpy")
    assert interval.points.shape == interval.estimate.shape == (2,)
    assert interval.n_resamples == 0 and interval.bootstrap_samples is None
    serialized = interval.to_dict()
    assert "bootstrap_samples" not in serialized
    assert serialized["points"] == [0., 1.]
    assert isinstance(serialized["lower"], list)
    assert serialized["metadata"]["method"] == "normal"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_cox_complete_reference_documents_public_numerical_exception(language):
    from statgpu.survival import CoxFitNumericalError

    assert issubclass(CoxFitNumericalError, FloatingPointError)
    text = (ROOT / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
    assert "CoxFitNumericalError" in text
    assert "FloatingPointError" in text
    assert "converged_" in text
    assert "broad exception" in text if language == "en" else "宽泛的异常捕获" in text
