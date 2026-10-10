"""Verify the reader-facing survival/smoothing reference against public APIs."""

import ast
import inspect
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from statgpu.nonparametric import (
    KDE,
    BandwidthSelectionResult,
    KDEBootstrapResult,
    KernelDensityEstimator,
    KernelRegression,
    KernelRegressionRegressor,
    fit_kde,
    fit_kernel_regression,
    kde_bootstrap_confidence_interval,
    kde_confidence_interval,
    kde_pdf,
    kernel_regression_predict,
    select_bandwidth,
    select_bandwidth_factor,
)
from statgpu.semiparametric import GAM
from statgpu.survival import CoxPH, CoxPHCV

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "reference/survival-smoothing-api.md"
PUBLIC = {
    obj.__name__: obj
    for obj in (
        CoxPH, CoxPHCV, GAM, KernelDensityEstimator, KernelRegression,
        fit_kde, kde_pdf, fit_kernel_regression, kernel_regression_predict,
        kde_confidence_interval, kde_bootstrap_confidence_interval,
        select_bandwidth, select_bandwidth_factor,
    )
}
for _class, _methods in (
    (CoxPH, ("fit", "predict", "predict_risk_score", "predict_hazard_ratio", "predict_survival", "score", "summary")),
    (CoxPHCV, ("fit", "predict", "predict_risk_score", "predict_hazard_ratio", "predict_survival", "score", "summary")),
    (GAM, ("fit", "predict", "summary")),
    (KernelDensityEstimator, ("fit", "pdf", "logpdf", "__call__", "predict", "score_samples", "score", "to_numpy_metadata")),
    (KernelRegression, ("fit", "predict", "__call__", "score", "to_numpy_metadata")),
    (KDEBootstrapResult, ("to_dict",)),
    (BandwidthSelectionResult, ("to_dict",)),
):
    PUBLIC.update({f"{_class.__name__}.{name}": getattr(_class, name) for name in _methods})


def _page(language, path=REFERENCE):
    return (ROOT / "docs" / language / path).read_text(encoding="utf-8")


def _normalized_signature(obj):
    signature = inspect.signature(obj)
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.name.startswith("_"):
            continue
        default = parameter.default
        if hasattr(default, "value"):
            default = default.value
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    return signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty)


def _signature_blocks(language):
    blocks = re.findall(
        r"<!-- signature: ([\w.]+) -->\s*```text\n(.*?)\n```",
        _page(language), re.DOTALL,
    )
    assert len(blocks) == len(dict(blocks)), "duplicate signature labels"
    return dict(blocks)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_complete_public_signature_inventory_and_defaults(language):
    blocks = _signature_blocks(language)
    assert set(blocks) == set(PUBLIC)
    for name, obj in PUBLIC.items():
        actual = name + str(_normalized_signature(obj))
        assert blocks[name] == actual, f"outdated public call: {language} {name}"
        # Signature fences are valid Python parameter lists, not pseudocode.
        parameter_list = blocks[name][len(name):]
        ast.parse(f"def documented{parameter_list}:\n    pass\n")


def test_language_signature_parity_and_aliases():
    assert _signature_blocks("en") == _signature_blocks("cn")
    assert _normalized_signature(KDE) == _normalized_signature(KernelDensityEstimator)
    assert _normalized_signature(KernelRegressionRegressor) == _normalized_signature(KernelRegression)
    for cls in (CoxPH, CoxPHCV, GAM, KernelDensityEstimator, KernelRegression):
        model = cls()
        assert set(model.get_params(deep=False)) == set(inspect.signature(cls).parameters)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_result_containers_and_shared_reference_are_documented(language):
    text = _page(language)
    for container in (KDEBootstrapResult, BandwidthSelectionResult):
        for field in fields(container):
            assert f"`{field.name}`" in text
    assert "estimator-api.md#parameter-management" in text
    assert "estimator-api.md#inference-helpers" in text
    for topic in ("coxph", "semiparametric", "nonparametric"):
        assert "../reference/survival-smoothing-api.md#" in _page(language, f"models/{topic}.md")


@pytest.fixture(scope="module", params=("en", "cn"))
def reference_examples(request):
    blocks = re.findall(
        r"<!-- example: ([\w-]+) -->\s*```python\n(.*?)```",
        _page(request.param), re.DOTALL,
    )
    assert set(dict(blocks)) == {
        "cox-output-reference-cpu", "bandwidth-selector-reference-cpu",
        "kde-normal-reference-cpu",
    }
    namespaces = {}
    for name, source in blocks:
        namespace = {}
        exec(compile(source, f"{request.param}/{REFERENCE}:{name}", "exec"), namespace)  # noqa: S102
        namespaces[name] = namespace
    return namespaces


def test_cox_coefficient_and_hazard_interval_scales(reference_examples):
    ns = reference_examples["cox-output-reference-cpu"]
    model = ns["model"]
    assert model.converged_
    assert ns["coefficient_ci"].shape == ns["hazard_ratio_ci"].shape == (2, 2)
    np.testing.assert_allclose(ns["coefficient_ci"], model._conf_int)
    np.testing.assert_allclose(ns["hazard_ratio_ci"], np.exp(model._conf_int))
    np.testing.assert_allclose(model._zvalues, model.coef_ / model._bse)
    np.testing.assert_allclose(model._pvalues, 2 * norm.sf(np.abs(model._zvalues)))
    assert ns["summary_result"] is None
    assert np.isclose(model.aic, -2 * model.log_likelihood + 2 * model.coef_.size)
    assert np.isclose(model.bic, -2 * model.log_likelihood + np.log(ns["event"].sum()) * model.coef_.size)
    penalized = CoxPH(penalty=0.2, device="cpu").fit(ns["X"], ns["time"], ns["event"])
    for name in ("aic", "bic"):
        with pytest.raises(RuntimeError):
            getattr(penalized, name)
    estimate_only = CoxPH(device="cpu", compute_inference=False, compute_cindex=False).fit(
        ns["X"], ns["time"], ns["event"],
    )
    assert estimate_only.concordance_index is None
    assert np.isfinite(estimate_only.log_likelihood)
    assert all(getattr(estimate_only, name) is None for name in ("_bse", "_zvalues", "_pvalues", "_conf_int"))


def test_cox_survival_formula_uses_breslow_baseline_with_efron_coefficients(reference_examples):
    ns = reference_examples["cox-output-reference-cpu"]
    assert ns["model"].ties == "efron"
    assert ns["event"].sum() > len(ns["failure_times"]), "exercise tied failures"
    assert ns["curves"].shape == (2, 3)
    np.testing.assert_allclose(ns["curves"], ns["survival_from_formula"], atol=2e-12)
    np.testing.assert_array_equal(ns["curves"][:, 0], 1.0)


def test_normal_interval_formula_and_result_semantics(reference_examples):
    ns = reference_examples["kde-normal-reference-cpu"]
    ci = ns["ci"]
    assert ci.n_resamples == 0 and ci.bootstrap_samples is None
    assert ci.metadata["method"] == "normal"
    np.testing.assert_allclose(ci.lower, ns["normal_lower"])
    np.testing.assert_allclose(ci.upper, ns["normal_upper"])
    assert ci.points.shape == ci.estimate.shape == (3,)
    assert "bootstrap_samples" not in ci.to_dict()
    np.testing.assert_allclose(ci.to_dict()["estimate"], ci.estimate)


def test_bandwidth_formulas_and_diagnostic_fields(reference_examples):
    ns = reference_examples["bandwidth-selector-reference-cpu"]
    result = ns["selection"]
    assert result.method == "scott"
    assert np.isclose(result.factor, 40 ** (-1 / 5))
    assert set(result.to_dict()) == {field.name for field in fields(BandwidthSelectionResult)}
    silverman = select_bandwidth("silverman", **ns["selector_inputs"])
    assert np.isclose(silverman.factor, (40 * 3 / 4) ** (-1 / 5))
    assert result.factor == select_bandwidth_factor("scott", **ns["selector_inputs"])


@pytest.mark.parametrize("language", ("en", "cn"))
def test_core_display_formulas_and_symbol_definitions(language):
    cox = _page(language, "models/coxph.md")
    smooth = _page(language, "models/nonparametric.md")
    gam = _page(language, "models/semiparametric.md")
    cox_math = "\n".join(re.findall(r"\$\$(.*?)\$\$", cox, re.DOTALL))
    smooth_math = "\n".join(re.findall(r"\$\$(.*?)\$\$", smooth, re.DOTALL))
    gam_math = "\n".join(re.findall(r"\$\$(.*?)\$\$", gam, re.DOTALL))
    for token in (r"h_s(t\mid x)", r"S_s(t\mid x)", r"\widehat H_{0s}", "d_{sk}", r"A^{-1}JA^{-1}"):
        assert token in cox_math
    for token in (r"K_H(u)", r"\arg\min_{a,b}", r"b_{\mathrm{Scott}}", r"b_{\mathrm{Silverman}}", r"Q_{\alpha/2}", r"\widehat{\mathrm{SE}}"):
        assert token in smooth_math
    for token in (r"\min_\beta", r"\operatorname{GCV}", r"\operatorname{edf}"):
        assert token in gam_math
    assert "A^-1 J A^-1" not in cox


def test_safe_centered_density_call_and_regression_target_shapes():
    raw = np.linspace(-1, 1, 80) + 1e9
    raw_query = np.array([-.5, 0, .5]) + 1e9
    offset = raw.mean()
    model = fit_kde(raw - offset, bandwidth=0.5, backend="numpy")
    np.testing.assert_allclose(model.logpdf(raw_query - offset), np.log(model.pdf(raw_query - offset)), atol=1e-12)
    for target in ((raw - offset), (raw - offset)[:, None]):
        reg = KernelRegressionRegressor(backend="numpy", device="cpu").fit(raw - offset, target)
        expected_shape = (3,) if target.ndim == 1 else (3, 1)
        assert reg.predict(raw_query - offset).shape == expected_shape
        assert isinstance(reg.score(raw - offset, target), float)
