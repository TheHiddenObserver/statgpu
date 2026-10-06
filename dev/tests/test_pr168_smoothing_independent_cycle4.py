"""Independent S1-S6 documentation regressions for the fourth PR168 review.

All numerical checks here run on CPU. Strict xfails assert the intended explicit
accelerator contract; they expose a separately reported routing defect and do not
claim that the documentation edits repair it or establish physical CUDA parity.
"""

import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import BSpline
from scipy.special import logsumexp

from statgpu.nonparametric import KernelDensityEstimator, KernelRegression
from statgpu.nonparametric.kernel_methods import KernelPCA, Nystroem, pairwise_kernels
from statgpu.nonparametric.splines import SplineTransformer

ROOT = Path(__file__).resolve().parents[2]


class _ExplicitAcceleratorPlacementError(Exception):
    """Only the known explicit-accelerator output-placement contract failed."""


def _example(language, name):
    path = ROOT / f"docs/{language}/models/kernel-methods.md"
    blocks = re.findall(
        r"<!-- example: ([\w-]+) -->\s*```python\n(.*?)```",
        path.read_text(), re.DOTALL,
    )
    assert len(blocks) == len(dict(blocks))
    namespace = {}
    exec(compile(dict(blocks)[name], str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("language", ["en", "cn"])
def test_seeded_kernel_workflow_keeps_held_out_rows_out_of_fit(language):
    example = _example(language, "kernel-methods-cpu")
    train, test = example["train"], example["test"]
    assert not np.intersect1d(train, test).size
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(240))
    X, y = example["X"], example["y"]
    selected = example["kr_cv"]
    assert selected.random_state == 42
    assert_allclose(selected.estimator_.X_fit_, X[train])
    assert_allclose(example["kr"].X_fit_, X[train])
    assert_allclose(example["kpca"].X_fit_, X[train])
    approximation = example["nystroem"]
    assert_allclose(approximation.components_, X[train][approximation.component_indices_])
    assert selected.cv_results_["mse_table"].shape == (4, 5, 1)
    assert np.isfinite(selected.cv_results_["mse_table"]).all()
    best = np.argmin(selected.cv_results_["mean_mse"].mean(axis=1))
    assert selected.alpha_ == selected.cv_results_["alphas"][best]
    # An independent RBF system checks the final training-only refit and query.
    kernel = np.exp(-0.7 * np.sum((X[train, None] - X[None, train]) ** 2, axis=2))
    query = np.exp(-0.7 * np.sum((X[test, None] - X[None, train]) ** 2, axis=2))
    prediction = query @ np.linalg.solve(kernel + selected.alpha_ * np.eye(180), y[train])
    assert_allclose(example["prediction"], prediction, atol=1e-10)
    errors = y[test] - prediction
    assert_allclose(example["held_out_mse"], np.mean(errors**2))
    assert_allclose(example["held_out_r2"], 1 - np.sum(errors**2) / np.sum((y[test] - y[test].mean())**2))
    assert example["held_out_r2"] > 0.9
    assert example["test_coordinates"].shape == (60, 3)
    assert example["test_features"].shape == (60, 40)
    assert 0 <= example["relative_kernel_error"] < 0.1


@pytest.mark.parametrize("language", ["en", "cn"])
def test_callable_example_handles_omitted_and_explicit_array_modules(language):
    example = _example(language, "kernel-callable-cpu")
    assert_allclose(example["implicit"], example["X"] @ example["X"].T)
    assert_allclose(example["explicit"], example["implicit"])


def test_callable_dispatch_forwards_none_and_explicit_module_unchanged():
    X = np.linspace(-1, 1, 7)[:, None]
    received = []

    def callable_kernel(X, Y=None, xp=None):
        received.append((Y, xp))
        module = np if xp is None else xp
        Y = X if Y is None else Y
        return module.asarray(X) @ module.asarray(Y).T

    assert_allclose(pairwise_kernels(X, metric=callable_kernel), X @ X.T)
    assert_allclose(pairwise_kernels(X, metric=callable_kernel, xp=np), X @ X.T)
    assert received == [(None, None), (None, np)]
    assert "including None" in inspect.getdoc(pairwise_kernels)
    assert "built-in kernels" in inspect.getdoc(pairwise_kernels)


def test_retained_kernel_dimensions_normalization_and_installed_help():
    X = np.linspace(-1, 1, 20)[:6, None]
    model = KernelPCA(n_components=30, device="cpu").fit(X)
    k = len(model.lambdas_)
    assert 0 < k < len(X) < model.n_components
    assert model.alphas_.shape == (len(X), k)
    assert model.transform(X[:2]).shape == (2, k)
    for field in ("lambdas_", "alphas_", "X_fit_"):
        assert isinstance(getattr(model, field), np.ndarray)
    # Recover unit-norm eigenvectors, rather than assuming alphas_ has unit norm.
    vectors = model.alphas_ * np.sqrt(model.lambdas_)[None, :]
    assert_allclose(vectors.T @ vectors, np.eye(k), atol=1e-12)
    assert not np.allclose(np.linalg.norm(model.alphas_, axis=0), 1)
    K = np.exp(-np.sum((X[:, None] - X[None, :])**2, axis=2))
    centered = K - K.mean(axis=0) - K.mean(axis=1)[:, None] + K.mean()
    assert_allclose(centered @ vectors, vectors * model.lambdas_, atol=1e-12)
    help_text = inspect.getdoc(KernelPCA)
    assert "shape (k,)" in help_text and "shape (n_samples, k)" in help_text
    assert "k <= min(n_components, n_samples)" in help_text
    assert "V / sqrt(lambda)" in help_text and "Host" in help_text
    assert "shape (n_samples, k)" in inspect.getdoc(KernelPCA.transform)
    assert "selected backend" in inspect.getdoc(KernelPCA.transform)

    approximation = Nystroem(n_components=30, random_state=23, device="cpu").fit(X)
    m = min(approximation.n_components, len(X))
    shapes = {
        "components_": (m, 1), "component_indices_": (m,),
        "normalization_": (m, m), "eigenvalues_": (m,),
    }
    for field, shape in shapes.items():
        value = getattr(approximation, field)
        assert isinstance(value, np.ndarray) and value.shape == shape
    assert approximation.transform(X[:2]).shape == (2, m)
    help_text = inspect.getdoc(Nystroem)
    assert "m = min(n_components, n_samples)" in help_text
    assert "shape (m, n_features)" in help_text
    assert "shape (m,)" in help_text and "shape (m, m)" in help_text
    assert "shape (n_components" not in help_text
    assert "shape (n_samples, m)" in inspect.getdoc(Nystroem.transform)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_public_device_surfaces_explain_actual_placement_checks(language):
    paths = ["models/nonparametric.md", "reference/survival-smoothing-api.md",
             "models/splines.md", "guides/device-and-memory.md"]
    for relative in paths:
        text = (ROOT / "docs" / language / relative).read_text()
        for phrase in ('device="torch"', 'device="cuda"', ".device", ".is_cuda", "NumPy"):
            assert phrase in text, (relative, phrase)
        assert ("CPU input" in text or "CPU 输入" in text), relative
        assert ("samples_" in text or "knots_" in text), relative
    guide = (ROOT / f"docs/{language}/guides/device-and-memory.md").read_text()
    assert "KernelDensityEstimator" in guide and "KernelRegression" in guide
    assert "SplineTransformer" in guide
    assert 'id="current-smoothing-and-spline-exceptions"' in guide
    for name in ("KernelPCA", "Nystroem", "fit_transform", "transform", "predict"):
        assert name in guide
    assert "Torch CUDA" in guide  # Preserve the strict intended device meaning.
    for cls in (KernelDensityEstimator, KernelRegression, SplineTransformer):
        text = inspect.getdoc(cls)
        assert "device='torch'" in text and "'cuda'" in text
        assert ".device/.is_cuda" in text and "CPU" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_kernel_page_leads_with_choices_and_interprets_held_out_results(language):
    text = (ROOT / f"docs/{language}/models/kernel-methods.md").read_text()
    opening = text.split("<!-- example: kernel-methods-cpu -->", 1)[0]
    for name in ("KernelRidge", "KernelRidgeCV", "KernelPCA", "Nystroem", "GAM"):
        assert name in opening
    assert "linear-regression.md" in opening
    assert "held-out" in opening if language == "en" else "留出" in opening
    reference = "## Complete estimator API" if language == "en" else "## 完整估计器 API"
    assert text.index("held_out_r2") < text.index(reference)
    assert "relative_kernel_error" in text and "best_score_" in text
    assert "including `None`" in text if language == "en" else "包括 `None`" in text


def test_chinese_cox_prose_and_spline_public_layering():
    cox = (ROOT / "docs/cn/models/coxph.md").read_text()
    for old in ("训练集一致性指数 当作", "一致性指数 前", "因此 标准误", "生存感知分支"):
        assert old not in cox
    assert "生存分析专用分支" in cox
    for language in ("en", "cn"):
        splines = (ROOT / f"docs/{language}/models/splines.md").read_text()
        assert "QR fallback" not in splines and "QR 回退" not in splines
        assert "1e6" in splines and "boundary_lo" in splines
        assert "cyclic_cubic_spline_basis" in splines
        assert "不再将完整数组交给 SciPy" not in splines


@pytest.mark.parametrize("kind", ["kde", "regression"])
def test_safe_explicit_numpy_cpu_smoothers_have_host_arrays_and_analytic_outputs(kind):
    X = np.linspace(-2, 2, 15)[:, None]
    q = np.array([[-0.7], [0.25], [1.2]])
    options = {"bandwidth": 0.6, "backend": "numpy", "device": "cpu"}
    if kind == "kde":
        model = KernelDensityEstimator(**options).fit(X)
    else:
        model = KernelRegression(**options).fit(X, np.sin(X[:, 0]))
    output = model.predict(q)
    assert isinstance(model.samples_, np.ndarray) and isinstance(output, np.ndarray)
    assert model.samples_.dtype == np.float64 and output.dtype == np.float64
    h = np.sqrt(model.covariance_[0, 0])
    weights = np.exp(-0.5 * ((q - X.T) / h)**2)
    if kind == "kde":
        expected = weights.mean(axis=1) / (np.sqrt(2 * np.pi) * h)
    else:
        expected = (weights @ np.sin(X[:, 0])) / weights.sum(axis=1)
    assert_allclose(output, expected, atol=1e-12)


def test_safe_numpy_cpu_spline_checks_knots_and_output_placement_against_scipy():
    X = np.linspace(-2, 2, 15)[:, None]
    q = np.array([[-0.7], [0.25], [1.2]])
    model = SplineTransformer(n_knots=5, device="cpu").fit(X)
    output = model.transform(q)
    assert all(isinstance(knots, np.ndarray) for knots in model.knots_)
    assert isinstance(output, np.ndarray) and output.dtype == np.float64
    knots = model.knots_[0]
    augmented = np.r_[np.repeat(knots[0], model.degree + 1),
                      knots[1:-1], np.repeat(knots[-1], model.degree + 1)]
    expected = BSpline(augmented, np.eye(model.n_features_out_), model.degree)(q[:, 0])
    assert_allclose(output, expected, atol=1e-12)


def _assert_accelerator_arrays(arrays, device, torch):
    if device == "torch":
        correctly_placed = all(isinstance(array, torch.Tensor) and array.is_cuda for array in arrays)
    else:
        correctly_placed = all(type(array).__module__.split(".")[0] == "cupy" for array in arrays)
    if not correctly_placed:
        observed = [(type(array).__module__, str(getattr(array, "device", "cpu"))) for array in arrays]
        raise _ExplicitAcceleratorPlacementError(
            f"Explicit device={device!r} must use its CUDA backend or raise; observed {observed!r}"
        )


@pytest.mark.xfail(strict=True, raises=_ExplicitAcceleratorPlacementError,
                   reason="Issue #231: explicit accelerators can silently execute on CPU")
@pytest.mark.parametrize("kind", ["kde", "regression"])
@pytest.mark.parametrize("input_kind", ["numpy", "torch_cpu"])
@pytest.mark.parametrize("device,backend", [
    ("torch", "auto"), ("torch", "torch"), ("cuda", "torch"),
    ("torch", "numpy"), ("cuda", "numpy"),
])
def test_smoothers_should_honor_explicit_accelerator_or_raise(kind, input_kind, device, backend):
    torch = pytest.importorskip("torch")
    X = np.linspace(-2, 2, 15)[:, None]
    if input_kind == "torch_cpu":
        X = torch.tensor(X, dtype=torch.float64, device="cpu")
    options = {"bandwidth": 0.6, "backend": backend, "device": device}
    try:
        if kind == "kde":
            model = KernelDensityEstimator(**options).fit(X)
        else:
            model = KernelRegression(**options).fit(X, np.linspace(0, 1, 15))
        output = model.predict(X[:2])
    except RuntimeError as error:
        # A corrected unavailable-backend error satisfies the intended contract.
        if re.search(r"unavailable|not available|not installed|no GPU backend|CuPy is required",
                     str(error), re.IGNORECASE):
            return
        raise
    _assert_accelerator_arrays([model.samples_, output], device, torch)


@pytest.mark.xfail(strict=True, raises=_ExplicitAcceleratorPlacementError,
                   reason="Issue #231: Torch CPU spline input overrides explicit device")
@pytest.mark.parametrize("device", ["torch", "cuda"])
def test_spline_should_honor_explicit_accelerator_with_torch_cpu_input_or_raise(device):
    torch = pytest.importorskip("torch")
    X = torch.linspace(-2, 2, 15, dtype=torch.float64, device="cpu")[:, None]
    try:
        model = SplineTransformer(n_knots=5, device=device).fit(X)
        output = model.transform(X[:2])
    except RuntimeError as error:
        if re.search(r"unavailable|not available|not installed|no GPU backend|CuPy is required",
                     str(error), re.IGNORECASE):
            return
        raise
    _assert_accelerator_arrays([*model.knots_, output], device, torch)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_kernel_feature_docs_distinguish_availability_from_actual_placement(language):
    text = (ROOT / f"docs/{language}/models/kernel-methods.md").read_text()
    for phrase in ("KernelPCA", "Nystroem", 'device="torch"', ".device", ".is_cuda",
                   "NumPy", "fit_transform", "current-smoothing-and-spline-exceptions"):
        assert phrase in text
    assert "CPU input" in text if language == "en" else "CPU 输入" in text
    for cls in (KernelPCA, Nystroem):
        help_text = inspect.getdoc(cls)
        assert "availability check succeeds" in help_text
        assert ".device/.is_cuda" in help_text
        assert "Public fitted arrays are deliberately NumPy" in help_text
        method_help = inspect.getdoc(cls.transform)
        assert "rejects unavailable" in method_help
        assert "CPU input can still" in method_help
        assert "does not guarantee placement" in method_help


@pytest.fixture(scope="module")
def kernel_feature_availability_only_records():
    """Fresh-process gate-only mock; no backend replacement or allocation mock.

    The real unavailable check runs first and remains separate from the mocked
    positive-resolution checks. No physical CUDA execution is established.
    """
    pytest.importorskip("torch")
    script = textwrap.dedent(r"""
        import json
        import re
        from unittest.mock import patch
        import numpy as np
        import torch
        from statgpu.nonparametric.kernel_methods import KernelPCA, Nystroem

        actual_available = torch.cuda.is_available()
        result = {"actual_cuda_available": actual_available, "records": []}
        X = np.linspace(-1, 1, 20)[:, None]
        for cls in (KernelPCA, Nystroem):
            for input_kind in ("numpy", "torch_cpu"):
                values = X if input_kind == "numpy" else torch.as_tensor(X, device="cpu")
                record = {"class": cls.__name__, "input": input_kind}
                if not actual_available:
                    try:
                        cls(n_components=3, device="torch").fit(values)
                    except RuntimeError as error:
                        record["unavailable_error"] = str(error)
                    else:
                        record["unavailable_error"] = None
                with patch.object(torch.cuda, "is_available", return_value=True):
                    model = cls(n_components=3, device="torch")
                    backend = model._get_backend(backend="auto")
                    record["resolved_name"] = backend.name
                    record["resolved_device"] = backend._device
                    try:
                        outputs = {"fit_transform": model.fit_transform(values)}
                        outputs.update({name: getattr(model, name)(values[:2])
                                        for name in ("transform", "predict")})
                    except (RuntimeError, AssertionError) as error:
                        # A future explicit CUDA rejection is valid; unrelated
                        # failures leave the subprocess nonzero and fail pytest.
                        if not re.search(
                            r"Torch not compiled with CUDA enabled|"
                            r"(?:CUDA|GPU).*(?:unavailable|not available)|"
                            r"(?:no|not found).*(?:CUDA|NVIDIA|GPU)",
                            str(error), re.IGNORECASE,
                        ):
                            raise
                        record["rejected"] = str(error)
                    else:
                        record["outputs"] = {
                            name: {"torch_tensor": isinstance(value, torch.Tensor),
                                   "device": str(value.device), "is_cuda": value.is_cuda}
                            for name, value in outputs.items()
                        }
                result["records"].append(record)
        print(json.dumps(result))
    """)
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = environment["OMP_NUM_THREADS"] = "1"
    with tempfile.TemporaryDirectory(prefix="pr168-kernel-device-cache-") as cache:
        environment["PYTHONPYCACHEPREFIX"] = cache
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=environment,
            text=True, capture_output=True, check=True, timeout=90,
        )
    evidence = json.loads(completed.stdout)
    # Routing identity is setup evidence, not the expected output-placement
    # assertion. A resolver regression must fail independently of the xfail.
    for record in evidence["records"]:
        if record["resolved_name"] != "torch" or record["resolved_device"] != "cuda":
            raise ValueError(f"Unexpected resolver identity in gate-only probe: {record}")
    return evidence


@pytest.mark.parametrize("cls_name", ["KernelPCA", "Nystroem"])
@pytest.mark.parametrize("input_kind", ["numpy", "torch_cpu"])
def test_kernel_feature_real_unavailable_torch_cuda_request_raises(
    kernel_feature_availability_only_records, cls_name, input_kind,
):
    evidence = kernel_feature_availability_only_records
    if evidence["actual_cuda_available"]:
        pytest.skip("Real CUDA is available; this is the actual-unavailability control")
    record = next(row for row in evidence["records"]
                  if row["class"] == cls_name and row["input"] == input_kind)
    assert record["unavailable_error"] is not None
    assert re.search(r"CUDA.*(?:unavailable|not available)", record["unavailable_error"],
                     re.IGNORECASE)


@pytest.mark.xfail(strict=True, raises=_ExplicitAcceleratorPlacementError,
                   reason="Issue #231: availability-only mock exposes missing Torch device conversion; no physical CUDA claim")
@pytest.mark.parametrize("cls_name", ["KernelPCA", "Nystroem"])
@pytest.mark.parametrize("input_kind", ["numpy", "torch_cpu"])
def test_kernel_feature_availability_only_mock_should_place_outputs_on_cuda_or_raise(
    kernel_feature_availability_only_records, cls_name, input_kind,
):
    record = next(row for row in kernel_feature_availability_only_records["records"]
                  if row["class"] == cls_name and row["input"] == input_kind)
    if "rejected" in record:
        return  # A clear corrected rejection prompts strict XPASS and marker removal.
    assert set(record["outputs"]) == {"fit_transform", "transform", "predict"}
    correctly_placed = all(value["torch_tensor"] and value["is_cuda"]
                           for value in record["outputs"].values())
    if not correctly_placed:
        raise _ExplicitAcceleratorPlacementError(record)


@pytest.mark.parametrize("cls", [KernelPCA, Nystroem])
def test_kernel_feature_explicit_numpy_cpu_alternative_returns_host_arrays(cls):
    X = np.linspace(-1, 1, 20)[:, None]
    options = {"n_components": 3, "device": "cpu"}
    if cls is Nystroem:
        options["random_state"] = 23
    model = cls(**options)
    fitted = model.fit_transform(X)
    transformed, predicted = model.transform(X[:2]), model.predict(X[:2])
    assert all(isinstance(value, np.ndarray) for value in (fitted, transformed, predicted))
    assert_allclose(transformed, fitted[:2], atol=1e-12)
    assert_allclose(predicted, transformed, atol=1e-12)


@pytest.mark.parametrize("bandwidth", [
    0.4, "scott", "silverman", "nrd", "nrd0", "ucv", "bcv", "sj-ste", "sj-dpi",
])
def test_kde_zero_weight_deletion_preserves_numeric_factor_not_every_string_selection(bandwidth):
    # Equal positive masses plus far-away zero masses exercise both raw-sample
    # scale rules and the selectors' uniform-vs-resampled weighting decision.
    X = np.r_[np.linspace(-1, 1, 20), np.repeat(1e6, 80)]
    weights = np.r_[np.ones(20), np.zeros(80)]
    keep = weights > 0
    options = {"backend": "numpy", "device": "cpu"}
    original = KernelDensityEstimator(bandwidth=bandwidth, weights=weights, **options).fit(X)
    reselected = KernelDensityEstimator(
        bandwidth=bandwidth, weights=weights[keep], **options,
    ).fit(X[keep])
    preserved = KernelDensityEstimator(
        bandwidth=original.bandwidth_factor_, weights=weights[keep], **options,
    ).fit(X[keep])
    assert preserved.bandwidth_info_ is None  # No second selector search.
    assert preserved.bandwidth_factor_ == original.bandwidth_factor_
    assert_allclose(preserved.covariance_, original.covariance_, rtol=2e-15)
    assert_allclose(preserved.pdf([0.0]), original.pdf([0.0]), rtol=2e-15)
    if bandwidth in (0.4, "scott", "silverman"):
        assert_allclose(reselected.bandwidth_factor_, original.bandwidth_factor_, rtol=2e-15)
    else:
        assert not np.isclose(reselected.bandwidth_factor_, original.bandwidth_factor_, rtol=0.01)
    # Removing zero terms also permits finite tail log density at the fixed H.
    query = np.array([0.0, 1e6])
    variance = float(preserved.covariance_[0, 0])
    expected = logsumexp(-0.5 * (query[:, None] - X[keep])**2 / variance, axis=1)
    expected -= np.log(keep.sum()) + 0.5 * np.log(2 * np.pi * variance)
    actual = preserved.logpdf(query)
    assert np.isfinite(actual).all()
    assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_zero_weight_guidance_distinguishes_reselection_and_preserved_factor(language):
    for relative in ("models/nonparametric.md", "reference/survival-smoothing-api.md"):
        text = (ROOT / "docs" / language / relative).read_text()
        assert "bandwidth=original.bandwidth_factor_" in text
        for selector in ("nrd", "nrd0", "ucv", "bcv", "sj-ste", "sj-dpi"):
            assert selector in text
        assert "preserves the intended fit" not in text
        assert "删除零权重行不会改变预期估计量" not in text
    for obj in (KernelDensityEstimator, KernelDensityEstimator.logpdf):
        text = inspect.getdoc(obj)
        assert "bandwidth=original.bandwidth_factor_" in text
        assert "Rerunning a string" in text
