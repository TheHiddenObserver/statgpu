"""Independently verify complete kernel API docs and current safe workarounds."""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline

from statgpu.nonparametric import kernel_methods as kernels
from statgpu.nonparametric.splines import SplineTransformer, bspline_basis

ROOT = Path(__file__).resolve().parents[2]


def _signature(obj):
    signature = inspect.signature(obj)
    parameters = []
    for parameter in signature.parameters.values():
        if parameter.name == 'self' or parameter.name.startswith('_'):
            continue
        default = parameter.default
        if hasattr(default, 'value'):
            default = default.value
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty,
                                            default=default))
    return str(signature.replace(parameters=parameters,
                                 return_annotation=inspect.Signature.empty))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_kernel_page_has_every_export_signature_and_default(language):
    text = (ROOT / f'docs/{language}/models/kernel-methods.md').read_text()
    lines = '\n'.join(re.findall(r'```text\n(.*?)```', text, re.DOTALL)).splitlines()
    for name in kernels.__all__:
        assert name + _signature(getattr(kernels, name)) in lines, name
    for cls in (kernels.KernelRidge, kernels.KernelRidgeCV,
                kernels.KernelPCA, kernels.Nystroem):
        methods = ('fit', 'predict', 'score') if 'Ridge' in cls.__name__ else (
            'fit', 'transform', 'fit_transform', 'predict')
        for name in methods:
            assert '`' + name + _signature(getattr(cls, name)) + '`' in text
    for field in ('dual_coef_', 'X_fit_', 'n_features_in_', 'alpha_',
                  'best_score_', 'cv_results_', 'estimator_', 'lambdas_',
                  'alphas_', 'n_samples_', 'components_', 'component_indices_',
                  'normalization_', 'eigenvalues_'):
        assert '`' + field + '`' in text
    assert 'kernel_params={"gamma": value}' in text
    assert '1e9' in text
    assert '1e-10' in text
    assert 'NumPy CPU' in text
    assert 'maintained' not in text and '维护的估计器路径' not in text


def test_kernel_shapes_and_scores_match_complete_api():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(16, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=16)
    for target in (y, y[:, None], np.column_stack([y, 2 * y])):
        n_targets = 1 if target.ndim == 1 else target.shape[1]
        direct = kernels.KernelRidge(device='cpu').fit(X, target)
        selected = kernels.KernelRidgeCV(alphas=[0.1, 1.0], cv=2,
                                        random_state=7, device='cpu').fit(X, target)
        shape = (4,) if n_targets == 1 else (4, n_targets)
        for model in (direct, selected):
            assert model.dual_coef_.shape == (16, n_targets)
            assert model.X_fit_.shape == X.shape
            assert model.predict(X[:4]).shape == shape
            assert isinstance(model.score(X, target), float)
        assert selected.cv_results_['mse_table'].shape == (2, 2, n_targets)
        assert selected.cv_results_['mean_mse'].shape == (2, n_targets)
        assert selected.estimator_.n_features_in_ == 3
    kpca = kernels.KernelPCA(n_components=30, device='cpu').fit(X)
    count = len(kpca.lambdas_)
    assert count < 16  # The centered kernel has a null direction.
    assert kpca.alphas_.shape == (16, count)
    assert_allclose(kpca.predict(X), kpca.transform(X))
    assert kpca.transform(X[:4]).shape == (4, count)
    nystroem = kernels.Nystroem(n_components=30, random_state=7,
                               device='cpu').fit(X)
    assert nystroem.components_.shape == (16, 3)
    assert nystroem.component_indices_.shape == (16,)
    assert nystroem.normalization_.shape == (16, 16)
    assert nystroem.eigenvalues_.shape == (16,)
    assert nystroem.transform(X[:4]).shape == (4, 16)


@pytest.mark.parametrize('metric', ['chi2', 'chi-squared'])
def test_chi2_gamma_dictionary_workaround_matches_analytic_system(metric):
    X = np.array([[0.2, 0.5], [0.4, 1.5], [1.0, 2.0], [2.0, 0.8]])
    y = np.array([0.0, 1.0, 0.5, 2.0])
    K = kernels.pairwise_kernels(X, metric=metric, gamma=7.0, xp=np)
    model = kernels.KernelRidge(kernel=metric, kernel_params={'gamma': 7.0},
                                alpha=1.0, device='cpu').fit(X, y)
    assert_allclose(model.dual_coef_.ravel(), np.linalg.solve(K + np.eye(4), y))


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='KernelRidge/CV omit constructor gamma for chi-squared kernels')
@pytest.mark.parametrize('cls', [kernels.KernelRidge, kernels.KernelRidgeCV])
@pytest.mark.parametrize('metric', ['chi2', 'chi-squared'])
def test_chi2_constructor_gamma_should_match_dictionary(cls, metric):
    X = np.array([[0.2, 0.5], [0.4, 1.5], [1.0, 2.0], [2.0, 0.8], [3.0, 3.0]])
    y = np.array([0.0, 1.0, 0.5, 2.0, 1.0])
    options = {'kernel': metric, 'device': 'cpu'}
    if cls is kernels.KernelRidgeCV:
        options.update(alphas=[0.1, 1.0, 10.0], cv=2, random_state=0)
    requested = cls(gamma=7.0, **options).fit(X, y)
    reference = cls(kernel_params={'gamma': 7.0}, **options).fit(X, y)
    if cls is kernels.KernelRidgeCV:
        assert_allclose(requested.cv_results_['mean_mse'],
                        reference.cv_results_['mean_mse'])
    assert_allclose(requested.predict(X), reference.predict(X))


def test_rbf_centering_workaround_preserves_pairwise_model():
    X = np.arange(6.0)[:, None] + 1e9
    center = X.mean(axis=0)
    queries = np.array([[1e9 + 0.5], [1e9 + 3.5]])
    expected = np.exp(-0.5 * np.sum((queries[:, None] - X[None, :]) ** 2, axis=2))
    assert_allclose(kernels.rbf_kernel(queries - center, X - center, gamma=0.5), expected)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='RBF norm/dot-product distances lose common-translation invariance')
def test_rbf_kernel_should_be_translation_invariant():
    X = np.arange(6.0)[:, None]
    expected = np.exp(-0.5 * np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    assert_allclose(kernels.rbf_kernel(X + 1e9, gamma=0.5), expected, atol=1e-12)


@pytest.mark.parametrize('degree', [0, 1, 2, 3, 5])
def test_repeated_boundary_knot_basis_matches_analytic_recurrence(degree):
    knots = np.array([0.2, 0.5, 0.8])
    x = np.r_[0.0, knots, np.linspace(0.0, 1.0, 71), 1.0]
    augmented = np.r_[np.zeros(degree + 1), knots, np.ones(degree + 1)]
    width = len(knots) + degree + 1
    reference = BSpline(augmented, np.eye(width), degree)(x)
    actual = bspline_basis(x, knots, degree=degree, xp=np)
    assert_allclose(actual, reference, atol=1e-12)
    assert_allclose(actual.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_spline_complete_methods_and_zero_denominator_convention(language):
    text = (ROOT / f'docs/{language}/models/splines.md').read_text()
    assert r'\frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}}' in text
    assert '0/0 = 0' not in text
    for name in ('fit', 'fit_transform', 'transform', 'predict',
                 'get_feature_names_out', 'get_params', 'set_params'):
        assert '`' + name + _signature(getattr(SplineTransformer, name)) + '`' in text


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Torch chi-squared kernel floors valid small positive denominators')
def test_torch_cpu_chi2_should_preserve_small_positive_denominators():
    torch = pytest.importorskip('torch')
    X = torch.tensor([[1e-12]], dtype=torch.float64)
    Y = torch.zeros((1, 1), dtype=torch.float64)
    actual = kernels.chi2_kernel(X, Y, gamma=1e12, xp=torch)
    assert_allclose(actual.numpy(), [[np.exp(-1.0)]], rtol=1e-12, atol=1e-12)
