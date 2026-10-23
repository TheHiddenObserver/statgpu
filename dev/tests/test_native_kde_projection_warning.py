"""Public warning aggregation for ambiguous native principal eigenspaces."""
import warnings
import json
import numpy as np
import pytest

from statgpu.nonparametric.kernel_smoothing import kde_bootstrap_confidence_interval


@pytest.mark.parametrize('backend', ['torch-cpu', 'torch-cuda', 'cupy'])
def test_public_bootstrap_aggregates_ambiguous_projection_warnings(backend, request):
    x = np.tile(np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]), (10, 1))
    q = np.zeros((1, 2))
    if backend.startswith('torch'):
        torch = pytest.importorskip('torch')
        device = 'cuda:0' if backend == 'torch-cuda' else 'cpu'
        if device != 'cpu' and not torch.cuda.is_available():
            pytest.skip('physical Torch CUDA unavailable')
        x, q = torch.asarray(x, device=device), torch.asarray(q, device=device)
        name = 'torch'
    else:
        cp = pytest.importorskip('cupy')
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('physical CuPy CUDA unavailable')
        except cp.cuda.runtime.CUDARuntimeError as error:
            pytest.skip(str(error))
        x, q, name = cp.asarray(x), cp.asarray(q), 'cupy'
    reports = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = kde_bootstrap_confidence_interval(
                x, q, bandwidth='nrd', backend=name, n_resamples=3,
                random_state=12, return_bootstrap_samples=True)
        relevant = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(relevant) == 1, [str(w.message) for w in caught]
        assert 'principal' in str(relevant[0].message).lower()
        report = result.metadata['projection_diagnostics']
        assert report['status'] == 'ambiguous_principal_eigenspace'
        assert 1 <= report['count'] <= 4
        assert report['exact_degeneracy_count'] >= 1
        assert result.bootstrap_samples.shape == (3, 1)
        assert np.isfinite(result.bootstrap_samples).all()
        reports.append(report)
    assert reports[0] == reports[1], 'warning counters must reset between public calls'
    request.node.user_properties.extend([
        ('backend', backend),
        ('projection_warning', str(relevant[0].message)),
        ('projection_diagnostics', json.dumps(reports[0], sort_keys=True)),
    ])


def test_numeric_native_bootstrap_has_no_pca_warning():
    torch = pytest.importorskip('torch')
    x = torch.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = kde_bootstrap_confidence_interval(
            x, torch.zeros((1, 2)), bandwidth=.5, backend='torch', n_resamples=2,
            random_state=12)
    assert not caught
    assert result.metadata['projection_diagnostics']['count'] == 0
    assert result.metadata['projection_diagnostics']['status'] == 'none'


def test_legacy_numpy_bootstrap_emits_no_new_pca_warning():
    x = np.tile(np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]), (10, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = kde_bootstrap_confidence_interval(
            x, np.zeros((1, 2)), bandwidth='nrd', backend='numpy',
            n_resamples=3, random_state=12)
    assert not caught
    assert np.isfinite(result.estimate).all()


def test_public_bootstrap_counts_initial_fit_and_all_ambiguous_refits(monkeypatch):
    torch = pytest.importorskip('torch')
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    x = torch.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float64)
    monkeypatch.setattr(native, '_draw_indices',
                        lambda cdf, rng, n, xp: xp.arange(n, device=cdf.device))
    with pytest.warns(RuntimeWarning, match='4 ambiguous projection') as caught:
        result = kde_bootstrap_confidence_interval(
            x, torch.zeros((1, 2)), bandwidth='nrd', backend='torch',
            n_resamples=3, random_state=12)
    assert len(caught) == 1
    assert result.metadata['projection_diagnostics']['count'] == 4
    assert result.metadata['projection_diagnostics']['exact_degeneracy_count'] == 4
