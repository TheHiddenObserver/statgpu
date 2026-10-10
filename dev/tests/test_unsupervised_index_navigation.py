"""Keep the learner index connected to every maintained estimator guide."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ESTIMATORS = {
    'PCA': 'pca.md',
    'KMeans': 'kmeans.md',
    'DBSCAN': 'dbscan.md',
    'GaussianMixture': 'gaussian-mixture.md',
    'NMF': 'nmf.md',
    'AgglomerativeClustering': 'agglomerative-clustering.md',
    'TruncatedSVD': 'truncated-svd.md',
    'MiniBatchKMeans': 'minibatch-kmeans.md',
    'IncrementalPCA': 'incremental-pca.md',
    'MiniBatchNMF': 'minibatch-nmf.md',
    'UMAP': 'umap.md',
    'TSNE': 'tsne.md',
}


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_index_choice_links_cover_real_public_estimators(language):
    from statgpu import unsupervised

    index = _ROOT / f'docs/{language}/unsupervised/README.md'
    text = index.read_text(encoding='utf-8')
    # The choice table must route to every model, not just repeat names.
    chooser = text.split('## ')[2]
    links = dict(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', chooser))
    assert {name: links.get(name) for name in _ESTIMATORS} == _ESTIMATORS
    for name, target in _ESTIMATORS.items():
        estimator = getattr(unsupervised, name)
        assert callable(estimator)
        assert callable(estimator.fit)
        assert (index.parent / target).is_file()
        assert f'| `{name}` |' in text


def test_index_relative_links_resolve_and_match_across_languages():
    routes = []
    for language in ('en', 'cn'):
        index = _ROOT / f'docs/{language}/unsupervised/README.md'
        targets = re.findall(r'\]\(([^)]+)\)', index.read_text(encoding='utf-8'))
        for target in targets:
            assert (index.parent / target.split("#", 1)[0]).resolve().is_file(), target
        routes.append({t.split("#", 1)[0] for t in targets if not t.startswith('../')})
    assert routes[0] == routes[1] == set(_ESTIMATORS.values()) | {'api-reference.md'}


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_index_cpu_workflow_reconstructs_held_out_rows(language):
    import numpy as np

    page = _ROOT / f'docs/{language}/unsupervised/README.md'
    text = page.read_text(encoding='utf-8')
    match = re.search(
        r'<!-- learner-example: unsupervised-pca -->\s*```python\n(.*?)```',
        text, flags=re.DOTALL,
    )
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), str(page), 'exec'), namespace)  # noqa: S102
    model = namespace['pca']
    assert namespace['Z_train'].shape == (90, 2)
    assert namespace['Z_test'].shape == (30, 2)
    assert namespace['X_reconstructed'].shape == (30, 3)
    np.testing.assert_allclose(model.mean_, namespace['X_train'].mean(axis=0))
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(2), atol=1e-12)
    assert model.explained_variance_ratio_.sum() > 0.99
    assert np.mean((namespace['X_test'] - namespace['X_reconstructed']) ** 2) < 0.002


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_index_gmm_covariance_literals_are_accepted_by_public_api(language):
    import numpy as np

    from statgpu.unsupervised import GaussianMixture

    text = (_ROOT / f'docs/{language}/unsupervised/README.md').read_text(encoding='utf-8')
    row = next(line for line in text.splitlines() if line.startswith('- [GaussianMixture]'))
    documented = re.findall(r'`([^`]+)`', row)
    assert set(documented) == {'diag', 'spherical', 'tied', 'full'}
    rng = np.random.default_rng(17)
    X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
    for covariance_type in documented:
        model = GaussianMixture(
            n_components=2, covariance_type=covariance_type,
            device='cpu', random_state=17,
        ).fit(X)
        assert np.all(np.isfinite(model.predict_proba(X)))


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_index_discloses_dbscan_runtime_neighbor_dependency(language, monkeypatch):
    import numpy as np

    from statgpu.unsupervised import DBSCAN

    neighbors = pytest.importorskip('sklearn.neighbors')
    original = neighbors.NearestNeighbors
    calls = []

    def track_neighbors(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(neighbors, 'NearestNeighbors', track_neighbors)
    X = np.random.default_rng(17).normal(size=(20, 13))
    model = DBSCAN(eps=2.0, min_samples=2, device='cpu').fit(X)
    assert calls and calls[0]['metric'] == 'euclidean'
    assert model.labels_.shape == (20,)
    text = (_ROOT / f'docs/{language}/unsupervised/README.md').read_text(encoding='utf-8')
    assert '`NearestNeighbors`' in text and 'scikit-learn' in text
    assert 'External packages serve as validation or benchmark references;' not in text
    assert '这些外部软件包仅作为验证或基准测试的对照' not in text


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_model_overview_distinguishes_internal_neighbor_search_from_exports(language):
    from statgpu import unsupervised

    text = (_ROOT / f'docs/{language}/models/unsupervised.md').read_text(encoding='utf-8')
    assert set(unsupervised.__all__) == set(_ESTIMATORS)
    overview = text.split('## ')[1]
    if language == 'en':
        assert 'not a separately exported estimator' in overview
        assert 'manifold embedding, and approximate nearest neighbor search' not in overview
    else:
        assert '不是单独导出的估计器' in overview
        assert '和近似最近邻搜索等估计器' not in overview
        assert 'Gaussian mixture' not in text
        assert '簇内平方欧氏距离之和' in text
    assert 'NNDescent' in text
    assert '../unsupervised/api-reference.md' in text
