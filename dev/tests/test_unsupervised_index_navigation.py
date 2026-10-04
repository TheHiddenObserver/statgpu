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
            assert (index.parent / target).resolve().is_file(), target
        routes.append({t for t in targets if not t.startswith('../')})
    assert routes[0] == routes[1] == set(_ESTIMATORS.values())


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
