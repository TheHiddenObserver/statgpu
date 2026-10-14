"""Scientific fixtures and bilingual teaching assets, without runtime changes.

Numerical/render checks need the optional plotting dependencies documented in
``dev/figures/README.md``. Asset/reference checks always run.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLUGS = {'kmeans':'kmeans', 'gaussian-mixture':'gmm', 'dbscan':'dbscan',
         'agglomerative-clustering':'agglomerative', 'minibatch-kmeans':'minibatch-kmeans'}


def generators():
    pytest.importorskip('matplotlib')
    from dev.figures.generate_clustering import GENERATORS
    return GENERATORS


@pytest.mark.parametrize('lang', ['en', 'cn'])
@pytest.mark.parametrize('page,asset', SLUGS.items())
def test_bilingual_images_exist_have_alt_and_precede_example(lang, page, asset):
    text = (ROOT / f'docs/{lang}/unsupervised/{page}.md').read_text()
    pattern = r'!\[([^\]]+)\]\((../../assets/unsupervised/' + asset + '-' + lang + r'\.svg)\)'
    match = re.search(pattern, text)
    assert match and len(match.group(1)) >= 30
    example = text.index(f'<!-- learner-example: {page} -->')
    if page == 'agglomerative-clustering':
        # The tree is an intuition aid; its technical height caption comes
        # after the linkage definitions (protected by opening-order tests).
        assert match.start() < text.index('$$') < example
    else:
        assert text.index('$$') < match.start() < example
    path = ROOT / f'docs/assets/unsupervised/{asset}-{lang}.svg'
    svg = ET.parse(path).getroot()
    assert svg.tag.endswith('svg')
    assert svg.attrib['viewBox'] == '0 0 720 316.8'
    assert not any(x.tag.endswith(('script', 'image', 'foreignObject')) for x in svg.iter())
    assert 'dc:date' not in path.read_text()


def test_kmeans_update_is_analytic_fixed_assignment_improvement():
    generators()
    from dev.figures.clustering_kmeans import build_kmeans_data
    d = build_kmeans_data()
    np.testing.assert_allclose(d['old'], [[0,0],[4,2]])
    np.testing.assert_allclose(d['new'], [[.5,.5],[3.75,1]])
    assert d['before'] == 11 and d['after'] == 4.75
    # Perturbing an updated center increases error for these fixed assignments.
    moved = d['new'].copy(); moved[0,0] += .2
    assert ((d['X'] - moved[d['labels']])**2).sum() > d['after']


@pytest.mark.parametrize('kind', ['gmm','dbscan','agglomerative','minibatch'])
def test_actual_fit_outputs_match_independent_scientific_checks(kind):
    generators()
    from dev.figures import clustering_density as density, clustering_hierarchy_stream as hs
    module = density if kind in ('gmm','dbscan') else hs
    data = getattr(module, 'build_' + kind + '_data')()
    assert getattr(module, 'validate_' + kind + '_data')(data)


@pytest.mark.parametrize('kind,key', [('gmm','responsibilities'),('dbscan','labels'),('agglomerative','heights'),('minibatch','centers')])
def test_scientific_validators_reject_corrupted_figure_data(kind, key):
    generators()
    from dev.figures import clustering_density as density, clustering_hierarchy_stream as hs
    module = density if kind in ('gmm','dbscan') else hs
    data = getattr(module, 'build_' + kind + '_data')()
    data[key] = data[key].copy()
    data[key].flat[0] += 1
    with pytest.raises(AssertionError):
        getattr(module, 'validate_' + kind + '_data')(data)


def test_all_bilingual_svg_generators_are_byte_deterministic(tmp_path):
    fns = generators()
    from dev.figures.clustering_common import FONT
    if not FONT.exists():
        pytest.skip('Noto Sans CJK Regular required for Chinese SVG regeneration')
    for lang in ('en','cn'):
        for fn in fns:
            first = fn(tmp_path/'first',lang).read_bytes()
            second = fn(tmp_path/'second',lang).read_bytes()
            assert first == second, (fn.__name__, lang)
