"""Keep independent intuition ahead of comparisons and implementation details.

These checks target the prerequisite mistakes found in the opening audit, not
one universal section template. Numerical, API and figure tests remain separate.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLUGS = ('agglomerative-clustering', 'minibatch-nmf', 'tsne',
         'incremental-pca', 'nmf', 'truncated-svd', 'umap')


def page(lang, slug):
    return (ROOT / f'docs/{lang}/unsupervised/{slug}.md').read_text()


def overview(text):
    return re.split(r'^## ', text, flags=re.M)[1].split('\n', 1)[1]


@pytest.mark.parametrize('lang', ['en', 'cn'])
def test_hierarchy_intuition_picture_math_comparison_and_example_order(lang):
    text = page(lang, 'agglomerative-clustering')
    intro = overview(text)
    assert 'KMeans' not in intro and 'DBSCAN' not in intro
    assert ('cannot be undone' if lang == 'en' else '不会撤销') in intro
    assert ('nested grouping' if lang == 'en' else '嵌套分组') in intro
    picture = text.index('![One Ward' if lang == 'en' else '![六个编号')
    reader = text.index('Read the numbered points' if lang == 'en' else '下图的编号点')
    rules = text.index('## Linkage and merging criteria' if lang == 'en' else '## 链接准则与合并规则')
    selection = text.index('## When to use it' if lang == 'en' else '## 何时使用')
    example = text.index('<!-- learner-example: agglomerative-clustering -->')
    assert reader < picture < rules < selection < example
    assert '(1, 2)' in text[reader:picture] and '(5, 6)' in text[reader:picture]
    # This checks prose only: the immutable SVG itself has technical labels.
    # The reader guide must bridge those labels rather than hide them.
    guide = text[reader:picture]
    assert '$' not in text[:picture]
    assert all(label in guide for label in ('`h=1.22`', '`k=3`', '`h=3.74`', '`k=2`'))
    explanations = (('height of the horizontal cut', 'number of groups',
                     'branches crossing the cut', 'spread around the group means',
                     'linkage section below', 'square-root scale') if lang == 'en' else
                    ('水平切线的高度', '切分后留下的组数', '切线穿过多少条',
                     '相对组均值的分散程度', '后面的连接准则', '平方根尺度'))
    assert all(explanation in guide for explanation in explanations)
    for formula in (r'\min_{x \in A', r'\max_{x \in A',
                    r'\frac{1}{|A||B|}', r'\Delta(A, B)'):
        assert rules < text.index(formula) < selection
    assert text.index(r'\Delta(A, B)') < text.index(r'\sqrt{2\Delta}') < selection
    assert text.index(r'\Delta(A, B)') < text.index('ΔSSE') < selection
    assert selection < text.index('KMeans') < example
    for implementation in ('Lance–Williams', 'SciPy hierarchy routines' if lang == 'en' else 'CPU 路径调用 SciPy'):
        assert text.index(implementation) > example


@pytest.mark.parametrize('lang', ['en', 'cn'])
@pytest.mark.parametrize('slug', ['nmf', 'tsne', 'umap'])
def test_other_models_are_choices_not_opening_prerequisites(lang, slug):
    text = page(lang, slug)
    assert 'PCA' not in overview(text)
    choice = text.index('## When to use it' if lang == 'en' else '## 何时使用')
    assert choice < text.index('PCA') < text.index(f'<!-- learner-example: {slug} -->')


@pytest.mark.parametrize('lang', ['en', 'cn'])
def test_embedding_openings_defer_distribution_and_neighbor_engine_details(lang):
    tsne = page(lang, 'tsne')
    early = overview(tsne)
    for term in ('Student', 'Gaussian', '高斯', '重尾', 'crowding', '拥挤'):
        assert term not in early
    assert 'NNDescent' not in overview(page(lang, 'umap'))


@pytest.mark.parametrize('lang', ['en', 'cn'])
@pytest.mark.parametrize('slug', SLUGS)
def test_core_math_precedes_complete_example_and_api_is_linked(lang, slug):
    text = page(lang, slug)
    before = text[:text.index(f'<!-- learner-example: {slug} -->')]
    assert '$$' in before
    assert '](api-reference.md#' in text


@pytest.mark.parametrize('lang', ['en', 'cn'])
def test_minibatch_tables_and_worked_reconstruction_precede_batch_learning(lang):
    text = page(lang, 'minibatch-nmf')
    intro = overview(text)
    worked = intro.index('<!-- worked-example: nonnegative-spectra -->')
    # Concrete table meanings and multiplication precede the streaming update.
    # Arithmetic, shapes, scale ambiguity, and stale coefficients are exercised
    # in test_plain_language_unsupervised_examples, not inferred from word order.
    assert all(shape in intro[:worked] for shape in ('(n, p)', '(k, p)', '(n, k)'))
    assert worked < intro.index('`partial_fit`')
    assert intro.index('`H`') < worked and intro.index('`W`') < worked
    assert intro.index('2*A + 0.5*B') < intro.index('dictionary' if lang == 'en' else '字典')
    assert ('compact statistics' if lang == 'en' else '紧凑统计量') not in intro


@pytest.mark.parametrize('lang', ['en', 'cn'])
def test_linear_representations_explain_conversion_before_naming_directions(lang):
    ipca = overview(page(lang, 'incremental-pca'))
    svd = overview(page(lang, 'truncated-svd'))
    # A reader can calculate one coordinate before meeting the formal name.
    assert ipca.index('2*0.6 - 1*0.8 = 0.4') < ipca.index(
        'called directions' if lang == 'en' else '称为一个方向')
    if lang == 'en':
        assert svd.index('scaling these patterns') < svd.index('singular directions')
    else:
        assert svd.index('相加') < svd.index('奇异方向')
