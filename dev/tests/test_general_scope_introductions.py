"""Keep general method explanations independent of their application examples.

These bounded prose checks do not certify comprehension or statistical runtime
correctness. Numerical examples and their mutation controls remain separately
executed by test_plain_language_*; no application word is globally prohibited.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


def page(language, path):
    return (ROOT / 'docs' / language / (path + '.md')).read_text()


def first_paragraph(text):
    section = re.split(r'^## .+\n', text, maxsplit=1, flags=re.M)[1]
    return section.strip().split('\n\n')[0]


# Each method has distinct input/process/output content, not a universal template.
CONCEPTS = {
    'nmf': (('nonnegative data table', 'two nonnegative tables', '`W @ H`'),
            ('非负数据表', '两张非负表', '`W @ H`')),
    'minibatch-nmf': (('nonnegative data table', 'two nonnegative tables', '`W @ H`'),
                      ('非负数据表', '两张非负表', '`W @ H`')),
    'incremental-pca': (('numeric observations', 'subtracting their means', 'batches', 'reconstruct'),
                        ('数值观测', '均值', '分批', '重构')),
    'truncated-svd': (('numeric matrix', 'without first subtracting column means', 'positive or negative'),
                      ('数值矩阵', '不先减去各列均值', '正数或负数')),
    'gaussian-mixture': (('numeric observations', 'Gaussian distributions', 'density', 'membership'),
                         ('数值观测', '高斯分布', '密度', '不确定性')),
    'pca': (('numeric features', 'variation around their means', 'reconstruct'),
            ('数值特征', '均值', '重构')),
    'kmeans': (('numeric observations', 'nearest center', 'squared distance'),
               ('数值观测', '最近的中心', '平方距离')),
    'dbscan': (('neighborhoods', 'border points', 'noise', 'cluster count'),
               ('邻域', '边界点', '噪声', '簇数')),
    'tsne': (('numeric observations', 'low-dimensional map', 'local similarities'),
             ('数值观测', '低维图', '局部相似')),
    'umap': (('numeric observations', 'weighted graph', 'cluster labels'),
             ('数值观测', '加权图', '聚类标签')),
}


def validate_general_opening(text, language, slug):
    paragraph = first_paragraph(text)
    required = CONCEPTS[slug][language == 'cn']
    assert all(term in paragraph for term in required), (slug, language)


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('slug', CONCEPTS)
def test_general_method_definition_survives_without_the_application(language, slug):
    text = page(language, 'unsupervised/' + slug)
    validate_general_opening(text, language, slug)
    opening = first_paragraph(text)
    # Deleting the standalone definition, or replacing it with a narrow example,
    # must not pass merely because a generic term appears somewhere later.
    application = ('Suppose the input consists of spectra from five wavelengths.'
                   if language == 'en' else '假设输入是五个波长的光谱测量。')
    for mutation in (text.replace(opening, '', 1),
                     text.replace(opening, application, 1)):
        with pytest.raises(AssertionError):
            validate_general_opening(mutation, language, slug)


def validate_nmf_mapping(text, language):
    marker = ('### Example: two basic spectra' if language == 'en'
              else '### 示例：两个基本光谱')
    general, example = text.split(marker, 1)
    assert all(shape in general for shape in ('(n, p)', '(k, p)', '(n, k)'))
    h_meaning = ('`H` contains one basic pattern per row' if language == 'en'
                 else '`H` 的每一行是一个基本模式')
    w_meaning = ('`W` contains one row per sample' if language == 'en'
                 else '`W` 的每一行对应一个样本')
    assert h_meaning in general and w_meaning in general
    assert ('need not sum to one' if language == 'en' else '不必加起来等于 1') in general
    assert ('need not be unique' if language == 'en' else '不一定唯一') in general
    assert 'nonnegative-spectra' in example and '2*A + 0.5*B' in example
    assert ('five wavelengths' if language == 'en' else '5 个波长') in example
    return h_meaning, w_meaning, marker


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('slug', ['nmf', 'minibatch-nmf'])
def test_nmf_table_roles_and_labeled_example_are_not_interchangeable(language, slug):
    text = page(language, 'unsupervised/' + slug)
    h_meaning, w_meaning, marker = validate_nmf_mapping(text, language)
    mutations = (
        text.replace(h_meaning, h_meaning.replace('`H`', '`W`'), 1),
        text.replace(w_meaning, w_meaning.replace('`W`', '`H`'), 1),
        # Keeping the entire example cannot substitute for the general tables.
        text[:text.index('## ')] + '## Overview\n\n' + marker + text.split(marker, 1)[1],
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_nmf_mapping(mutation, language)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_true_specialized_data_requirements_are_retained(language):
    cox = page(language, 'models/coxph')
    panel = page(language, 'panel/random-effects')
    nmf = page(language, 'unsupervised/nmf')
    assert '`event=0`' in cox and '`event=1`' in cox
    assert ('censoring is independent' if language == 'en' else '删失') in cox
    assert ('`entity_ids` is required' if language == 'en' else '`entity_ids` 必须提供') in panel
    assert ('conditional mean zero' if language == 'en' else '条件均值必须为零') in panel
    assert ('nonnegative' if language == 'en' else '非负') in nmf
    assert 'device="cuda"' in nmf and 'device="torch"' in nmf
