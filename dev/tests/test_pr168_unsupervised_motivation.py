"""Protect distinct learner questions before unsupervised mechanics and examples.

These checks preserve a few substantive distinctions, not complete paragraphs or
one mandatory prose template. Existing numerical and API tests remain unchanged.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Each entry names a simple baseline, the method's idea, and an interpretation
# boundary. EN/CN assertions deliberately use independently natural phrasing.
# Revised quantitative examples are also executed in
# test_plain_language_unsupervised_examples; these are only placement guards.
CONCEPTS = {
    'kmeans': {
        'en': ('One overall average', 'nearest center', 'even when the data do not contain distinct populations'),
        'cn': ('只看总体均值', '最近的中心', '没有截然分开的真实群体'),
    },
    'minibatch-kmeans': {
        'en': ('small subset', 'successive batches', 'no forgetting mechanism'),
        'cn': ('只抽取一个小样本', '后续批次', '没有遗忘机制'),
    },
    'agglomerative-clustering': {
        'en': ('single flat partition', 'nested grouping', 'not a significance test'),
        'cn': ('一次平面划分', '嵌套分组', '不是显著性检验'),
    },
    'dbscan': {
        'en': ('nearest-center partition', 'density connectivity', 'does not establish a measurement error'),
        'cn': ('按最近中心分组', '局部密度连通性', '不能据此认定'),
    },
    'gaussian-mixture': {
        'en': ('One Gaussian distribution', '0.7 for component A and 0.3 for B', 'not a noise detector'),
        'cn': ('只用一个高斯分布', '软归属', '不等于噪声检测'),
    },
    'pca': {
        'en': ('Dropping correlated columns', 'weighted combinations', 'not a search for the causes'),
        'cn': ('手动删除相关列', '加权组合', '并不识别变化的成因'),
    },
    'incremental-pca': {
        'en': ('every file separately', 'difference between the old and new means', 'information loss'),
        'cn': ('分别对每天的数据', '旧均值与新批次均值', '损失信息'),
    },
    'truncated-svd': {
        'en': ('Selecting only a few common terms', "does not first subtract each term's average weight", 'need not correspond to a topic you can name'),
        'cn': ('只保留几个常见词', '不先减去每列的平均词权重', '不一定对应可以命名的主题'),
    },
    'nmf': {
        'en': ('assigning each sample to one type', '2*A + 0.5*B', 'not the number of basic spectra or probabilities'),
        'cn': ('把每份样品硬分成某一种类型', '2*A + 0.5*B', '不是基本光谱的个数，也不是概率'),
    },
    'minibatch-nmf': {
        'en': ('current A and B', 'after that pass', 'Earlier coefficients are not updated automatically'),
        'cn': ('当前 A 和 B', '整轮结束后才更新', '旧系数不会自动更新'),
    },
    'tsne': {
        'en': ('scatter plot of two columns', 'neighbor affinities', 'does not establish a distinct population'),
        'cn': ('只画两列特征', '近邻亲和度', '不能证明存在一个独立群体'),
    },
    'umap': {
        'en': ('Plotting two sensors', 'attraction along graph edges', 'cannot establish time order'),
        'cn': ('只画两个传感器', '沿图边的吸引力', '不能确定时间先后'),
    },
}


def page(language, slug):
    return (ROOT / 'docs' / language / 'unsupervised' / (slug + '.md')).read_text(encoding='utf-8')


def introduction(text, language):
    # Import-only fences are not executable learner examples.
    end = '## Path\n' if language == 'en' else '## 导入路径\n'
    return text.split(end, 1)[0]


@pytest.mark.parametrize('language', ('en', 'cn'))
@pytest.mark.parametrize('slug', CONCEPTS)
def test_distinct_problem_intuition_and_boundary_precede_mechanics(language, slug):
    intro = introduction(page(language, slug), language)
    for concept in CONCEPTS[slug][language]:
        assert concept in intro, (slug, language, concept)
    assert re.search(r'\]\([a-z-]+\.md\)', intro), 'Provide a nearby method choice'
    for configuration in ('device="cpu"', 'device="cuda"', 'n_jobs=', 'random_state='):
        assert configuration not in intro


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_index_starts_from_distinct_questions_without_labels(language):
    text = page(language, 'README')
    overview = text.split('## ')[1]
    concepts = (('without a known target label', '**Representation:**', '**Grouping:**',
                 '**Visualization:**', 'information outside the fit') if language == 'en'
                else ('没有已知目标标签', '**表示数据：**', '**组织观测：**',
                      '**探索图形：**', '拟合之外的信息'))
    for concept in concepts:
        assert concept in overview
    assert 'NNDescent' in overview  # preserve internal/export distinction


@pytest.mark.parametrize('language', ('en', 'cn'))
@pytest.mark.parametrize('slug', tuple(CONCEPTS))
def test_necessary_mathematics_still_precedes_the_learner_example(language, slug):
    text = page(language, slug)
    marker = '<!-- learner-example: ' + slug + ' -->'
    assert marker in text
    before = text.split(marker, 1)[0]
    assert '$$' in before, (language, slug)
    assert '](api-reference.md#' in text


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_umap_math_bridge_is_before_the_example_and_does_not_promise_exact_objective(language):
    text = page(language, 'umap')
    heading = '## From neighborhoods to a map' if language == 'en' else '## 从近邻关系到低维图'
    start = text.index(heading)
    stop = text.index('<!-- learner-example: umap -->')
    assert start < stop
    bridge = text[start:stop]
    assert 'q_{ij}' in bridge and 'a,b>0' in bridge
    assert ('rather than exactly differentiating' if language == 'en' else '并非该目标的精确梯度') in bridge


def test_bilingual_revision_dates_match():
    for slug in ('README',) + tuple(CONCEPTS):
        en = re.search(r'^> Last updated: (\d{4}-\d{2}-\d{2})$', page('en', slug), re.M)
        cn = re.search(r'^> 最后更新：\s*(\d{4}-\d{2}-\d{2})$', page('cn', slug), re.M)
        assert en and cn
        assert en.group(1) == cn.group(1), slug
