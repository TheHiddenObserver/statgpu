"""Keep custom-fold principles, configuration advice and API limits distinct."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RIDGE_SUPPORT_LINK = '../reference/linear-model-api.md#custom-ridgecv-training-subsets'


def _read(language, page):
    return (ROOT / f'docs/{language}/{page}.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('guide', ['cross-validation', 'cross-validation-design'])
def test_shared_cv_guides_link_canonical_custom_split_support(language, guide):
    text = _read(language, f'guides/{guide}')
    assert RIDGE_SUPPORT_LINK in text
    reference = _read(language, 'reference/linear-model-api')
    # The English heading produces the anchor; Chinese keeps an explicit ID.
    anchor = ('### Custom RidgeCV training subsets' if language == 'en'
              else '<a id="custom-ridgecv-training-subsets"></a>')
    assert anchor in reference


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_public_cv_design_keeps_custom_split_principles_without_bug_details(language):
    text = _read(language, 'guides/cross-validation-design')
    section = text.split('\n## 4.', 1)[1].split('\n## 5.', 1)[0]
    required = (
        ('nonempty', 'disjoint', 'without repeated rows', 'support varies by estimator',
         'scientifically appropriate') if language == 'en' else
        ('非空', '互不重叠', '没有重复行', '支持范围因估计器而异', '符合科学问题')
    )
    for term in required:
        assert term in section
    for detail in ('sample_weight', 'full complement', 'exactly once',
                   '完整补集', '恰好覆盖'):
        assert detail not in section, detail
    for detail in ('Known RidgeCV custom-split issue',
                   'RidgeCV 自定义划分的已知问题', '/issues/243'):
        assert detail not in text, detail


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_cv_configuration_guide_keeps_actionable_ridge_limit_without_bug_cause(language):
    text = _read(language, 'guides/cross-validation')
    start, end = (('## Folds', '## Tuning grids') if language == 'en'
                  else ('## 数据折', '## 调参网格'))
    section = text.split(start, 1)[1].split(end, 1)[0]
    required = (
        ('RidgeCV', 'not reliably supported', 'excludes training rows', 'external CV loop')
        if language == 'en' else
        ('RidgeCV', '不能得到可靠支持', '另行排除训练行', '外部交叉验证循环')
    )
    for term in required:
        assert term in section
    assert RIDGE_SUPPORT_LINK in section
    for detail in ('cover every observation exactly once', 'full complement',
                   '恰好覆盖每个观测一次', '完整补集', '/issues/243'):
        assert detail not in section, detail
