"""Keep shared CV guidance consistent with the documented custom-fold limit."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('guide', ['cross-validation', 'cross-validation-design'])
def test_shared_cv_guides_disclose_custom_ridge_training_rows(language, guide):
    text = (ROOT / f'docs/{language}/guides/{guide}.md').read_text()
    assert 'RidgeCV' in text and 'sample_weight' in text
    assert '../reference/linear-model-api.md#custom-ridgecv-training-subsets' in text
    if language == 'en':
        for term in ['cover every observation exactly once', 'full complement',
                     'external CV loop', 'disjoint', 'acceptance alone']:
            assert term in text
    else:
        for term in ['恰好覆盖每个观测一次', '完整补集', '外部交叉验证循环',
                     '互不重叠', '并不证明划分有效']:
            assert term in text
