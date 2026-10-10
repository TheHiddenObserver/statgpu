"""Focused regressions for the reviewed guide wording and statistical meaning."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.nonparametric import fit_kde
from statgpu.nonparametric.splines import _penalized

_ROOT = Path(__file__).resolve().parents[2]


def _page(language, relative_path):
    return (_ROOT / 'docs' / language / relative_path).read_text(encoding='utf-8')


@pytest.mark.parametrize('correction', [1.0, 0.5, 5e-11, 0.0, -0.25])
def test_gcv_checks_the_correction_before_squaring(monkeypatch, correction):
    """A positive squared denominator must not admit a negative correction."""
    design = np.ones((4, 2))
    response = np.arange(1.0, 5.0)
    gamma = 4.0
    edf = (1.0 - correction) * len(response) / gamma

    def fixed_fit(*args, **kwargs):
        return np.zeros(design.shape[1]), edf

    monkeypatch.setattr(_penalized, 'penalized_ls', fixed_fit)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = _penalized.generalized_cross_validation(
            design, response, np.eye(design.shape[1]), 1.0, np, gamma=gamma,
        )
    if correction <= 1e-10:
        assert np.isposinf(score)
    else:
        expected = np.mean(response**2) / correction**2
        np.testing.assert_allclose(score, expected)


@pytest.mark.parametrize('scale', [1.0, 7.0])
def test_kde_numeric_factor_is_not_an_absolute_width(scale):
    samples = scale * np.array([-2.0, -0.5, 0.0, 0.25, 1.0, 3.0]) + 10.0
    model = fit_kde(samples, bandwidth=0.35, backend='numpy')
    assert model.bandwidth_factor_ == 0.35
    expected_width = 0.35 * samples.std(ddof=1)
    np.testing.assert_allclose(np.sqrt(model.covariance_[0, 0]), expected_width)


@pytest.mark.parametrize('language, factor_term', [('en', 'factor'), ('cn', '因子')])
def test_kde_first_example_explains_the_bandwidth_factor(language, factor_term):
    text = _page(language, 'models/nonparametric.md')
    # Check the immediate example section, not an explanation much later on.
    introduction = text.split('<!-- example: kde-cpu -->', 1)[0].rsplit('\n## ', 1)[-1]
    assert 'bandwidth=0.35' in introduction
    assert factor_term in introduction
    assert '$h$' in introduction


@pytest.mark.parametrize('page, fragments', [
    ('models/semiparametric.md', (
        '请求的基数', '实际基数', '增加基数', '按 job 并行',
        '分配更多分辨率', '调整后的分母非正',
    )),
    ('models/nonparametric.md', ('条件化的对象', '规范源码签名', 'device/jobs/cleanup')),
    ('guides/distribution-api.md', (
        '最小支持整数', '内部概率', '正侧临界值', '拒绝判断',
        '正支持分布', '位置界值', '非 R 历史名称',
    )),
    ('unsupervised/README.md', ('平方欧氏惯性', 'Gaussian mixture')),
])
def test_reviewed_chinese_guides_do_not_restore_ambiguous_fragments(page, fragments):
    text = _page('cn', page)
    prose = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    for fragment in fragments:
        assert fragment not in prose, (page, fragment)
