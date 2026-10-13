"""Guard local concept-to-operation bridges, not a formula-first page template."""

from pathlib import Path
import re

import numpy as np
import pytest

from statgpu.feature_selection._knockoff_utils import _corr_diff_statistics

ROOT = Path(__file__).resolve().parents[2]


def _text(language, page):
    return (ROOT / 'docs' / language / (page + '.md')).read_text(encoding='utf-8')


def _compact(text):
    return re.sub(r'\s+', ' ', text)


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('method', ['fixed-x', 'model-x'])
def test_knockoff_rule_is_defined_before_its_lower_bound_and_tie_warning(language, method):
    text = _text(language, 'models/feature-selection/knockoff-' + method)
    heading = ('## From feature competition to a selection rule' if language == 'en'
               else '## 从变量竞争到筛选规则')
    boundary = ('## Before the first call:' if language == 'en' else '## 首次调用前')
    bridge = _compact(text[text.index(heading):text.index(boundary)])
    assert 'W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|' in bridge
    assert '(1+#{W≤−T})/max(1,#{W≥T})' in bridge
    if language == 'en':
        for meaning in ('sample response mean', 'not normalized correlation coefficients or p-values',
                        'each distinct positive value of |W|', 'including every tie',
                        'Choose the smallest T', 'select the original features with W≥T',
                        'If no candidate qualifies, select none', 'prespecified FDR target q'):
            assert meaning in bridge
    else:
        for meaning in ('响应的样本均值', '不是归一化相关系数或 p 值',
                        '每个不同的正值', '包括所有并列值', '最小 T',
                        '保留 W≥T 的原变量', '不选择任何变量', '预设 FDR 目标 q'):
            assert meaning in bridge
    assert text.index(heading) < text.index(boundary) < text.index('1/p')
    assert text.index('1/p') < text.index('W=[8,8,-8]') < text.index('```python')


def _complete_threshold(w, q):
    candidates = [t for t in np.unique(abs(w)) if t > 0]
    eligible = [t for t in candidates
                if (1 + np.count_nonzero(w <= -t)) / max(1, np.count_nonzero(w >= t)) <= q]
    return min(eligible) if eligible else np.inf


def test_corr_diff_bridge_means_centered_inner_products_not_correlation():
    x = np.array([[1., 4.], [2., 0.], [5., 3.]])
    xk = np.array([[3., 1.], [0., 2.], [1., 4.]])
    y = np.array([4., -2., 7.])
    expected = abs(x.T @ (y - y.mean())) - abs(xk.T @ (y - y.mean()))
    actual = _corr_diff_statistics(x, xk, y, np)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(_corr_diff_statistics(x, xk, y + 10, np), expected, atol=1e-14)
    np.testing.assert_allclose(_corr_diff_statistics(3*x, 3*xk, y, np), 3*expected, atol=1e-14)


def test_complete_rule_counts_all_ties_and_chooses_smallest_eligible_value():
    # A teaching-rule oracle, deliberately not an assertion that runtime ties are fixed.
    assert np.isinf(_complete_threshold(np.array([8., 8., -8.]), .5))
    w = np.array([9., 8., 7., 6., 5., -1.])
    threshold = _complete_threshold(w, .25)
    assert threshold == 5.
    np.testing.assert_array_equal(np.flatnonzero(w >= threshold), [0, 1, 2, 3, 4])
    assert np.isinf(_complete_threshold(np.array([0., 0.]), .25))


PANEL_FORMULAS = {
    'pooled-ols': (r'y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}',
                   r'\sum_{i,t}(y_{it}-\alpha-x_{it}^{\top}\beta)^2'),
    'panel-ols': (r'y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}',
                  r'y_{it}-\bar y_i=(x_{it}-\bar x_i)^{\top}\beta+(\varepsilon_{it}-\bar\varepsilon_i)'),
    'random-effects': (r'y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it}',
                       r'y_{it}^{*}=y_{it}-\theta_i\bar y_i',
                       r'x_{it}^{*}=x_{it}-\theta_i\bar x_i'),
    'between-ols': (r'y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}',
                    r'\bar y_i=\alpha+\bar x_i^{\top}\beta+a_i+\bar\varepsilon_i'),
    'first-difference-ols': (r'y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}',
                             r'\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}'),
    'fama-macbeth': (r'y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it}',
                     r'\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^{T}\widehat\beta_t'),
}


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page,formulas', list(PANEL_FORMULAS.items()))
def test_panel_short_model_and_variable_mapping_precede_the_first_fit(language, page, formulas):
    text = _text(language, 'panel/' + page)
    before = text[:text.index('<!-- example: cpu-basic -->')]
    for formula in formulas:
        assert '$' + formula + '$' in before
    # Symbols must connect to the input columns and observational indices locally.
    if language == 'en':
        for explanation in ('indexes entities', 'indexes periods', 'response', 'predictor vector'):
            assert explanation in before
    else:
        for symbol in ('$i$', '$t$', '$y_{it}$', '$x_{it}$', '个体', '时期', '响应'):
            assert symbol in before
    # Keep longer reference derivations after the example; no global formula-first rewrite.
    assert text.index('<!-- example-end: cpu-basic -->') < text.index('$$')


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page,constructor', [('kernel-pca', 'kpca = KernelPCA('),
                                               ('kernel-nystroem', 'nystroem = Nystroem(')])
def test_rbf_distance_scale_is_explained_before_the_example_constructor(language, page, constructor):
    text = _text(language, 'models/nonparametric/' + page)
    formula = r"$k(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$"
    start = text.index(formula)
    bridge = _compact(text[start:text.index(constructor)])
    assert '`gamma=0.7`' in bridge
    assert ('makes similarity decay faster with distance' if language == 'en'
            else '衰减') in bridge
    assert text.index(constructor) < text.index('.fit_transform(X_train)')


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_other_optional_bridges_precede_their_specific_use(language):
    text = _text(language, 'models/covariance/covariance-graphical-lasso-cv')
    objective = text.index(r'\operatorname{tr}(S\Theta)')
    before = _compact(text[:objective])
    after = _compact(text[objective:text.index('```python')])
    for symbol in ('$S$', '$\\Theta$'):
        assert symbol in before
    if language == 'en':
        assert 'training-row count as denominator' in before
        assert 'positive definite precision matrix (inverse covariance)' in before
        assert 'only the off-diagonal entries' in after and 'diagonal is unpenalized' in after
    else:
        assert '训练' in before and '正定' in before and '逆协方差' in before
        assert '非对角' in after and '不受惩罚' in after

    text = _text(language, 'models/nonparametric/splines')
    before = _compact(text[:text.index('transformer = SplineTransformer(')])
    assert '`degree=3`' in before and '`n_knots=6`' in before
    for meaning in (('piecewise cubic', 'including the two boundaries', 'not the number of output basis columns')
                    if language == 'en' else ('分段三次', '两个边界节点', '不等于输出基列数')):
        assert meaning in before

    text = _text(language, 'unsupervised/gaussian-mixture')
    before = _compact(text[:text.index('<!-- learner-example: gaussian-mixture -->')])
    assert '$\\sum_k\\pi_k=1$' in before
    for meaning in (('nonnegative mixing weight', 'mean vector', 'covariance', 'mixture density')
                    if language == 'en' else ('非负', '混合权重', '均值向量', '协方差', '混合密度')):
        assert meaning in before

    text = _text(language, 'models/testing/multiple-testing')
    call = text.index('statistic, global_p = combine_pvalues(')
    formula = text.index('$-2\\log p_i$')
    bridge = _compact(text[formula:call])
    assert ('smaller p-value gives a larger positive contribution' if language == 'en'
            else 'P 值越小，正向贡献越大') in bridge
    assert ('rather than averaging' if language == 'en' else '不是对 P 值取平均') in bridge

    text = _text(language, 'guides/regression-diagnostics')
    before = text[:text.index('print(diag.leverage)')]
    for meaning in (('design space', 'influence on the fitted regression', 'collinearity among predictors')
                    if language == 'en' else ('设计空间', '对回归拟合的影响', '预测变量间的共线性')):
        assert meaning in before
