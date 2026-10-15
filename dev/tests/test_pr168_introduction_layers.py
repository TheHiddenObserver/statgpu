"""Keep the repaired introductions conceptual without deleting API contracts.

These bounded section checks deliberately allow class names, statistical symbols,
and input concepts in learner prose. They are not a repository-wide token ban.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


def page(language, name):
    return (ROOT / 'docs' / language / (name + '.md')).read_text(encoding='utf-8')


def section(text, heading):
    body = text.split('## ' + heading + '\n', 1)[1]
    return re.split(r'^## ', body, maxsplit=1, flags=re.M)[0]


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_fixed_effect_introduction_explains_identification_before_configuration(language):
    text = page(language, 'panel/panel-ols')
    intro = section(text, 'When to use fixed effects' if language == 'en' else '什么时候使用固定效应')
    data = section(text, 'Prepare your panel data' if language == 'en' else '先准备面板数据')
    effects = section(text, 'Choose effects and standard errors' if language == 'en' else '选择固定效应与标准误')
    estimator = section(text, 'Estimator' if language == 'en' else '估计量')
    for control in ('entity_effects=True', 'time_effects=True', 'False'):
        assert control not in intro
        assert control in data and control in effects
    for concept in (('never changes within an entity', 'time-varying confounding',
                     'conditional mean zero', 'both kinds of differences') if language == 'en'
                    else ('从不变化', '随时间变化的混杂', '条件均值为零', '双向效应')):
        assert concept in intro
    assert r'y_{it}-\bar y_i=' in intro
    implementation = 'dummy-variable matrix' if language == 'en' else '虚拟变量矩阵'
    assert implementation not in intro and implementation in estimator
    assert ('Supplying IDs' in data or 'supplying IDs' in data or '仅提供标签' in data)
    assert 'pooled-ols.md' in effects
    assert text.index('## ' + ('Prepare your panel data' if language == 'en' else '先准备面板数据')) < text.index('<!-- example: cpu-basic -->')


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_pooled_labels_are_explained_with_data_not_between_assumptions_and_model(language):
    text = page(language, 'panel/pooled-ols')
    intro = section(text, 'When one pooled relationship is appropriate' if language == 'en' else '什么时候适合合并回归')
    data = section(text, 'Prepare the stacked observations' if language == 'en' else '准备堆叠后的观测')
    assert 'entity_ids' not in intro
    for token in ('entity_ids', 'LM', 'time_index'):
        assert token in data
    for meaning in (('without changing the coefficients', 'does not automatically cluster')
                    if language == 'en' else ('不改变系数', '不会自动启用聚类')):
        assert meaning in data
    assert ('conditional mean zero' if language == 'en' else '条件均值为零') in intro
    assert r'\sum_{i,t}' in intro


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_random_effects_retains_quasi_demeaning_but_moves_covariance_control(language):
    text = page(language, 'panel/random-effects')
    intro = section(text, 'When a random intercept is appropriate' if language == 'en' else '什么时候适合随机截距')
    covariance = section(text, 'Choose uncertainty settings and refit' if language == 'en' else '选择标准误并重新拟合')
    assert 'cov_type' not in intro and 'cov_type' in covariance
    assert r'y_{it}^{*}=y_{it}-\theta_i\bar y_i' in intro
    assert ('Swamy-Arora' if language == 'en' else 'Swamy–Arora') in intro
    assert ('not the variance components or coefficients' if language == 'en' else '不改变方差分量与系数') in covariance


def test_english_diagnostic_selection_uses_test_names_and_retains_conditions():
    text = page('en', 'panel/diagnostics')
    selection = section(text, 'Choose the question, then check applicability')
    operation = section(text, 'Run the comparisons on one sample')
    for term in ('pooling F', 'Breusch-Pagan LM', 'Hausman'):
        assert term in selection
    for call in ('pooling_f_test()', 'breusch_pagan_lm_test()', 'hausman_test()', 'entity_ids'):
        assert call not in selection and call in operation
    for condition in ('aligned sample', 'coefficient design', 'uniquely identified', 'nonrobust', 'one-way entity'):
        assert condition in selection
    chinese = section(page('cn', 'panel/diagnostics'), '如何选择检验')
    for term in ('Pooling F', 'Breusch–Pagan LM', 'Hausman', '同样本', '可唯一识别', '非稳健'):
        assert term in chinese


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_cox_overview_keeps_model_and_assumptions_without_support_inventory(language):
    text = page(language, 'models/coxph')
    intro = section(text, 'Overview' if language == 'en' else '概览')
    for term in ('CoxPHCV', 'NumPy', 'CuPy', 'Torch', 'Breslow', 'Efron', 'Exact'):
        assert term not in intro
        assert term in text
    assert r'h_s(t\mid x)=h_{0s}(t)\exp(x^\top\beta)' in intro
    for meaning in (('censoring', 'proportional-hazards', 'shared coefficient', 'risk sets')
                    if language == 'en' else ('删失', '比例风险', '共享', '风险集')):
        assert meaning in intro


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_dbscan_execution_details_follow_the_example_with_gpu_link(language):
    text = page(language, 'unsupervised/dbscan')
    intro = section(text, 'Overview' if language == 'en' else '概览')
    execution_heading = 'Execution and memory' if language == 'en' else '计算路径与内存'
    execution = section(text, execution_heading)
    for token in ('CuPy', 'Torch', 'float32', 'Cython'):
        assert token not in intro and token in execution
    assert text.index('<!-- learner-example: dbscan -->') < text.index('## ' + execution_heading)
    assert r'\left\|' in text[:text.index('<!-- learner-example: dbscan -->')]
    assert ('#execution-and-memory' if language == 'en' else '#计算路径与内存') in text
    for meaning in (('quadratic memory', 'Large common feature offsets', 'sparse-core')
                    if language == 'en' else ('二次', '偏移', '核心')):
        assert meaning in text
