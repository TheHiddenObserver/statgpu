"""Guard user-facing interpretation and bilingual audit refinements."""
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def doc(lang, page):
    return (ROOT / 'docs' / lang / (page + '.md')).read_text(encoding='utf-8')


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_shared_resampling_limits_are_retrievable(language):
    text = doc(language, 'reference/estimator-api')
    title = '## Module-function differences' if language == 'en' else '## 与模块函数的区别'
    section = text.split(title)[1]
    # Independent concepts need their own paragraphs, without deleting edge cases.
    assert len([p for p in section.split('\n\n') if p.startswith('**')]) == 4
    for term in ('force_vectorized=True', 'statistic_hint="mean"', '"pearson_corr"',
                 '(n, 1)', 'backend="auto"'):
        assert term in section
    assert ('final one-row batch' if language == 'en' else '最后只有一行的批次') in section
    assert ('broadcast error' if language == 'en' else '广播错误') in section
    assert ('do not verify' if language == 'en' else '不会检查') in section


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_diagnostics_explain_outputs_without_universal_deletion_rule(language):
    text = doc(language, 'guides/regression-diagnostics')
    required = ('design space', 'influence', 'collinearity', 'no single cutoff') if language == 'en' else ('设计空间', '影响', '共线性', '删样本规则')
    for term in required:
        assert term in text
    assert ('runs on CPU' if language == 'en' else 'CPU') in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_umap_cautions_have_distinct_topics_and_keep_failure_limits(language):
    section = doc(language, 'unsupervised/api-reference').split('## UMAP')[1].split('## TSNE')[0]
    assert len(re.findall(r'^- \*\*', section, re.M)) == 6
    for term in ('float32', 'float64', 'SciPy', 'NumPy 2', 'nn_method="nndescent"', 'random_state'):
        assert term in section
    # One seeded-initialization recommendation, rather than the old duplicate.
    assert section.count('`init="random"`') == 1
    assert ('not the exact gradient' if language == 'en' else '不是标准 UMAP 交叉熵的精确梯度') in section
    assert ('one common positive' if language == 'en' else '同一个正尺度') in section
    assert '[unsupervised index](README.md)' not in doc(language, 'unsupervised/README')
    assert '[无监督学习索引](README.md)' not in doc(language, 'unsupervised/README')


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page', ['pca', 'kmeans', 'nmf', 'minibatch-nmf', 'gaussian-mixture', 'agglomerative-clustering'])
def test_unsupervised_inference_limits_are_observable(language, page):
    text = doc(language, 'unsupervised/' + page)
    assert ('standard errors' if language == 'en' else '标准误') in text
    assert 'strict inference' not in text
    assert '严格推断' not in text
    assert '严格的统计推断' not in text


@pytest.mark.parametrize('language, page, anchor', [
    ('en','models/nonparametric/nonparametric','nonparametric-methods'),
    ('cn','models/nonparametric/nonparametric','非参数方法'),
    ('en','models/feature-selection/knockoff','a-complete-cpu-example-with-a-valid-supplied-pair'),
    ('cn','models/feature-selection/knockoff','使用有效外部配对的完整-cpu-示例'),
    ('en','unsupervised/dbscan','objective-function--loss-function'),
    ('cn','unsupervised/dbscan','目标函数--损失函数'),
    ('en','unsupervised/agglomerative-clustering','estimating-equation'),
    ('cn','unsupervised/agglomerative-clustering','估计方程'),
])
def test_renamed_teaching_headings_preserve_previous_anchors(language, page, anchor):
    assert f'id="{anchor}"' in doc(language, page)


def test_cn_analytic_weights_and_permutation_have_single_names():
    for name in ('inference-modes','penalized-glm-inference','cross-validation',
                 'lbfgs-float32-precision-contract','nodewise-alpha-migration',
                 'loss-penalty-solver-framework','solver-penalty-matrix','solver-algorithms'):
        text = doc('cn', 'guides/' + name)
        prose = re.sub(r'<a[^>]*></a>', '', text)
        assert '解析权重' not in prose
        assert '分析权重' in prose
    text = doc('cn', 'guides/inference-api')
    assert '## 置换检验' in text
    assert '<a id="排列检验"></a>' in text
    assert 'permutation test' in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_changelog_uses_final_pr129_source_and_preserves_history_boundary(language):
    text = doc(language, 'changelog')
    for term in ('nodewise_alpha', 'PR #139', '8d9f422e2fcf648e412838747b9ab5881b3fb7ef',
                 'issuecomment-5479959733', 'b17c8a6c161065bfbe9efa9f60d0fad69a464773',
                 'gaussian_cupy_rank_recovery_8d9f422e.json'):
        assert term in text
    assert ('not to later versions' if language == 'en' else '不是对后续版本的重新验证') in text
    assert 'PR #129 当前保持 open/unmerged' not in text


def test_singular_kernel_ridge_minimizes_objective_not_fitted_function_values():
    # A singular PSD Gram matrix still gives an invertible regularized system.
    K = np.array([[1., 1.], [1., 1.]])
    y = np.array([2., 3.])
    alpha = .4
    coef = np.linalg.solve(K + alpha * np.eye(2), y)
    objective = lambda c: np.sum((y - K @ c) ** 2) + alpha * c @ K @ c
    null = np.array([1., -1.])
    np.testing.assert_allclose(objective(coef + null), objective(coef), rtol=1e-13)
    for offset in (-2., -.5, .5, 2.):
        assert objective(coef + offset * np.ones(2)) > objective(coef)
    assert '使上述目标函数最小' in doc('cn', 'models/nonparametric/kernel-ridge')
    assert '使拟合函数达到' not in doc('cn', 'models/nonparametric/kernel-ridge')
