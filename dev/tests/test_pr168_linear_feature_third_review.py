"""Executable learner workflows and current knockoff cache disclosures.

The changed-input defect is observed conditionally rather than required to
persist. Fresh-array/process routes assert correct current-data statistics.
"""
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from doc_examples import run_example

from statgpu.feature_selection import FixedXKnockoffSelector, fixed_x_knockoff_filter

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_knockoff_caches(monkeypatch):
    from statgpu.feature_selection import _knockoff_utils as utilities
    from statgpu.linear_model.wrappers import _lasso

    monkeypatch.setattr(utilities, '_LASSO_DIFF_CACHE', OrderedDict())
    monkeypatch.setattr(_lasso, '_LASSO_CV_ALPHA_CACHE', OrderedDict())


def _paired_design():
    rng = np.random.default_rng(23)
    n, p = 80, 4
    basis, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
    return basis[:, 1:p+1].copy(), basis[:, p+1:2*p+1].copy()


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / (name + '.md')).read_text()


def _execute_marked_example(text, marker):
    return run_example(text, marker.split(': ', 1)[1], marker, allow_legacy=True)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_knockoff_learner_example_is_complete_and_interprets_selection(language):
    text = _page(language, 'models', 'knockoff')
    namespace = _execute_marked_example(text, 'learner-example: knockoff-selection')
    result = namespace['result']
    np.testing.assert_array_equal(result.selected_features, [0, 1, 2, 3, 4, 5, 15])
    assert result.threshold == pytest.approx(1.019324242097392)
    assert result.estimated_fdr == pytest.approx(1 / 7)
    assert result.metadata['xk_source'] == 'provided'
    assert namespace['q'] == .20
    assert np.count_nonzero(namespace['beta']) == 6
    assert ('not the actual fraction' in text if language == 'en' else '不是这一次样本' in text)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_documented_retained_copy_workflow_uses_valid_current_pairs(language):
    namespace = _execute_marked_example(
        _page(language, 'reference', 'feature-selection-api'),
        'api-example: knockoff-fresh-inputs',
    )
    X, Xk = namespace['X'], namespace['Xk']
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(Xk.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(X.T @ X, Xk.T @ Xk, atol=1e-14)
    np.testing.assert_allclose(X.T @ Xk, np.zeros((4, 4)), atol=1e-14)
    snapshots = namespace['inputs_kept_alive']
    assert len(snapshots) == 2
    assert all(not np.shares_memory(old, new) for old, new in zip(*snapshots))
    assert np.argmax(namespace['results'][0].W) == 0
    assert np.argmax(namespace['results'][1].W) == 1


@pytest.mark.parametrize('implementation', ['statgpu', 'sklearn'])
def test_fresh_input_snapshot_agrees_with_fresh_process_after_response_change(implementation):
    if implementation == 'sklearn':
        pytest.importorskip('sklearn')
    X, Xk = _paired_design()
    y = 5 * X[:, 0]
    kwargs = {'method': 'lasso_coef_diff', 'random_state': 17,
              'backend': 'numpy', 'lasso_cv_impl': implementation}
    old = fixed_x_knockoff_filter(X, y, Xk=Xk, **kwargs)
    y[:] = 5 * X[:, 1]
    # A new selector is not sufficient to isolate the process-global caches.
    changed = FixedXKnockoffSelector(**kwargs).fit(X, y, Xk=Xk).result_
    retained_inputs = [(X, y, Xk)]
    snapshot = tuple(np.array(a, dtype=np.float64, copy=True) for a in (X, y, Xk))
    retained_inputs.append(snapshot)
    fresh = fixed_x_knockoff_filter(snapshot[0], snapshot[1], Xk=snapshot[2], **kwargs)
    code = '''
import json
import numpy as np
from statgpu import fixed_x_knockoff_filter
rng = np.random.default_rng(23)
n, p = 80, 4
basis, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = basis[:, 1:p+1].copy(), basis[:, p+1:2*p+1].copy()
y = 5 * X[:, 1]
result = fixed_x_knockoff_filter(X, y, Xk=Xk, method='lasso_coef_diff',
    random_state=17, backend='numpy', lasso_cv_impl=IMPLEMENTATION)
print(json.dumps(result.W.tolist()))
'''.replace('IMPLEMENTATION', repr(implementation))
    process = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                             capture_output=True, text=True, check=True, timeout=60)
    cold = np.asarray(json.loads(process.stdout))
    np.testing.assert_allclose(fresh.W, cold, atol=1e-10, rtol=1e-10)
    assert np.argmax(old.W) == 0
    assert np.argmax(cold) == 1
    if not np.allclose(changed.W, cold):
        for language in ['en', 'cn']:
            text = _page(language, 'reference', 'feature-selection-api')
            learner = _page(language, 'models', 'knockoff')
            assert 'repeated-lasso-statistic-calls' in text
            assert 'repeated-lasso-statistic-calls' in learner
            assert ('fresh Python process' in text if language == 'en' else '新的 Python 进程' in text)
            assert ('memory identity' in text if language == 'en' else '内存标识' in text)
        assert 'reuse stale statistics' in fixed_x_knockoff_filter.__doc__


def test_sklearn_import_failure_can_resolve_to_native_lasso(monkeypatch):
    X, Xk = _paired_design()
    y = 5 * X[:, 0]
    kwargs = {'Xk': Xk, 'method': 'lasso_coef_diff', 'random_state': 17, 'backend': 'numpy'}
    expected = fixed_x_knockoff_filter(X, y, lasso_cv_impl='statgpu', **kwargs)
    monkeypatch.setitem(sys.modules, 'sklearn', None)
    fallback = fixed_x_knockoff_filter(X, y, lasso_cv_impl='sklearn', **kwargs)
    np.testing.assert_allclose(fallback.W, expected.W, atol=1e-10)
    if fallback.metadata['lasso_cv_impl'] == 'sklearn':
        for language in ['en', 'cn']:
            text = _page(language, 'reference', 'feature-selection-api')
            assert 'metadata["lasso_cv_impl"]' in text
            assert ('not necessarily' in ' '.join(text.split()) if language == 'en' else '不一定' in text)


def test_prediction_docs_distinguish_numerical_backend_from_coefficient_storage():
    from statgpu import ElasticNet

    doc = ElasticNet.predict.__doc__
    assert "fitted model's numerical backend" in doc
    assert 'Public ``coef_`` storage' in doc
    assert 'NumPy even after GPU fitting' in doc
    X = np.arange(12.0).reshape(6, 2)
    model = ElasticNet(device='cpu').fit(X, X[:, 0])
    assert isinstance(model.coef_, np.ndarray)
    assert isinstance(model.predict(X, return_cpu=False), np.ndarray)


def test_chinese_knockoff_distinguishes_statistical_construction_from_precision():
    text = _page('cn', 'models', 'knockoff')
    assert '不同的设计或特征分布假设' in text
    assert '不能把两者当作速度或数值精度档位' in text
    assert '性能与精度权衡主要体现在 `fixed_x`/`model_x` 选择' not in text
