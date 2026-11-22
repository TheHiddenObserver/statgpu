"""Ownership and compatibility regressions for the shared LassoCV selector."""

import inspect
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.linear_model import LassoCV
from statgpu.linear_model import _lassocv_device_affinity_contract as affinity
from statgpu.linear_model import _lassocv_uniform_weight_identity_contract as uniform
from statgpu.linear_model import _weighted_lassocv_review_contract as weighted
from statgpu.linear_model.wrappers import _lasso as lasso


def test_canonical_entries_own_code_and_explicit_delegates():
    assert Path(lasso._select_lasso_alpha_cv.__code__.co_filename).name == '_lasso.py'
    assert Path(LassoCV._prepare_cv_inputs_for_resolved_device.__code__.co_filename).name == '_lasso_cv.py'
    assert weighted._ORIGINAL_SELECT is lasso._select_lasso_alpha_cv_unweighted
    assert uniform._REVIEWED_SELECT is weighted._weighted_select_lasso_alpha_cv
    assert affinity._ORIGINAL_SELECT is uniform._select_lasso_alpha_cv
    assert affinity._ORIGINAL_PREPARE is LassoCV._prepare_cv_inputs_without_affinity
    assert inspect.signature(lasso._select_lasso_alpha_cv) == inspect.signature(weighted._weighted_select_lasso_alpha_cv)


def test_compatibility_installers_do_not_replace_canonical_or_later_entries(monkeypatch):
    selector = lasso._select_lasso_alpha_cv
    prepare = LassoCV._prepare_cv_inputs_for_resolved_device
    for _ in range(3):
        weighted.install_weighted_lassocv_review_contract()
        uniform.install_lassocv_uniform_weight_identity_contract()
        affinity.install_lassocv_device_affinity_contract()
        assert lasso._select_lasso_alpha_cv is selector
        assert LassoCV._prepare_cv_inputs_for_resolved_device is prepare
    later_selector = lambda *a, **k: None
    later_prepare = lambda *a, **k: None
    monkeypatch.setattr(lasso, '_select_lasso_alpha_cv', later_selector)
    monkeypatch.setattr(LassoCV, '_prepare_cv_inputs_for_resolved_device', later_prepare)
    weighted.install_weighted_lassocv_review_contract()
    uniform.install_lassocv_uniform_weight_identity_contract()
    affinity.install_lassocv_device_affinity_contract()
    assert lasso._select_lasso_alpha_cv is later_selector
    assert LassoCV._prepare_cv_inputs_for_resolved_device is later_prepare


@pytest.mark.parametrize('kind', ['none', 'uniform', 'nonuniform'])
def test_shared_entry_preserves_uniform_weight_routing(monkeypatch, kind):
    X = np.arange(12.).reshape(6, 2)
    y = np.arange(6.)
    weights = {'none': None, 'uniform': np.full(6, 2.), 'nonuniform': np.arange(1., 7.)}[kind]
    seen = []
    sentinel = object()
    def delegate(*args, **kwargs):
        seen.append(kwargs)
        return sentinel
    monkeypatch.setattr(uniform, '_REVIEWED_SELECT', delegate)
    assert lasso._select_lasso_alpha_cv(X, y, sample_weight=weights, cache_key=('owned',), device='cpu') is sentinel
    assert seen[0]['cache_key'] == ('owned',)
    assert seen[0]['sample_weight'] is (weights if kind == 'nonuniform' else None)


@pytest.mark.parametrize('fails', [False, True])
def test_canonical_device_context_precedes_dispatch_and_restores(monkeypatch, fails):
    events = []
    X = SimpleNamespace(device=SimpleNamespace(id=3))
    class Context:
        def __init__(self, device): assert device == 3
        def __enter__(self): events.append('enter')
        def __exit__(self, *exc): events.append('exit')
    def delegate(*args, **kwargs):
        events.append('dispatch')
        if fails: raise ValueError('sentinel')
        return 42
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(Device=Context)))
    monkeypatch.setattr(affinity, '_is_cupy_array', lambda value: value is X)
    monkeypatch.setattr(affinity, '_ORIGINAL_SELECT', delegate)
    if fails:
        with pytest.raises(ValueError, match='sentinel'): lasso._select_lasso_alpha_cv(X, object())
    else:
        assert lasso._select_lasso_alpha_cv(X, object()) == 42
    assert events == ['enter', 'dispatch', 'exit']


def test_prepare_retains_captured_boundary_and_dynamic_helper(monkeypatch):
    class Child(LassoCV):
        def _prepare_cv_inputs_without_affinity(self, *args):
            raise AssertionError('captured implementation must not use subclass dispatch')
    model = Child(device='cpu')
    events = []
    def original(self, X, y, w, device):
        events.append('original')
        return X, y, w
    def convert(self, X, y, w):
        events.append('convert')
        return 'prepared', y, w
    monkeypatch.setattr(affinity, '_ORIGINAL_PREPARE', original)
    monkeypatch.setattr(affinity, '_convert_cpu_cv_inputs', convert)
    assert model._prepare_cv_inputs_for_resolved_device('X', 'y', None, 'cpu') == ('prepared', 'y', None)
    assert events == ['original', 'convert']


def test_old_pickle_globals_resolve_to_current_implementations():
    # Protocol 0 GLOBAL references reproduce historical pickle lookup paths.
    refs = [
        ('statgpu.linear_model.cv._lasso_cv', 'LassoCV', LassoCV),
        ('statgpu.linear_model._weighted_lassocv_review_contract', '_weighted_select_lasso_alpha_cv', weighted._weighted_select_lasso_alpha_cv),
        ('statgpu.linear_model._lassocv_device_affinity_contract', '_align_cupy_cv_inputs', affinity._align_cupy_cv_inputs),
        ('statgpu.linear_model._lassocv_uniform_weight_identity_contract', '_uniform_positive_weight', uniform._uniform_positive_weight),
    ]
    for module, name, expected in refs:
        assert pickle.loads(f'c{module}\n{name}\n.'.encode()) is expected
    prepare = LassoCV._prepare_cv_inputs_for_resolved_device
    assert pickle.loads(pickle.dumps(prepare)) is prepare


def test_knockoff_consumer_uses_canonical_selector(monkeypatch):
    from statgpu.feature_selection import _knockoff_utils as knockoff
    X = np.arange(24., dtype=float).reshape(12, 2)
    calls = []
    monkeypatch.setattr(knockoff, '_lasso_diff_cache_get', lambda key: None)
    monkeypatch.setattr(knockoff, '_lasso_diff_cache_put', lambda *args: None)
    def select(X, y, **kwargs):
        calls.append(('select', X.shape, kwargs['device']))
        return .1
    def fit(X, y, **kwargs):
        calls.append(('fit', kwargs['alpha']))
        return {'coef': np.arange(X.shape[1], dtype=float)}
    monkeypatch.setattr(affinity, '_ORIGINAL_SELECT', select)
    monkeypatch.setattr(lasso, '_fit_lasso_single_alpha_fast', fit)
    result = knockoff._lasso_coef_diff_statistics(X, X + .2, np.arange(12.), np, random_state=5, cv_folds=2, n_alphas=2)
    assert result.shape == (2,)
    assert calls == [('select', (12, 4), 'cpu'), ('fit', .1)]
