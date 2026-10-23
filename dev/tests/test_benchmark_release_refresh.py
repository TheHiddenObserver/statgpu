"""Harness contracts only; these tests do not certify physical GPU timings."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks import benchmark_release_refresh as bench


@pytest.mark.parametrize('alpha', [0.0, 0.1, 3.0])
@pytest.mark.parametrize('weighted', [False, True])
def test_reference_satisfies_average_loss_normal_equations(alpha, weighted):
    X, y, query, w, _ = bench.make_data(48, 4, 'float64', 7, weighted)
    beta, pred = bench.analytic_reference(X, y, query, w, alpha)
    weights = np.ones(len(y)) if w is None else w
    D = np.column_stack((np.ones(len(y)), X))
    penalty = np.diag(np.r_[0., np.repeat(alpha * weights.sum(), X.shape[1])])
    np.testing.assert_allclose((D.T @ (weights[:, None]*D)+penalty) @ beta,
                               D.T @ (weights*y), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred, beta[0]+query @ beta[1:])


def test_weight_global_scaling_does_not_change_reference():
    X, y, query, w, _ = bench.make_data(32, 3, 'float64', 8, True)
    first = bench.analytic_reference(X, y, query, w, .2)
    second = bench.analytic_reference(X, y, query, w*17, .2)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, atol=1e-12)


def kwargs(**overrides):
    result = dict(model_name='ridge', n=48, p=4, dtype='float64', weighted=True,
                  seed=9, alpha=.1, warmup=1, repeats=2)
    result.update(overrides)
    return result


@pytest.mark.parametrize('model', ['ols', 'ridge'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('weighted', [True, False])
def test_cpu_smoke_rows_have_correctness_and_separate_scopes(model, dtype, weighted):
    row = bench.run_case(bench.Runtime('numpy'), **kwargs(model_name=model, dtype=dtype, weighted=weighted))
    assert row['status'] == 'passed'
    assert row['correctness']['status'] == 'pass'
    assert row['problem']['dtype'] == dtype
    assert row['backend']['fit_ingress']['X']['device'] == 'cpu'
    assert row['timing']['synchronized_device'] is None
    assert [r['scope'] for r in row['timing']['records']] == ['fit-only-public-api', 'predict-only-public-api']
    assert all(len(r['raw_seconds']) == 2 for r in row['timing']['records'])
    assert all(r['median_seconds'] > 0 for r in row['timing']['records'])
    assert 'speedup' not in row
    assert 'inference' not in row


def test_bad_reference_prevents_timing(monkeypatch):
    original = bench.analytic_reference
    def bad(*args):
        beta, pred = original(*args)
        return beta+1, pred+1
    monkeypatch.setattr(bench, 'analytic_reference', bad)
    row = bench.run_case(bench.Runtime('numpy'), **kwargs())
    assert row['status'] == 'failed'
    assert row['correctness']['status'] == 'fail'
    assert 'timing' not in row


def test_later_failed_repeat_discards_all_timing(monkeypatch):
    original = bench.correctness
    calls = []
    def late_failure(*args):
        result = original(*args)
        calls.append(None)
        if len(calls) == 3:
            result['status'] = 'fail'
        return result
    monkeypatch.setattr(bench, 'correctness', late_failure)
    row = bench.run_case(bench.Runtime('numpy'), **kwargs(warmup=0))
    assert row['status'] == 'failed'
    assert 'timing' not in row


def test_finite_wrong_values_and_nan_fail_correctness():
    X, y, query, w, _ = bench.make_data(32, 3, 'float64', 8, True)
    beta, pred = bench.analytic_reference(X, y, query, w, .2)
    for incorrect in (beta + 1, np.full_like(beta, np.nan)):
        result = bench.correctness(X, y, w, .2, incorrect, pred, beta, pred, 'float64')
        assert result['status'] == 'fail'


def test_timing_synchronizes_before_and_after_on_actual_device(monkeypatch):
    events = []
    class Runtime:
        def sync(self, device):
            events.append(('sync', device))
    value, elapsed = bench.time_call(lambda: events.append(('call',)) or 42, Runtime(), 'cuda:3')
    assert value == 42 and elapsed >= 0
    assert events == [('sync', 'cuda:3'), ('call',), ('sync', 'cuda:3')]


def test_wrong_device_is_rejected():
    with pytest.raises(RuntimeError, match='Execution device mismatch'):
        bench.Runtime('numpy').sync('cuda:0')
    model = bench.ObservedOLS(device='cpu', compute_inference=False).fit(np.eye(4), np.arange(4.))
    model._selected_backend_device = 'cuda:0'
    with pytest.raises(RuntimeError, match='Selected device'):
        bench.verify_route(model, bench.Runtime('numpy'))


def test_identity_reproducible_and_sensitive_to_dtype_weight_seed():
    data = bench.make_data(32, 3, 'float64', 1, True)
    assert data[-1] == bench.make_data(32, 3, 'float64', 1, True)[-1]
    assert len({data[-1], bench.make_data(32, 3, 'float32', 1, True)[-1],
                bench.make_data(32, 3, 'float64', 2, True)[-1],
                bench.make_data(32, 3, 'float64', 1, False)[-1]}) == 4


@pytest.mark.parametrize('required,exit_code', [(False, 0), (True, 2)])
def test_unavailable_is_explicit_and_require_backends_fails(tmp_path, monkeypatch, required, exit_code):
    original = bench.Runtime
    def fake_runtime(name, ordinal):
        if name == 'cupy':
            raise bench.BackendUnavailable('test: no CUDA device')
        return original(name, ordinal)
    monkeypatch.setattr(bench, 'Runtime', fake_runtime)
    output = tmp_path/'result.json'
    argv = ['--backends', 'numpy', 'cupy', '--shapes', '24x3', '--models', 'ols',
            '--weights', 'none', '--warmup', '0', '--repeats', '1', '--output', str(output)]
    if required:
        argv.append('--require-backends')
    assert bench.main(argv) == exit_code
    result = json.loads(output.read_text())
    assert result['backends']['cupy']['status'] == 'unavailable'
    assert len(result['rows']) == 1
    assert result['validation']['status'] == ('failed' if required else 'partial')


def test_runtime_error_is_not_an_unavailable_skip(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        raise ValueError('wrong device configuration')
    monkeypatch.setattr(bench, 'Runtime', failing)
    path = tmp_path/'result.json'
    assert bench.main(['--backends', 'numpy', '--output', str(path)]) == 2
    assert json.loads(path.read_text())['backends']['numpy']['status'] == 'failed'


@pytest.mark.parametrize('entry', [['dev/benchmarks/benchmark_release_refresh.py'],
                                  ['-m', 'dev.benchmarks.benchmark_release_refresh']])
def test_direct_and_module_smoke_are_runnable(tmp_path, entry):
    path = tmp_path/'result.json'
    result = subprocess.run([sys.executable, *entry, '--smoke', '--output', str(path)],
                            cwd=bench.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = json.loads(path.read_text())
    assert result['validation']['status'] == 'complete'
    assert result['validation']['canonical_dashboard_registered'] is False
    assert len(result['source']['git_commit']) == 40
    assert len(result['source']['runner_sha256']) == 64
    assert len(result['environment']['environment_id']) == 64
    assert result['source']['worktree_fingerprint'] == result['source_after']['worktree_fingerprint']


@pytest.mark.parametrize('argument', [['--repeats', '0'], ['--warmup', '-1'],
                                     ['--alpha', 'nan'], ['--cuda-device', '-1'],
                                     ['--shapes', '4x8']])
def test_bad_cli_rejected(tmp_path, argument):
    with pytest.raises(SystemExit) as exc:
        bench.main([*argument, '--output', str(tmp_path/'bad.json')])
    assert exc.value.code == 2


def test_untracked_ignored_runner_cannot_report_clean_source():
    source = bench.source_identity()
    if source['runner_tracked'] is False:
        assert source['working_tree'] == 'dirty'
        assert source['source_overlay'] == 'untracked-or-ignored runner'


def test_zero_executed_cases_is_not_success(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise bench.BackendUnavailable('test missing backend')
    monkeypatch.setattr(bench, 'Runtime', missing)
    path = tmp_path/'result.json'
    assert bench.main(['--backends', 'cupy', '--output', str(path)]) == 2
    result = json.loads(path.read_text())
    assert result['rows'] == []
    assert result['validation']['status'] == 'failed'
