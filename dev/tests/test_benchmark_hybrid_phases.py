"""CPU harness and negative contracts; these do not establish physical GPU evidence."""
import json
import sys
from dataclasses import replace
import numpy as np
import pytest
from dev.benchmarks import benchmark_hybrid_phases as bench
from dev.benchmarks.hybrid_phase_observer import PhaseObserver


@pytest.mark.parametrize('case', bench.CASES)
def test_tiny_cpu(case):
    row = bench.run_case(case, bench.Runtime('numpy'), bench.PRESETS['tiny'], 0, 1)
    assert row['status'] == 'pass'
    assert row['timing']['instrumented'] is False
    assert len(row['timing']['samples_seconds']) == 1
    assert row['backend_evidence']['device'] == 'cpu'
    assert row['native_gpu_variant']['status'] == 'not-implemented'
    assert row['performance_eligible'] is (not case.startswith('knockoff'))
    assert row['diagnostic']['records']
    assert sys.getprofile() is None


def test_resource_block_does_not_downscale():
    p = bench.PRESETS['large']
    with pytest.raises(bench.Blocked, match='no downscaling'):
        bench.guard('kde_bootstrap', p, 10)
    assert p.n == 5000


@pytest.mark.parametrize('field', ['n', 'p', 'queries', 'resamples'])
def test_bad_shape(field):
    with pytest.raises(ValueError):
        bench.guard('kde_bootstrap', replace(bench.PRESETS['tiny'], **{field: 0}), 1000)


def test_wrong_bootstrap_replicate_rejected():
    problem = bench.PRESETS['tiny']
    call, arrays = bench.make_call('kde_bootstrap', bench.Runtime('numpy'), problem)
    reference = call()
    value = {k: v.copy() for k, v in reference.items()}
    value['draws'][0, 0] += .1
    with pytest.raises(AssertionError):
        bench.check('kde_bootstrap', value, reference, arrays, problem)


def test_wrong_selector_rejected():
    problem = bench.PRESETS['tiny']
    call, arrays = bench.make_call('bandwidth_nrd', bench.Runtime('numpy'), problem)
    reference = call()
    value = dict(reference, factor=reference['factor'] * 2)
    with pytest.raises(AssertionError):
        bench.check('bandwidth_nrd', value, reference, arrays, problem)


def test_missing_knockoff_oracle_rejected():
    with pytest.raises(AssertionError, match='missing realized'):
        bench.check('knockoff_fixed', {}, {}, (), bench.PRESETS['tiny'])


def test_observer_exception_restores_profiler():
    with pytest.raises(ValueError):
        with PhaseObserver(bench.Runtime('numpy'), []):
            raise ValueError('failure')
    assert sys.getprofile() is None


def test_observer_rejects_existing_profiler():
    previous = lambda *args: None
    sys.setprofile(previous)
    try:
        with pytest.raises(RuntimeError, match='existing profiler'):
            with PhaseObserver(bench.Runtime('numpy'), []):
                pass
        assert sys.getprofile() is previous
    finally:
        sys.setprofile(None)


def test_torch_bootstrap_explicit_unsupported():
    class FakeTorch:
        name = 'torch'
    with pytest.raises(bench.Unsupported, match='unsupported'):
        bench.run_case('kde_bootstrap', FakeTorch(), bench.PRESETS['tiny'])


def test_cli_blocked_no_comparison(tmp_path):
    output = tmp_path/'result.json'
    assert bench.main(['--cases', 'kde_bootstrap', '--max-work', '1', '--output', str(output)]) == 2
    result = json.loads(output.read_text())
    assert result['rows'][0]['status'] == 'blocked'
    assert result['performance']['paired_end_to_end'] == []
    assert not result['validation']['physical_gpu_claim']


def test_source_fingerprints_observer():
    source = bench.fingerprint()
    assert 'dev/benchmarks/hybrid_phase_observer.py' in source['content_hashes']
    assert 'dev/benchmarks/learning_execution_trace.py' in source['content_hashes']


def test_timed_calls_are_uninstrumented(monkeypatch):
    original = bench.make_call
    profiles = []
    def factory(*args):
        call, arrays = original(*args)
        def checked():
            profiles.append(sys.getprofile())
            return call()
        return checked, arrays
    monkeypatch.setattr(bench, 'make_call', factory)
    bench.run_case('bandwidth_nrd', bench.Runtime('numpy'), bench.PRESETS['tiny'], 1, 2)
    assert all(p is None for p in profiles[-3:])
    assert profiles[-4] is not None


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', bench.CASES)
def test_physical_gpu_optional(case, backend):
    try:
        rt = bench.Runtime(backend)
    except bench.BackendUnavailable as exc:
        pytest.skip(str(exc))
    if backend == 'torch' and case == 'kde_bootstrap':
        with pytest.raises(bench.Unsupported):
            bench.run_case(case, rt, bench.PRESETS['tiny'], 0, 1)
    else:
        row = bench.run_case(case, rt, bench.PRESETS['tiny'], 0, 1)
        assert row['backend_evidence']['device'] == rt.device


def test_cli_refuses_overwrite_and_duplicates(tmp_path):
    output = tmp_path/'keep.json'
    output.write_text('keep')
    with pytest.raises(SystemExit):
        bench.main(['--output', str(output)])
    assert output.read_text() == 'keep'
    with pytest.raises(SystemExit):
        bench.main(['--cases', 'bandwidth_nrd', 'bandwidth_nrd', '--output', str(tmp_path/'new.json')])


def test_source_mutation_invalidates_physical_claim(monkeypatch, tmp_path):
    states = iter([{'worktree_fingerprint': 'before'}, {'worktree_fingerprint': 'after'}])
    monkeypatch.setattr(bench, 'fingerprint', lambda: next(states))
    monkeypatch.setattr(bench, 'Runtime', lambda *a: object())
    def fake_run(case, rt, *args):
        return {'case': case, 'backend': 'cupy', 'status': 'pass', 'performance_eligible': True}
    monkeypatch.setattr(bench, 'run_case', fake_run)
    output = tmp_path/'changed.json'
    assert bench.main(['--cases', 'bandwidth_nrd', '--backends', 'cupy', '--output', str(output)]) == 2
    result = json.loads(output.read_text())
    assert result['source_changed']
    assert not result['validation']['physical_gpu_claim']
    assert not result['rows'][0]['performance_eligible']
    assert result['performance']['paired_end_to_end'] == []


def test_gpu_only_never_emits_ratio(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, 'Runtime', lambda *a: object())
    monkeypatch.setattr(bench, 'run_case', lambda case, rt, *a: {
        'case': case, 'backend': 'cupy', 'status': 'pass', 'performance_eligible': True})
    output = tmp_path/'gpu-only.json'
    assert bench.main(['--cases', 'bandwidth_nrd', '--backends', 'cupy', '--output', str(output)]) == 0
    assert json.loads(output.read_text())['performance']['paired_end_to_end'] == []


def test_convergence_warning_rejects_timing(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    original = bench.make_call
    def factory(*args):
        call, arrays = original(*args)
        def warning_call():
            warnings.warn('not converged', ConvergenceWarning)
            return call()
        return warning_call, arrays
    monkeypatch.setattr(bench, 'make_call', factory)
    with pytest.raises(ConvergenceWarning):
        bench.run_case('bandwidth_nrd', bench.Runtime('numpy'), bench.PRESETS['tiny'], 0, 1)
    assert sys.getprofile() is None


def test_timed_repeat_corruption_rejected(monkeypatch):
    original = bench.make_call
    def factory(*args):
        call, arrays = original(*args)
        count = [0]
        def altered():
            count[0] += 1
            result = call()
            if count[0] >= 3:
                result['factor'] *= 2
            return result
        return altered, arrays
    monkeypatch.setattr(bench, 'make_call', factory)
    with pytest.raises(AssertionError):
        bench.run_case('bandwidth_nrd', bench.Runtime('numpy'), bench.PRESETS['tiny'], 0, 1)


def test_model_draw_average_finite_threshold_and_corruption():
    X = np.eye(6)
    y = np.arange(6, dtype=float)
    captures = [(X, np.zeros_like(X), y), (X, .5*X, y)]
    W = .75*np.abs(y-y.mean())
    result = {'W': W, 'threshold': np.array(.375), 'selected': np.arange(6)}
    checked = bench.check('knockoff_model', result, {}, (), replace(bench.PRESETS['tiny'], p=6), captures)
    assert checked['status'] == 'pass'
    assert not checked['performance_eligible']
    for field, bad in [('W', .5*np.abs(y-y.mean())), ('threshold', np.array(1.125)), ('selected', np.array([0]))]:
        with pytest.raises(AssertionError):
            bench.check('knockoff_model', dict(result, **{field: bad}), {}, (), replace(bench.PRESETS['tiny'], p=6), captures)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0., -1.])
def test_nonfinite_or_nonpositive_bandwidth_rejected(bad):
    problem = bench.PRESETS['tiny']
    value = {'factor': np.array(bad), 'prediction': np.zeros(problem.queries)}
    with pytest.raises(AssertionError):
        bench.check('bandwidth_nrd', value, value, (), problem)


def test_wrong_prediction_shape_rejected():
    problem = bench.PRESETS['tiny']
    value = {'factor': np.array(.4), 'prediction': np.zeros((problem.queries, 1))}
    with pytest.raises(AssertionError):
        bench.check('bandwidth_nrd', value, value, (), problem)


@pytest.mark.parametrize('mismatch', ['data', 'scope', None])
def test_ratio_requires_matching_identity_and_scope(monkeypatch, tmp_path, mismatch):
    monkeypatch.setattr(bench, 'Runtime', lambda name, *a: name)
    def fake_run(case, backend, *args):
        return {'case': case, 'backend': backend, 'status': 'pass', 'performance_eligible': True,
                'problem': {'data_id': backend if mismatch == 'data' else 'same'},
                'timing': {'scope': backend if mismatch == 'scope' else 'same', 'median_seconds': 1.}}
    monkeypatch.setattr(bench, 'run_case', fake_run)
    output = tmp_path/'ratio.json'
    assert bench.main(['--cases', 'bandwidth_nrd', '--backends', 'numpy', 'cupy', '--output', str(output)]) == 0
    pairs = json.loads(output.read_text())['performance']['paired_end_to_end']
    assert len(pairs) == (1 if mismatch is None else 0)
