"""Harness/negative evidence checks; CPU tests never certify physical GPU use."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dev.benchmarks import benchmark_native_kde_bootstrap as bench
from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, stage


@pytest.mark.parametrize('weights', ['none', 'unequal', 'sparse'])
def test_cpu_controlled_workflow_matches_every_draw_and_linear_quantile(weights):
    problem = replace(bench.PRESETS['tiny'], weights=weights)
    data = bench.make_data(problem)
    before = bench.data_identity(data)
    reference = bench.analytic_reference(data, problem)
    call = bench.make_call(bench.VARIANTS[0], bench.Runtime('numpy'), data, problem,
                           64 * 1024**2, 3)
    actual = call()
    assert set(bench.check(actual, reference, problem)) == {k + '_max_abs_diff' for k in bench.FIELDS}
    assert bench.data_identity(data) == before
    alpha = (1 - problem.confidence_level) / 2
    np.testing.assert_allclose(actual['lower'], np.quantile(actual['draws'], alpha, axis=0))


def test_identical_indices_not_merely_shared_seed():
    problem = bench.PRESETS['tiny']
    first, second = bench.make_data(problem), bench.make_data(problem)
    assert bench.data_identity(first) == bench.data_identity(second)
    modified = dict(first, indices=first['indices'].copy())
    modified['indices'][0, 0] = (modified['indices'][0, 0] + 1) % problem.n
    assert bench.data_identity(first) != bench.data_identity(modified)
    assert all(not value.flags.writeable for value in first.values() if value is not None)


def test_resampling_reweights_original_weights_again():
    problem = replace(bench.PRESETS['tiny'], weights='unequal')
    data = bench.make_data(problem)
    expected = bench.analytic_reference(data, problem)
    incorrectly_unweighted = bench.analytic_reference(dict(data, weights=None), problem)
    assert np.max(np.abs(expected['draws'] - incorrectly_unweighted['draws'])) > 1e-4


@pytest.mark.parametrize('field', bench.FIELDS)
def test_wrong_estimate_draw_or_interval_is_rejected(field):
    problem = bench.PRESETS['tiny']
    expected = bench.analytic_reference(bench.make_data(problem), problem)
    actual = {key: value.copy() for key, value in expected.items()}
    actual[field].flat[0] += .01
    with pytest.raises(AssertionError):
        bench.check(actual, expected, problem)


def test_nonfinite_wrong_shape_and_dtype_rejected():
    problem = bench.PRESETS['tiny']
    expected = bench.analytic_reference(bench.make_data(problem), problem)
    for wrong in (np.full(problem.queries, np.nan), np.ones(1), np.ones(problem.queries, dtype=np.float32)):
        with pytest.raises(AssertionError):
            bench.check(dict(expected, estimate=wrong), expected, problem)


@pytest.mark.parametrize('field', ['n', 'queries', 'resamples'])
def test_bad_dimension(field):
    with pytest.raises(ValueError):
        bench.guard(replace(bench.PRESETS['tiny'], **{field: 0}), 2, 7, 64 * 1024**2)


def test_resource_guard_never_downscales():
    problem = replace(bench.PRESETS['gpu'], n=1_000_000)
    with pytest.raises(bench.Blocked, match='no downscaling'):
        bench.guard(problem, 2, 7, 64 * 1024**2)
    assert problem.n == 1_000_000
    with pytest.raises(bench.Blocked):
        bench.guard(bench.PRESETS['tiny'], 2, 7, bench.MAX_WORKSPACE_BYTES + 1)


def test_gpu_preset_fits_fixed_guard():
    guard = bench.guard(bench.PRESETS['gpu'], 2, 7, 64 * 1024**2)
    assert guard['total_kernel_evaluations_upper_bound'] <= bench.MAX_TOTAL_WORK


def test_explicit_variant_cannot_fall_back():
    with pytest.raises(ValueError, match='fall back'):
        bench.make_call(bench.VARIANTS[2], bench.Runtime('numpy'), {}, bench.PRESETS['tiny'], 1000, None)


def test_measure_is_uninstrumented_and_syncs_both_sides():
    events = []
    def call():
        assert sys.getprofile() is None
        events.append('call')
        return 'result'
    measured = bench.measure(call, lambda: events.append('sync'), lambda value: events.append('check'), 1, 2)
    assert events == ['sync', 'call', 'sync', 'check'] * 3
    assert measured['instrumented'] is False
    assert len(measured['samples_seconds']) == 2


def test_existing_profiler_rejects_timing_and_is_not_changed():
    profiler = lambda *args: None
    sys.setprofile(profiler)
    try:
        with pytest.raises(TraceIncomplete, match='profiler'):
            bench.measure(lambda: None, lambda: None, lambda x: None, 0, 1)
        assert sys.getprofile() is profiler
    finally:
        sys.setprofile(None)


def test_trace_rejects_replaced_primitive():
    with pytest.raises(TraceIncomplete, match='Untrusted replacement'):
        NumericalTrace._verify_native_callable(lambda value: value, np, 'numpy', 'exp')


@pytest.mark.parametrize('failure', ['missing', 'wrong-device', 'no-result'])
def test_trace_rejects_incomplete_primitive_evidence(failure):
    def density():
        pass
    observer = bench.CandidateTrace(SimpleNamespace(name='cupy', device='cuda:1'),
                                    [stage(density, 'density', ('exp',))])
    if failure != 'missing':
        identity = {'backend': 'cupy', 'device': 'cuda:0' if failure == 'wrong-device' else 'cuda:1', 'dtype': 'float64'}
        observer.records = [{'stage': 'density', 'completed': True, 'operations': [
            {'operation': 'exp', 'backend': 'cupy', 'inputs': [identity],
             'outputs': [] if failure == 'no-result' else [identity]}]}]
    with pytest.raises(TraceIncomplete):
        observer.report()


def _complete_fixture():
    """Synthetic acceptance inputs ONLY, not fabricated physical measurements."""
    source = {'git_commit': 'test', 'worktree_fingerprint': 'fixed',
              'missing_sources': [], 'exact_import_paths': True,
              'working_tree': 'clean', 'all_sources_tracked': True}
    rows = []
    for name in bench.VARIANTS:
        device = 'cpu' if name == bench.VARIANTS[0] else 'cuda:1'
        rows.append({'variant': name, 'status': 'pass', 'device': device,
                     'data_identity': {'indices': 'same'}, 'correctness': {'status': 'pass'},
                     'timing': {'median_seconds': 1., 'samples_seconds': [1.] * 7,
                                'instrumented': False, 'synchronized_device': device},
                     'operation_evidence': {'status': 'observed', 'backend': 'cupy',
                                            'device': 'cuda:1',
                                            'host_conversion_guard': {'status': 'pass'}},
                     'memory': {'status': 'measured', 'device': 'cuda:1'}})
    return {'source': source.copy(), 'source_after': source.copy(), 'rows': rows,
            'data_identity': {'indices': 'same'}, 'data_identity_after': {'indices': 'same'},
            'timing': {'warmup': 2, 'repeats': 7, 'scope': 'synthetic-unit-test'},
            'environment': {'environment_id': 'synthetic-unit-test', 'gpu_device': 'cuda:1'}}


@pytest.mark.parametrize('failure', ['source', 'commit', 'imports', 'missing-source', 'indices',
                                    'missing-gpu', 'wrong-output', 'trace', 'current-trace',
                                    'host-conversion', 'memory', 'warmup', 'repeats', 'dirty', 'untracked', 'tree-transition'])
def test_acceptance_fails_closed(failure):
    result = _complete_fixture()
    native = result['rows'][2]
    if failure == 'source':
        result['source_after']['worktree_fingerprint'] = 'changed'
    elif failure == 'commit':
        result['source_after']['git_commit'] = 'changed'
    elif failure == 'imports':
        result['source']['exact_import_paths'] = False
    elif failure == 'missing-source':
        result['source']['missing_sources'] = ['candidate.py']
    elif failure == 'indices':
        result['data_identity_after']['indices'] = 'changed'
    elif failure == 'missing-gpu':
        result['rows'] = result['rows'][:1]
    elif failure == 'wrong-output':
        native['status'] = 'failed'
    elif failure == 'trace':
        native['operation_evidence'] = {}
    elif failure == 'current-trace':
        result['rows'][1]['operation_evidence'] = {}
    elif failure == 'host-conversion':
        native['operation_evidence']['host_conversion_guard']['status'] = 'failed'
    elif failure == 'memory':
        native['memory'] = {'status': 'unavailable'}
    elif failure == 'dirty':
        result['source']['working_tree'] = 'dirty'
        result['source_after']['working_tree'] = 'dirty'
    elif failure == 'untracked':
        result['source']['all_sources_tracked'] = False
        result['source_after']['all_sources_tracked'] = False
    elif failure == 'tree-transition':
        result['source_after']['working_tree'] = 'dirty'
    elif failure in ('warmup', 'repeats'):
        result['timing'][failure] = 0 if failure == 'warmup' else 1
    bench.finalize(result)
    assert result['validation']['status'] == 'partial'
    assert result['validation']['physical_gpu_claim'] is False
    assert result['performance'] == {'eligible': False, 'paired_end_to_end': []}
    assert not any(row['performance_eligible'] for row in result['rows'])


def test_cpu_only_cli_is_partial_and_does_not_fallback(tmp_path):
    output = tmp_path / 'cpu-only.json'
    assert bench.main(['--preset', 'tiny', '--cpu-only', '--warmup', '0', '--repeats', '1',
                       '--output', str(output)]) == 2
    result = json.loads(output.read_text())
    assert result['rows'][0]['status'] == 'pass'
    assert all(row['status'] == 'unavailable' for row in result['rows'][1:])
    assert result['performance']['eligible'] is False
    assert result['validation']['physical_gpu_claim'] is False
    assert result['benchmark']['not_public_bootstrap_api_timing'] is True
    assert result['source_changed'] is False


def test_missing_gpu_stays_partial(monkeypatch):
    original = bench.Runtime
    def runtime(name, *args, **kwargs):
        if name == 'cupy':
            raise bench.BackendUnavailable('test: CUDA not present')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(bench, 'Runtime', runtime)
    result = bench.run_benchmark(bench.PRESETS['tiny'], warmup=0, repeats=1)
    assert all(row['status'] == 'unavailable' for row in result['rows'][1:])
    assert not result['validation']['physical_gpu_claim']


def test_cli_blocked_budget_keeps_requested_shape(tmp_path):
    output = tmp_path / 'blocked.json'
    assert bench.main(['--preset', 'tiny', '--workspace-bytes', '1', '--output', str(output)]) == 2
    result = json.loads(output.read_text())
    assert result['problem']['n'] == bench.PRESETS['tiny'].n
    assert all(row['status'] == 'blocked' for row in result['rows'])
    assert not result['performance']['eligible']


def test_cli_refuses_overwrite(tmp_path):
    output = tmp_path / 'keep.json'
    output.write_text('keep')
    with pytest.raises(SystemExit):
        bench.main(['--output', str(output)])
    assert output.read_text() == 'keep'


def test_fingerprint_covers_numerical_source_runner_tests_and_trace():
    source = bench.fingerprint()
    assert not source['missing_sources']
    assert source['exact_import_paths']
    for path in ('dev/benchmarks/native_kde_bootstrap.py',
                 'dev/benchmarks/benchmark_native_kde_bootstrap.py',
                 'dev/tests/test_native_kde_bootstrap.py',
                 'dev/tests/test_benchmark_native_kde_bootstrap.py',
                 'dev/benchmarks/learning_execution_trace.py',
                 'statgpu/nonparametric/kernel_smoothing/_kde.py'):
        assert path in source['content_hashes']


def test_physical_gpu_optional():
    try:
        runtime = bench.Runtime('cupy')
    except bench.BackendUnavailable as exc:
        pytest.skip(str(exc))
    problem = bench.PRESETS['tiny']
    data = bench.make_data(problem)
    reference = bench.analytic_reference(data, problem)
    value, trace = bench.trace_candidate(runtime, data, problem, 64 * 1024**2, 3)
    bench.check(value, reference, problem)
    assert trace['status'] == 'observed'
    assert trace['host_conversion_guard']['status'] == 'pass'
    value, diagnostic = bench.memory_diagnostic(runtime, data, problem, 64 * 1024**2, 3)
    bench.check(value, reference, problem)
    assert diagnostic['status'] == 'measured'


def test_oracle_failure_writes_failed_artifact(monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError('test: degenerate effective degrees of freedom')
    monkeypatch.setattr(bench, 'analytic_reference', fail)
    output = tmp_path / 'oracle-failed.json'
    assert bench.main(['--preset', 'tiny', '--cpu-only', '--output', str(output)]) == 2
    result = json.loads(output.read_text())
    assert all(row['status'] == 'failed' for row in result['rows'])
    assert all('workload/oracle' in row['reason'] for row in result['rows'])
    assert not result['performance']['eligible']


def test_invalid_gpu_ordinal_is_failed_not_fallback(monkeypatch):
    original = bench.Runtime
    def runtime(name, *args, **kwargs):
        if name == 'cupy':
            raise ValueError('CUDA ordinal 99 outside device count')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(bench, 'Runtime', runtime)
    result = bench.run_benchmark(bench.PRESETS['tiny'], device=99, warmup=0, repeats=1)
    assert all(row['status'] == 'failed' for row in result['rows'][1:])
    assert all('ordinal' in row['reason'] for row in result['rows'][1:])
    assert not result['performance']['eligible']
    assert all('timing' not in row for row in result['rows'])


def test_failed_correctness_blocks_all_timing(monkeypatch):
    original = bench.make_call
    def make_call(*args, **kwargs):
        call = original(*args, **kwargs)
        def corrupted():
            value = call()
            value['draws'][0, 0] += 1
            return value
        return corrupted
    monkeypatch.setattr(bench, 'make_call', make_call)
    monkeypatch.setattr(bench, 'measure', lambda *args: pytest.fail('timing ran after correctness failure'))
    result = bench.run_benchmark(bench.PRESETS['tiny'], cpu_only=True, warmup=0, repeats=1)
    assert result['rows'][0]['status'] == 'failed'
    assert not result['performance']['eligible']


def test_candidate_conversion_guard_allows_only_scalar_observed_get():
    class DeviceArray:
        def __init__(self, ndim):
            self.ndim = ndim
    runtime = SimpleNamespace(name='cupy', device='cuda:1',
                              module=SimpleNamespace(ndarray=DeviceArray))
    trace = bench.CandidateTrace(runtime, [])
    frame = SimpleNamespace(f_code=None)
    trace._profile(frame, 'c_call', SimpleNamespace(__self__=DeviceArray(0), __name__='get'))
    with pytest.raises(TraceIncomplete, match='forbidden host numeric conversion'):
        trace._profile(frame, 'c_call', SimpleNamespace(__self__=DeviceArray(1), __name__='get'))
    assert trace.host_conversion_attempts == ['ndarray.get']


@pytest.mark.parametrize('shape', [(65000, 1000, 1), (1000000, 1, 40)])
def test_host_temporaries_guard_rejects_before_data_allocation(monkeypatch, shape):
    n, queries, resamples = shape
    problem = replace(bench.PRESETS['tiny'], n=n, queries=queries, resamples=resamples)
    monkeypatch.setattr(bench, 'make_data', lambda *args: pytest.fail('allocated guarded workload'))
    result = bench.run_benchmark(problem, warmup=0, repeats=1, cpu_only=True)
    assert all(row['status'] == 'blocked' for row in result['rows'])
    assert all('fixed resource guard' in row['reason'] for row in result['rows'])
    assert result['problem']['n'] == n
    assert not result['performance']['eligible']


def test_host_estimate_explicitly_counts_managed_temporaries():
    problem = bench.PRESETS['tiny']
    result = bench.guard(problem, 2, 7, 64 * 1024**2)
    assert result['host_buffer_estimate_bytes'] == 8 * (
        3 * problem.resamples * problem.n + 6 * problem.n * problem.queries
        + 5 * problem.resamples * problem.queries + 10 * problem.n)
    assert 'not a hard process RSS cap' in result['host_buffer_estimate_scope']


def test_synthetic_acceptance_fixture_is_self_consistent():
    # This verifies only acceptance plumbing, not physical evidence.
    result = bench.finalize(_complete_fixture())
    assert result['performance']['eligible']
    assert len(result['performance']['paired_end_to_end']) == 2


@pytest.mark.parametrize('variant', bench.VARIANTS)
@pytest.mark.parametrize('failure', ['instrumented', 'missing-timing', 'sample-count', 'nan',
                                    'inf', 'zero', 'negative', 'median', 'sync-device',
                                    'data-identity', 'correctness'])
def test_row_acceptance_requires_complete_consistent_evidence(variant, failure):
    result = _complete_fixture()
    row = next(item for item in result['rows'] if item['variant'] == variant)
    if failure == 'instrumented':
        row['timing']['instrumented'] = True
    elif failure == 'missing-timing':
        del row['timing']
    elif failure == 'sample-count':
        row['timing']['samples_seconds'].pop()
    elif failure in ('nan', 'inf', 'zero', 'negative'):
        row['timing']['samples_seconds'][0] = {'nan': float('nan'), 'inf': float('inf'),
                                               'zero': 0., 'negative': -1.}[failure]
    elif failure == 'median':
        row['timing']['median_seconds'] = 2.
    elif failure == 'sync-device':
        row['timing']['synchronized_device'] = 'cuda:99'
    elif failure == 'data-identity':
        row['data_identity']['indices'] = 'different'
    elif failure == 'correctness':
        row['correctness']['status'] = 'failed'
    bench.finalize(result)
    assert result['validation']['status'] == 'partial'
    assert not result['validation']['physical_gpu_claim']
    assert result['performance'] == {'eligible': False, 'paired_end_to_end': []}


@pytest.mark.parametrize('failure', ['environment', 'current-device', 'native-device',
                                    'current-trace-device', 'native-trace-device',
                                    'current-trace-backend', 'native-trace-backend',
                                    'memory-device', 'cpu-device'])
def test_acceptance_requires_same_concrete_gpu_for_both_variants(failure):
    result = _complete_fixture()
    current, native = result['rows'][1:]
    if failure == 'environment':
        result['environment']['gpu_device'] = 'cuda'
    elif failure in ('current-device', 'native-device'):
        row = current if failure == 'current-device' else native
        row['device'] = 'cuda:2'
        row['timing']['synchronized_device'] = 'cuda:2'
    elif failure.endswith('trace-device'):
        row = current if failure.startswith('current') else native
        row['operation_evidence']['device'] = 'cuda:2'
    elif failure.endswith('trace-backend'):
        row = current if failure.startswith('current') else native
        row['operation_evidence']['backend'] = 'numpy'
    elif failure == 'memory-device':
        native['memory']['device'] = 'cuda:2'
    elif failure == 'cpu-device':
        result['rows'][0]['device'] = 'cuda:1'
        result['rows'][0]['timing']['synchronized_device'] = 'cuda:1'
    bench.finalize(result)
    assert not result['validation']['physical_gpu_claim']
    assert result['performance'] == {'eligible': False, 'paired_end_to_end': []}


@pytest.mark.parametrize('option', ['--bandwidth', '--confidence-level'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_cli_rejects_nonfinite_parameters_without_creating_artifact(tmp_path, option, value):
    output = tmp_path / 'must-not-exist.json'
    with pytest.raises(SystemExit) as exc:
        bench.main(['--preset', 'tiny', '--cpu-only', option + '=' + value,
                    '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()
