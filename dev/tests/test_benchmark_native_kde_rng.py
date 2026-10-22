"""RNG-inclusive benchmark contracts; synthetic checks are not GPU evidence."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from dev.benchmarks import benchmark_native_kde_rng as bench


@pytest.mark.parametrize('weights', ['none', 'unequal', 'sparse'])
def test_public_numpy_fast_path_matches_own_realized_indices(weights):
    problem = replace(bench.PRESETS['tiny'], weights=weights)
    data = bench.make_data(problem)
    before = bench.data_identity(data)
    reference, metrics, rng = bench.diagnose(bench.VARIANTS[0], bench.Runtime('numpy'), data, problem, 64*1024**2, 3)
    assert set(metrics) == {key + '_max_abs_diff' for key in bench.FIELDS}
    assert rng['same_seed_indices_reproduced']
    assert rng['status'] == 'pass'
    assert bench.data_identity(data) == before
    assert reference['draws'].shape == (problem.resamples, problem.queries)


def test_data_does_not_pregenerate_bootstrap_indices():
    data = bench.make_data(bench.PRESETS['tiny'])
    assert set(data) == {'samples', 'points', 'weights'}
    assert all(not value.flags.writeable for value in data.values() if value is not None)


def test_capture_calls_original_generator_and_restores_factory():
    factory = np.random.default_rng
    expected = factory(15).choice(7, 10, replace=True)
    value, indices = bench.capture_public(lambda: np.random.default_rng(15).choice(7, 10, replace=True))
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(indices[0], expected)
    assert np.random.default_rng is factory


def test_capture_restores_factory_on_failure():
    factory = np.random.default_rng
    def fail():
        raise ValueError('test')
    with pytest.raises(ValueError):
        bench.capture_public(fail)
    assert np.random.default_rng is factory


def test_capture_requires_observed_sampling():
    with pytest.raises(AssertionError, match='not observed'):
        bench.capture_public(lambda: None)


@pytest.mark.parametrize('seed', [-1, 2**32, True, None, 1.5])
def test_bad_seed_is_rejected_before_work(seed):
    with pytest.raises(ValueError, match='seed'):
        bench.guard(replace(bench.PRESETS['tiny'], seed=seed), 2, 7, 64*1024**2)


def test_guard_includes_extra_diagnostic_calls():
    p = bench.PRESETS['tiny']
    guarded = bench.guard(p, 2, 7, 64*1024**2)
    assert guarded['total_kernel_evaluations_upper_bound'] >= p.n*p.queries*(p.resamples+1)*(3*(2+7)+12)


def test_invalid_indices_or_zero_mass_are_rejected():
    p = bench.PRESETS['tiny']
    data = bench.make_data(p)
    for indices in (np.zeros((1, p.n), dtype=int), np.full((p.resamples,p.n), p.n), np.zeros((p.resamples,p.n), dtype=float)):
        with pytest.raises(AssertionError):
            bench.validate_indices(indices, data, p)
    data['weights'] = np.ones(p.n)
    data['weights'][0] = 0
    with pytest.raises(AssertionError, match='zero-probability'):
        bench.validate_indices(np.zeros((p.resamples,p.n), dtype=int), data, p)


def test_grossly_wrong_sampling_distribution_fails():
    p = replace(bench.PRESETS['tiny'], weights='none')
    with pytest.raises(AssertionError, match='CDF'):
        bench.validate_indices(np.zeros((p.resamples,p.n), dtype=int), bench.make_data(p), p)


def test_explicit_gpu_variant_cannot_fallback():
    with pytest.raises(ValueError, match='fall back'):
        bench.make_call(bench.VARIANTS[2], bench.Runtime('numpy'), {}, bench.PRESETS['tiny'], 1, None)


def _complete():
    # Synthetic metadata fixture tests the gate only; not an execution claim.
    source = {'git_commit':'fixture', 'worktree_fingerprint':'fixture', 'working_tree':'clean',
              'all_sources_tracked':True, 'exact_import_paths':True, 'missing_sources':[]}
    rows=[]
    for i,name in enumerate(bench.VARIANTS):
        device='cpu' if i==0 else 'cuda:1'
        rows.append({'variant':name, 'status':'pass', 'device':device,
            'data_identity':{'samples':'fixture'}, 'correctness':{'status':'pass'},
            'rng_validation':{'status':'pass','same_seed_indices_reproduced':True,
                              'indices_sha256':'a'*64,'seed':15,'index_shape':[5,32],
                              'cdf_max_abs_error':0.0,'dkw_epsilon':float(np.sqrt(np.log(2./1e-6)/(2.*5*32))),'dkw_alpha':1e-6},
            'timing':{'instrumented':False,'samples_seconds':[.1]*7,'median_seconds':.1,'synchronized_device':device},
            'operation_evidence':{'status':'observed','backend':'cupy','device':'cuda:1',
                'host_conversion_guard':{'status':'pass'},
                'rng_generation':{'status':'observed','backend':'cupy','device':'cuda:1','outputs':[{'backend':'cupy','device':'cuda:1','dtype':'float64','shape':[5,32]}]}},
            'memory':{'status':'measured','device':'cuda:1'}})
    return {'source':source.copy(),'source_after':source.copy(),'rows':rows,
        'data_identity':{'samples':'fixture'},'data_identity_after':{'samples':'fixture'},
        'timing':{'warmup':2,'repeats':7,'scope':'synthetic'},
        'problem':{'seed':15,'resamples':5,'n':32},
        'environment':{'environment_id':'fixture','gpu_device':'cuda:1'}}


def test_valid_fixture_exposes_not_same_indices_comparison():
    result=bench.finalize(_complete())
    assert result['validation']['status']=='complete'
    assert 'NOT identical' in result['performance']['pairing']
    assert [p['baseline'] for p in result['performance']['end_to_end_comparisons']] == list(bench.VARIANTS[:2])


@pytest.mark.parametrize('failure', ['rng','repro','seed','shape','hash','rngtrace','rngdevice','rngoutput',
    'rngoutputdevice','rngoutputshape','rngoutputdtype','law','nanlaw','badthreshold','duplicate','unknown','missing','memory','currenttrace','hostconversion','timing','source','dirty','data'])
def test_finalizer_fails_closed(failure):
    result=_complete()
    native=result['rows'][2]
    if failure=='rng': native.pop('rng_validation')
    elif failure=='repro': native['rng_validation']['same_seed_indices_reproduced']=False
    elif failure=='seed': native['rng_validation']['seed']=7
    elif failure=='shape': native['rng_validation']['index_shape']=[1,1]
    elif failure=='hash': native['rng_validation']['indices_sha256']=''
    elif failure=='law': native['rng_validation']['cdf_max_abs_error']=1.0
    elif failure=='nanlaw': native['rng_validation']['cdf_max_abs_error']=float('nan')
    elif failure=='badthreshold': native['rng_validation']['dkw_epsilon']=1.0
    elif failure=='rngtrace': native['operation_evidence'].pop('rng_generation')
    elif failure=='rngdevice': native['operation_evidence']['rng_generation']['device']='cuda:0'
    elif failure=='rngoutput': native['operation_evidence']['rng_generation']['outputs']=[]
    elif failure=='rngoutputdevice': native['operation_evidence']['rng_generation']['outputs'][0]['device']='cuda:0'
    elif failure=='rngoutputshape': native['operation_evidence']['rng_generation']['outputs'][0]['shape']=[1,1]
    elif failure=='rngoutputdtype': native['operation_evidence']['rng_generation']['outputs'][0]['dtype']='float32'
    elif failure=='duplicate': result['rows'].append(copy.deepcopy(native))
    elif failure=='unknown': result['rows'].append(dict(native,variant='unknown'))
    elif failure=='missing': result['rows'].pop()
    elif failure=='memory': native['memory']['status']='unavailable'
    elif failure=='currenttrace': result['rows'][1]['operation_evidence']['status']='missing'
    elif failure=='hostconversion': native['operation_evidence']['host_conversion_guard']['status']='failed'
    elif failure=='timing': native['timing']['instrumented']=True
    elif failure=='source': result['source_after']['git_commit']='changed'
    elif failure=='dirty': result['source']['working_tree']='dirty'
    elif failure=='data': result['data_identity_after']={}
    bench.finalize(result)
    assert result['validation']['status']=='partial'
    assert result['validation']['physical_gpu_claim'] is False
    assert result['performance']['eligible'] is False
    assert not result['performance']['end_to_end_comparisons']


def test_cpu_only_cli_is_honestly_partial(tmp_path):
    output=tmp_path/'partial.json'
    assert bench.main(['--preset','tiny','--cpu-only','--warmup','0','--repeats','1','--output',str(output)])==2
    result=json.loads(output.read_text())
    assert result['rows'][0]['status']=='pass'
    assert all(row['status']=='unavailable' for row in result['rows'][1:])
    assert not result['performance']['eligible']
    assert result['source_changed'] is False
    assert 'rng-inclusive' in result['timing']['scope']
    assert result['benchmark']['public_baseline_api']=='kde_bootstrap_confidence_interval'


def test_timing_calls_have_no_rng_capture_wrapper(monkeypatch):
    factory=np.random.default_rng
    original=bench.controlled.measure
    def measure(call,sync,validate,warmup,repeats):
        assert np.random.default_rng is factory
        return original(call,sync,validate,warmup,repeats)
    monkeypatch.setattr(bench.controlled,'measure',measure)
    result=bench.run_benchmark(bench.PRESETS['tiny'],cpu_only=True,warmup=0,repeats=1)
    assert result['rows'][0]['status']=='pass'


def test_failed_correctness_blocks_all_timing(monkeypatch):
    def fail(*args): raise AssertionError('test oracle rejection')
    monkeypatch.setattr(bench,'diagnose',fail)
    result=bench.run_benchmark(bench.PRESETS['tiny'],cpu_only=True)
    assert result['rows'][0]['status']=='failed'
    assert all('timing' not in row for row in result['rows'])


def test_no_overwrite(tmp_path):
    path=tmp_path/'existing.json';path.write_text('preserve')
    with pytest.raises(SystemExit): bench.main(['--output',str(path)])
    assert path.read_text()=='preserve'


def test_fingerprint_includes_rng_candidate_and_tests():
    fp=bench.fingerprint()
    for name in ('dev/benchmarks/native_kde_rng.py','dev/benchmarks/benchmark_native_kde_rng.py',
                 'dev/tests/test_benchmark_native_kde_rng.py'):
        assert name in fp['content_hashes']
    assert fp['exact_import_paths']


@pytest.mark.parametrize('weights', ['none','unequal','sparse'])
def test_physical_native_rng_diagnostic_trace_memory(weights):
    try: runtime=bench.Runtime('cupy')
    except bench.BackendUnavailable as exc: pytest.skip(str(exc))
    p=replace(bench.PRESETS['tiny'],weights=weights)
    data=bench.make_data(p)
    ref,_,rng=bench.diagnose(bench.VARIANTS[2],runtime,data,p,64*1024**2,3)
    assert rng['same_seed_indices_reproduced']
    value,trace=bench.trace_native(runtime,data,p,64*1024**2,3)
    bench.check(value,ref,p)
    assert trace['rng_generation']['status']=='observed'
    assert trace['host_conversion_guard']['status']=='pass'
    value,memory=bench.memory_native(runtime,data,p,64*1024**2,3)
    bench.check(value,ref,p)
    assert memory['status']=='measured'


@pytest.mark.parametrize('key,value', [('warmup',True),('warmup',2.5),('repeats',7.5)])
def test_noninteger_timing_counts_fail_closed(key,value):
    result=_complete()
    result['timing'][key]=value
    assert bench.finalize(result)['validation']['status']=='partial'


@pytest.mark.parametrize('option', ['--bandwidth','--confidence-level'])
def test_cli_nonfinite_rejected_without_artifact(tmp_path,option):
    path=tmp_path/'invalid.json'
    with pytest.raises(SystemExit):
        bench.main([option,'nan','--output',str(path)])
    assert not path.exists()
