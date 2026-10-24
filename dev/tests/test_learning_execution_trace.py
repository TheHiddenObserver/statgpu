"""Observer contracts; Torch CPU tests are not physical CUDA evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import json

import numpy as np
import pytest

from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, identities, stage, learning_stages
from dev.benchmarks import benchmark_learning_coverage as b


def cpu_runtime():
    return SimpleNamespace(name='numpy', module=np, device='cpu')


def test_each_stage_invocation_requires_numerical_work():
    def target(x, compute=True):
        return np.exp(x) if compute else x.copy()
    trace=NumericalTrace(cpu_runtime(), [stage(target,'density',('exp',))])
    with trace:
        target(np.ones(4));target(np.ones(4),False)
    with pytest.raises(TraceIncomplete,match='density'):
        trace.report()


def test_allocation_and_copy_do_not_prove_computation():
    def target(x):return np.array(x,copy=True)
    trace=NumericalTrace(cpu_runtime(),[stage(target,'numeric',('sum',))])
    with trace:target(np.ones(4))
    with pytest.raises(TraceIncomplete):trace.report()


def test_trace_records_primitives_and_restores_observer_on_error():
    original=np.exp
    def target(x):return np.exp(x)
    trace=NumericalTrace(cpu_runtime(),[stage(target,'density',('exp',))])
    with trace:target(np.ones(4))
    assert np.exp is original and sys.getprofile() is None
    assert trace.report()['stages'][0]['operations'][0]['operation']=='exp'
    with pytest.raises(ValueError):
        with NumericalTrace(cpu_runtime(),[stage(target,'density',('exp',))]):
            target(np.ones(4));raise ValueError('test')
    assert np.exp is original and sys.getprofile() is None


def test_existing_profiler_is_not_overwritten():
    def prior(*args):pass
    sys.setprofile(prior)
    try:
        with pytest.raises(TraceIncomplete,match='Existing profiler'):
            with NumericalTrace(cpu_runtime(),[]):pass
        assert sys.getprofile() is prior
    finally:sys.setprofile(None)


def test_source_code_not_function_name_is_required():
    def target(x):return np.exp(x)
    def replacement(x):return np.exp(x)
    replacement.__name__=target.__name__
    trace=NumericalTrace(cpu_runtime(),[stage(target,'density',('exp',))])
    with trace:replacement(np.ones(4))
    with pytest.raises(TraceIncomplete,match='Missing'):trace.report()


def test_oracle_operations_outside_stage_do_not_count():
    def target(x):return x.copy()
    trace=NumericalTrace(cpu_runtime(),[stage(target,'density',('exp',))])
    with trace:
        target(np.ones(4));np.exp(np.ones(4))
    with pytest.raises(TraceIncomplete):trace.report()


def torch_runtime(device='cpu'):
    torch=pytest.importorskip('torch')
    return SimpleNamespace(name='torch',module=torch,device=device)


def test_real_torch_cpu_native_arithmetic_and_cpu_recast_rejection():
    rt=torch_runtime();torch=rt.module;x=torch.ones(4,dtype=torch.float64)
    def good(x):return torch.exp(x)
    trace=NumericalTrace(rt,[stage(good,'density',('exp',))])
    with trace:good(x)
    assert trace.report()['device']=='cpu'
    def bad(x):return torch.as_tensor(np.exp(x.numpy()))
    trace=NumericalTrace(rt,[stage(bad,'density',('exp',))])
    with trace:bad(x)
    with pytest.raises(TraceIncomplete):trace.report()


def test_wrong_concrete_device_rejected_with_real_cpu_tensor():
    rt=torch_runtime('cuda:1');torch=rt.module
    def target(x):return torch.exp(x)
    trace=NumericalTrace(rt,[stage(target,'density',('exp',))])
    with trace:target(torch.ones(4))
    with pytest.raises(TraceIncomplete):trace.report()


def test_returned_host_array_from_native_primitive_rejected(monkeypatch):
    rt=torch_runtime();torch=rt.module
    monkeypatch.setattr(torch,'exp',lambda x:np.exp(x.numpy()))
    def target(x):return torch.exp(x)
    trace=NumericalTrace(rt,[stage(target,'density',('exp',))])
    with pytest.raises(TraceIncomplete,match='replacement'):
        with trace:target(torch.ones(4))


def test_real_torch_cpu_normal_ci_internal_stages():
    from statgpu.nonparametric import kde_confidence_interval
    rt=torch_runtime();torch=rt.module
    x=torch.as_tensor(np.random.default_rng(2).normal(size=(32,1)))
    q=torch.linspace(-1,1,9,dtype=torch.float64)
    trace=NumericalTrace(rt,learning_stages('kde_normal_ci','torch'))
    with trace:result=kde_confidence_interval(x,q,bandwidth=.4,backend='torch',method='normal')
    report=trace.report()
    assert result.estimate.shape==(9,)
    assert {'kde-fit','kde-density','host-interval-assembly'}=={r['stage'] for r in report['stages']}


def test_bandwidth_validation_is_not_misclassified_by_case_whitelist():
    row=b.run_case('bandwidth_helpers',b.Runtime('numpy'),32,4,42,0,1)
    assert row['performance_eligible'] is False
    assert row['execution_qualification']['compute_placement']=='host'
    assert row['execution_qualification']['observations']['requested_backend_array_events'] > 0
    assert len(row['execution_trace'][0]['stages'])==2
    assert all(r['placement']=='array-input-validation-with-host-scalar-rule' for r in row['execution_trace'][0]['stages'])


def test_torch_cpu_bootstrap_observes_native_rng_and_quantiles():
    from statgpu.nonparametric import kde_bootstrap_confidence_interval
    rt = torch_runtime()
    x = rt.module.as_tensor(np.random.default_rng(7).normal(size=(32, 1)))
    q = rt.module.linspace(-1, 1, 9, dtype=rt.module.float64)
    traces = []
    result = b.traced_call('kde_bootstrap', lambda: kde_bootstrap_confidence_interval(
        x, q, backend='torch', bandwidth=.4, n_resamples=20, random_state=7), rt, traces)
    assert result.estimate.shape == (9,)
    assert traces[0]['device'] == 'cpu'  # Never physical CUDA evidence.
    stages = traces[0]['stages']
    assert len([r for r in stages if r['stage'] == 'native-resampling']) == 20
    assert {'native-probability-cdf', 'native-interval-quantiles'} <= {r['stage'] for r in stages}


def test_bootstrap_requires_every_replication_stage():
    specs=learning_stages('kde_bootstrap','cupy')
    assert {s['label']:s['minimum_calls'] for s in specs}['kde-density']==21
    def target(x,execute):return np.exp(x) if execute else x.copy()
    trace=NumericalTrace(cpu_runtime(),[stage(target,'replicate',('exp',),minimum_calls=21)])
    with trace:
        for i in range(21):target(np.ones(4),i!=20)
    with pytest.raises(TraceIncomplete):trace.report()


@pytest.mark.parametrize('case',['bandwidth_nrd','bandwidth_ucv','bandwidth_cv','kde_normal_ci','kde_bootstrap','knockoff_fixed','knockoff_model'])
def test_bounded_trace_integrates_without_changing_cpu_oracle(case):
    row=b.run_case(case,b.Runtime('numpy'),48,4,42,0,1)
    assert row['status']=='pass' and row['execution_trace']
    assert row['performance_eligible'] is False
    assert 'overhead' in row['timing']['instrumentation']
    assert row['correctness']['status']=='pass'


def test_observer_source_changes_invalidate_identity(tmp_path,monkeypatch):
    root=tmp_path/'repo';directory=root/'dev'/'benchmarks';directory.mkdir(parents=True)
    (root/'statgpu').mkdir();(root/'statgpu'/'fake.py').write_text('x=1\n')
    runner=directory/'benchmark_learning_coverage.py';runner.write_text('runner')
    (directory/'benchmark_release_refresh.py').write_text('runtime')
    helper=directory/'learning_execution_trace.py';helper.write_text('observer v1')
    monkeypatch.setattr(b,'ROOT',root);monkeypatch.setattr(b,'__file__',str(runner))
    before=b.source_identity();helper.write_text('observer v2');after=b.source_identity()
    assert before['source_content_sha256']!=after['source_content_sha256']
    assert before['runner_overlay'] and before['working_tree']=='dirty'


@pytest.mark.parametrize('backend',['cupy','torch'])
@pytest.mark.parametrize('case',['bandwidth_helpers','bandwidth_scott','bandwidth_silverman','bandwidth_nrd','bandwidth_nrd0','bandwidth_ucv',
                                'bandwidth_bcv','bandwidth_sj','bandwidth_sj-ste','bandwidth_sj-dpi',
                                'bandwidth_cv','kde_normal_ci','kde_bootstrap','knockoff_fixed','knockoff_model'])
def test_physical_gpu_stage_observation_when_available(backend,case):
    try:
        runtime=b.Runtime(backend,0)
    except b.BackendUnavailable as exc:
        pytest.skip(str(exc))
    row=b.run_case(case,runtime,48,4,42,0,1)
    assert row['status']=='pass' and row['correctness']['status']=='pass'
    assert row['execution_trace']
    assert all(trace['device']=='cuda:0' for trace in row['execution_trace'])
    assert row['performance_eligible'] is False
    assert row['execution_qualification']['compute_placement'] in ('host-scalar','mixed-host-and-observed-native-arithmetic','observed-native-arithmetic-with-host-control-and-api-boundaries')


def test_native_primitive_cpu_compute_then_recast_is_rejected(monkeypatch):
    rt=torch_runtime();torch=rt.module
    monkeypatch.setattr(torch,'exp',lambda x:torch.as_tensor(np.exp(x.numpy())))
    def target(x):return torch.exp(x)
    with pytest.raises(TraceIncomplete,match='replacement'):
        with NumericalTrace(rt,[stage(target,'density',('exp',))]):target(torch.ones(4))
    assert sys.getprofile() is None


def test_wrong_device_event_cannot_hide_behind_matching_event():
    rt=cpu_runtime()
    def target(x):return np.exp(x)
    trace=NumericalTrace(rt,[stage(target,'density',('exp',))])
    with trace:target(np.ones(4))
    bad=dict(trace.records[0]['operations'][0])
    bad['outputs']=[{'backend':'numpy','device':'wrong-device','dtype':'float64'}]
    trace.records[0]['operations'].append(bad)
    with pytest.raises(TraceIncomplete,match='wrong backend/device'):trace.report()


def test_unsupported_boundary_is_machine_readable(monkeypatch,tmp_path):
    import json
    rt=SimpleNamespace(name='torch',device='cuda:0',hardware={},context=nullcontext)
    monkeypatch.setattr(b,'Runtime',lambda *args:rt)
    received=[]
    def unsupported(*args, **kwargs):
        received.append(kwargs)
        raise b.UnsupportedCapability('unsupported kernel/shape combination')
    monkeypatch.setattr(b,'run_case',unsupported)
    output=tmp_path/'unsupported.json'
    assert b.main(['--cases','kde_bootstrap','--backends','torch','--require-backends','--output',str(output)])==2
    row=json.loads(output.read_text())['results'][0]
    assert received == [{'dtype':'float64','weights':'none','solver':'default',
                         'sparse':'dense','covariance_type':'default'}]
    assert row['status']=='blocked' and row['blocked_kind']=='unsupported-capability'
    assert row['reason']=='unsupported kernel/shape combination'
    assert 'timing' not in row


def test_callable_instance_cannot_spoof_backend_module(monkeypatch):
    rt=torch_runtime();torch=rt.module
    class Recast:
        __module__='torch'
        def __call__(self,x):return torch.as_tensor(np.exp(x.numpy()))
    monkeypatch.setattr(torch,'exp',Recast())
    def target(x):return torch.exp(x)
    with pytest.raises(TraceIncomplete,match='replacement'):
        with NumericalTrace(rt,[stage(target,'density',('exp',))]):target(torch.ones(4))
    assert sys.getprofile() is None


def test_numpy_generated_wrapper_does_not_trust_foreign_implementation():
    # Relevant to the generated NumPy 1.24 __array_function__ wrapper. Newer
    # NumPy uses a C dispatcher, exercised by the general native-call tests.
    if getattr(getattr(np.sum,'__code__',None),'co_filename',None)!='<__array_function__ internals>':
        return
    from numpy.core import overrides
    def dispatcher(x):return (x,)
    def foreign(x):return np.exp(x)
    fake=overrides.array_function_dispatch(dispatcher)(foreign)
    with pytest.raises(TraceIncomplete,match='replacement'):
        NumericalTrace._verify_native_callable(fake,np,'numpy','sum')


def test_current_numpy_array_function_primitives_are_recognized():
    for name in ('sum','mean','std','var','quantile'):
        NumericalTrace._verify_native_callable(getattr(np,name),np,'numpy',name)


@pytest.mark.parametrize('method', ['scott', 'silverman', 'nrd', 'nrd0', 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi'])
def test_torch_cpu_native_selector_stages(method):
    from statgpu.nonparametric import fit_kde
    rt = torch_runtime()
    x = rt.module.as_tensor(np.random.default_rng(8).normal(size=(48, 2)))
    trace = NumericalTrace(rt, learning_stages('bandwidth_' + method, 'torch'))
    with trace:
        fit_kde(x, bandwidth=method, backend='torch')
    assert trace.report()['device'] == 'cpu'


def test_native_rng_replacement_cannot_pass_by_returning_tensor(monkeypatch):
    rt = torch_runtime()
    original = rt.module.rand
    monkeypatch.setattr(rt.module, 'rand', lambda *args, **kwargs: original(*args, **kwargs))
    with pytest.raises(TraceIncomplete, match='replacement'):
        with NumericalTrace(rt, learning_stages('kde_bootstrap', 'torch')):
            pass
    assert sys.getprofile() is None


def test_native_rng_missing_from_one_replication_is_rejected():
    rt = torch_runtime()
    def draw():
        return rt.module.rand(3, dtype=rt.module.float64)
    trace = NumericalTrace(rt, [stage(draw, 'rng', ('rand',), minimum_calls=2)])
    with trace:
        draw()
        draw()
    trace.report()
    trace.records[-1]['operations'].clear()
    with pytest.raises(TraceIncomplete, match='expected rand'):
        trace.report()


def test_subclass_repeated_primitive_names_are_instrumented_once():
    class ExtendedTrace(NumericalTrace):
        OPS = NumericalTrace.OPS + ('cumsum', 'searchsorted')
    def target(x):
        return np.cumsum(x)
    trace = ExtendedTrace(cpu_runtime(), [stage(target, 'cdf', ('cumsum',))])
    with trace:
        target(np.ones(3))
    assert len(trace.report()['stages'][0]['operations']) == 1


def scalar_control_protocol_trace():
    """Synthetic observer protocol only, never CuPy execution evidence.

    CuPy v13.6.0 cupy/linalg/_norms.py:257-258 passes dtype.type(0/-inf)
    into where alongside native arrays. These scalars are genuinely NumPy
    scalars; native identities below are explicit metadata fixtures, not arrays.
    """
    def target():
        pass
    native = dict(backend='cupy', device='cuda:0', dtype='float64', kind='array', shape=[])
    scalar = identities(np.dtype('float64').type(0))[0]
    trace = NumericalTrace(SimpleNamespace(name='cupy', device='cuda:0'),
        [stage(target, 'source-derived-slogdet', ('where',))])
    event = dict(operation='where', backend='cupy',
        inputs=[dict(native, dtype='bool'), scalar, native.copy()], outputs=[native.copy()])
    trace.records = [dict(stage='source-derived-slogdet', completed=True, operations=[event])]
    return trace, event


def test_numpy_scalar_control_identity_is_not_zero_dimensional_array():
    scalar, array = identities(np.float64(0))[0], identities(np.asarray(0.))[0]
    assert scalar == dict(backend='numpy', device='cpu', dtype='float64', kind='numpy-scalar')
    assert array == dict(backend='numpy', device='cpu', dtype='float64', kind='array', shape=[])
    assert identities(0.) == []


@pytest.mark.parametrize('value', (0., -float('inf')))
def test_source_derived_numpy_scalar_control_protocol_roundtrips(value):
    trace, event = scalar_control_protocol_trace()
    event['inputs'][1] = identities(np.dtype('float64').type(value))[0]
    trace.records = json.loads(json.dumps(trace.records))
    report = trace.report()
    assert report['stages'][0]['operations'][0]['inputs'][1]['kind'] == 'numpy-scalar'
    assert report['stages'][0]['operations'][0]['outputs'][0]['shape'] == []


@pytest.mark.parametrize('failure', ('host-scalar-output', 'host-array-input', 'host-0d-input',
    'host-array-output', 'host-0d-output', 'wrong-device-input', 'wrong-device-output',
    'foreign-backend-input', 'foreign-backend-output', 'scalar-only-input',
    'forged-native-scalar-input', 'forged-native-scalar-output', 'scalar-with-array-shape',
    'missing-input-kind', 'unknown-input-kind', 'missing-output-kind', 'unknown-output-kind'))
def test_scalar_control_protocol_rejects_non_native_array_evidence(failure):
    trace, event = scalar_control_protocol_trace()
    if failure == 'host-scalar-output': event['outputs'].append(identities(np.float64(1.))[0])
    elif failure == 'host-array-input': event['inputs'].append(identities(np.ones(2))[0])
    elif failure == 'host-0d-input': event['inputs'].append(identities(np.asarray(0.))[0])
    elif failure == 'host-array-output': event['outputs'].append(identities(np.ones(2))[0])
    elif failure == 'host-0d-output': event['outputs'].append(identities(np.asarray(0.))[0])
    elif failure == 'wrong-device-input': event['inputs'][0]['device'] = 'cuda:1'
    elif failure == 'wrong-device-output': event['outputs'][0]['device'] = 'cuda:1'
    elif failure == 'foreign-backend-input': event['inputs'][0]['backend'] = 'torch'
    elif failure == 'foreign-backend-output': event['outputs'][0]['backend'] = 'torch'
    elif failure == 'scalar-only-input': event['inputs'] = [event['inputs'][1]]
    elif failure == 'forged-native-scalar-input': event['inputs'][0]['kind'] = 'numpy-scalar'
    elif failure == 'forged-native-scalar-output': event['outputs'][0]['kind'] = 'numpy-scalar'
    elif failure == 'scalar-with-array-shape': event['inputs'][1]['shape'] = []
    elif failure == 'missing-input-kind': event['inputs'][0].pop('kind')
    elif failure == 'unknown-input-kind': event['inputs'][0]['kind'] = 'unknown'
    elif failure == 'missing-output-kind': event['outputs'][0].pop('kind')
    elif failure == 'unknown-output-kind': event['outputs'][0]['kind'] = 'unknown'
    with pytest.raises(TraceIncomplete):
        trace.report()


def test_scalar_control_protocol_diagnostic_identifies_offending_operation_and_operands():
    trace, event = scalar_control_protocol_trace()
    event['outputs'].append(identities(np.float64(1.))[0])
    with pytest.raises(TraceIncomplete) as error:
        trace.report()
    text = str(error.value)
    for detail in ('source-derived-slogdet', 'operation=where', 'expected=cupy/cuda:0',
                   'inputs=', 'outputs=', 'numpy-scalar'):
        assert detail in text


def test_numpy_scalar_control_only_does_not_prove_cpu_array_work():
    def target(value):
        return np.exp(value)
    trace = NumericalTrace(cpu_runtime(), [stage(target, 'scalar-only', ('exp',))])
    with trace:
        target(np.float64(1.))
    with pytest.raises(TraceIncomplete, match='No matching native arithmetic'):
        trace.report()


def test_real_numpy_reduction_scalar_output_retains_cpu_array_evidence():
    def target(value):
        return np.sum(value)
    trace = NumericalTrace(cpu_runtime(), [stage(target, 'reduction', ('sum',))])
    with trace:
        target(np.ones(3))
    event = trace.report()['stages'][0]['operations'][0]
    assert event['inputs'][0]['kind'] == 'array'
    assert event['outputs'][0]['kind'] == 'numpy-scalar'


def test_numpy_scalar_control_with_array_shape_cannot_prove_cpu_array_work():
    trace, event = scalar_control_protocol_trace()
    trace.runtime = cpu_runtime()
    event['backend'] = 'numpy'
    event['inputs'] = [dict(identities(np.float64(1.))[0], shape=[])]
    event['outputs'] = identities(np.asarray(1.))
    with pytest.raises(TraceIncomplete):
        trace.report()


def test_cupy_ufunc_bound_method_metadata_without_gpu():
    """Real extension identity only; no CUDA allocation, device query or kernel."""
    cp = pytest.importorskip('cupy')
    assert cp.add.at.__self__ is cp.add
    NumericalTrace._verify_native_callable(cp.add.at, cp, 'cupy', 'add.at')
    original_add, original_sum = cp.add, cp.sum
    def target():
        raise AssertionError('Metadata-only test must not execute a GPU stage')
    runtime = SimpleNamespace(name='cupy', module=cp, device='cuda:0')
    with NumericalTrace(runtime, [stage(target, 'scatter', ('add.at',))]):
        assert cp.add is not original_add
    assert cp.add is original_add and cp.sum is original_sum
    assert sys.getprofile() is None
    # Audit every currently registered primitive, not only the reported method.
    for name in NumericalTrace.OPS:
        fn = getattr(cp, name, None)
        if callable(fn):
            NumericalTrace._verify_native_callable(fn, cp, 'cupy', name)
    for name in NumericalTrace.LINALG:
        fn = getattr(cp.linalg, name, None)
        if callable(fn):
            NumericalTrace._verify_native_callable(fn, cp, 'cupy', 'linalg.' + name)
    NumericalTrace._verify_native_callable(cp.random.RandomState.random_sample,
                                          cp, 'cupy', 'RandomState.random_sample')


@pytest.mark.parametrize('replacement', ['subtract.at', 'add.reduce', 'sum', 'add'])
def test_cupy_add_at_requires_exact_registered_method(replacement):
    cp = pytest.importorskip('cupy')
    fn = cp
    for part in replacement.split('.'):
        fn = getattr(fn, part)
    with pytest.raises(TraceIncomplete, match='bound method'):
        NumericalTrace._verify_native_callable(fn, cp, 'cupy', 'add.at')


@pytest.mark.parametrize('replacement', [np.add.at, [].append, list.append,
                                         lambda *args: None])
def test_cupy_add_at_rejects_foreign_builtins_and_python(replacement):
    module = SimpleNamespace(add=np.add, ufunc=np.ufunc)
    with pytest.raises(TraceIncomplete, match='bound method'):
        NumericalTrace._verify_native_callable(replacement, module, 'cupy', 'add.at')


def test_cupy_add_at_rejects_forged_callable_metadata():
    class Forged:
        __module__ = 'cupy._core._kernel'
        __name__ = 'at'
        __self__ = np.add
        def __call__(self, *args):
            raise AssertionError('Must not execute a replacement to authenticate it')
    with pytest.raises(TraceIncomplete, match='bound method'):
        NumericalTrace._verify_native_callable(Forged(),
            SimpleNamespace(add=np.add, ufunc=np.ufunc), 'cupy', 'add.at')

    class ufunc(list):
        __module__ = 'cupy._core._kernel'
        __name__ = 'cupy_add'
        at = list.append
    owner = ufunc()
    with pytest.raises(TraceIncomplete, match='bound method'):
        NumericalTrace._verify_native_callable(owner.at,
            SimpleNamespace(add=owner, ufunc=ufunc), 'cupy', 'add.at')


def test_cupy_add_at_rejects_replaced_owner_and_restores_startup(monkeypatch):
    cp = pytest.importorskip('cupy')
    original = cp.add
    original_sum = cp.sum
    monkeypatch.setattr(cp, 'add', cp.subtract)
    rt = SimpleNamespace(name='cupy', module=cp, device='cuda:0')
    def target():
        raise AssertionError('Startup rejection must precede execution')
    with pytest.raises(TraceIncomplete, match='bound method'):
        with NumericalTrace(rt, [stage(target, 'add-at', ('add.at',))]):
            target()
    assert cp.sum is original_sum
    assert cp.add is cp.subtract
    assert sys.getprofile() is None
    monkeypatch.setattr(cp, 'add', original)


def test_mutating_primitive_observer_records_destination_not_none():
    """CPU observer regression for the same in-place-output protocol as add.at."""
    def target(x):
        assert observed(x, [0, 0], [2., 3.]) is None
        return x
    trace = NumericalTrace(cpu_runtime(), [stage(target, 'scatter', ('add.at',))])
    observed = trace._wrap(np.add.at, 'numpy', 'add.at', mutated_argument=0)
    destination = np.zeros(2, dtype=np.float64)
    with trace:
        assert target(destination) is destination
    report = trace.report()
    assert report
    np.testing.assert_array_equal(destination, [5., 0.])
    event = trace.records[0]['operations'][0]
    assert event['outputs'][0]['dtype'] == 'float64'
    assert event['outputs'][0]['backend'] == 'numpy'
    assert event['outputs'][0]['shape'] == [2]
