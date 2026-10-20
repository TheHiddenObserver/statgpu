"""Observer contracts; Torch CPU tests are not physical CUDA evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, stage, learning_stages
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


def test_host_scalar_rule_is_reported_without_gpu_compute_claim():
    row=b.run_case('bandwidth_helpers',b.Runtime('numpy'),32,4,42,0,1)
    assert row['performance_eligible'] is False
    assert row['execution_qualification']['compute_placement']=='host-scalar'
    assert len(row['execution_trace'][0]['stages'])==2
    assert all(r['placement']=='host-scalar-no-GPU-work-required' for r in row['execution_trace'][0]['stages'])


def test_torch_bootstrap_stays_genuine_unsupported():
    rt=SimpleNamespace(name='torch')
    called=[]
    with pytest.raises(b.Blocked,match='unsupported: Torch KDE bootstrap'):
        b.traced_call('kde_bootstrap',lambda:called.append(True),rt,[])
    assert called==[]


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
    if backend=='torch' and case=='kde_bootstrap':
        with pytest.raises(b.Blocked,match='unsupported: Torch KDE bootstrap'):
            b.run_case(case,runtime,48,4,42,0,1)
        return
    row=b.run_case(case,runtime,48,4,42,0,1)
    assert row['status']=='pass' and row['correctness']['status']=='pass'
    assert row['execution_trace']
    assert all(trace['device']=='cuda:0' for trace in row['execution_trace'])
    assert row['performance_eligible'] is False
    assert row['execution_qualification']['compute_placement'] in ('host-scalar','mixed-host-and-observed-native-arithmetic')


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
    output=tmp_path/'unsupported.json'
    assert b.main(['--cases','kde_bootstrap','--backends','torch','--require-backends','--output',str(output)])==2
    row=json.loads(output.read_text())['results'][0]
    assert row['status']=='blocked' and row['blocked_kind']=='unsupported-capability'
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
