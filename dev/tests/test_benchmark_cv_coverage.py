import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dev.benchmarks import benchmark_cv_coverage as b


def record():
    run={'framework':'statgpu','backend':'numpy','status':'success','reason':None,
         'selected_parameters':{'alpha':0.1},'timing':{'cv_evaluation_ms':1.,'final_refit_ms':1.,'total_fit_ms':3.},
         'repeat_samples':[{'cv_evaluation_ms':1.,'final_refit_ms':1.,'total_fit_ms':3.}],
         'scores':{'validation_score':.1,'final_score':.2},
         'convergence':{'fold_count':3,'candidate_count':3,'final_refit_converged':True,'failed_candidates':0,'failed_folds':0}}
    return {'cases':[{'model_id':s.model_id,'case_id':s.model_id,'cv':{'fold_count':3},'grid':{'candidate_count':3},'runs':[copy.deepcopy(run)]} for s in b.legacy.CASE_SPECS]}


def test_selected_backend_valid_without_padding():b.validate_scoped_source(record(),['numpy'],1)


@pytest.mark.parametrize('field,value', [('total_fit_ms',float('nan')),('cv_evaluation_ms',-1),('total_fit_ms',.5)])
def test_invalid_durations_rejected(field,value):
    x=record();x['cases'][0]['runs'][0]['timing'][field]=value
    with pytest.raises(ValueError):b.validate_scoped_source(x,['numpy'],1)


def test_missing_family():
    x=record();x['cases'].pop()
    with pytest.raises(ValueError):b.validate_scoped_source(x,['numpy'],1)


def test_duplicate_case():
    x=record();x['cases'][0]['case_id']=x['cases'][1]['case_id']
    with pytest.raises(ValueError):b.validate_scoped_source(x,['numpy'],1)


def test_repeat_mismatch():
    with pytest.raises(ValueError):b.validate_scoped_source(record(),['numpy'],2)


def test_convergence_rejected():
    x=record();x['cases'][0]['runs'][0]['convergence']['failed_folds']=1
    with pytest.raises(ValueError):b.validate_scoped_source(x,['numpy'],1)


def test_non_success_timing_forbidden():
    x=record();x['cases'][0]['runs'][0].update(status='unavailable',reason='absent')
    with pytest.raises(ValueError):b.validate_scoped_source(x,['numpy'],1)


def test_missing_backend():
    with pytest.raises(ValueError):b.validate_scoped_source(record(),['numpy','torch'],1)


def test_external_mismatch_rejected():
    case=record()['cases'][0];ref=copy.deepcopy(case['runs'][0]);ref.update(framework='sklearn',backend=None);ref['scores']['final_score']=20.;case['runs'].append(ref)
    with pytest.raises(AssertionError):b.check_external_alignment(case)


def test_source_records_helpers():
    s=b.source_identity()
    assert isinstance(s['runner_overlay'],bool)
    assert s['working_tree'] in {'clean','dirty'}
    assert 'dev/benchmarks/benchmark_cv_models.py' in s['files_sha256']
    assert 'dev/benchmarks/cv_external.py' in s['files_sha256']
    assert 'dev/benchmarks/cv_source.py' in s['files_sha256']


def test_cpu_smoke(tmp_path):
    out=tmp_path/'cv.json'
    assert b.main(['--smoke','--output',str(out)])==0
    d=json.loads(out.read_text());assert len(d['results'])==6
    assert {r['status'] for r in d['results']}=={'pass'}
    assert d['validation']['canonical_ready'] is False
    assert d['source']['content_sha256']==d['source_after']['content_sha256']
    for row in d['results']:assert row['correctness']['direct_refit']['status']=='pass'


def test_no_available_backend_partial(monkeypatch):
    monkeypatch.setattr(b.legacy,'_backend_status',lambda x:(False,x,'not installed'))
    a=SimpleNamespace(sizes=[60],backends=['torch'],features=4,repeats=1,warmup=0,seed=1,smoke=False)
    d=b.run(a);assert d['validation']['status']=='partial'
    assert len(d['results'])==6
    assert all(r['timing'] is None for r in d['results'])


def fake_torch(monkeypatch):
    class Tensor:
        def __init__(self,index=0):self.is_cuda=True;self.device=SimpleNamespace(index=index)
    module=SimpleNamespace(Tensor=Tensor,cuda=SimpleNamespace(device=lambda i:nullcontext(),synchronize=lambda **kw:None))
    monkeypatch.setitem(sys.modules,'torch',module)
    monkeypatch.setattr(b.legacy.RegionProfiler,'__exit__',lambda *a:False)
    return Tensor


def test_gpu_proof_must_cover_both_regions(monkeypatch):
    fake_torch(monkeypatch)
    with b.observed_backend('torch'):
        p=b.legacy.RegionProfiler(object(),60,lambda:None)
        with pytest.raises(RuntimeError,match='both'):p.__exit__(None,None,None)


def test_gpu_observation_and_sync_restore(monkeypatch):
    Tensor=fake_torch(monkeypatch);original=b.legacy.RegionProfiler
    with b.observed_backend('torch') as proofs:
        p=b.legacy.RegionProfiler(object(),60,lambda:None)
        p.selector_stack=[-1];p.refit_stack=[-1]
        frame=SimpleNamespace(f_globals={'__name__':'statgpu.solvers._fista'},f_code=SimpleNamespace(co_name='fista_solver'),f_locals={'X':Tensor()})
        p._profile(frame,'call',None);p.__exit__(None,None,None)
        assert proofs[0]['synchronized_device']=='cuda:0'
        assert all(proofs[0]['observed_numerical_calls'].values())
    assert b.legacy.RegionProfiler is original


def test_wrong_cuda_device_rejected(monkeypatch):
    Tensor=fake_torch(monkeypatch)
    with b.observed_backend('torch'):
        p=b.legacy.RegionProfiler(object(),60,lambda:None)
        frame=SimpleNamespace(f_globals={'__name__':'statgpu.solvers._fista'},f_code=SimpleNamespace(co_name='fista_solver'),f_locals={'X':Tensor(1)})
        with pytest.raises(RuntimeError,match='unexpected'):p._profile(frame,'call',None)


def test_gpu_cpu_dispatch_rejected(monkeypatch):
    fake_torch(monkeypatch)
    with b.observed_backend('torch'):
        p=b.legacy.RegionProfiler(object(),60,lambda:None);p.selector_stack=[-1]
        frame=SimpleNamespace(f_globals={'__name__':'statgpu.linear_model.penalized._fit_mixin'},f_code=SimpleNamespace(co_name='_fit_cpu'),f_locals={})
        with pytest.raises(RuntimeError,match='CPU fit dispatch'):p._profile(frame,'call',None)


def test_invalid_source_is_failed_report(monkeypatch):
    monkeypatch.setattr(b.legacy,'build_source',lambda args:{'cases':[]})
    a=SimpleNamespace(sizes=[60],backends=['numpy'],features=4,repeats=1,warmup=0,seed=1,smoke=False)
    d=b.run(a)
    assert d['validation']['status']=='failed'
    assert len(d['results'])==6
    assert all(r['status']=='failed' and r['timing'] is None for r in d['results'])
