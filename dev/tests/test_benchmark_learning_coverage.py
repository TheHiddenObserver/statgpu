"""Harness tests, not physical CUDA or production benchmark evidence."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dev.benchmarks import benchmark_learning_coverage as b


@pytest.mark.parametrize('case',b.CASES)
def test_cpu_cases_have_real_gate(case):
    rt=b.Runtime('numpy')
    row=b.run_case(case,rt,48,4,42,0,1)
    assert row['status']=='pass'
    assert row['correctness']['status']=='pass'
    assert row['correctness']['reference']
    assert 'timing' in row
    assert row['problem']['data_id']


def test_oracle_rejects_corrupt_krr():
    X,y,labels=b.data(32,4,42)
    m=b.estimator('kernel_ridge','cpu',42,32).fit(X,y)
    with pytest.raises(AssertionError):
        b.validate('kernel_ridge',m,X,y,np.zeros(32),labels)


def test_cv_selection_gate_checks_refit_and_folds():
    X,y,labels=b.data(32,4,42)
    m=b.estimator('kernel_ridge_cv','cpu',42,32).fit(X,y)
    out=m.predict(X);m.alpha_=999.
    with pytest.raises(AssertionError):b.validate('kernel_ridge_cv',m,X,y,out,labels)


def test_source_captures_ignored_runner_overlay():
    source=b.source_identity()
    assert len(source['script_sha256'])==64
    assert len(source['source_content_sha256'])==64
    if source['runner_overlay']:assert source['working_tree']=='dirty'


def test_gpu_cannot_use_host_array_as_proof():
    class FakeRuntime:
        name='torch';device='cuda:0'
    class FakeModel:
        components_=np.eye(2)
    with pytest.raises(b.Blocked,match='ownership'):
        b.verify_device(FakeModel(),np.ones((2,2)),FakeRuntime())


def test_scoped_diagnostic_case_not_performance_eligible():
    row=b.run_case('nmf',b.Runtime('numpy'),32,4,42,0,1)
    assert row['performance_eligible'] is False
    assert row['correctness']['strength']=='invariant-only-no-speedup'


def test_knockoff_threshold_rejects_wrong_selection():
    result=b.knockoff('knockoff_fixed',b.Runtime('numpy'),*b.data(32,4,42)[:2],42)
    result.selected_features=np.arange(4)
    with pytest.raises(AssertionError):b.validate_knockoff(result,[0])


def test_smoke_full_registry_is_complete(tmp_path):
    output=tmp_path/'smoke.json'
    assert b.main(['--smoke','--output',str(output)])==0
    j=json.loads(output.read_text())
    assert len(j['results'])==len(b.CASES)
    assert j['validation']['status']=='complete'
    blocked=[r for r in j['results'] if r['status']=='blocked']
    assert blocked==[]
    assert all('timing' not in r for r in blocked)
    assert j['source']==j['source_after']
    assert 'scikit-learn' in j['environment']['packages']
    assert j['environment']['environment_id']


def test_required_blocked_case_fails_exit(monkeypatch,tmp_path):
    def unavailable(*args,**kwargs):raise b.Blocked('controlled runtime resource limitation')
    monkeypatch.setattr(b,'run_case',unavailable)
    assert b.main(['--smoke','--cases','cyclic_spline','--require-backends','--output',str(tmp_path/'x.json')])==2


def test_independent_selection_seeds_not_timing_repeats(tmp_path):
    out=tmp_path/'selection.json'
    assert b.main(['--smoke','--cases','knockoff_fixed','--selection-seeds','1','2','3','--output',str(out)])==0
    j=json.loads(out.read_text());summary=j['selection_summary'][0]
    assert summary['independent_trial_count']==3
    assert summary['seeds']==[1,2,3]
    assert summary['fdp_mc_standard_error'] is not None


@pytest.mark.parametrize('entry',[['-m','dev.benchmarks.benchmark_learning_coverage'],['dev/benchmarks/benchmark_learning_coverage.py']])
def test_direct_module_entry(entry,tmp_path):
    out=tmp_path/'entry.json'
    result=subprocess.run([sys.executable,*entry,'--smoke','--cases','pca','--output',str(out)],cwd=b.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(out.read_text())['results'][0]['status']=='pass'


def test_unavailable_backend_recorded_without_fallback(monkeypatch,tmp_path):
    real=b.Runtime
    def runtime(name,*args):
        if name=='cupy':raise b.BackendUnavailable('test unavailable')
        return real(name,*args)
    monkeypatch.setattr(b,'Runtime',runtime)
    out=tmp_path/'missing.json'
    assert b.main(['--cases','pca','--backends','cupy','--require-backends','--output',str(out)])==2
    rows=json.loads(out.read_text())['results']
    assert rows==[{'case':'pca','backend':'cupy','seed':42,'status':'skipped','reason':'test unavailable',
                   'performance_eligible':False,'oracle_comparison_eligible':False}]


def test_failure_does_not_publish_timing(monkeypatch,tmp_path):
    def fail(*args,**kwargs):raise AssertionError('oracle mismatch')
    monkeypatch.setattr(b,'validate',fail)
    out=tmp_path/'fail.json'
    assert b.main(['--smoke','--cases','pca','--output',str(out)])==2
    row=json.loads(out.read_text())['results'][0]
    assert row['status']=='failed' and 'timing' not in row
    assert row['seed']==42


@pytest.mark.parametrize('extra',[['--selection-seeds','1','1'],['--backends','numpy','numpy'],['--cases','pca','pca']])
def test_duplicate_trial_identity_rejected(extra,tmp_path):
    with pytest.raises(SystemExit):b.main([*extra,'--output',str(tmp_path/'bad.json')])


def test_resource_cap_is_explicit_block_not_smaller_case(tmp_path):
    out=tmp_path/'cap.json'
    assert b.main(['--smoke','--cases','kernel_ridge','--max-pairwise-elements','100','--output',str(out)])==0
    row=json.loads(out.read_text())['results'][0]
    assert row['status']=='blocked' and 'resource cap' in row['reason']
    assert 'timing' not in row


def test_gpu_output_ownership_never_certifies_compute_placement():
    class Runtime:
        name='cupy';device='cuda:1'
    row=b.classify_placement({'status':'pass','performance_eligible':True},Runtime())
    assert row['performance_eligible'] is False
    assert row['execution_qualification']['compute_placement']=='unverified'
    assert row['execution_qualification']['array_device']=='cuda:1'


def test_broken_periodic_constraints_remain_numerical_failure(monkeypatch):
    from statgpu import nonparametric as N
    # An unconstrained clamped basis must still fail the unchanged boundary gate.
    monkeypatch.setattr(N,'cyclic_cubic_spline_basis',N.bspline_basis)
    with pytest.raises(AssertionError,match='boundary constraint violation'):
        b.special_case('cyclic_spline',b.Runtime('numpy'),48,42,0,1)
