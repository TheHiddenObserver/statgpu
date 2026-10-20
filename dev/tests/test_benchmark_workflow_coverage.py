"""Deterministic harness tests; no GPU/performance acceptance claim."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from dev.benchmarks import benchmark_workflow_coverage as bench


def args():
    return SimpleNamespace(seed=123,targets=3,warmup=0,repeats=1)


@pytest.mark.parametrize('case',bench.CASES)
def test_all_cpu_workflows(case):
    if case=='diagnostics':pytest.importorskip('statsmodels.api')
    if case=='formula':pytest.importorskip('pandas');pytest.importorskip('patsy')
    X,y=bench.fixture(48,4,123)
    row=bench.IMPLEMENTATIONS[case](bench.common.Runtime('numpy'),X,y,args())
    assert row['correctness']['status']=='pass'
    assert row['timing']
    for record in row['timing']:
        assert len(record['raw_seconds'])==1
        assert record['median_seconds']>=0
        assert record['synchronized_device']=='cpu'
    if case=='diagnostics':assert row['execution']['not_gpu_native']


@pytest.mark.parametrize('direction',['forward','backward','both'])
@pytest.mark.parametrize('criterion',['aic','bic'])
def test_reference_scores_match_statsmodels(direction,criterion):
    sm=pytest.importorskip('statsmodels.api')
    X,y=bench.fixture(48,4,123)
    for features,scores in bench.selection_reference(X,y,direction,criterion):
        reference=sm.OLS(y,np.column_stack((np.ones(len(y)),X[:,list(features)]))).fit()
        np.testing.assert_allclose([scores['aic'],scores['bic']],[reference.aic,reference.bic],atol=1e-9)


def test_failure_gate_has_no_successful_timing(tmp_path,monkeypatch):
    def broken(*args):raise AssertionError('oracle mismatch')
    monkeypatch.setitem(bench.IMPLEMENTATIONS,'multitarget',broken)
    out=tmp_path/'out.json'
    assert bench.main(['--smoke','--cases','multitarget','--output',str(out)])==2
    result=json.loads(out.read_text());row=result['rows'][0]
    assert row['status']=='failed' and 'timing' not in row
    assert result['validation']['status']=='failed'


def test_measure_validates_every_repetition():
    checks=[]
    _,record=bench.measure(lambda:3,lambda x:checks.append(x),bench.common.Runtime('numpy'),2,3,'test')
    assert checks==[3]*5 and len(record['raw_seconds'])==3


@pytest.mark.parametrize('value,expected',[(np.nan,0.),(np.inf,0.),([1,2],[1]),([1],[4])])
def test_correctness_rejects_bad_values(value,expected):
    with pytest.raises(AssertionError):bench.close(value,expected,'bad')


def test_source_records_helper_and_ignored_overlay():
    source=bench.source_identity()
    assert source['working_tree']=='dirty'
    assert 'dev/benchmarks/benchmark_workflow_coverage.py' in source['runner_dependencies']
    assert 'dev/benchmarks/benchmark_release_refresh.py' in source['runner_dependencies']
    assert len(source['worktree_fingerprint'])==64


def test_required_missing_backend_fails(tmp_path,monkeypatch):
    original=bench.common.Runtime
    def runtime(name,ordinal=0):
        if name=='cupy':raise bench.common.BackendUnavailable('test missing GPU')
        return original(name,ordinal)
    monkeypatch.setattr(bench.common,'Runtime',runtime)
    out=tmp_path/'out.json'
    command=['--cases','multitarget','--backends','numpy','cupy','--sizes','48','--features','4','--warmup','0','--repeats','1','--output',str(out)]
    assert bench.main(command)==0
    assert json.loads(out.read_text())['validation']['status']=='partial'
    assert bench.main(command+['--require-backends'])==2
    result=json.loads(out.read_text())
    assert result['rows'][-1]['status']=='unavailable'


def test_source_change_fails(tmp_path,monkeypatch):
    original=bench.source_identity;calls=[]
    def identity():
        value=original();calls.append(1)
        if len(calls)>1:value['worktree_fingerprint']='changed'
        return value
    monkeypatch.setattr(bench,'source_identity',identity)
    out=tmp_path/'out.json'
    assert bench.main(['--smoke','--cases','multitarget','--output',str(out)])==2
    assert json.loads(out.read_text())['rows'][-1]['reason']=='source changed during execution'


@pytest.mark.parametrize('mode',['module','direct'])
def test_cli_invocations(tmp_path,mode):
    output=tmp_path/f'{mode}.json'
    invocation=['-m','dev.benchmarks.benchmark_workflow_coverage'] if mode=='module' else ['dev/benchmarks/benchmark_workflow_coverage.py']
    proc=subprocess.run([sys.executable,*invocation,'--smoke','--cases','multitarget','--output',str(output)],cwd=bench.ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    result=json.loads(output.read_text())
    assert result['validation']['status']=='complete'
    assert result['source']==result['source_after']


@pytest.mark.parametrize('flag,value',[('--features','1'),('--repeats','0'),('--targets','0'),('--sizes','3')])
def test_invalid_cli(flag,value,tmp_path):
    with pytest.raises(SystemExit):bench.main([flag,value,'--output',str(tmp_path/'x')])
