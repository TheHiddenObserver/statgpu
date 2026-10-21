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


@pytest.fixture
def identity_checkout(tmp_path, monkeypatch):
    """Isolate executable bytes; emulate Git rather than depend on caller dirt."""
    scripts = tmp_path/'dev'/'benchmarks'
    scripts.mkdir(parents=True)
    helper = scripts/'benchmark_release_refresh.py'
    runner = scripts/'benchmark_workflow_coverage.py'
    helper.write_text('# helper\n')
    runner.write_text('# runner\n')
    package = tmp_path/'statgpu'
    package.mkdir()
    (package/'__init__.py').write_text('# package\n')
    monkeypatch.setattr(bench.common, 'ROOT', tmp_path)
    monkeypatch.setattr(bench.common, '__file__', str(helper))
    monkeypatch.setattr(bench, '__file__', str(runner))
    state = {'dirty': False, 'overlay': False, 'unavailable': False}

    def git(command, **kwargs):
        if state['unavailable']:
            raise FileNotFoundError('git unavailable')
        args = command[1:]
        if args[:1] == ['rev-parse']:
            return b'0123456789abcdef\n'
        if args[:1] == ['diff']:
            return b'changed executable' if state['dirty'] else b''
        if args[:1] == ['ls-files']:
            return b'' if state['overlay'] and args[-1].endswith('benchmark_workflow_coverage.py') else args[-1].encode()
        if args[:1] == ['status']:
            return b' M tracked-file' if state['dirty'] else b''
        raise AssertionError(command)

    monkeypatch.setattr(bench.common.subprocess, 'check_output', git)
    return state, runner, helper


@pytest.mark.parametrize('state_key,expected,overlay', [
    (None, 'clean', 'tracked'),
    ('dirty', 'dirty', 'tracked'),
    ('overlay', 'dirty', 'untracked-or-ignored runner'),
    ('unavailable', 'unknown', 'unknown: git provenance unavailable'),
])
def test_source_reports_real_identity(identity_checkout, state_key, expected, overlay):
    state, runner, helper = identity_checkout
    if state_key:
        state[state_key] = True
    source = bench.source_identity()
    assert source['working_tree'] == expected
    assert source['source_overlay'] == overlay
    assert source == bench.common.source_identity(extra_paths=(runner,))
    assert set(source['runner_dependencies']) == {
        'dev/benchmarks/benchmark_workflow_coverage.py',
        'dev/benchmarks/benchmark_release_refresh.py'}
    assert len(source['worktree_fingerprint']) == 64
    assert source == bench.source_identity()
    if expected == 'unknown':
        assert source['git_commit'] is None
        assert all(value is None for value in source['dependencies_tracked'].values())


def test_source_fingerprint_tracks_dependency_bytes_and_provenance(identity_checkout):
    state, runner, helper = identity_checkout
    clean = bench.source_identity()
    runner.write_text('# changed runner\n')
    changed_runner = bench.source_identity()
    assert changed_runner['worktree_fingerprint'] != clean['worktree_fingerprint']
    helper.write_text('# changed helper\n')
    changed_helper = bench.source_identity()
    assert changed_helper['worktree_fingerprint'] != changed_runner['worktree_fingerprint']
    state['unavailable'] = True
    unknown = bench.source_identity()
    assert unknown['worktree_fingerprint'] != changed_helper['worktree_fingerprint']
    assert unknown['runner_dependencies'] == changed_helper['runner_dependencies']


def test_source_git_command_failure_is_unknown(identity_checkout, monkeypatch):
    def failed(command, **kwargs):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(bench.common.subprocess, 'check_output', failed)
    assert bench.source_identity()['working_tree'] == 'unknown'


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
