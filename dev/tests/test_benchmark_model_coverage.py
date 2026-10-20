"""Contract checks for supervised benchmarks; CPU smoke is not GPU evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.benchmarks import benchmark_model_coverage as b

@pytest.mark.parametrize('case',b.CASES)
def test_cpu_case_has_independent_gate_and_raw_time(case):
    row=b.run_case(case,b.Runtime('numpy'),48,3,20261020,0,1)
    assert row['status']=='passed'
    assert row['correctness']['status']=='pass'
    assert row['contract']['reference']
    assert len(row['timing']['raw_seconds'])==1
    assert row['timing']['median_seconds']>=0
    assert row['timing']['synchronized_device'] is None

@pytest.mark.parametrize('name',b.PENALTIES)
def test_orthogonal_reference_does_not_use_production_penalty(name):
    X,y,_=b.orthogonal_data(40,4,7)
    target=b.orthogonal_oracle(X,y,.08,name)
    assert np.isfinite(target).all()
    np.testing.assert_allclose(X.T@X/40,np.eye(4),atol=1e-12)
    assert target.shape==(5,)


def test_wrong_reference_is_not_timed(monkeypatch):
    old=b.make_case
    def bad(*args,**kwargs):
        factory,values,kw,target,contract,infer=old(*args,**kwargs)
        return factory,values,kw,target+10,contract,infer
    monkeypatch.setattr(b,'make_case',bad)
    row=b.run_case('glm:gaussian',b.Runtime('numpy'),48,3,5,0,1)
    assert row['status']=='failed'
    assert 'oracle mismatch' in row['correctness']['reason']
    assert 'timing' not in row


def test_gpu_provenance_rejects_request_only_and_mixed_routes():
    with pytest.raises(RuntimeError,match='could not be verified'):b.verify_execution([], 'torch','cuda:1')
    good={'backend':'torch','device':'cuda:1','function':'statgpu.solvers.fista'}
    b.verify_execution([good],'torch','cuda:1')
    with pytest.raises(RuntimeError,match='contradictory'):
        b.verify_execution([good,{'backend':'numpy','device':'cpu','function':'_fit_cpu'}],'torch','cuda:1')
    with pytest.raises(RuntimeError):b.verify_execution([good],'torch','cuda:0')


def test_source_hash_captures_ignored_runner():
    s=b.source_identity();name='dev/benchmarks/benchmark_model_coverage.py'
    assert name in s['file_hashes']
    assert len(s['worktree_fingerprint'])==64
    if s['runner_overlay']:assert s['working_tree']=='dirty'

@pytest.mark.parametrize('mode',['module','direct'])
def test_cli_smoke(mode,tmp_path):
    command=[sys.executable]+(['-m','dev.benchmarks.benchmark_model_coverage'] if mode=='module' else [str(ROOT/'dev/benchmarks/benchmark_model_coverage.py')])
    output=tmp_path/'smoke.json'
    result=subprocess.run(command+['--smoke','--cases','glm:poisson','panel:within','--output',str(output)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    data=json.loads(output.read_text());assert len(data['rows'])==2
    assert data['validation']['canonical_ready'] is False
    assert data['environment']['statsmodels']
    assert data['source']==data['source_after']


def test_require_unavailable_is_nonzero(monkeypatch,tmp_path):
    def missing(*args,**kwargs):raise b.BackendUnavailable('controlled absent GPU')
    monkeypatch.setattr(b,'Runtime',missing)
    out=tmp_path/'unavailable.json'
    code=b.main(['--cases','glm:gaussian','--backends','cupy','--require-backends','--output',str(out)])
    assert code==2
    data=json.loads(out.read_text());assert data['rows'][0]['status']=='skipped'
    assert 'timing' not in data['rows'][0]


def test_cox_zero_coefficient_risk_set_objective():
    X=np.array([[0.],[1.],[2.]])
    actual=b.cox_objective(np.zeros(1),X,np.array([1.,2.,3.]),np.ones(3))
    assert actual==pytest.approx((np.log(3)+np.log(2))/3)


def test_bad_shape_rejected(tmp_path):
    with pytest.raises(SystemExit):b.main(['--n','3','--p','3','--output',str(tmp_path/'a.json')])


def test_profile_only_observes_preflight(monkeypatch):
    count=[]
    original=b.Observe.__enter__
    def enter(self):count.append(1);return original(self)
    monkeypatch.setattr(b.Observe,'__enter__',enter)
    row=b.run_case('glm:gaussian',b.Runtime('numpy'),48,3,7,1,3)
    assert len(count)==1
    assert len(row['timing']['raw_seconds'])==3
    assert 'unprofiled' in row['timing']['instrumentation']


GLM_INFERENCE_REPAIR_CASES = [
    f'glm_inference:{family}:{cov}'
    for family in ('inverse_gaussian', 'negative_binomial')
    for cov in ('hc0', 'nonrobust')
]


@pytest.mark.parametrize('case', GLM_INFERENCE_REPAIR_CASES)
def test_glm_covariance_oracle_realigns_reporting_params_real_torch_cpu(case, monkeypatch):
    """Real tensor arithmetic exercises the mixed-state bug without claiming CUDA."""
    torch = pytest.importorskip('torch')
    from statgpu.backends import TorchBackend
    from statgpu.inference import _sandwich

    factory, args, fitkw, target, contract, inference = b.make_case(case, 96, 4, 42, 'numpy')
    model = factory()
    # Public device='torch' means CUDA. This is an explicit test-only Torch CPU
    # backend injection, not an availability fallback in production or a runner.
    monkeypatch.setattr(model, '_get_backend', lambda **kwargs: TorchBackend(device='cpu'))
    model.fit(*(torch.as_tensor(value) for value in args), **fitkw)
    assert isinstance(model._params, np.ndarray)
    assert isinstance(model._X_design, torch.Tensor)
    assert model._X_design.device.type == 'cpu'
    reporting_params = model._params
    saved_params = reporting_params.copy()
    original = _sandwich.m_estimation_inference
    observed = []

    def check_native(loss, X, y, coef, **kwargs):
        assert isinstance(coef, torch.Tensor)
        assert coef.device == X.device
        assert coef.dtype == X.dtype
        observed.append(coef)
        return original(loss, X, y, coef, **kwargs)

    monkeypatch.setattr(_sandwich, 'm_estimation_inference', check_native)
    metrics = b.check_fit(model, target, contract, inference)
    assert observed and 'covariance_max_abs_error' in metrics
    assert model._params is reporting_params
    np.testing.assert_array_equal(model._params, saved_params)
    # A device conversion fix must not weaken the independent covariance gate.
    corrupted = dict(inference, covariance=inference['covariance'] + 1.)
    with pytest.raises(AssertionError, match='full GLM covariance oracle mismatch'):
        b.check_fit(model, target, contract, corrupted)


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', GLM_INFERENCE_REPAIR_CASES)
def test_glm_inference_benchmark_on_physical_gpu_when_available(backend, case):
    try:
        runtime = b.Runtime(backend, 0)
    except b.BackendUnavailable as exc:
        pytest.skip(str(exc))
    row = b.run_case(case, runtime, 96, 4, 42, 0, 1)
    assert row['status'] == 'passed', row
    assert row['correctness']['status'] == 'pass'
    assert 'covariance_max_abs_error' in row['correctness']['metrics']
    assert row['timing']['synchronized_device'] == 'cuda:0'
