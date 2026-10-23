"""Harness tests, not a CUDA or statistical-coverage certification."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from dev.benchmarks import benchmark_statistical_coverage as bench


@pytest.mark.parametrize('case', bench.case_ids())
def test_each_registered_adapter_cpu_smoke(case):
    row=bench.run_case(case,bench.Runtime('numpy'),48,3,20261020,12,0,1)
    assert row['status']=='passed'
    assert row['correctness']['status']=='pass'
    assert len(row['timing']['seconds'])==1
    assert row['timing']['median_seconds']>=0
    assert row['performance_claim_eligible'] is False


def test_registry_has_unique_real_cases():
    cases=bench.case_ids()
    assert len(cases)==len(set(cases))==138
    assert {c.split('/')[0] for c in cases}==set(bench.FAMILIES)
    for name in bench.DISTRIBUTIONS:
        assert f'distributions/{name}/rvs' in cases


def test_corrupted_values_rejected():
    call,check,_,_=bench.prepare('distributions/norm/cdf',bench.Runtime('numpy'),48,3,3,12)
    with pytest.raises(AssertionError):check(call()+.1)


def test_wrong_gpu_output_rejected():
    class Fake:
        name='cupy';device='cuda:0'
    with pytest.raises(AssertionError):bench.native(np.ones(4),Fake())


def test_case_selector_and_strict_json(tmp_path):
    out=tmp_path/'out.json'
    assert bench.main(['--smoke','--cases','multiple-testing','--output',str(out)])==0
    result=json.loads(out.read_text(),parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
    assert len(result['results'])==13
    assert result['source']==result['source_after']
    tracked = all(subprocess.run(
        ['git', 'ls-files', '--error-unmatch', path],
        cwd=bench.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0 for path in (
        'dev/benchmarks/benchmark_statistical_coverage.py',
        'dev/benchmarks/benchmark_release_refresh.py',
    ))
    assert result['source']['runner_overlay'] is (not tracked)
    assert result['environment']['packages']['scikit-learn']
    assert result['environment']['packages']['statsmodels']
    assert result['environment']['environment_id']


def test_bad_selector_rejected(tmp_path):
    with pytest.raises(SystemExit):bench.main(['--smoke','--cases','made-up','--output',str(tmp_path/'x.json')])


def test_source_fingerprint_includes_reused_runtime():
    identity=bench.source_identity()
    assert 'dev/benchmarks/benchmark_statistical_coverage.py' in identity['file_sha256']
    assert 'dev/benchmarks/benchmark_release_refresh.py' in identity['file_sha256']


def test_failure_emits_no_timing_and_nonzero(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise TypeError("injected production error")
    monkeypatch.setattr(bench, "run_case", fail)
    out=tmp_path/'failed.json'
    assert bench.main(['--smoke','--cases','distributions/f/rvs','--output',str(out)])==2
    row=json.loads(out.read_text())['results'][0]
    assert row['status']=='failed' and 'timing' not in row
    assert 'TypeError' in row['reason']


def test_backend_unavailable_never_cpu_fallback(monkeypatch,tmp_path):
    real=bench.Runtime
    def runtime(name,ordinal):
        if name!='numpy':raise bench.BackendUnavailable('test unavailable')
        return real(name,ordinal)
    monkeypatch.setattr(bench,'Runtime',runtime)
    out=tmp_path/'missing.json'
    args=['--backends','cupy','--cases','anova/f_oneway','--sizes','48','--output',str(out)]
    assert bench.main(args+['--require-backends'])==2
    data=json.loads(out.read_text());assert data['results'][0]['status']=='skipped'
    assert 'timing' not in data['results'][0]


@pytest.mark.parametrize('module_mode',[False,True])
def test_entry_points(module_mode,tmp_path):
    target=['-m','dev.benchmarks.benchmark_statistical_coverage'] if module_mode else [str(Path(bench.__file__))]
    out=tmp_path/'direct.json'
    proc=subprocess.run([sys.executable,*target,'--smoke','--cases','anova','--output',str(out)],cwd=bench.ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert len(json.loads(out.read_text())['results'])==7


def test_missing_external_reference_is_explicit(monkeypatch,tmp_path):
    def missing(*a,**kw):raise ModuleNotFoundError('No statsmodels',name='statsmodels')
    monkeypatch.setattr(bench,'prepare',missing)
    out=tmp_path/'reference-missing.json'
    assert bench.main(['--smoke','--cases','ordered','--output',str(out)])==0
    result=json.loads(out.read_text())
    assert result['validation']['status']=='partial'
    assert all(row['status']=='skipped' and 'Reference dependency unavailable' in row['reason'] for row in result['results'])
    assert all('timing' not in row for row in result['results'])


@pytest.mark.parametrize('quantity', ['statistic', 'pvalue'])
@pytest.mark.parametrize('operation', ['f_oneway', 'f_welch'])
def test_anova_failure_reports_full_precision_context(quantity, operation):
    from dataclasses import replace
    if operation == 'f_welch':
        pytest.importorskip('statsmodels')
    call, check, _, _ = bench.prepare('anova/'+operation, bench.Runtime('numpy'),
                                      128, 4, 20261020, 12)
    out = call()
    corrupted = replace(out, **{quantity:getattr(out, quantity)+.001})
    with pytest.raises(bench.ComparisonError) as caught:
        check(corrupted)
    detail = caught.value.diagnostics
    assert detail['quantity'] == quantity
    assert detail['actual'] == getattr(corrupted, quantity)
    assert detail['actual_dtype'] == detail['expected_dtype'] == 'float64'
    assert detail['rtol'] == 1e-6 and detail['atol'] == 1e-7
    assert detail['max_abs_error'] > .00099
    context = detail['context']
    assert context['statistic'] == corrupted.statistic
    assert context['dfn'] == out.df_between and context['dfd'] == out.df_within
    assert context['backend'] == 'numpy' and context['device'] == 'cpu'
    assert context['input_dtype'] == 'float64'
    json.dumps(detail, allow_nan=False)


def test_failed_anova_json_retains_comparison_not_timing(monkeypatch, tmp_path):
    from dataclasses import replace
    real = bench.prepare

    def corrupted(*args, **kwargs):
        call, check, problem, reference = real(*args, **kwargs)
        def changed():
            out = call()
            return replace(out, pvalue=out.pvalue+.001)
        return changed, check, problem, reference

    monkeypatch.setattr(bench, 'prepare', corrupted)
    out = tmp_path/'anova-failure.json'
    assert bench.main(['--smoke', '--cases', 'anova/f_oneway', '--output', str(out)]) == 2
    row = json.loads(out.read_text())['results'][0]
    assert row['correctness']['status'] == 'fail'
    assert row['correctness']['comparison']['quantity'] == 'pvalue'
    assert 'timing' not in row


def test_nonfinite_diagnostic_serializes_as_strict_json():
    with pytest.raises(bench.ComparisonError) as caught:
        bench.close(np.nan, 0.3, label='pvalue', context={'statistic':np.inf})
    detail = caught.value.diagnostics
    assert detail['actual'] == 'nan' and detail['context']['statistic'] == 'inf'
    json.dumps(detail, allow_nan=False)
