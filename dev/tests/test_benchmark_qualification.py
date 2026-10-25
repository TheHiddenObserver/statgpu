import copy
import json
import pytest
from dev.benchmarks import benchmark_qualification as q


def artifact(backend='cupy', case='example'):
    source={'git_commit':'a'*40, 'working_tree':'clean', 'worktree_fingerprint':'b'*64}
    return {'schema_version':'statgpu-benchmark-v2', 'source':source,
            'source_after':copy.deepcopy(source), 'validation':{'status':'complete'},
            'results':[{'case':case,'backend':backend,'status':'pass',
                        'correctness':{'status':'pass'},
                        'timing':{'raw_seconds':[.1,.2],'repeats':2}}]}


@pytest.mark.parametrize('suite', q.SUITES)
@pytest.mark.parametrize('backend',['numpy','cupy','torch'])
def test_all_suites_share_nonpromotional_policy(suite,backend):
    doc=artifact(backend)
    if suite == 'six-family-cv':
        doc['schema_version']='statgpu-scoped-cv-candidate-v2'
        doc['results'][0]['correctness']={'direct_refit':{'status':'pass'},'external':{'status':'pass'}}
        doc['results'][0]['clean_public_fit']={'status':'pass','instrumented':False,'raw_seconds':[.1,.2],'repeats':2}
        doc['results'][0]['model_id']='example'
        doc['results'][0]['case']={'runs':[{'framework':'statgpu','backend':backend,'status':'success'}]}
    original=json.dumps(doc,sort_keys=True)
    result=q.evaluate_artifact(doc,suite,backend,expected_source='a'*40,expected_cases=['example'])
    assert result['correctness']['status']=='complete'
    assert result['performance']['eligible'] is False
    assert result['performance']['comparisons']==[]
    assert result['performance']['exclusive_gpu_ownership']=='UNVERIFIED_EXCLUSIVITY'
    assert not result['whole_pipeline_native_gpu']
    assert json.dumps(doc,sort_keys=True)==original


@pytest.mark.parametrize('change',[
    lambda d:d.update(schema_version='other'),
    lambda d:d.update(validation=None),
    lambda d:d.update(source=None),
    lambda d:d.update(source_after={}),
    lambda d:d['source'].update(git_commit='bad'),
    lambda d:d['source'].pop('worktree_fingerprint'),
    lambda d:d.update(results=[]),
    lambda d:d.update(results=[None]),
    lambda d:d['results'][0].update(backend='numpy'),
    lambda d:d['results'][0].update(status='not-applicable'),
    lambda d:d['results'][0].update(status=[]),
    lambda d:d['results'][0]['correctness'].update(status='failed'),
    lambda d:d['results'][0].update(execution_trace={'status':'failed'}),
    lambda d:d['results'][0]['timing'].update(raw_seconds=[float('nan'),.1]),
    lambda d:d['results'][0]['timing'].update(raw_seconds=[True,.1]),
    lambda d:d['results'][0]['timing'].update(raw_seconds=[-.1,.1]),
    lambda d:d['results'][0]['timing'].update(repeats=7),
])
def test_malformed_or_failed_artifacts_never_pass(change):
    doc=artifact();change(doc)
    assert q.evaluate_artifact(doc,'model-families','cupy')['correctness']['status']=='failed'


@pytest.mark.parametrize('status',['blocked','unavailable','skipped','partial'])
def test_missing_coverage_stays_partial(status):
    doc=artifact();doc['results'][0]['status']=status
    assert q.evaluate_artifact(doc,'learning-families','cupy')['correctness']['status']=='partial'


@pytest.mark.parametrize('backend',['cupy','torch'])
@pytest.mark.parametrize('case',sorted(q.HOST_ONLY))
def test_only_exact_host_not_applicable_contract(backend,case):
    doc=artifact(backend,case);doc['results'][0]['status']='not-applicable'
    assert q.evaluate_artifact(doc,'statistical-families',backend)['correctness']['status']=='complete'
    assert q.evaluate_artifact(doc,'model-families',backend)['correctness']['status']=='failed'


def test_producer_flags_and_output_cuda_are_not_gpu_proof():
    doc=artifact();doc['performance_eligible']=True
    doc['results'][0].update(performance_eligible=True,output_device='cuda:0',dtype='float32')
    got=q.evaluate_artifact(doc,'classification-metrics','cupy')
    assert got['rows'][0]['device_claim']=='not-established-by-this-artifact'
    assert got['rows'][0]['compute_dtype_claim']=='not-inferred-from-input-or-output'
    assert not got['performance']['eligible']


def test_exact_source_cases_and_nonzero_dominate():
    for kw in ({'exit_code':2},{'expected_source':'c'*40},{'expected_cases':['missing']}):
        assert q.evaluate_artifact(artifact(),'model-families','cupy',**kw)['correctness']['status']=='failed'


def test_dirty_local_run_not_canonical():
    doc=artifact();doc['source']['working_tree']='dirty';doc['source_after']=copy.deepcopy(doc['source'])
    got=q.evaluate_artifact(doc,'model-families','cupy')
    assert got['correctness']['status']=='partial'
    assert not got['performance']['canonical_dashboard_ready']


def test_gaussian_plain_fits_only_in_measured_region(monkeypatch):
    from dev.benchmarks import benchmark_release_refresh as runner
    original=runner.time_call
    calls=[]
    def checked(call,runtime,device):
        result,seconds=original(call,runtime,device)
        if hasattr(result,'fit'):
            calls.append(type(result))
            assert not isinstance(result,runner._ObserveFit)
        return result,seconds
    monkeypatch.setattr(runner,'time_call',checked)
    row=runner.run_case(runner.Runtime('numpy'),model_name='ridge',n=20,p=2,
                        dtype='float64',weighted=True,seed=2,alpha=.2,warmup=1,repeats=2)
    assert row['status']=='passed' and len(calls)==3
    assert row['timing']['instrumented'] is False
    assert row['backend']['fit_ingress']['route']=='numpy'


def test_workflow_preflight_and_host_scope_do_not_claim_cuda_compute():
    from dev.benchmarks import benchmark_workflow_coverage as runner
    events=[]
    class Runtime:
        device='cuda:0'
        def sync(self, device): events.append('sync')
    def call(): events.append('call'); return 1
    def check(value): events.append('check'); assert value==1
    _,timing=runner.measure(call,check,Runtime(),0,2,'cached-host-leverage-access')
    assert events[:2]==['call','check']
    assert events.count('call')==3
    assert timing['compute_placement']=='host-cpu'
    assert timing['synchronized_device']=='cuda:0'  # barrier is not compute placement
    assert not timing['instrumented']


@pytest.mark.parametrize('field,value',[
    ('correctness',None),('correctness',[]),('correctness',{'status':'bogus'}),
    ('backend_provenance',{'status':'failed'}),('execution_qualification',{'status':'failed'}),
    ('execution_trace',{'status':'bogus'}),('clean_public_fit',{'status':'failed'}),
])
def test_missing_or_contradictory_evidence_is_not_pass(field,value):
    doc=artifact();doc['results'][0][field]=value
    assert q.evaluate_artifact(doc,'model-families','cupy')['correctness']['status']=='failed'


@pytest.mark.parametrize('timing',[None,{}, {'raw_seconds':None}, {'raw_seconds':[]}, {'seconds':.1}])
def test_missing_timing_samples_cannot_be_complete(timing):
    doc=artifact();doc['results'][0]['timing']=timing
    assert q.evaluate_artifact(doc,'model-families','cupy')['correctness']['status']=='failed'


def test_statistical_seconds_array_is_validated():
    doc=artifact();doc['results'][0]['timing']={'seconds':[.1,.2],'repeats':2}
    assert q.evaluate_artifact(doc,'statistical-families','cupy')['correctness']['status']=='complete'
    doc['results'][0]['timing']['seconds']=[float('inf')]
    assert q.evaluate_artifact(doc,'statistical-families','cupy')['correctness']['status']=='failed'


def cv_artifact():
    doc=artifact();doc['schema_version']='statgpu-scoped-cv-candidate-v2'
    row=doc['results'][0];row['model_id']='RidgeCV'
    row['clean_public_fit']={'status':'pass','instrumented':False,'raw_seconds':[.1,.2],'repeats':2}
    row['case']={'runs':[{'framework':'statgpu','backend':'cupy','status':'success'},
                         {'framework':'sklearn','status':'success'}]}
    row['correctness']={'direct_refit':{'status':'pass'},'external':{'status':'pass'}}
    return doc


def test_failed_cv_reference_cannot_hide_beneath_parent_pass():
    doc=cv_artifact()
    assert q.evaluate_artifact(doc,'six-family-cv','cupy')['correctness']['status']=='complete'
    doc['results'][0]['case']['runs'][1]['status']='failed'
    assert q.evaluate_artifact(doc,'six-family-cv','cupy')['correctness']['status']=='failed'


def test_cv_external_not_applicable_is_exactly_scoped():
    doc=cv_artifact();doc['results'][0]['correctness']['external']['status']='not_applicable'
    assert q.evaluate_artifact(doc,'six-family-cv','cupy')['correctness']['status']=='failed'


def test_arbitrary_precision_integer_duration_fails_without_crashing():
    doc=artifact();doc['results'][0]['timing']['raw_seconds']=[10**400,.1]
    assert q.evaluate_artifact(doc,'model-families','cupy')['correctness']['status']=='failed'


def test_cv_v2_requires_clean_lane_not_just_profiled_timing():
    doc=cv_artifact();doc['results'][0].pop('clean_public_fit')
    assert q.evaluate_artifact(doc,'six-family-cv','cupy')['correctness']['status']=='failed'


@pytest.mark.parametrize("suite", ["gaussian-refresh", "classification-metrics"])
@pytest.mark.parametrize("declared", [
    {"requested": "cupy", "executed": "numpy"},
    {"requested": "cupy", "executed": None},
    {"requested": "numpy", "executed": "cupy"},
])
def test_contradictory_backend_declarations_fail(suite, declared):
    doc = artifact()
    doc["results"][0]["backend"] = declared
    result = q.evaluate_artifact(doc, suite, "cupy")
    assert result["correctness"]["status"] == "failed"


def test_top_level_requested_backend_cannot_contradict_nested_identity():
    doc = artifact()
    doc['results'][0].update(requested_backend='numpy', backend={'requested': 'cupy', 'executed': 'cupy'})
    assert q.evaluate_artifact(doc, 'classification-metrics', 'cupy')['correctness']['status'] == 'failed'
