"""Small CPU fixtures for real registered-runner axes; not benchmark evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from dev.benchmarks import benchmark_cv_coverage as cv
from dev.benchmarks import benchmark_cv_models as legacy
from dev.benchmarks import benchmark_learning_coverage as learning
from dev.benchmarks import benchmark_model_coverage as models
from dev.benchmarks import cv_external


def test_model_default_data_and_oracle_are_unchanged():
    old = models._make_default_case('glm:gaussian', 24, 2, 11, 'numpy')
    new = models.make_case('glm:gaussian', 24, 2, 11, 'numpy')
    assert old[4]['data_id'] == new[4]['data_id']
    for a, b in zip(old[1], new[1]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(old[3], new[3])
    assert new[0]().C == 1e12


@pytest.mark.parametrize('case,solver', [('glm:gaussian', 'lbfgs'), ('inference:nonrobust', 'default'), ('inference:hc0', 'default'), ('inference:hc3', 'default')])
def test_weighted_quantized_model_fixture_matches_actual_oracle(case, solver):
    factory, args, kw, target, contract, inference = models.make_case(
        case, 24, 2, 19, 'numpy', dtype='float32', weights='analytic', solver=solver)
    assert all(a.dtype == np.dtype('float32') for a in args)
    weight = kw['sample_weight']
    assert weight.dtype == np.dtype('float32') and np.ptp(weight) > 0
    model = factory().fit(*args, **kw)
    models.check_fit(model, target, contract, inference)
    assert contract['parameters']['weights'] == 'analytic'
    if solver != 'default':
        assert model.solver == solver
    unweighted = models.make_case(case, 24, 2, 19, 'numpy', dtype='float32')[3]
    assert not np.allclose(target, unweighted, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('case,settings', [
    ('penalty:l1', {'weights':'analytic'}),
    ('cox:breslow', {'dtype':'float32'}),
    ('glm_inference:gaussian:hc0', {'weights':'analytic'}),
    ('glm:gaussian', {'sparse':'csr'}),
    ('penalty:l1', {'solver':'lbfgs'}),
    ('wrapper:LogisticRegression', {'solver':'lbfgs'}),
])
def test_model_unsupported_axes_fail_before_factory(case, settings):
    with pytest.raises(models.UnsupportedCapability):
        models.make_case(case, 24, 2, 11, 'numpy', **settings)


def test_quantile_budget_is_a_named_capability_exclusion():
    with pytest.raises(models.UnsupportedCapability, match='n<=2000'):
        models.make_case('quantile:0.5', 2001, 2, 11, 'numpy')


@pytest.mark.parametrize('solver', ['fista', 'fista_bb', 'admm'])
def test_penalty_solver_axis_changes_public_factory(solver):
    factory, args, kw, target, contract, inference = models.make_case('penalty:l1', 24, 2, 11, 'numpy', solver=solver)
    model=factory().fit(*args, **kw)
    assert model.solver == model._selected_solver == solver
    models.check_fit(model,target,contract,inference)


def test_model_observer_records_real_numerical_output_dtype():
    factory, args, kw, *_ = models.make_case('inference:nonrobust', 24, 2, 11, 'numpy', dtype='float32')
    with models.Observe() as observer:
        model=factory().fit(*args, **kw)
    computed = [r for r in observer.rows if r['evidence_kind']=='numerical-return-local']
    assert computed and all(r['dtype'] for r in computed)
    assert any(r['dtype'] == str(model.coef_.dtype) for r in computed)


@pytest.mark.parametrize('covariance_type', ['diag','full','tied','spherical'])
@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_learning_covariance_axis_matches_analytic_shape(covariance_type, dtype):
    X,y,labels = learning.data(24, 3, 13)
    X,y = X.astype(dtype), y.astype(dtype)
    estimator = learning.estimator('gaussian_mixture', 'cpu', 13, len(X), covariance_type=covariance_type)
    estimator.fit(X)
    result = learning.validate('gaussian_mixture', estimator, X, y, estimator.predict(X), labels)
    assert result['metrics']['covariance_type'] == covariance_type
    assert result['strength'] == 'external-or-analytic'


@pytest.mark.parametrize('solver', ['full','covariance','randomized'])
def test_learning_pca_solver_axis_executes_against_svd_oracle(solver):
    X,y,labels = learning.data(24, 3, 13)
    estimator = learning.estimator('pca', 'cpu', 13, len(X), solver=solver)
    estimator.fit(X)
    assert estimator.svd_solver == solver
    learning.validate('pca', estimator, X, y, estimator.transform(X), labels)


@pytest.mark.parametrize('case,settings', [
    ('pca', {'weights':'analytic'}), ('pca', {'sparse':'csr'}),
    ('kde', {'dtype':'float32'}), ('kmeans', {'solver':'randomized'}),
    ('pca', {'covariance_type':'full'}),
])
def test_learning_unsupported_axes_are_named(case, settings):
    assert learning.capability_exclusion(case, **settings)


def test_cv_default_specs_preserve_identity_and_grid():
    assert legacy.configured_specs() == legacy.CASE_SPECS
    assert len(legacy.configured_specs()) == 6


def test_cv_configured_grid_folds_reach_native_and_reference():
    spec = legacy.configured_specs(folds=2, grid_size=4)[0]
    assert spec.cv_folds == 2 and legacy._candidate_count(spec) == 4
    assert legacy._model_kwargs(spec, 'numpy', 13)['cv'] == 2
    assert legacy.CASE_SPECS[0].cv_folds == 3
    model, data, identity = legacy._prepare_fit(spec, 'numpy', 13, 24, 3)
    assert len(identity['folds']) == 2
    external_folds = cv_external._folds(spec, data[0], data[1], 13)
    for (train, valid), fingerprints in zip(external_folds, identity['folds']):
        assert legacy._array_identity(train) == fingerprints[0]
        assert legacy._array_identity(valid) == fingerprints[1]
    model.fit(*legacy._fit_args(spec, data[0], data[1]))
    assert float(model.alpha_) in spec.grid_parameters['alphas']
    direct = cv.direct_refit_check(spec, 13, 24, 3)
    assert direct['status'] == 'pass'
    reference = cv_external._run_once(spec, 13, 24, 3, legacy._regression_data)
    assert reference['selected_parameters'] == legacy._selected_parameters(model)
    np.testing.assert_allclose(reference['final_score'], legacy._score_model(model, spec, data[2], data[3]), rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize('folds,grid_size', [(1,3),(3,1)])
def test_cv_invalid_knobs_rejected(folds, grid_size):
    with pytest.raises(ValueError):
        legacy.configured_specs(folds=folds,grid_size=grid_size)


def test_cv_unsupported_dtype_blocks_before_execution(monkeypatch):
    monkeypatch.setattr(cv, 'source_identity', lambda: {'git_commit':'fixture','content_sha256':'fixture'})
    monkeypatch.setattr(legacy, '_backend_status', lambda *_: pytest.fail('unsupported option reached numerical work'))
    args=SimpleNamespace(sizes=[30],backends=['numpy'],features=3,folds=2,grid_size=4,
                         dtype='float32',weights='none',sparse='dense',smoke=False)
    result=cv.run(args)
    assert len(result['results']) == 6
    assert result['validation']['status'] == 'partial'
    assert all(r['status']=='blocked' and r['blocked_kind']=='unsupported-capability' for r in result['results'])
    assert result['benchmark']['arguments']['dtype'] == 'float32'


@pytest.mark.parametrize('case,configuration', [
    ('pca',{'solver':'full'}), ('pca',{'solver':'covariance'}), ('pca',{'solver':'randomized'}),
    ('gaussian_mixture',{'covariance_type':'diag'}), ('gaussian_mixture',{'covariance_type':'full'}),
    ('gaussian_mixture',{'covariance_type':'tied'}), ('gaussian_mixture',{'covariance_type':'spherical'}),
])
def test_learning_configured_trace_observes_actual_algorithm(case,configuration):
    row=learning.run_case(case,learning.Runtime('numpy'),24,3,13,0,1,dtype='float32',**configuration)
    assert row['status']=='pass',row.get('reason')
    assert row['input_provenance']['X']['dtype']=='float32'
    assert row['problem']['input_dtype']=='float32'
    assert row['execution_trace'] and all(t['status']=='observed' for t in row['execution_trace'])
    stages=[stage['stage'] for trace in row['execution_trace'] for stage in trace['stages']]
    if configuration.get('solver')=='covariance':
        assert 'pca-covariance-eigendecomposition' in stages
        assert 'pca-centered-svd' not in stages
    if configuration.get('solver')=='randomized':
        assert 'pca-randomized-qr-svd' in stages
        assert 'pca-randomized-projection-rng' in stages


def test_gmm_covariance_oracle_distinguishes_regularization_conventions():
    X=np.array([[0.,0.],[1.,2.],[2.,0.],[3.,2.]])
    _,y,labels=learning.data(len(X),2,13)
    for covariance_type in ('diag','spherical','full','tied'):
        model=learning.estimator('gaussian_mixture','cpu',13,len(X),covariance_type=covariance_type).fit(X)
        check=learning.validate('gaussian_mixture',model,X,y,model.predict(X),labels)
        assert check['metrics']['covariance_max_abs_error']<1e-12


def test_wrong_default_path_trace_cannot_certify_selected_pca_solver(monkeypatch):
    monkeypatch.setattr(learning,'configured_estimator_stages',
                        lambda case,backend,phase,**kwargs: learning.estimator_stages(case,backend,phase))
    row=learning.run_case('pca',learning.Runtime('numpy'),24,3,13,0,1,solver='covariance')
    assert row['status']=='blocked'
    assert row['correctness']['status']=='pass'
    assert 'timing' not in row


def test_cv_grid_and_folds_reach_all_public_cv_constructors():
    import importlib
    for spec in legacy.configured_specs(folds=4,grid_size=5):
        cls=getattr(importlib.import_module(spec.import_module),spec.import_name)
        model=legacy._construct(cls,legacy._model_kwargs(spec,'numpy',13))
        assert model.get_params(deep=False)['cv']==4
        assert legacy._candidate_count(spec)==5


@pytest.mark.parametrize('case,solver', [('glm:gaussian','lbfgs'),('inference:hc3','default')])
def test_weighted_model_runner_emits_input_and_compute_evidence(case,solver):
    row=models.run_case(case,models.Runtime('numpy'),24,2,19,0,1,
                        dtype='float32',weights='analytic',solver=solver)
    assert row['status']=='passed',row.get('reason')
    assert row['input_provenance']['arguments'][0]['dtype']=='float32'
    assert row['input_provenance']['fit_keywords']['sample_weight']['dtype']=='float32'
    assert row['backend_provenance']['compute_dtypes']
    assert row['problem']['weights']=='analytic'


def test_cli_model_unsupported_axes_retain_requested_configuration(tmp_path):
    import json
    output=tmp_path/'unsupported.json'
    assert models.main(['--cases','glm:gaussian','--backends','numpy','--sparse','csr','--output',str(output)])==2
    result=json.loads(output.read_text())
    assert result['benchmark']['arguments']['sparse']=='csr'
    assert result['rows'][0]['blocked_kind']=='unsupported-capability'
    assert result['rows'][0]['problem']['sparse']=='csr'
    assert 'timing' not in result['rows'][0]


def test_default_panel_group_metadata_keeps_original_transfer_boundary():
    runtime=models.Runtime('numpy')
    converted=[]
    original=runtime.array
    def record(value):
        converted.append(np.asarray(value).dtype.kind)
        return original(value)
    runtime.array=record
    row=models.run_case('panel:within',runtime,24,2,13,0,1)
    assert row['status']=='passed',row.get('reason')
    assert converted==['f','f']
    assert row['input_provenance']['fit_keywords']['entity_ids']['backend']=='numpy'


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_gmm_does_not_relax_tolerance_and_metrics_are_json_serializable(dtype):
    import json
    X, y, labels = learning.data(24, 3, 13)
    X = X.astype(dtype)
    model = learning.estimator('gaussian_mixture', 'cpu', 13, len(X)).fit(X)
    check = learning.validate('gaussian_mixture', model, X, y, model.predict(X), labels)
    json.dumps(check, allow_nan=False)
    assert check['metrics']['thresholds']['atol'] == 2e-6
    assert check['metrics']['thresholds']['rtol'] == 2e-7
    model.covariances_[0, 0] += 8e-6
    with pytest.raises(AssertionError):
        learning.validate('gaussian_mixture', model, X, y, model.predict(X), labels)


def test_float32_gmm_cli_writes_strict_json(tmp_path):
    import json
    output = tmp_path / 'gmm32.json'
    code = learning.main(['--cases', 'gaussian_mixture', '--backends', 'numpy',
                          '--n', '24', '--p', '3', '--seed', '13', '--warmup', '0',
                          '--repeats', '1', '--dtype', 'float32', '--covariance-type', 'full',
                          '--output', str(output)])
    assert code == 0
    document = json.loads(output.read_text())
    assert document['validation']['status'] == 'complete'
    assert document['results'][0]['correctness']['metrics']['thresholds']['rtol'] == 2e-7


@pytest.mark.parametrize('covariance', ['nonrobust', 'hc0', 'hc3'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('weights', ['none', 'analytic'])
def test_inference_actual_cpu_arithmetic_not_input_dtype(covariance, dtype, weights):
    row = models.run_case('inference:' + covariance, models.Runtime('numpy'), 48, 3, 19, 0, 1,
                          dtype=dtype, weights=weights)
    assert row['status'] == 'passed', row
    assert row['correctness']['status'] == 'pass'
    assert row['input_provenance']['arguments'][0]['dtype'] == dtype
    assert row['backend_provenance']['compute_dtypes'] == ['float64']
    assert row['execution_trace']['status'] == 'observed'
    assert row['timing']['repeats'] == 1


@pytest.mark.parametrize('covariance', ['nonrobust', 'hc0', 'hc3'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('weights', ['none', 'analytic'])
def test_inference_actual_torch_cpu_covariance_primitive_contract(covariance, dtype, weights):
    """Real production helper calls on CPU tensors, not CUDA evidence."""
    torch = pytest.importorskip('torch')
    import statsmodels.api as sm
    from statgpu.linear_model.wrappers._linear import LinearRegression
    from statgpu.backends._gpu_inference_torch import compute_inference_torch
    from dev.benchmarks.learning_execution_trace import NumericalTrace, model_stages
    _, args, fitkw, _, _, _ = models.make_case('inference:' + covariance, 48, 3, 19,
                                             'numpy', dtype=dtype, weights=weights)
    X, y = args
    sw = fitkw.get('sample_weight', np.ones(len(y)))
    reference = sm.WLS(y.astype('float64'), sm.add_constant(X.astype('float64')),
                      weights=sw.astype('float64')).fit(
                          cov_type='nonrobust' if covariance == 'nonrobust' else covariance.upper())
    # Mirror the documented existing Torch compute promotion, not an assertion
    # derived from input dtype. Observation below independently checks every op.
    design = torch.as_tensor(sm.add_constant(X.astype('float64')), dtype=torch.float64)
    root_weights = torch.sqrt(torch.as_tensor(sw, dtype=torch.float64))
    design = design * root_weights[:, None]
    response = torch.as_tensor(y, dtype=torch.float64) * root_weights
    coef = torch.linalg.lstsq(design, response).solution
    resid = response - design @ coef
    scale = torch.sum(resid ** 2) / (len(y) - design.shape[1])
    rt = SimpleNamespace(name='torch', module=torch, device='cpu', dtype=dtype)
    trace = NumericalTrace(rt, model_stages('inference:' + covariance, 'torch', input_dtype=dtype, weights=weights))
    with trace:
        if covariance == 'nonrobust':
            bse, _, _, _ = compute_inference_torch(design, resid, scale, len(y)-design.shape[1], coef, device='cpu')
        else:
            model = LinearRegression(cov_type=covariance)
            covariance_result = model._robust_covariance_torch(
                design, resid, torch.linalg.inv(design.T @ design), device='cpu')
            bse = torch.sqrt(torch.diag(covariance_result))
    report = trace.report()
    assert models.observed_compute_dtypes(report, []) == ['float64']
    np.testing.assert_allclose(bse.numpy(), reference.bse, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize('mode', ['missing-stage', 'empty-stage-list'])
def test_inference_missing_arithmetic_never_accepts_input_only(monkeypatch, mode):
    from dev.benchmarks.learning_execution_trace import stage
    def never_called(x):
        return np.sqrt(x)
    monkeypatch.setattr(models, 'model_stages', lambda *args, **kwargs:
                        [] if mode == 'empty-stage-list' else [stage(never_called, 'missing', ('sqrt',))])
    row = models.run_case('inference:hc3', models.Runtime('numpy'), 48, 3, 19, 0, 1,
                          dtype='float32', weights='analytic')
    assert row['correctness']['status'] == 'pass'
    assert row['status'] == 'blocked'
    assert 'timing' not in row


@pytest.mark.parametrize('corruption', ['backend', 'device', 'dtype', 'missing'])
def test_inference_arithmetic_evidence_corruption_rejected(corruption):
    from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, model_stages
    from statgpu.linear_model._gaussian_inference import _compute_single_native
    rt = SimpleNamespace(name='numpy', module=np, device='cpu', dtype='float64')
    trace = NumericalTrace(rt, model_stages('inference:hc3', 'numpy'))
    values = np.ones(3)
    trace.records = [{'stage':'ols-inference-standard-errors','completed':True,
                     'operations':[{'operation':'sqrt','backend':'numpy','status':'completed',
                         'inputs':[{'kind':'array','backend':'numpy','device':'cpu','dtype':'float64','shape':[3]}],
                         'outputs':[{'kind':'array','backend':'numpy','device':'cpu','dtype':'float64','shape':[3]}]}]}]
    event = trace.records[0]['operations'][0]
    if corruption == 'missing':
        trace.records[0]['operations'].clear()
    else:
        event['outputs'][0][corruption] = {'backend':'torch','device':'cuda:1','dtype':'float32'}[corruption]
    with pytest.raises(TraceIncomplete):
        trace.report()


def test_inference_native_primitive_monkeypatch_rejected(monkeypatch):
    original = np.sqrt
    monkeypatch.setattr(np, 'sqrt', lambda *args, **kwargs: original(*args, **kwargs))
    row = models.run_case('inference:hc3', models.Runtime('numpy'), 48, 3, 19, 0, 1,
                          dtype='float32', weights='analytic')
    assert row['status'] == 'failed'
    assert 'replacement' in row['reason']
    assert 'timing' not in row


def test_inference_clean_repeats_exclude_numerical_observer(monkeypatch):
    import sys
    from statgpu.linear_model.wrappers._linear import LinearRegression
    original = LinearRegression.fit
    instrumented = []
    def recorded(self, *args, **kwargs):
        instrumented.append(sys.getprofile() is not None)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(LinearRegression, 'fit', recorded)
    row = models.run_case('inference:hc3', models.Runtime('numpy'), 48, 3, 19, 1, 2,
                          dtype='float32', weights='analytic')
    assert row['status'] == 'passed', row
    assert instrumented == [True, False, False, False]
    assert len(row['timing']['raw_seconds']) == 2
