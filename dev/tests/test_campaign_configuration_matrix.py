"""Expanded campaign contracts: synthetic JSON/planning only, no benchmark jobs."""
import ast
import copy
import json
import sys

import pytest

from test_qualification_campaign import ROOT, SOURCE, SHA, artifact, campaign, plan


def configured_artifact(job):
    document = artifact(job)
    document['benchmark'] = {'arguments': copy.deepcopy(job.get('expected_arguments', {}))}
    for row, identity in zip(document['results'], job['expected_rows']):
        timing = row.get('clean_public_fit') if job['suite'] == 'six-family-cv' else row.get('timing')
        if isinstance(timing, dict): timing['warmup'] = job['expected_warmup']
        axes = identity.get('configuration', {})
        if job['suite'] == 'six-family-cv':
            row.update({key: axes[key] for key in ('dtype', 'weights', 'sparse') if key in axes})
            row['case']['cv'] = {'fold_count': axes.get('folds', 3)}
            key = 'Cs' if identity['case'] == 'LogisticRegressionCV' else {'PenalizedGLM_CV': 'alpha_grid', 'CoxPHCV': 'penalties'}.get(identity['case'], 'alphas')
            row['case']['grid'] = {'parameters': {key: campaign.cv_grid(key, axes.get('grid_size', 3))}}
        else:
            row.setdefault('problem', {}).update(axes)
        parameters = identity.get('parameters', {})
        if job['suite'] == 'gaussian-refresh': row['problem']['parameters'].update(parameters)
        elif job['suite'] == 'statistical-families': row['problem'].update(parameters)
        elif job['suite'] == 'workflow-families' and parameters:
            row['problem_extra'] = {'n_targets': parameters['targets']}
    return document


def assess_document(tmp_path, job, document):
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    return campaign.assess(output, job, 0, SHA)


def model_profile(**axes):
    return plan('--suites', 'model-families', '--cases', 'model-families=glm:gaussian',
                '--backends', 'numpy', '--seeds', '17', '--config',
                'model-families:configured=' + json.dumps(axes))['jobs'][0]


def test_default_is_unchanged_and_expanded_is_explicit_finite():
    original, expanded = plan(), plan('--profile', 'expanded')
    assert (len(original['jobs']), original['expected_total_rows']) == (21, 1692)
    assert expanded['plan_only'] and expanded['profile'] == 'expanded'
    assert (len(expanded['jobs']), expanded['expected_total_rows']) == (990, 8316)
    assert len(expanded['capability_exclusions']) == 24
    assert expanded['excluded_slot_count'] == 216
    assert {job['seed'] for job in expanded['jobs']} == {17, 42, 20261020}
    assert all(len(job['cases']) <= 8 for job in expanded['jobs'])
    assert len({job['id'] for job in expanded['jobs']}) == len(expanded['jobs'])
    slots = [slot for job in expanded['jobs'] for slot in job['expected_slots']]
    assert len({slot['id'] for slot in slots}) == len(slots)
    assert all(slot['source'] == expanded['source'] and slot['configuration']['seed'] in (17, 42, 20261020) for slot in slots)
    assert all(job['resource_bounds']['timeout_seconds'] <= 1800 for job in expanded['jobs'])


def test_multiple_seeds_explicit_shapes_and_no_slot_overwrite():
    value = plan('--suites', 'model-families', '--cases', 'model-families=glm:gaussian',
                 '--seeds', '3', '7', '--shapes', 'model-families=64x3,96x4')
    assert len(value['jobs']) == 12 and value['expected_total_rows'] == 12
    tuples = {(j['backend'], r['n'], r['p'], r['seed']) for j in value['jobs'] for r in j['expected_rows']}
    assert len(tuples) == 12
    assert {j['seed'] for j in value['jobs']} == {3, 7}
    assert all(j['source'] == value['source'] for j in value['jobs'][0]['expected_slots'])


def test_batches_partition_original_ids_not_renumber_or_repeat():
    args = ['--profile', 'expanded', '--suites', 'learning-families']
    original = plan(*args)
    batches = [plan(*args, '--batch-count', '7', '--batch-index', str(i)) for i in range(7)]
    all_ids = [j['id'] for batch in batches for j in batch['jobs']]
    assert len(all_ids) == len(set(all_ids)) == len(original['jobs'])
    assert set(all_ids) == {j['id'] for j in original['jobs']}
    assert sum(batch['expected_total_rows'] for batch in batches) == original['expected_total_rows']
    assert all(batch['batch']['all_expected_rows'] == original['expected_total_rows'] for batch in batches)


def test_named_profiles_execute_real_direct_options_and_precision_products():
    value = plan('--suites', 'model-families', '--cases', 'model-families=glm:gaussian',
        '--backends', 'numpy', '--dtypes', 'float32', 'float64', '--weights', 'none', 'analytic',
        '--config', 'model-families:irls={"solver":"irls"}',
        '--config', 'model-families:newton={"solver":"newton"}')
    assert len(value['jobs']) == value['expected_total_rows'] == 8
    for job in value['jobs']:
        axes = job['configuration']['axes']
        assert job['expected_rows'][0]['configuration'] == axes
        assert job['expected_rows'][0]['input_dtype'] == axes['dtype']
        for key, value in axes.items():
            assert job['argv'][job['argv'].index('--' + key) + 1] == value


@pytest.mark.parametrize('axes', [
    {'dtype': 'float32', 'weights': 'analytic', 'solver': 'irls', 'sparse': 'dense'},
    {'dtype': 'float64', 'weights': 'none', 'solver': 'newton'},
])
def test_actual_configured_rows_are_required_for_success(tmp_path, axes):
    job = model_profile(**axes)
    result = assess_document(tmp_path, job, configured_artifact(job))
    assert result['correctness']['status'] == 'complete', result
    assert result['expanded_inventory']['exact_match']
    assert not result['performance']['eligible']


@pytest.mark.parametrize('mutation', ['seed', 'dtype', 'weights', 'solver', 'sparse', 'backend', 'duplicate', 'missing', 'partial', 'arguments', 'source'])
def test_new_dimensions_cannot_be_hidden_by_matching_case_set(tmp_path, mutation):
    job = model_profile(dtype='float32', weights='analytic', solver='irls', sparse='dense')
    document = configured_artifact(job)
    row = document['results'][0]
    if mutation == 'seed': row['problem']['seed'] += 1
    elif mutation == 'dtype': row['problem']['dtype'] = 'float64'
    elif mutation == 'weights': row['problem']['weights'] = 'none'
    elif mutation == 'solver': row['problem']['solver'] = 'fista'
    elif mutation == 'sparse': row['problem']['sparse'] = 'csr'
    elif mutation == 'backend': row['backend'] = 'cupy'
    elif mutation == 'duplicate': document['results'].append(copy.deepcopy(row))
    elif mutation == 'missing': document['results'].clear()
    elif mutation == 'partial': document['validation']['status'] = 'partial'
    elif mutation == 'arguments': document['benchmark']['arguments']['solver'] = 'newton'
    elif mutation == 'source': document['source_after'] = dict(SOURCE, git_commit='f' * 40)
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] != 'complete', result


@pytest.mark.parametrize('mutation', [None, 'folds', 'grid_size', 'grid_values', 'metadata_only'])
def test_cv_grid_folds_validate_effective_case_not_just_cli(tmp_path, mutation):
    job = plan('--suites', 'six-family-cv', '--backends', 'numpy', '--config',
               'six-family-cv:folds={"folds":5,"grid_size":4,"dtype":"float64","weights":"none","sparse":"dense"}')['jobs'][0]
    document = configured_artifact(job)
    case = document['results'][0]['case']
    if mutation == 'folds': case['cv']['fold_count'] = 3
    elif mutation == 'grid_size': case['grid']['parameters']['alphas'].pop()
    elif mutation == 'grid_values': case['grid']['parameters']['alphas'][1] *= 2
    elif mutation == 'metadata_only': case.pop('cv'); case.pop('grid')
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] == ('complete' if mutation is None else 'failed'), result


def test_learning_all_44_have_case_resource_policies_not_old_allowlist():
    cases = campaign.registry()['learning-families']['case_ids']
    for case in cases:
        value = plan('--suites', 'learning-families', '--cases', 'learning-families=' + case,
                     '--shapes', 'learning-families=256x4', '--backends', 'numpy')
        assert case in value['jobs'][0]['resource_bounds']['case_policies']
    medium = plan('--scale', 'medium', '--suites', 'learning-families',
                  '--cases', 'learning-families=kernel_ridge,kernel_pca,bandwidth_ucv,knockoff_model')
    assert medium['expected_total_rows'] == 12
    with pytest.raises(ValueError, match='iterative_embedding'):
        plan('--suites', 'learning-families', '--cases', 'learning-families=tsne', '--shapes', 'learning-families=1024x4')
    with pytest.raises(ValueError, match='kernel_cv'):
        plan('--suites', 'learning-families', '--cases', 'learning-families=kernel_ridge_cv', '--shapes', 'learning-families=1024x4')


def test_quantile_2048_is_excluded_or_rejected_never_silently_downscaled():
    value = plan('--profile', 'expanded', '--suites', 'model-families')
    excluded = value['capability_exclusions']
    assert len(excluded) == 4 and all(e['n'] == 2048 and 'n<=2000' in e['reason'] for e in excluded)
    with pytest.raises(ValueError, match='n<=2000'):
        plan('--suites', 'model-families', '--cases', 'model-families=quantile:0.5', '--shapes', 'model-families=2048x16')
    bounded = plan('--suites', 'model-families', '--cases', 'model-families=quantile:0.5', '--shapes', 'model-families=2000x16')
    assert all(row['n'] == 2000 for j in bounded['jobs'] for row in j['expected_rows'])


@pytest.mark.parametrize('size', [5000, 10000])
def test_large_cv_approval_resource_boundary_cannot_be_relaxed(size):
    with pytest.raises(ValueError, match='separate approval and resource review'):
        plan('--suites', 'six-family-cv', '--shapes', f'six-family-cv={size}x16')


@pytest.mark.parametrize('args,reason', [
    (['--seeds','17','17'], 'unique'),
    (['--seeds','-1'], 'nonnegative'),
    (['--batch-count','0'], 'batch-count'),
    (['--batch-count','3','--batch-index','3'], 'batch-count'),
    (['--shapes','model-families=64x3,64x3'], 'distinct'),
    (['--suites','model-families','--shapes','model-families=64x3','--scale','medium'], 'cannot be combined'),
    (['--suites','statistical-families','--dtypes','float32'], 'float64 only'),
    (['--weights','analytic'], 'weight capability'),
    (['--max-pairwise-elements','4000001'], '[Rr]esource policy'),
    (['--resamples','1001'], 'resamples'),
    (['--targets','17'], 'targets'),
    (['--device','1'], 'device ordinal'),
    (['--config','model-families:x={"loss":"poisson"}'], 'unsupported'),
    (['--config','model-families:x={"dtype":"float32","dtype":"float64"}'], 'Duplicate'),
    (['--config','model-families:x={"sparse":"csr"}'], 'sparse'),
    (['--config','six-family-cv:x={"folds":5,"grid_size":9}'], 'fit resource budget'),
    (['--config','six-family-cv:x={"folds":true}'], 'scalar types'),
])
def test_invalid_or_unsupported_axis_fails_closed(args, reason):
    with pytest.raises(ValueError, match=reason): plan(*args)


def test_duplicate_named_effective_configurations_fail_instead_of_overwrite():
    with pytest.raises(ValueError, match='Duplicate campaign configuration'):
        plan('--suites', 'model-families', '--cases', 'model-families=glm:gaussian',
             '--config', 'model-families:a={"solver":"irls"}',
             '--config', 'model-families:b={"solver":"irls"}')


def test_selection_seed_trials_and_feature_dimension_are_distinct_from_repeats():
    value = plan('--suites', 'learning-families', '--cases', 'learning-families=knockoff_fixed,pca',
                 '--backends', 'numpy', '--seeds', '23', '--selection-seeds', '1', '2', '3', '--knockoff-p', '24')
    job = value['jobs'][0]
    assert job['expected_row_count'] == 4
    selected = [row for row in job['expected_rows'] if row['case'].startswith('knockoff')]
    assert {row['seed'] for row in selected} == {1, 2, 3}
    assert all(row['p'] == 24 for row in selected)
    assert next(row for row in job['expected_rows'] if row['case'] == 'pca')['seed'] == 23
    with pytest.raises(ValueError, match='repeat selection trials'):
        plan('--suites', 'learning-families', '--seeds', '1', '2', '--selection-seeds', '3', '4')


@pytest.mark.parametrize('suite,cases,args', [
    ('gaussian-refresh', 'ridge', ['--alpha', '.3']),
    ('statistical-families', 'resampling/bootstrap/iid', ['--resamples', '200']),
    ('workflow-families', 'multitarget', ['--targets', '8']),
])
def test_supported_direct_knobs_have_row_level_parameter_checks(tmp_path, suite, cases, args):
    # Registry spells the iid resampling strategy as ordinary.
    if suite == 'statistical-families':
        cases = next(c for c in campaign.registry()[suite]['case_ids'] if c.startswith('resampling/bootstrap/'))
    job = plan('--suites', suite, '--cases', suite + '=' + cases, '--backends', 'numpy', *args)['jobs'][0]
    document = configured_artifact(job)
    assert campaign.row_identity(document['results'][0], suite, document)['case'] == cases
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] == 'complete', result
    row = document['results'][0]
    if suite == 'gaussian-refresh': row['problem']['parameters']['alpha'] = .9
    elif suite == 'statistical-families': row['problem']['resamples'] = 10
    else: row['problem_extra']['n_targets'] = 2
    output = tmp_path / 'bad.json'; campaign.write_json(output, document)
    assert campaign.assess(output, job, 0, SHA)['correctness']['status'] == 'failed'


def test_expanded_plan_imports_no_runner_or_numerical_packages(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('Implicit process'))
    before = set(sys.modules)
    value = plan('--profile', 'expanded')
    assert value['plan_only']
    assert not ({'numpy', 'torch', 'cupy', 'statgpu'} & (set(sys.modules) - before))
    for module in {job['module'] for job in value['jobs']}:
        job = next(job for job in value['jobs'] if job['module'] == module)
        source = ROOT / ('/'.join(job['module'].split('.')) + '.py')
        tree = ast.parse(source.read_text())
        flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'add_argument' for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        assert {arg for arg in job['argv'] if arg.startswith('--')} <= flags


def test_changed_source_prevents_launch_of_locked_configuration(tmp_path, monkeypatch):
    value = plan('--suites', 'model-families', '--cases', 'model-families=glm:gaussian', '--seeds', '17')
    monkeypatch.setattr(campaign, 'fingerprint', lambda: copy.deepcopy(SOURCE))
    monkeypatch.setattr(campaign, 'launch', lambda *a, **k: pytest.fail('Stale plan launched'))
    destination = tmp_path / 'stale'
    assert campaign.execute(value, destination, 1) == 2
    summary = campaign.read_json(destination / 'summary.json')
    assert summary['attempted_invocations'] == 0
    assert 'Source changed since configuration plan generation' in summary['stop_reason']


def test_source_lock_includes_unified_dispatch_dependencies():
    names = {path.name for path in campaign.source_files()}
    assert {'run_full_benchmark_matrix.py', 'independent_benchmark_registry.py', 'run_independent_benchmarks.py',
            'independent_benchmark_catalog.json', 'cv_external.py'} <= names


def test_expanded_execution_needs_a_bounded_explicit_batch():
    with pytest.raises(ValueError, match='exceeding max-invocations'):
        plan('--profile', 'expanded', '--execute')
    bounded = plan('--profile', 'expanded', '--execute', '--batch-count', '48', '--batch-index', '0')
    assert bounded['execution_budget']['within_budget']
    assert len(bounded['jobs']) <= 21
    assert bounded['batch']['all_invocations'] == 990
    with pytest.raises(ValueError, match='1..64'):
        plan('--max-invocations', '990')


@pytest.mark.parametrize('actual', ['cuda:1', 'cuda:0', None])
def test_selected_device_ordinal_uses_observed_device_not_cli(tmp_path, actual):
    job = plan('--suites', 'gaussian-refresh', '--cases', 'gaussian-refresh=ridge', '--backends', 'cupy', '--device', '1')['jobs'][0]
    document = configured_artifact(job)
    for row in document['results']:
        row['backend'] = {'requested': 'cupy', 'executed': 'cupy'}
        if actual is not None:
            row['backend']['device'] = actual
            row['timing']['synchronized_device'] = actual
        row['correctness']['reference_arrays'] = {'device': 'cpu'}
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] == ('complete' if actual == 'cuda:1' else 'failed')


@pytest.mark.parametrize('args, reason', [
    (['--suites', 'six-family-cv', '--shapes', 'six-family-cv=16x1'], '30<=n'),
    (['--suites', 'statistical-families', '--shapes', 'statistical-families=16x1'], '2<=p'),
    (['--suites', 'workflow-families', '--shapes', 'workflow-families=16x1'], '2<=p'),
    (['--suites', 'statistical-families', '--resamples', '1'], '2..1000'),
])
def test_planner_minima_match_direct_cli_capabilities(args, reason):
    with pytest.raises(ValueError, match=reason): plan(*args)


@pytest.mark.parametrize('mutation', [None, 'wrong_gpu', 'unknown_host_scope', 'no_gpu_timing', 'wrong_host_device'])
def test_formula_host_preparation_is_narrow_device_exception(tmp_path, mutation):
    job = plan('--suites', 'workflow-families', '--cases', 'workflow-families=formula',
               '--backends', 'cupy', '--device', '1')['jobs'][0]
    document = configured_artifact(job)
    for row in document['results']:
        row['executed_device'] = 'cuda:1'
        row['timing'] = [dict(scope=scope, synchronized_device=device, raw_seconds=[.1]*3, warmup=1, repeats=3)
                        for scope, device in [('host-formula-model-matrix-preparation-only', 'cpu'),
                            ('formula-parse-plus-fit-including-required-host-to-device-transfer', 'cuda:1'),
                            ('resident-array-fit-only-same-formula-design', 'cuda:1')]]
        if mutation == 'wrong_gpu': row['timing'][1]['synchronized_device'] = 'cuda:0'
        elif mutation == 'unknown_host_scope': row['timing'][0]['scope'] = 'arbitrary-host-stage'
        elif mutation == 'no_gpu_timing': row['timing'] = row['timing'][:1]
        elif mutation == 'wrong_host_device': row['timing'][0]['synchronized_device'] = 'cuda:1'
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] == ('complete' if mutation is None else 'failed'), result


@pytest.mark.parametrize('suite', ['gaussian-refresh', 'classification-metrics', 'model-families',
                                   'statistical-families', 'learning-families', 'workflow-families', 'six-family-cv'])
@pytest.mark.parametrize('mutation', [None, 'wrong', 'missing', 'bool'])
def test_selected_warmup_requires_actual_timing_protocol(tmp_path, suite, mutation):
    job = plan('--suites', suite, '--backends', 'numpy', '--warmup', '7')['jobs'][0]
    document = configured_artifact(job)
    document['benchmark']['arguments']['warmup'] = 7  # Not sufficient evidence.
    for row in document['results']:
        section = row['clean_public_fit'] if suite == 'six-family-cv' else row['timing']
        if suite == 'workflow-families':
            row['timing'] = [copy.deepcopy(section), copy.deepcopy(section)]
            section = row['timing'][-1]
        if mutation == 'wrong': section['warmup'] = 0
        elif mutation == 'missing': section.pop('warmup')
        elif mutation == 'bool': section['warmup'] = True
    result = assess_document(tmp_path, job, document)
    assert result['correctness']['status'] == ('complete' if mutation is None else 'failed'), result


def test_observed_default_warmup_mismatch_is_rejected(tmp_path):
    job = plan('--suites', 'model-families', '--backends', 'numpy')['jobs'][0]
    document = configured_artifact(job)
    document['results'][0]['timing']['warmup'] = job['expected_warmup'] + 1
    assert assess_document(tmp_path, job, document)['correctness']['status'] == 'failed'
