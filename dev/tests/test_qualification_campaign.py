"""Synthetic harness/contract tests only: never execute numerical/GPU campaigns."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'dev/benchmarks/run_qualification_campaign.py'
spec = importlib.util.spec_from_file_location('qualification_campaign', PATH)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
SHA = 'a' * 40
SOURCE = {'git_commit': SHA, 'working_tree': 'clean', 'content_sha256': 'b' * 64,
          'source_content_sha256': 'b' * 64, 'files_sha256': {}}


def plan(*arguments):
    return campaign.build_plan(campaign.parser().parse_args(list(arguments)))


def artifact(job):
    """Hand-transcribe the seven producers' distinct row layouts, without imports."""
    rows = []
    for identity in job['expected_rows']:
        case, backend = identity['case'], identity['backend']
        row = {'case': case, 'backend': backend, 'status': 'pass', 'correctness': {'status': 'pass'},
               'timing': {'raw_seconds': [0.1] * int(job['argv'][job['argv'].index('--repeats') + 1]),
                          'repeats': int(job['argv'][job['argv'].index('--repeats') + 1])}}
        problem = {k: identity[k] for k in ('n', 'p', 'seed') if k in identity}
        problem['dtype'] = identity.get('input_dtype')
        row['problem'] = problem
        if identity.get('not_applicable'):
            row = {'case_id': case, 'backend': backend, 'status': 'not-applicable'}
        elif job['suite'] == 'gaussian-refresh':
            row.pop('case')
            row['backend'] = {'requested': backend}
            row['problem'] = {'shape': {'n': identity['n'], 'p': identity['p']},
                'dtype': identity['input_dtype'], 'seed': identity['seed'],
                'parameters': {'model': case, 'weighted': identity['weighted']}}
        elif job['suite'] == 'classification-metrics':
            row.pop('case')
            row['operation'] = case
            row['requested_backend'] = backend
            row['backend'] = {'executed': backend, 'output_device_verified': True, 'device': 'cpu' if backend == 'numpy' else 'cuda:0'}
            row['problem'] = {'shape': {'n': identity['n']}, 'input_dtype': identity['input_dtype'],
                              'seed': identity['seed'], 'scenario': identity['scenario']}
        elif job['suite'] == 'six-family-cv':
            row.pop('problem')
            row['correctness'] = {'direct_refit': {'status': 'pass'}, 'external': {'status': 'pass'}}
            row['clean_public_fit'] = dict(row.pop('timing'), status='pass', instrumented=False)
            runs = [{'framework': 'statgpu', 'backend': backend, 'status': 'success'}]
            if case in ('RidgeCV', 'LassoCV', 'ElasticNetCV', 'LogisticRegressionCV'):
                runs.append({'framework': 'sklearn', 'status': 'success'})
            row.update(model_id=case, case={'model_name': case, 'runs': runs}, n_samples=identity['n'], n_features=identity['p'],
                       protocol={'seeds': [identity['seed']]})
        elif job['suite'] == 'statistical-families':
            row['case_id'] = row.pop('case')
            row['problem']['input_dtype'] = row['problem'].pop('dtype')
        rows.append(row)
    return {'schema_version': 'statgpu-scoped-cv-candidate-v2' if job['suite'] == 'six-family-cv' else 'statgpu-benchmark-v2',
            'source': SOURCE, 'source_after': SOURCE, 'validation': {'status': 'complete'},
            'coverage': {'dtype': 'float64'}, 'results': rows}


def test_default_plan_has_21_invocations_1692_exact_rows():
    value = plan()
    assert value['plan_only'] is True
    assert len(value['jobs']) == 21
    assert value['expected_total_rows'] == 1692
    assert sum(bool(r.get('not_applicable')) for j in value['jobs'] for r in j['expected_rows']) == 16
    assert [j['expected_row_count'] for j in value['jobs'] if j['backend'] == 'numpy'] == [24, 126, 74, 276, 44, 8, 12]
    assert all('--require-backends' in j['argv'] and '--smoke' not in j['argv'] for j in value['jobs'])
    assert all(j['cpu_threads'] == 1 for j in value['jobs'])


def test_plan_never_launches_process_or_imports_numerical_packages(monkeypatch, capsys):
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('Implicit process execution'))
    modules = set(sys.modules)
    assert campaign.main([]) == 0
    assert not ({'numpy', 'torch', 'cupy', 'statgpu'} & (set(sys.modules) - modules))
    assert json.loads(capsys.readouterr().out)['plan_only'] is True


def test_cli_arguments_match_source_parser_without_runner_imports():
    for job in plan()['jobs']:
        tree = ast.parse((ROOT / ('/'.join(job['module'].split('.')) + '.py')).read_text())
        flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                 for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--')}
        assert {arg for arg in job['argv'] if arg.startswith('--')} <= flags


@pytest.mark.parametrize('arguments', [
    ('--scale', 'medium'),
    ('--scale', 'large', '--suites', 'learning-families'),
    ('--scale', 'large', '--suites', 'learning-families', '--cases', 'learning-families=tsne'),
    ('--cases', 'model-families=all'),
    ('--cases', 'gaussian-refresh=ols,ols'),
    ('--cases', 'gaussian-refresh=ols', '--suites', 'model-families'),
    ('--suites', 'six-family-cv', '--cases', 'six-family-cv=RidgeCV'),
    ('--timeout', '1801'), ('--timeout', 'nan'), ('--timeout', '0'),
])
def test_invalid_or_unsafe_plan_rejected(arguments):
    with pytest.raises(ValueError):
        plan(*arguments)


def test_selective_scale_caps_and_no_5000_10000_cv():
    value = plan('--scale', 'large', '--suites', 'learning-families', 'six-family-cv',
                 '--cases', 'learning-families=pca,kde')
    for job in value['jobs']:
        assert job['expected_row_count'] in (2, 6)
        assert max(r['n'] for r in job['expected_rows']) <= 2048
        if job['suite'] == 'learning-families':
            assert '4000000' in job['argv']
    assert not any('5000' in j['argv'] or '10000' in j['argv'] for j in plan()['jobs'] if j['suite'] == 'six-family-cv')


def test_thread_bounds_and_separate_numpy_profiles(monkeypatch):
    monkeypatch.setattr(campaign, 'cpu_limits', lambda: {'thread_cap': 4})
    value = plan('--cpu-threads', '1', '2', '4')
    assert len(value['jobs']) == 35
    assert all(j['backend'] == 'numpy' for j in value['jobs'] if j['cpu_threads'] != 1)
    for threads in [('1', '8'), ('2',), ('1', '1')]:
        with pytest.raises(ValueError):
            plan('--cpu-threads', *threads)


def test_actual_affinity_and_fractional_cgroup_quota(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.os, 'sched_getaffinity', lambda _: set(range(8)))
    (tmp_path / 'cpu.max').write_text('250000 100000')
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 2
    (tmp_path / 'cpu.max').write_text('50000 100000')
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 1
    assert campaign.cpu_limits(tmp_path)['effective_cpus'] == 0.5


@pytest.mark.parametrize('job', plan()['jobs'], ids=lambda j: j['id'])
def test_real_producer_layouts_exact_expanded_identity(job, tmp_path):
    output = tmp_path / 'raw.json'
    campaign.write_json(output, artifact(job))
    result = campaign.assess(output, job, 0, SHA)
    assert result['expanded_inventory']['exact_match'], result
    assert result['correctness']['status'] == 'complete', result
    assert not result['performance']['eligible']
    assert not result['whole_pipeline_native_gpu']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'dtype', 'n', 'scenario', 'weight', 'backend'])
def test_same_case_set_cannot_hide_multiplicity_or_axis_mismatch(mutation, tmp_path):
    suite = 'classification-metrics' if mutation == 'scenario' else 'gaussian-refresh'
    job = plan('--suites', suite)['jobs'][0]
    document = artifact(job)
    row = document['results'][0]
    if mutation == 'missing':
        document['results'].pop()
    elif mutation == 'duplicate':
        document['results'][-1] = copy.deepcopy(row)
    elif mutation == 'dtype':
        row['problem']['dtype'] = 'float16'
    elif mutation == 'n':
        row['problem']['shape']['n'] += 1
    elif mutation == 'scenario':
        row['problem']['scenario'] = 'unknown'
    elif mutation == 'weight':
        row['problem']['parameters']['weighted'] = True
    else:
        row['backend']['requested'] = 'cupy'
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    result = campaign.assess(output, job, 0, SHA)
    assert result['correctness']['status'] == 'failed'
    assert not result['expanded_inventory']['exact_match']


@pytest.mark.parametrize('content', [None, '', '{', '[]', '{"results":NaN}', '{"results": [null]}'])
def test_missing_or_malformed_json_is_failure(tmp_path, content):
    output = tmp_path / 'raw.json'
    if content is not None:
        output.write_text(content)
    result = campaign.assess(output, plan()['jobs'][0], 0, SHA)
    assert result['correctness']['status'] == 'failed'


@pytest.mark.parametrize('exit_code,state', [(7, 'complete'), (None, 'complete'), (0, 'partial'), (0, 'failed')])
def test_process_and_partial_failures_never_become_complete(tmp_path, exit_code, state):
    job = plan()['jobs'][0]
    document = artifact(job)
    document['validation']['status'] = state
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    assert campaign.assess(output, job, exit_code, SHA)['correctness']['status'] != 'complete'


def test_fake_gpu_output_only_and_unknown_docker_pid_never_qualify_performance(tmp_path):
    job = plan('--suites', 'classification-metrics', '--backends', 'cupy')['jobs'][0]
    document = artifact(job)
    document['environment'] = {'unknown_docker_pid': 123456, 'exclusive_gpu_ownership': True}
    document['performance_eligible'] = True
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    qualified = campaign.assess(output, job, 0, SHA)
    assert qualified['correctness']['status'] == 'complete'
    assert qualified['performance']['exclusive_gpu_ownership'] == 'UNVERIFIED_EXCLUSIVITY'
    assert not qualified['performance']['eligible'] and not qualified['whole_pipeline_native_gpu']
    # Explicitly contradictory backend is not output placement evidence either.
    document['results'][0]['backend']['executed'] = 'numpy'
    other = tmp_path / 'fake.json'
    campaign.write_json(other, document)
    assert campaign.assess(other, job, 0, SHA)['correctness']['status'] == 'failed'


def test_source_mutation_in_raw_artifact_fails(tmp_path):
    job = plan()['jobs'][0]
    document = artifact(job)
    document['source_after'] = dict(SOURCE, git_commit='c' * 40)
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    assert campaign.assess(output, job, 0, SHA)['correctness']['status'] == 'failed'


def test_destination_must_be_fresh_and_external(tmp_path):
    for path in (ROOT / 'results' / 'new-campaign', tmp_path):
        with pytest.raises(ValueError):
            campaign.validate_destination(path)
    destination = tmp_path / 'new'
    assert campaign.validate_destination(destination) == destination
    (tmp_path / 'new.tar.gz').write_bytes(b'existing')
    with pytest.raises(ValueError):
        campaign.validate_destination(destination)


def test_launch_timeout_retains_log_and_actual_exit(tmp_path):
    result = campaign.launch([sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(30)'],
                             tmp_path / 'log', 0.1, os.environ.copy(), lambda: None)
    assert result['reason'] == 'timeout'
    assert result['exit_code'] != 0
    assert (tmp_path / 'log').exists()


def test_launch_resource_guard_stops_only_owned_child(tmp_path):
    result = campaign.launch([sys.executable, '-c', 'import time; time.sleep(30)'],
                             tmp_path / 'log', 10, os.environ.copy(), lambda: 'memory reserve')
    assert result['reason'] == 'resource_guard: memory reserve'
    assert result['exit_code'] != 0


def test_launch_never_overwrites_log(tmp_path):
    log = tmp_path / 'log'
    log.write_bytes(b'old')
    with pytest.raises(FileExistsError):
        campaign.launch([sys.executable, '-c', 'pass'], log, 1, os.environ.copy(), lambda: None)
    assert log.read_bytes() == b'old'


@pytest.fixture
def synthetic_execution(monkeypatch):
    monkeypatch.setattr(campaign, 'fingerprint', lambda: copy.deepcopy(SOURCE))
    monkeypatch.setattr(campaign, 'resource_guard', lambda *a: None)
    monkeypatch.setattr(campaign, 'cpu_limits', lambda: {'thread_cap': 8})
    calls = []
    def fake_launch(command, log, timeout, environment, guard):
        # No workload is run. Persist exactly the wrapper's output contracts.
        output = Path(command[-1])
        job = next(j for j in calls.plan['jobs'] if j['id'] == output.parent.name)
        calls.append(job['id'])
        Path(log).write_text('synthetic child only\n')
        assert all(environment[key] == str(job['cpu_threads']) for key in campaign.THREAD_ENV)
        campaign.write_json(output, artifact(job))
        campaign.write_json(output.parent / 'runtime.after.json', {
            'before': {'cap_verified': True}, 'after': {'cap_verified': True},
            'process_peak_rss_bytes': 1024, 'gpu_allocator_peak_bytes': {'status': 'not_measured'}})
        return {'exit_code': 0, 'reason': None}
    class Calls(list):
        pass
    calls = Calls()
    calls.plan = plan('--suites', 'gaussian-refresh', '--backends', 'numpy', 'cupy')
    monkeypatch.setattr(campaign, 'launch', fake_launch)
    return calls


def test_serial_execution_preserves_raw_hashes_and_archive(tmp_path, synthetic_execution):
    calls = synthetic_execution
    destination = tmp_path / 'campaign'
    assert campaign.execute(calls.plan, destination, 10) == 0
    assert calls == [j['id'] for j in calls.plan['jobs']]
    checksums = campaign.read_json(destination / 'SHA256SUMS.json')
    for relative, digest in checksums.items():
        path = destination / relative
        assert campaign.hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert path.stat().st_mode & 0o222 == 0
    assert len(list(destination.glob('*/raw.json'))) == 2
    with tarfile.open(tmp_path / 'campaign.tar.gz') as archive:
        assert 'campaign/summary.json' in archive.getnames()
    assert (tmp_path / 'campaign.tar.gz.sha256').exists()


def test_ordinary_failure_continues_and_archives(tmp_path, synthetic_execution, monkeypatch):
    calls = synthetic_execution
    normal = campaign.launch
    def fail_first(*args):
        result = normal(*args)
        if len(calls) == 1:
            result['exit_code'] = 7
        return result
    monkeypatch.setattr(campaign, 'launch', fail_first)
    destination = tmp_path / 'campaign'
    assert campaign.execute(calls.plan, destination, 10) == 2
    assert len(calls) == 2
    assert (tmp_path / 'campaign.tar.gz').exists()
    assert campaign.read_json(destination / 'summary.json')['results'][0]['status'] == 'failed'


def test_source_change_stops_remaining_jobs_and_archives(tmp_path, synthetic_execution, monkeypatch):
    calls = synthetic_execution
    monkeypatch.setattr(campaign, 'fingerprint', lambda: dict(SOURCE, content_sha256='c' * 64) if calls else dict(SOURCE))
    destination = tmp_path / 'campaign'
    assert campaign.execute(calls.plan, destination, 10) == 2
    assert len(calls) == 1
    assert campaign.read_json(destination / 'summary.json')['stop_reason'] == 'source_changed'
    assert (tmp_path / 'campaign.tar.gz').exists()


def test_resource_guard_stops_and_archives_even_before_first_child(tmp_path, synthetic_execution, monkeypatch):
    monkeypatch.setattr(campaign, 'resource_guard', lambda *a: 'disk reserve')
    destination = tmp_path / 'campaign'
    assert campaign.execute(synthetic_execution.plan, destination, 10) == 2
    assert not synthetic_execution
    assert campaign.read_json(destination / 'summary.json')['stop_reason'].startswith('resource_guard:')
    assert (tmp_path / 'campaign.tar.gz').exists()


def test_exception_archives_partial_evidence(tmp_path, synthetic_execution, monkeypatch):
    monkeypatch.setattr(campaign, 'fingerprint', lambda: (_ for _ in ()).throw(RuntimeError('synthetic read error')))
    destination = tmp_path / 'campaign'
    assert campaign.execute(synthetic_execution.plan, destination, 10) == 2
    assert (destination / 'summary.json').exists()
    assert (tmp_path / 'campaign.tar.gz').exists()


def test_no_allow_incomplete_or_implicit_execution_escape_hatch():
    with pytest.raises(SystemExit):
        campaign.parser().parse_args(['--allow-incomplete'])
    with pytest.raises(SystemExit):
        campaign.main(['--execute'])
    tree = ast.parse(campaign.CHILD_BOOTSTRAP)
    assert tree
    assert "resource.getrusage" in campaign.CHILD_BOOTSTRAP
    assert "threadpool_info()" in campaign.CHILD_BOOTSTRAP
    assert "gpu_allocator_peak_bytes" in campaign.CHILD_BOOTSTRAP


def test_duplicate_json_keys_cannot_hide_failed_disposition(tmp_path):
    output = tmp_path / 'raw.json'
    output.write_text('{"validation":{"status":"failed","status":"complete"}}')
    assert campaign.assess(output, plan()['jobs'][0], 0, SHA)['correctness']['status'] == 'failed'


@pytest.mark.parametrize('pool_threads,backend,exit_code', [(1, 'numpy', 0), (2, 'numpy', 86), (1, 'torch', 0)])
def test_real_child_wrapper_uses_preimport_env_and_actual_stub_pool_inspection(tmp_path, pool_threads, backend, exit_code):
    # Pure-Python stand-ins, not NumPy/Torch, exercise actual subprocess control.
    (tmp_path / 'threadpoolctl.py').write_text(
        'def threadpool_info():\n    return [{"num_threads": ' + str(pool_threads) + ', "internal_api": "synthetic"}]\n')
    (tmp_path / 'torch.py').write_text('''intra = inter = 99
def set_num_threads(n):
    global intra
    intra = n
def set_num_interop_threads(n):
    global inter
    inter = n
def get_num_threads(): return intra
def get_num_interop_threads(): return inter
''')
    (tmp_path / 'synthetic_runner.py').write_text('''import os
assert os.environ['OMP_NUM_THREADS'] == '1'
assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
def main(argv): return 0
''')
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(campaign.CHILD_BOOTSTRAP)
    environment = os.environ.copy()
    environment.update({key: '1' for key in campaign.THREAD_ENV})
    result = subprocess.run([sys.executable, str(bootstrap), str(tmp_path), 'synthetic_runner',
                             backend, '1', str(tmp_path / 'runtime')], env=environment, capture_output=True, timeout=10)
    assert result.returncode == exit_code, result.stderr.decode()
    runtime = campaign.read_json(tmp_path / 'runtime.after.json')
    assert runtime['process_peak_rss_bytes'] > 0
    assert runtime['gpu_allocator_peak_bytes'] == {'status': 'not_measured'}
    if exit_code == 0:
        assert runtime['before']['cap_verified'] and runtime['after']['cap_verified']
        assert (tmp_path / 'runtime.before.json').exists()
    else:
        assert runtime['resource_guard']


def test_real_child_wrapper_detects_late_thread_pool_violation(tmp_path):
    (tmp_path / 'threadpoolctl.py').write_text('''threads = 1
def threadpool_info(): return [{'num_threads': threads}]
''')
    (tmp_path / 'synthetic_runner.py').write_text('''def main(argv):
    import threadpoolctl
    threadpoolctl.threads = 8
    return 0
''')
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(campaign.CHILD_BOOTSTRAP)
    result = subprocess.run([sys.executable, str(bootstrap), str(tmp_path), 'synthetic_runner',
                             'numpy', '1', str(tmp_path / 'runtime')], capture_output=True, timeout=10)
    assert result.returncode == 86
    assert campaign.read_json(tmp_path / 'runtime.before.json')['before']['cap_verified']
    assert campaign.read_json(tmp_path / 'runtime.after.json')['resource_guard']


def test_cv_seed_list_and_duplicate_na_counts_are_not_inferred_from_passes(tmp_path):
    job = plan('--suites', 'six-family-cv', '--backends', 'numpy')['jobs'][0]
    document = artifact(job)
    document['results'][0]['protocol']['seeds'] = [1, 2]
    output = tmp_path / 'cv.json'
    campaign.write_json(output, document)
    assert campaign.assess(output, job, 0, SHA)['correctness']['status'] == 'failed'
    job = plan('--suites', 'statistical-families', '--backends', 'cupy')['jobs'][0]
    document = artifact(job)
    index = next(i for i, row in enumerate(document['results']) if row['status'] == 'not-applicable')
    document['results'].pop(index)
    output = tmp_path / 'statistics.json'
    campaign.write_json(output, document)
    assert campaign.assess(output, job, 0, SHA)['correctness']['status'] == 'failed'


def test_unreadable_linux_quota_cannot_authorize_extra_threads(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.sys, 'platform', 'linux')
    monkeypatch.setattr(campaign.os, 'sched_getaffinity', lambda _: set(range(32)))
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 1
    assert campaign.cpu_limits(tmp_path)['quota_status'] == 'unknown'


def test_cgroup_v1_quota_and_v2_unlimited_are_distinct(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign.os, 'sched_getaffinity', lambda _: set(range(8)))
    cpu = tmp_path / 'cpu'
    cpu.mkdir()
    (cpu / 'cpu.cfs_quota_us').write_text('200000')
    (cpu / 'cpu.cfs_period_us').write_text('100000')
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 2
    (cpu / 'cpu.cfs_quota_us').write_text('-1')
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 8
    (tmp_path / 'cpu.max').write_text('max 100000')
    assert campaign.cpu_limits(tmp_path)['thread_cap'] == 8


def test_cli_repeat_override_must_match_artifact_samples(tmp_path):
    job = plan('--suites', 'gaussian-refresh', '--backends', 'numpy', '--repeats', '7')['jobs'][0]
    document = artifact(job)
    document['results'][0]['timing'] = {'repeats': 2, 'raw_seconds': [0.1, 0.2]}
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    result = campaign.assess(output, job, 0, SHA)
    assert result['correctness']['status'] == 'failed'
    assert 'Timing sample count differs from selected CLI repeats' in result['correctness']['errors']


def test_thread_guard_exit_stops_remaining_jobs(tmp_path, synthetic_execution, monkeypatch):
    calls = synthetic_execution
    normal = campaign.launch
    def guarded(*args):
        result = normal(*args)
        result['exit_code'] = 86
        return result
    monkeypatch.setattr(campaign, 'launch', guarded)
    destination = tmp_path / 'campaign'
    assert campaign.execute(calls.plan, destination, 10) == 2
    assert len(calls) == 1
    assert campaign.read_json(destination / 'summary.json')['stop_reason'].startswith('resource_guard:')
    assert (tmp_path / 'campaign.tar.gz').exists()


def test_extra_cpu_profiles_require_numpy(monkeypatch):
    monkeypatch.setattr(campaign, 'cpu_limits', lambda: {'thread_cap': 8})
    with pytest.raises(ValueError, match='require NumPy'):
        plan('--backends', 'cupy', '--cpu-threads', '1', '2')


def test_default_axes_match_manifest_and_literal_runner_defaults():
    """Regression anchor for the prior fullscan, rather than a reduced case-set plan."""
    entries = campaign.registry()
    for job in plan()['jobs']:
        argv = job['argv']
        def values(flag):
            tail = argv[argv.index(flag) + 1:]
            return tail[:next((i for i, v in enumerate(tail) if v.startswith('--')), len(tail))]
        if job['suite'] == 'gaussian-refresh':
            template = entries[job['suite']]['argv']
            assert values('--shapes') == template[template.index('--shapes') + 1:template.index('--dtypes')]
            assert values('--dtypes') == ['float32', 'float64']
            assert values('--weights') == ['none', 'analytic']
        elif job['suite'] == 'classification-metrics':
            assert values('--sizes') == ['1000', '10000', '100000']
            assert values('--dtypes') == ['float32', 'float64']
            assert values('--scenarios') == list(campaign.SCENARIOS)
        elif job['suite'] == 'statistical-families':
            assert values('--sizes') == ['128', '1024']
        elif job['suite'] == 'workflow-families':
            assert values('--sizes') == ['256', '1024']
        elif job['suite'] == 'six-family-cv':
            assert values('--sizes') == ['240', '1024']
        assert job['cases'] == entries[job['suite']]['case_ids']
        assert values('--repeats') == [str(job['expected_repeats'])]


@pytest.mark.parametrize('content', ['1e999', '-1e999', '{"error":1e999}', '{"values":[-1e999]}'])
def test_json_overflow_is_rejected_not_silently_converted_to_infinity(tmp_path, content):
    output = tmp_path / 'raw.json'
    output.write_text(content)
    with pytest.raises(ValueError, match='Nonfinite JSON number'):
        campaign.read_json(output)
    assert campaign.assess(output, plan()['jobs'][0], 0, SHA)['correctness']['status'] == 'failed'


@pytest.mark.skipif(os.name != 'posix', reason='Owned POSIX process-group lifecycle contract')
def test_exited_leader_descendants_cannot_keep_writing_log(tmp_path):
    descendant_marker = tmp_path / 'descendant-wrote'
    child = (
        'import pathlib,time; time.sleep(0.5); '
        f'pathlib.Path({str(descendant_marker)!r}).write_text("unexpected"); '
        'print("LATE DESCENDANT OUTPUT", flush=True)'
    )
    leader = (
        'import subprocess,sys; '
        f'subprocess.Popen([sys.executable,"-c",{child!r}]); '
        'print("LEADER EXITING",flush=True)'
    )
    log = tmp_path / 'log'
    result = campaign.launch([sys.executable, '-c', leader], log, 5, os.environ.copy(), lambda: None)
    before = log.read_bytes()
    assert result['exit_code'] == 0
    assert result['reason'] == 'process_group_cleanup: descendants outlived exited leader'
    import time
    time.sleep(0.7)
    assert log.read_bytes() == before
    assert b'LATE DESCENDANT OUTPUT' not in before
    assert not descendant_marker.exists()


@pytest.mark.parametrize('suite', ['gaussian-refresh', 'classification-metrics'])
def test_requested_gpu_cannot_hide_contradictory_executed_backend(tmp_path, suite):
    job = plan('--suites', suite, '--backends', 'cupy')['jobs'][0]
    document = artifact(job)
    document['results'][0]['backend'] = {'requested': 'cupy', 'executed': 'numpy', 'device': 'cpu'}
    output = tmp_path / 'raw.json'
    campaign.write_json(output, document)
    assert campaign.assess(output, job, 0, SHA)['correctness']['status'] == 'failed'
