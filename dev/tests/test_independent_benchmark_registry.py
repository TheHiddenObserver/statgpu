"""CPU/stdlib-only registry and synthetic-child contracts. No benchmark jobs."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dev.benchmarks import independent_benchmark_registry as reg
from dev.benchmarks import run_independent_benchmarks as runner


SHA = '1' * 40


def public_payload(config=None):
    c = reg.normalize_configuration('public-native-kde-bootstrap', config)
    return {'schema_version': 'statgpu-benchmark-v2',
            'problem': {k: v for k, v in c.items() if k not in ('warmup', 'repeats')},
            'timing': {'warmup': c['warmup'], 'repeats': c['repeats']},
            'validation': {'status': 'complete'},
            'rows': [{'backend': b, 'status': 'pass', 'correctness': {'status': 'pass'},
                      'timing': {'samples_seconds': [0.1] * c['repeats']},
                      'operation_evidence': {'status': 'observed', 'device': 'cuda:0'} if b != 'numpy' else {}}
                     for b in reg.BACKENDS]}


def test_full_historical_inventory_and_all_runner_dispositions():
    catalog = reg.load_catalog()
    assert len(catalog['files']) == 202
    assert len(reg.registry()) == 41
    assert len(reg.ADAPTERS) >= 25
    assert set(reg.ADAPTERS).isdisjoint(reg.BLOCK_REASONS)
    assert len(set(reg.ADAPTERS) | set(reg.BLOCK_REASONS)) == 41
    assert catalog['audit_baseline_sha'] == '75426d42f4ac1c053584bc8e06a0608f4350de20'
    assert set(catalog['audit_inputs_sha256']) == {'independent-runners.json', 'independent-command-plan.json'}
    assert all(len(v) == 64 for v in catalog['audit_inputs_sha256'].values())
    for row in catalog['files']:
        assert row['disposition'] and len(row['sha256']) == 64
    assert reg.registry()['cv-models']['disposition'] == 'deduplicated'
    assert reg.registry()['nonparametric-vs-r']['disposition'] == 'conditional-external-dependencies'


def test_plan_has_no_imports_or_subprocesses():
    code = """
import sys, subprocess
subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No process in planning'))
from dev.benchmarks.independent_benchmark_registry import build_plan
from dev.benchmarks.run_independent_benchmarks import main
plan = build_plan()
assert len(plan['jobs']) == 41
assert not {'numpy', 'scipy', 'torch', 'cupy', 'statgpu'} & set(sys.modules)
assert not any(k.startswith('dev.benchmarks.benchmark_') for k in sys.modules)
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=reg.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_audit_drift_is_not_permanent_execution_pin(tmp_path):
    (tmp_path / 'dev/benchmarks').mkdir(parents=True)
    first = reg.load_catalog()['files'][0]['path']
    (tmp_path / first).write_text('# changed baseline\n')
    new = tmp_path / 'dev/benchmarks/new_support.py'
    new.write_text('# new\n')
    drift = reg.verify_catalog_sources(tmp_path)
    assert first in drift['changed_since_audit']
    assert 'dev/benchmarks/new_support.py' in drift['new_since_audit']
    assert len(drift['missing_since_audit']) == 201


def test_every_adapter_uses_existing_main_guard_and_declared_flags():
    for identifier, entry in reg.registry().items():
        if entry['adapter'] is None:
            continue
        source = (reg.ROOT / entry['script']).read_text()
        tree = ast.parse(source)
        assert any(isinstance(node, ast.If) and '__name__' in ast.unparse(node.test) for node in tree.body)
        flags = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                flags.update(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
        # Three maintained scripts construct their size flags in small loops.
        flags |= {'--n', '--queries', '--resamples', '--dimension', '--seed', '--p'} if identifier in ('public-native-kde-bootstrap', 'native-kde-rng', 'hybrid-phases') else set()
        spec = reg.build_plan([identifier])['jobs'][0]['adapter']
        assert {a for a in spec['arguments'] if a.startswith('--')} <= flags, identifier
        assert '--remote' not in spec['arguments']
        assert 'ssh' not in spec['arguments']


def test_parameter_matrix_identity_is_unique_and_scope_preserved():
    configurations = {'public-native-kde-bootstrap': [{'kernel': 'gaussian'}, {'kernel': 'epanechnikov', 'dimension': 2, 'bandwidth': 'scott'}]}
    plan = reg.build_plan(['public-native-kde-bootstrap'], configurations=configurations)
    assert len(plan['jobs']) == 2
    assert len({j['config_id'] for j in plan['jobs']}) == 2
    assert plan['jobs'][1]['configuration']['dimension'] == 2
    assert plan['jobs'][1]['expected_cells'] == ['numpy', 'cupy', 'torch']
    assert not plan['performance_eligible']
    assert 'partial' in plan['coverage_status']


def test_public_existing_gpu_shape_is_plannable_without_execution():
    plan = reg.build_plan(['public-native-kde-bootstrap'], configurations={'public-native-kde-bootstrap': [{'n': 4096, 'queries': 512, 'resamples': 100}]})
    assert plan['jobs'][0]['configuration']['n'] == 4096


@pytest.mark.parametrize('configuration', [
    {'kernel': 'made-up'}, {'kernel': 'uniform'}, {'bandwidth': 'cv'},
    {'kernel': 'cosine', 'dimension': 2}, {'n': True}, {'queries': 0},
    {'bandwidth': float('nan')}, {'bandwidth': float('inf')},
    {'n': 4096, 'queries': 512, 'resamples': 500, 'dimension': 8},
    {'output': '/tmp/injected'}, {'extra_args': '--remote'}, {'repeats': 100},
])
def test_bad_or_overbudget_configurations_rejected(configuration):
    with pytest.raises(ValueError):
        reg.build_plan(['public-native-kde-bootstrap'], configurations={'public-native-kde-bootstrap': [configuration]})


def test_duplicate_configurations_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        reg.build_plan(['public-native-kde-bootstrap'], configurations={'public-native-kde-bootstrap': [{}, {}]})


def test_compile_scales_expand_explicitly_and_have_expected_cells():
    plan = reg.build_plan(['torch-compile-scale'], configurations={'torch-compile-scale': [{'preset': 'standard', 'cases': ['lasso']}]})
    assert len(plan['jobs'][0]['configuration']['scales']) == 6
    assert len(plan['jobs'][0]['expected_cells']) == 12


@pytest.mark.parametrize('field,value', [('n', 64), ('queries', 2), ('resamples', 2), ('dimension', 2), ('kernel', 'triangular'), ('bandwidth', 'scott'), ('weights', 'none'), ('seed', 44)])
def test_wrong_raw_public_configuration_cannot_receive_requested_identity(field, value):
    payload = public_payload()
    payload['problem'][field] = value
    with pytest.raises(ValueError, match='Wrong configured KDE'):
        reg.validate_artifact('public-native-kde-bootstrap', payload)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unavailable', 'no-correctness', 'short-timing', 'nonfinite', 'wrong-repeats'])
def test_public_missing_or_failed_contract_fails_closed(mutation):
    payload = public_payload()
    if mutation == 'missing': payload['rows'].pop()
    if mutation == 'duplicate': payload['rows'][2] = copy.deepcopy(payload['rows'][1])
    if mutation == 'unavailable': payload['rows'][1]['status'] = 'unavailable'
    if mutation == 'no-correctness': payload['rows'][1].pop('correctness')
    if mutation == 'short-timing': payload['rows'][1]['timing']['samples_seconds'].pop()
    if mutation == 'nonfinite': payload['rows'][1]['timing']['samples_seconds'][0] = float('inf')
    if mutation == 'wrong-repeats': payload['timing']['repeats'] = 3
    with pytest.raises(ValueError): reg.validate_artifact('public-native-kde-bootstrap', payload)


def test_qualified_preserves_raw_and_scoped_evidence_but_not_performance():
    payload = public_payload()
    original = copy.deepcopy(payload)
    result = reg.validate_artifact('public-native-kde-bootstrap', payload)
    assert payload == original
    assert result['correctness_status'] == 'pass'
    assert result['device_claim'] == 'recorded-scoped-evidence-requires-review'
    assert result['performance_eligible'] is False
    assert result['canonical_dashboard_ready'] is False


def test_hybrid_must_match_actual_problem():
    config = reg.normalize_configuration('hybrid-phases', {'cases': ['kde_bootstrap']})
    payload = {'schema_version': 'statgpu-benchmark-v2', 'validation': {'status': 'complete'},
               'rows': [{'backend': b, 'case': 'kde_bootstrap', 'status': 'pass',
                         'correctness': {'status': 'pass'}, 'problem': {**config, 'n': 999},
                         'timing': {'samples_seconds': [.1] * 3}} for b in reg.BACKENDS]}
    with pytest.raises(ValueError, match='Wrong hybrid'):
        reg.validate_artifact('hybrid-phases', payload, config)


@pytest.mark.parametrize('identifier', sorted(reg.ADAPTERS))
def test_every_adapter_rejects_empty_artifact(identifier):
    with pytest.raises((ValueError, KeyError)):
        reg.validate_artifact(identifier, {})


def test_memory_adapter_does_not_claim_peak_or_correctness():
    config = reg.normalize_configuration('gpu-memory-cleanup', {'models': ['linear']})
    payload = {'rows': [dict(model='linear', cleanup=c, sample_index=i, fit_ms=1.0,
                             used_after_fit=1, total_after_fit=2, used_after_reset=1, total_after_reset=1)
                        for c in (False, True) for i in range(3)]}
    result = reg.validate_artifact('gpu-memory-cleanup', payload, config)
    assert result['correctness_status'] == 'unavailable'
    assert result['device_claim'].startswith('unavailable')
    assert 'NOT peak' in result['limitations']


def test_assess_requires_runtime_and_leaves_raw_bytes(tmp_path):
    job = reg.build_plan(['public-native-kde-bootstrap'])['jobs'][0]
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps(public_payload()))
    original = raw.read_bytes()
    process = {'exit_code': 0, 'reason': None}
    assert runner.assess_artifact(job, tmp_path, process)['status'] == 'failed'
    (tmp_path / 'runtime.before.json').write_text(json.dumps({'before': {'cap_verified': True}}))
    (tmp_path / 'runtime.after.json').write_text(json.dumps({'exit_code': 0, 'after': {'cap_verified': True}}))
    result = runner.assess_artifact(job, tmp_path, process)
    assert result['status'] == 'complete'
    assert raw.read_bytes() == original
    assert len(result['raw_sha256']) == 64
    assert not result['performance_eligible']


@pytest.mark.parametrize('body', ['{"x": NaN}', '{"x": 1e999}', '{"x": 1, "x": 2}'])
def test_nonfinite_and_duplicate_json_rejected(tmp_path, body):
    path = tmp_path / 'bad.json'
    path.write_text(body)
    with pytest.raises(ValueError): runner.read_json(path)


def test_unknown_lanes_and_duplicate_lanes_rejected():
    for selection in (['not-a-lane'], ['public-native-kde-bootstrap'] * 2, []):
        with pytest.raises(ValueError): reg.build_plan(selection)


@pytest.mark.parametrize('change', ['module', 'arguments', 'budget', 'identity'])
def test_modified_plan_cannot_execute(tmp_path, monkeypatch, change):
    plan = reg.build_plan(['public-native-kde-bootstrap'], expected_sha=SHA, environment_id='test')
    if change == 'module': plan['jobs'][0]['adapter']['module'] = 'os'
    if change == 'arguments': plan['jobs'][0]['adapter']['arguments'].append('--remote')
    if change == 'budget': plan['jobs'][0]['adapter']['budget']['cpu_threads'] = 24
    if change == 'identity': plan['jobs'][0]['config_id'] = 'wrong'
    monkeypatch.setattr(runner, 'source_fingerprint', lambda: pytest.fail('Must reject before source/process access'))
    with pytest.raises(ValueError, match='reviewed registry'):
        runner.execute(plan, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_execute_requires_source_environment_and_global_budget(tmp_path):
    with pytest.raises(ValueError, match='full exact'):
        runner.execute(reg.build_plan(['public-native-kde-bootstrap']), tmp_path / 'out')
    with pytest.raises(ValueError, match='environment'):
        runner.execute(reg.build_plan(['public-native-kde-bootstrap'], expected_sha=SHA), tmp_path / 'out')
    plan = reg.build_plan(['public-native-kde-bootstrap'], configurations={'public-native-kde-bootstrap': [{'seed': i} for i in range(22)]}, expected_sha=SHA, environment_id='test')
    with pytest.raises(ValueError, match='invocation budget'):
        runner.execute(plan, tmp_path / 'out')
    with pytest.raises(ValueError, match='1..64'):
        runner.execute(plan, tmp_path / 'out', max_invocations=65)


def test_only_selected_blocked_disposition_emits_no_process(tmp_path, monkeypatch):
    source = {'git_commit': SHA, 'working_tree': 'clean', 'content_sha256': 'a' * 64}
    monkeypatch.setattr(runner, 'source_fingerprint', lambda: dict(source))
    monkeypatch.setattr(runner, 'launch', lambda *a, **k: pytest.fail('Blocked adapter must never launch'))
    monkeypatch.setattr(runner, 'freeze_and_archive', lambda p: None)
    plan = reg.build_plan(['cox-fit-timing'], expected_sha=SHA, environment_id='synthetic-only')
    out = tmp_path / 'out'
    report = runner.execute(plan, out)
    assert report['status'] == 'partial'
    assert report['results'][0]['status'] == 'unsupported-contract'
    assert (out / 'source-after.json').exists()


def test_cli_negative_cases_launch_nothing(tmp_path):
    cases = [['--execute'], ['--execute', '--output-root', str(tmp_path / 'a')],
             ['--lanes', 'unknown'], ['--output-root', str(tmp_path / 'b')],
             ['--extra-args', '--remote'], ['--validation-tier', 'physical-cuda']]
    for args in cases:
        result = subprocess.run([sys.executable, '-B', '-m', 'dev.benchmarks.run_independent_benchmarks', *args],
                                cwd=reg.ROOT, capture_output=True, text=True)
        assert result.returncode == 2, result.stderr
    assert not list(tmp_path.iterdir())


def test_synthetic_child_success_and_timeout_only_own_group(tmp_path):
    result = runner.launch([sys.executable, '-B', '-c', 'print("synthetic child only")'],
                           tmp_path / 'success.log', 5, os.environ.copy(), lambda: None)
    assert result['exit_code'] == 0 and result['reason'] is None
    result = runner.launch([sys.executable, '-B', '-c', 'import time; time.sleep(20)'],
                           tmp_path / 'timeout.log', .15, os.environ.copy(), lambda: None)
    assert result['reason'] == 'timeout'
    assert result['elapsed_seconds'] < 5


def test_nested_torch_thread_initialization_without_torch_or_gpu(monkeypatch):
    from types import SimpleNamespace
    state = {'intraop': 16, 'interop': 16}
    fake = SimpleNamespace(set_num_threads=lambda n: state.update(intraop=n),
        set_num_interop_threads=lambda n: state.update(interop=n),
        get_num_threads=lambda: state['intraop'], get_num_interop_threads=lambda: state['interop'])
    compiler = SimpleNamespace(compile_threads=32)
    pool_info = lambda: [{'num_threads': 1}]
    result = reg.configure_torch_child_threads(fake, pool_info=pool_info, compile_config=compiler)
    assert result['cap_verified'] and compiler.compile_threads == 1
    assert os.environ['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
    state['interop'] = 4
    with pytest.raises(RuntimeError, match='budget exceeded'):
        reg.torch_child_thread_state(fake, pool_info=pool_info, compile_config=compiler)
    with pytest.raises(RuntimeError, match='pool is not serial'):
        reg.torch_child_thread_state(fake, pool_info=lambda: [{'num_threads': 4}], compile_config=compiler)


def test_nested_compiler_report_cannot_reuse_outer_cap_only():
    with pytest.raises(ValueError, match='nested workload'):
        reg._validate_nested_thread_limits({'cap_verified': True})
    state = dict(cap_verified=True, intraop=1, interop=1, inductor_compile_threads=1, pools=[{'num_threads': 1}])
    reg._validate_nested_thread_limits({'before': state, 'after': state})
    bad = copy.deepcopy(state)
    bad['inductor_compile_threads'] = 32
    with pytest.raises(ValueError):
        reg._validate_nested_thread_limits({'before': state, 'after': bad})


def test_compile_sources_initialize_and_report_nested_actual_caps():
    for name in ('benchmark_torch_compile_maintenance', 'benchmark_torch_compile_scale'):
        source = (reg.ROOT / 'dev/benchmarks' / (name + '.py')).read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_child')
        text = ast.get_source_segment(source, fn)
        assert text.index('configure_torch_child_threads(torch)') < text.index('torch.cuda.is_available')
        assert '"thread_limits"' in text and 'torch_child_thread_state(torch)' in text
    source = (reg.ROOT / 'dev/benchmarks/benchmark_torch_compile_scale.py').read_text()
    assert '"disable": eager["thread_limits"], "default": compiled["thread_limits"]' in source


def test_nonparametric_bogus_reference_sections_do_not_pass():
    payload = {'metrics': {name: {'timing_ms': {'bogus': 0}, 'precision': {'bogus': 0}}
                           for name in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear', 'kde_confidence_interval')}}
    with pytest.raises(ValueError):
        reg.validate_artifact('nonparametric-vs-r', payload)


def test_extension_omitted_external_references_fail_closed():
    payload = {'models': {name: {backend: {'status': 'ok', 'fit_ms_mean': 1.0}
                                for backend in ('statgpu_cpu', 'statgpu_cuda', 'statgpu_torch')}
                         for name in ('IncrementalPCA', 'MiniBatchNMF')}}
    with pytest.raises(ValueError, match='reference'):
        reg.validate_artifact('unsupervised-phase3c', payload)


@pytest.mark.parametrize('field,value', [('features', 9), ('seed', 99), ('rows_per_stratum', 16)])
def test_exact_strata_wrong_configuration_fails(field, value):
    payload = {'status': 'complete', 'features': 4, 'seed': 20260727, 'rows_per_stratum': 8}
    payload[field] = value
    with pytest.raises(ValueError, match='Wrong Exact configuration'):
        reg.validate_artifact('exact-strata-count-scaling', payload)


@pytest.mark.parametrize('field,value', [('features', 9), ('seed', 99), ('scaling_scenario', 'strata'), ('strata_count_requested', 8)])
def test_exact_ties_wrong_configuration_fails(field, value):
    payload = {'status': 'complete', 'features': 4, 'seed': 88031, 'scaling_scenario': 'right_censored', 'strata_count_requested': 3}
    payload[field] = value
    with pytest.raises(ValueError, match='Wrong Exact'):
        reg.validate_artifact('exact-ties-scaling', payload)


@pytest.mark.parametrize('reason,exit_code', [('interrupted', -9), ('resource_guard: no memory', -9), (None, 86)])
def test_interrupt_and_resource_guards_stop_before_next_lane(tmp_path, monkeypatch, reason, exit_code):
    source = {'git_commit': SHA, 'working_tree': 'clean', 'content_sha256': 'a' * 64}
    monkeypatch.setattr(runner, 'source_fingerprint', lambda: dict(source))
    monkeypatch.setattr(runner, 'freeze_and_archive', lambda p: None)
    monkeypatch.setattr(runner, 'resource_guard', lambda *a: None)
    monkeypatch.setattr(runner, '_host_memory_guard', lambda *a: None)
    calls = []
    def fake_launch(*args):
        calls.append(args)
        return {'reason': reason, 'exit_code': exit_code}
    monkeypatch.setattr(runner, 'launch', fake_launch)
    plan = reg.build_plan(['public-native-kde-bootstrap', 'gpu-memory-cleanup'], expected_sha=SHA, environment_id='synthetic-only')
    report = runner.execute(plan, tmp_path / 'out')
    assert len(calls) == 1
    assert report['stop_reason']
    assert len(report['results']) == 1


def test_preflight_resource_stop_does_not_launch_any_job(tmp_path, monkeypatch):
    source = {'git_commit': SHA, 'working_tree': 'clean', 'content_sha256': 'a' * 64}
    monkeypatch.setattr(runner, 'source_fingerprint', lambda: dict(source))
    monkeypatch.setattr(runner, 'freeze_and_archive', lambda p: None)
    monkeypatch.setattr(runner, 'resource_guard', lambda *a: 'out of space')
    monkeypatch.setattr(runner, 'launch', lambda *a: pytest.fail('Resource preflight must stop execution'))
    plan = reg.build_plan(['public-native-kde-bootstrap', 'gpu-memory-cleanup'], expected_sha=SHA, environment_id='synthetic-only')
    report = runner.execute(plan, tmp_path / 'out')
    assert len(report['results']) == 1
    assert report['stop_reason'].startswith('Resource preflight:')


def test_source_drift_invalidates_and_stops(tmp_path, monkeypatch):
    source = {'git_commit': SHA, 'working_tree': 'clean', 'content_sha256': 'a' * 64}
    state = {'calls': 0}
    def fingerprint():
        state['calls'] += 1
        return dict(source, content_sha256='b' * 64) if state['calls'] > 2 else dict(source)
    monkeypatch.setattr(runner, 'source_fingerprint', fingerprint)
    monkeypatch.setattr(runner, 'freeze_and_archive', lambda p: None)
    monkeypatch.setattr(runner, 'resource_guard', lambda *a: None)
    monkeypatch.setattr(runner, '_host_memory_guard', lambda *a: None)
    calls = []
    monkeypatch.setattr(runner, 'launch', lambda *a: calls.append(a) or {'exit_code': 1, 'reason': None})
    plan = reg.build_plan(['public-native-kde-bootstrap', 'gpu-memory-cleanup'], expected_sha=SHA, environment_id='synthetic-only')
    report = runner.execute(plan, tmp_path / 'out')
    assert len(calls) == 1
    assert report['results'][0]['status'] == 'invalidated'
    assert 'Source changed' in report['stop_reason']


def test_source_clean_and_exact_head_required_before_output_creation(tmp_path, monkeypatch):
    plan = reg.build_plan(['public-native-kde-bootstrap'], expected_sha=SHA, environment_id='synthetic-only')
    for source in ({'git_commit': SHA, 'working_tree': 'dirty-or-unknown'}, {'git_commit': '2' * 40, 'working_tree': 'clean'}):
        monkeypatch.setattr(runner, 'source_fingerprint', lambda: source)
        with pytest.raises(ValueError, match='clean checkout'):
            runner.execute(plan, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_all_admitted_protocols_have_nonempty_unique_expected_cells():
    for job in reg.build_plan()['jobs']:
        cells = job['expected_cells']
        assert len(cells) == len(set(cells))
        assert bool(cells) == (job['adapter'] is not None)


def test_known_mislabeled_partial_legacy_module_is_not_admitted():
    entry = reg.registry()['new-modules']
    assert entry['adapter'] is None
    assert entry['disposition'] == 'unsupported-contract'
    assert 'partial' in entry['blocked_reason'] and 'synchronization' in entry['blocked_reason']


@pytest.mark.parametrize('lane', ['unsupervised-external-matrix', 'unsupervised-phase2'])
def test_external_reference_duplicate_cannot_hide_missing_case(lane):
    if lane == 'unsupervised-external-matrix':
        identities = [(m, 'statgpu', b, v) for m in ('PCA', 'KMeans') for b in ('cpu', 'cuda', 'torch') for v in (2, 4)]
        identities += [(m, f, 'cpu', v) for m in ('PCA', 'KMeans') for f in (('sklearn', 'statsmodels', 'R') if m == 'PCA' else ('sklearn', 'R')) for v in (2, 4)]
        rows = [dict(method=m, framework=f, backend=b, **({'n_components': v} if m == 'PCA' else {'k': v}), status='ok', fit_ms=1., fit_ms_all=[1.] * 3) for m, f, b, v in identities]
    else:
        identities = [(m, 'statgpu', b) for m in ('DBSCAN', 'GaussianMixture', 'NMF') for b in ('cpu', 'cuda', 'torch')]
        identities += [(m, 'sklearn', 'cpu') for m in ('DBSCAN', 'GaussianMixture', 'NMF')]
        identities += [('AgglomerativeClustering', f, 'cpu') for f in ('statgpu', 'sklearn', 'scipy', 'R cluster')]
        identities += [('UMAP', 'umap-learn', 'cpu'), ('TSNE', 'openTSNE', 'cpu')]
        rows = [dict(method=m, framework=f, backend=b, status='ok', fit_ms=1., fit_ms_all=[1.] * 3) for m, f, b in identities]
    assert reg.validate_artifact(lane, {'rows': rows})['protocol_status'] == 'complete'
    rows[-1] = copy.deepcopy(rows[-2])
    with pytest.raises(ValueError, match='duplicate'):
        reg.validate_artifact(lane, {'rows': rows})
