"""Exact-ten orchestration tests with fake jobs; no numerical/GPU workloads."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'benchmarks/data_update'
sys.path.insert(0, str(TOOLS))

import batch_driver as d
import rerun_thread_bound_jobs as r


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    (repo / 'statgpu').mkdir(parents=True)
    (repo / 'statgpu/__init__.py').write_text('# Synthetic source only\n')
    evidence = tmp_path / 'old-native'
    evidence.mkdir()
    selected = iter(r.EXPECTED_IDS)
    indexes = {int(item[:4]) for item in r.EXPECTED_IDS}
    jobs = []
    for index in range(21):
        identifier = next(selected) if index in indexes else '%04d-synthetic-unselected' % index
        backend = 'cupy' if '-cupy-' in identifier else 'numpy'
        row = {'case': 'synthetic', 'backend': backend, 'n': 16, 'p': 1, 'seed': 42}
        job = {'id': identifier, 'backend': backend, 'cpu_threads': 1,
            'expected_row_count': 1, 'expected_rows': [row],
            'expected_slots': [{'id': identifier + '-r0', 'row': row, 'source': {'git_commit': r.BASE_SHA}}],
            'expected_source': {'git_commit': r.BASE_SHA},
            'argv': ['--backends', backend, '--require-backends', '--warmup', '1', '--repeats', '3'],
            'resource_bounds': {'timeout_seconds': 1800}}
        jobs.append(job)
        folder = evidence / identifier
        folder.mkdir()
        d.write_json(folder / 'runtime.after.json', {'after': {
            'torch_threads': {'intraop': 1, 'interop': 12 if index in indexes else 1}}})
    original = {'profile': 'representative', 'expected_total_rows': 1692,
                'source': {'git_commit': r.BASE_SHA}, 'jobs': jobs}
    d.write_json(evidence / 'plan.json', original)
    old_source = {'git_commit': r.BASE_SHA, 'working_tree': 'clean'}
    d.write_json(evidence / 'source-before.json', old_source)
    d.write_json(evidence / 'source-after.json', old_source)
    d.seal_directory(evidence, archive=True)
    monkeypatch.setattr(r, 'OLD_PLAN_SHA256', d.file_hash(evidence / 'plan.json'))
    monkeypatch.setattr(r, 'OLD_SEAL_SHA256', d.file_hash(evidence / 'SHA256SUMS.json'))
    source = {'git_commit': r.REPAIRED_SHA, 'working_tree': 'clean', 'content_sha256': 'b' * 64}
    monkeypatch.setattr(d, 'source_identity', lambda *args: copy.deepcopy(source))
    calls = []
    native = types.SimpleNamespace(fingerprint=lambda: copy.deepcopy(source),
        cpu_limits=lambda: {'thread_cap': 1}, CHILD_BOOTSTRAP='# synthetic ' + r.POLICY)
    def execute(plan, output, timeout):
        calls.append(copy.deepcopy(plan))
        output.mkdir()
        d.write_json(output / 'plan.json', plan)
        d.write_json(output / 'summary.json', {'status': 'complete', 'attempted_invocations': 10,
            'results': [{'id': item['id'], 'status': 'complete'} for item in plan['jobs']]})
        d.seal_directory(output, archive=True)
        return 0
    native.execute = execute
    monkeypatch.setattr(r, 'load_native', lambda _repo: native)
    return types.SimpleNamespace(repo=repo, evidence=evidence, original=original,
        source=source, native=native, calls=calls, output=tmp_path / 'output')


def test_default_is_plan_only_and_exact_original_workloads(fixture):
    before_modules = set(sys.modules)
    assert r.run(fixture.repo, fixture.evidence, fixture.output) == 0
    assert fixture.calls == []
    plan = d.read_json(fixture.output / 'plan.json')
    assert plan['plan_only'] is True
    assert [job['id'] for job in plan['jobs']] == list(r.EXPECTED_IDS)
    assert plan['execution_budget']['max_invocations'] == 10
    assert plan['execution_budget']['worst_case_timeout_seconds'] == 18000
    assert plan['thread_bound_rerun']['operator_attestation'] is None
    originals = {job['id']: job for job in fixture.original['jobs']}
    for job in plan['jobs']:
        old = originals[job['id']]
        assert job['argv'] == old['argv']
        assert job['expected_rows'] == old['expected_rows']
        assert [slot['id'] for slot in job['expected_slots']] == [slot['id'] for slot in old['expected_slots']]
        assert job['expected_source']['git_commit'] == r.REPAIRED_SHA
        assert all(slot['source']['git_commit'] == r.REPAIRED_SHA for slot in job['expected_slots'])
    assert not ({'numpy', 'torch', 'cupy', 'statgpu'} & (set(sys.modules) - before_modules))
    d.verify_seal(fixture.output)
    assert not fixture.output.with_suffix('.tar.gz').exists()


def test_execute_requires_explicit_flag_preserves_native_and_outer_archives(fixture):
    assert r.run(fixture.repo, fixture.evidence, fixture.output, execute=True) == 0
    assert len(fixture.calls) == 1
    assert len(fixture.calls[0]['jobs']) == 10
    assert fixture.calls[0]['plan_only'] is False
    d.verify_seal(fixture.output, archive=True)
    d.verify_seal(fixture.output / 'native', archive=True)
    summary = d.read_json(fixture.output / 'summary.json')
    assert summary['attempted_invocations'] == 10
    assert summary['performance_eligible'] is False
    assert summary['canonical_dashboard_ready'] is False
    for path in (Path(r.__file__), Path(d.__file__)):
        assert (fixture.output / 'tooling' / path.name).read_bytes() == path.read_bytes()


@pytest.mark.parametrize('timeout', [0, -1, 1800.01, float('inf'), float('nan'), True])
def test_timeout_is_finite_positive_and_never_expands_budget(fixture, timeout):
    with pytest.raises(ValueError, match='Timeout'):
        r.run(fixture.repo, fixture.evidence, fixture.output, timeout=timeout)
    assert fixture.calls == []
    assert not fixture.output.exists()


def test_shorter_timeout_is_explicit_and_keeps_args(fixture):
    plan, _, _ = r.build_plan(fixture.repo, fixture.evidence, timeout=2.5)
    assert plan['execution_budget']['worst_case_timeout_seconds'] == 25
    assert all(job['resource_bounds']['timeout_seconds'] == 2.5 for job in plan['jobs'])
    assert plan['thread_bound_rerun']['timeout_seconds_per_job'] == 2.5


def test_existing_output_or_archive_is_never_overwritten(fixture):
    fixture.output.mkdir()
    with pytest.raises(ValueError, match='existing'):
        r.run(fixture.repo, fixture.evidence, fixture.output, execute=True)
    assert fixture.calls == []


def test_output_cannot_change_historical_evidence(fixture):
    with pytest.raises(ValueError, match='disjoint'):
        r.run(fixture.repo, fixture.evidence, fixture.evidence / 'new-output')


def test_tampered_old_runtime_or_manifest_is_rejected_before_launch(fixture):
    path = fixture.evidence / r.EXPECTED_IDS[0] / 'runtime.after.json'
    path.chmod(0o644)
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash/file inventory'):
        r.run(fixture.repo, fixture.evidence, fixture.output, execute=True)
    assert fixture.calls == []


def test_different_old_seal_is_rejected(fixture, monkeypatch):
    monkeypatch.setattr(r, 'OLD_SEAL_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='seal differs'):
        r.run(fixture.repo, fixture.evidence, fixture.output, execute=True)
    assert fixture.calls == []


def test_wrong_repaired_source_fails_before_any_launch(fixture, monkeypatch):
    def denied(*args):
        raise ValueError('Checkout HEAD differs from the full expected SHA')
    monkeypatch.setattr(d, 'source_identity', denied)
    with pytest.raises(ValueError, match='HEAD differs'):
        r.run(fixture.repo, fixture.evidence, fixture.output, execute=True)
    assert fixture.calls == []


def test_native_failure_is_preserved_and_nonzero_with_final_archive(fixture):
    fixture.native.execute = lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic native failure'))
    assert r.run(fixture.repo, fixture.evidence, fixture.output, execute=True) == 2
    d.verify_seal(fixture.output, archive=True)
    report = d.read_json(fixture.output / 'summary.json')
    assert report['status'] == 'failed'
    assert 'synthetic native failure' in report['error']


def test_interrupt_is_reported_and_sealed(fixture):
    fixture.native.execute = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt('synthetic interrupt'))
    assert r.run(fixture.repo, fixture.evidence, fixture.output, execute=True) == 2
    assert d.read_json(fixture.output / 'summary.json')['status'] == 'interrupted'
    d.verify_seal(fixture.output, archive=True)


def test_repeated_termination_is_ignored_until_finalization(monkeypatch):
    old = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with r.one_interrupt_then_finalize():
            handler = signal.getsignal(signal.SIGTERM)
            try:
                handler(signal.SIGTERM, None)
            except KeyboardInterrupt:
                assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
                assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
                raise
    assert signal.getsignal(signal.SIGTERM) == old


def real_source():
    path = Path(os.environ.get('STATGPU_THREAD_RERUN_SOURCE',
        str(Path(__file__).resolve().parents[2])))
    if not (path / 'dev/benchmarks/run_qualification_campaign.py').is_file():
        pytest.skip('Set STATGPU_THREAD_RERUN_SOURCE for real-source integration tests')
    return path.resolve()


def external_python(tmp_path, code, *args):
    environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run([sys.executable, '-I', '-c', code,
        str(TOOLS), *map(str, args)], cwd=tmp_path,
        env=environment, capture_output=True, timeout=30, text=True)


def test_native_import_and_assessment_work_outside_checkout_without_pythonpath(tmp_path):
    result = external_python(tmp_path, '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import rerun_thread_bound_jobs as r
native = r.load_native(Path(sys.argv[2]))
from dev.benchmarks import benchmark_qualification
assert Path(benchmark_qualification.__file__).resolve().is_relative_to(Path(sys.argv[2]))
job = {'expected_row_count': 1}
assessment = native.assess(Path('not-present.json'), job, 1, r.REPAIRED_SHA)
assert assessment['correctness']['status'] == 'failed'
assert not {'numpy', 'cupy', 'torch', 'statgpu'}.intersection(sys.modules)
''', real_source())
    assert result.returncode == 0, result.stderr


def test_stale_validator_module_from_different_checkout_is_rejected(tmp_path):
    result = external_python(tmp_path, '''import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import rerun_thread_bound_jobs as r
sys.modules['dev.benchmarks.benchmark_qualification'] = types.SimpleNamespace(__file__='/unrelated/checkout/dev/benchmarks/benchmark_qualification.py')
try:
    r.load_native(Path(sys.argv[2]))
except ValueError as exc:
    assert 'outside the exact checkout' in str(exc)
else:
    raise AssertionError('Foreign validator was not rejected')
''', real_source())
    assert result.returncode == 0, result.stderr


def test_cli_has_no_expansion_or_independent_escape_hatch():
    with pytest.raises(SystemExit):
        r.parser().parse_args(['--repo', 'repo', '--evidence', 'old', '--output-dir', 'new', '--profile', 'expanded'])
    with pytest.raises(SystemExit):
        r.parser().parse_args(['--repo', 'repo', '--evidence', 'old', '--output-dir', 'new', '--max-invocations', '11'])
