"""Coordinator controls without importing numerical runners or launching jobs."""
import json
import types
import sys
from pathlib import Path

import pytest

from dev.benchmarks import run_full_benchmark_matrix as full


def config(tmp_path, value):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize('value', [None, [], {}, {'schema_version': full.SCHEMA, 'unexpected': 1},
    {'schema_version': full.SCHEMA, 'registered': {'argv': '--execute'}},
    {'schema_version': full.SCHEMA, 'registered': {'argv': ['--execute']}},
    {'schema_version': full.SCHEMA, 'registered': {'argv': ['--output-dir=/tmp/foo']}},
    {'schema_version': full.SCHEMA, 'independent': {'command': ['ssh', 'host']}},
    {'schema_version': full.SCHEMA, 'independent': {'lanes': []}},
    {'schema_version': full.SCHEMA, 'independent': {'lanes': ['a', 'a']}}])
def test_rejects_unreviewed_configuration(tmp_path, value):
    with pytest.raises(ValueError):
        full.load_configuration(config(tmp_path, value))


def test_default_plan_retains_denominator_without_running(monkeypatch):
    name = 'dev.benchmarks.independent_benchmark_registry'
    fake = types.ModuleType(name)
    fake.build_plan = lambda **kwargs: {'jobs': [], 'catalog_dispositions': 'all'}
    monkeypatch.setitem(sys.modules, name, fake)
    import dev.benchmarks
    monkeypatch.setattr(dev.benchmarks, 'independent_benchmark_registry', fake, raising=False)
    result = full.build_plan(full.load_configuration())
    assert result['plan_only'] is True
    assert result['lanes']['registered']['expected_total_rows'] == 1692
    assert len(result['lanes']['registered']['jobs']) == 21
    assert result['performance_eligible'] is False
    assert result['comparisons'] == []


def test_registered_only_is_not_entire_catalog():
    result = full.build_plan({'schema_version': full.SCHEMA, 'registered': {'argv': []}})
    assert set(result['lanes']) == {'registered'}
    assert 'not all APIs' in result['coverage_scope']


def test_inventory_execution_requires_explicit_lanes(tmp_path):
    with pytest.raises(ValueError, match='explicit independent lanes'):
        full.execute({'configuration': {'independent': {'lanes': None}}}, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('child_code,child_report,expected', [
    (0, {'status': 'complete'}, 'complete_for_selected_protocols'),
    (2, {'status': 'failed'}, 'incomplete_or_failed'),
    (2, {'status': 'failed', 'stop_reason': 'resource_guard: memory'}, 'failed'),
    (2, {'status': 'failed', 'source_changed': True}, 'failed'),
    (2, {'status': 'failed', 'stop_reason': 'Source changed during execution'}, 'failed'),
    (2, {'status': 'failed', 'stop_reason': 'interrupted'}, 'failed'),
])
def test_selected_result_and_safety_stop(tmp_path, monkeypatch, child_code, child_report, expected):
    destination = tmp_path / 'out'
    monkeypatch.setattr(full.registered, 'validate_destination', lambda path: path)
    monkeypatch.setattr(full.registered, 'fingerprint', lambda: {'git_commit': 'a'*40, 'working_tree': 'clean', 'content_sha256': 'b'*64})
    monkeypatch.setattr(full.registered, 'resource_guard', lambda *a: None)
    monkeypatch.setattr(full.registered, 'freeze_and_archive', lambda path: None)
    def child(plan, target, timeout):
        target.mkdir()
        (target / 'summary.json').write_text(json.dumps(child_report))
        return child_code
    monkeypatch.setattr(full.registered, 'execute', child)
    plan = {'configuration': {'registered': {'argv': []}}, 'lanes': {'registered': {}},
            'source': full.registered.fingerprint()}
    code = full.execute(plan, destination)
    result = json.loads((destination / 'summary.json').read_text())
    assert result['status'] == expected
    assert (code == 0) == (expected == 'complete_for_selected_protocols')
    assert result['performance_eligible'] is False
    assert result['canonical_dashboard_ready'] is False


def test_source_drift_before_dispatch_archives_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(full.registered, 'validate_destination', lambda path: path)
    monkeypatch.setattr(full.registered, 'fingerprint', lambda: {'git_commit': 'b'*40, 'working_tree': 'clean', 'content_sha256': 'c'*64})
    monkeypatch.setattr(full.registered, 'freeze_and_archive', lambda path: None)
    monkeypatch.setattr(full.registered, 'execute', lambda *a: pytest.fail('must not dispatch'))
    plan = {'configuration': {'registered': {'argv': []}}, 'lanes': {'registered': {}},
            'source': {'git_commit': 'a'*40, 'content_sha256': 'c'*64}}
    assert full.execute(plan, tmp_path / 'out') == 2
    report = json.loads((tmp_path / 'out/summary.json').read_text())
    assert 'Source changed since plan' in report['error']
    assert report['lanes'] == {}


def test_independent_configuration_forwarding_and_hash(monkeypatch):
    name = 'dev.benchmarks.independent_benchmark_registry'
    fake = types.ModuleType(name)
    captured = []
    fake.build_plan = lambda **kwargs: captured.append(kwargs) or {'jobs': []}
    monkeypatch.setitem(sys.modules, name, fake)
    import dev.benchmarks
    monkeypatch.setattr(dev.benchmarks, 'independent_benchmark_registry', fake, raising=False)
    settings = {'lanes': ['public-native-kde-bootstrap'], 'configurations': {
        'public-native-kde-bootstrap': [{'seed': 17, 'weights': 'sparse', 'bandwidth': 0.4}]}}
    first = full.build_plan({'schema_version': full.SCHEMA, 'independent': settings})
    assert captured == [settings]
    second_settings = json.loads(json.dumps(settings))
    second_settings['configurations']['public-native-kde-bootstrap'][0]['seed'] = 18
    second = full.build_plan({'schema_version': full.SCHEMA, 'independent': second_settings})
    assert first['configuration_sha256'] != second['configuration_sha256']
    assert first['performance_eligible'] is False


def test_coordinator_never_raises_planned_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(full.registered, 'validate_destination', lambda path: path)
    monkeypatch.setattr(full.registered, 'fingerprint', lambda: {'git_commit': 'a'*40, 'working_tree': 'clean', 'content_sha256': 'b'*64})
    monkeypatch.setattr(full.registered, 'resource_guard', lambda *a: None)
    monkeypatch.setattr(full.registered, 'freeze_and_archive', lambda path: None)
    observed = []
    def child(plan, target, timeout):
        observed.append(timeout)
        target.mkdir()
        (target / 'summary.json').write_text('{"status":"complete"}')
        return 0
    monkeypatch.setattr(full.registered, 'execute', child)
    plan = {'configuration': {'registered': {'argv': ['--timeout', '1']}},
            'lanes': {'registered': {'jobs': [{'resource_bounds': {'timeout_seconds': 1}}]}},
            'source': full.registered.fingerprint()}
    assert full.execute(plan, tmp_path / 'out', timeout=1800) == 0
    assert observed == [1]
    assert json.loads((tmp_path/'out/summary.json').read_text())['lanes']['registered']['effective_timeout_seconds'] == 1


@pytest.mark.parametrize('timeout', [float('nan'), float('inf'), -1, 0, 1801, True, '1'])
def test_execute_api_rejects_invalid_timeout_before_output(tmp_path, timeout):
    with pytest.raises(ValueError, match='Timeout'):
        full.execute({}, tmp_path / 'out', timeout=timeout)
    assert not (tmp_path / 'out').exists()
