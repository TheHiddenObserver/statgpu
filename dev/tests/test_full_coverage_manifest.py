"""Prevent a broad suite label from concealing unclassified public capabilities."""
import ast
import hashlib
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'dev/benchmarks/full_coverage_manifest.json'


def exported_names():
    tree=ast.parse((ROOT/'statgpu/__init__.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__all__' for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError('Could not resolve public __all__ inventory')


def test_every_top_level_export_is_explicitly_classified():
    m=json.loads(MANIFEST.read_text())
    names=[c['export_name'] for c in m['capabilities'] if c.get('export_name')]
    assert len(names)==len(set(names))
    assert set(names)==exported_names()


def test_each_capability_has_real_entry_or_explicit_reason():
    m=json.loads(MANIFEST.read_text());entries={e['id']:e for e in m['entries']}
    for c in m['capabilities']:
        assert c.get('api')
        assert c.get('support_status')
        if c.get('entry_id'):
            assert c['entry_id'] in entries
        else:
            assert c.get('reason')
            assert c['support_status']=='not_separate_numerical_benchmark'


def test_only_reviewed_local_runners_enabled():
    m=json.loads(MANIFEST.read_text())
    for e in m['entries']:
        assert (ROOT/e['script']).is_file()
        if e['safe_to_execute']:
            assert e['argv'][0]=='{python}'
            assert e['smoke_argv'][0]=='{python}'
            assert '{output}' in e['argv']
            assert '{output}' in e['smoke_argv']
    for e in m.get('legacy_runners',[]):
        assert not e.get('safe_to_execute',False)


def test_final_manifest_does_not_hide_unimplemented_work():
    m=json.loads(MANIFEST.read_text())
    unresolved={'implementation_in_progress','implementation_gap','blocked_runner_schema'}
    missing=[c['api'] for c in m['capabilities'] if c['support_status'] in unresolved]
    assert not missing, 'Unfinished executable coverage: '+', '.join(missing)


def test_active_repaired_execution_status_is_not_historical_failure():
    m=json.loads(MANIFEST.read_text())
    entries={e['id']:e for e in m['entries']}
    assert m['local_repair_state']['status']=='local-cpu-validated-unreleased'
    for name,count in [('model-families',74),('statistical-families',138),('learning-families',44)]:
        assert entries[name]['execution_status']==f'CPU{count}passed'
        assert entries[name]['historical_v2_execution_status']
    for c in m['capabilities']:
        assert not any(old in c.get('evidence_status','') for old in ['72pass2','42pass2','137pass1'])


def test_learning_execution_evidence_dependencies_are_pinned():
    manifest = json.loads(MANIFEST.read_text())
    entry = next(e for e in manifest['entries'] if e['id'] == 'learning-families')
    assert entry['script_sha256'] == hashlib.sha256((ROOT / entry['script']).read_bytes()).hexdigest()
    dependencies = entry['source_dependencies_sha256']
    assert 'dev/benchmarks/learning_execution_trace.py' in dependencies
    for path, digest in dependencies.items():
        assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_model_benchmark_adapter_source_is_pinned():
    manifest = json.loads(MANIFEST.read_text())
    entry = next(e for e in manifest['entries'] if e['id'] == 'model-families')
    assert entry['script_sha256'] == hashlib.sha256((ROOT / entry['script']).read_bytes()).hexdigest()
