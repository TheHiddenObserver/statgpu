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
    assert {
        'dev/benchmarks/learning_execution_trace.py',
        'statgpu/nonparametric/kernel_smoothing/_native_bandwidth.py',
        'statgpu/nonparametric/kernel_smoothing/_native_bootstrap.py',
    } <= dependencies.keys()
    for path, digest in dependencies.items():
        assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_current_native_learning_support_does_not_reuse_historical_acceptance():
    manifest = json.loads(MANIFEST.read_text())
    learning = next(e for e in manifest['entries'] if e['id'] == 'learning-families')
    assert (
        'Torch KDE bootstrap is implemented in the current public native path; '
        'physical acceptance of this new source remains pending. '
        'Historical learning-suite evidence treated Torch bootstrap as unsupported '
        'and does not certify this implementation.'
    ) in learning['evidence_gaps']
    assert not any('Torch KDE bootstrap remains explicitly unsupported' in gap
                   for gap in learning['evidence_gaps'])


def test_model_benchmark_adapter_source_is_pinned():
    manifest = json.loads(MANIFEST.read_text())
    entry = next(e for e in manifest['entries'] if e['id'] == 'model-families')
    assert entry['script_sha256'] == hashlib.sha256((ROOT / entry['script']).read_bytes()).hexdigest()


def test_repaired_failures_are_historical_not_current_evidence_gaps():
    manifest = json.loads(MANIFEST.read_text())
    entries = {entry['id']: entry for entry in manifest['entries']}
    # These fields are copied into fresh dispatcher artifacts, not just an archive.
    for name in ('model-families', 'statistical-families'):
        gaps = ' '.join(entries[name]['evidence_gaps'])
        assert 'failure retained' not in gaps
        assert 'retained as2failed' not in gaps
        assert 'full' in gaps.lower() and 'unvalidated' in gaps
    glm = next(c for c in manifest['capabilities']
               if c['api'] == 'statgpu.GeneralizedLinearModel')
    assert glm['inference_status'] == '14passed-cpu-reference-aligned'
    assert glm['inference_failures'] == []
    assert 'observed-Hessian bread' in glm['inference_note']
    assert glm['historical_v2_inference_status'] == '12passed2failed-reference-alignment'
    assert {row['case'] for row in glm['historical_v2_inference_failures']} == {
        'glm_inference:inverse_gaussian:hc0',
        'glm_inference:negative_binomial:hc0',
    }
    assert len(entries['model-families']['historical_v2_failures']) == 2
    assert any(row['case'] == 'distributions/f/rvs' and row['outcome'] == 'failed'
               for row in manifest['historical_v2_runtime_limits'])


def test_targeted_physical_evidence_is_not_full_coverage_claim():
    manifest = json.loads(MANIFEST.read_text())
    status = manifest['local_repair_state']['physical_cuda_validation']
    assert '07f12e2ca10e294153bbfd8da5e449bcfd425cbe' in status
    assert 'full seven-suite physical acceptance remains unestablished' in status
    assert any('Full seven-suite physical CUDA acceptance' in gap
               for gap in manifest['global_uncovered_axes'])


def test_repaired_learning_and_distribution_physical_status_is_source_specific():
    manifest = json.loads(MANIFEST.read_text())
    learning = next(e for e in manifest['entries'] if e['id'] == 'learning-families')
    assert '07f12e2ca10e294153bbfd8da5e449bcfd425cbe' in learning['invocation_validation']
    assert 'full learning-suite physical closure' in learning['invocation_validation']
    repaired = [c for c in manifest['capabilities']
                if c.get('support_status') == 'direct_case_with_unreleased_repair']
    assert len(repaired) == 3
    for case in repaired:
        assert '9a6145d187da1000703bc4a20d990ef72d44b893' in case['evidence_status']
        assert 'does not certify the full physical suite or later changes' in case['evidence_status']


def test_targeted_distribution_and_umap_status_matches_recorded_acceptance():
    manifest = json.loads(MANIFEST.read_text())
    cases = [c for c in manifest['capabilities']
             if c.get('case') in ('umap', 'distributions')]
    assert len(cases) == 16
    for case in cases:
        assert '9a6145d187da1000703bc4a20d990ef72d44b893' in case['evidence_status']
        assert 'physical' in case['evidence_status'] and 'not' in case['evidence_status']


def test_fullscan_repair_keeps_failed_physical_evidence_and_repins_dependencies():
    manifest = json.loads(MANIFEST.read_text())
    state = manifest['fullscan_repair_state']
    assert state['base_sha'] == '521680370e0e0071bf881a13fcf5750420c71187'
    assert state['status'] == 'local-repairs-physical-validation-pending'
    historical = state['historical_fullscan']
    assert historical['executed_suites'] == 21
    assert historical['passed_suites'] == 14
    assert historical['failed_suites'] == 7
    assert historical['failed_rows'] == 20
    assert historical['blocked_rows'] == 32
    assert 'exclusivity unverified' in state['performance_qualification']
    for entry in manifest['entries']:
        assert entry['script_sha256'] == hashlib.sha256((ROOT / entry['script']).read_bytes()).hexdigest()
        for path, digest in entry.get('source_dependencies_sha256', {}).items():
            assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_qualification_state_is_separate_from_historical_physical_evidence():
    manifest = json.loads(MANIFEST.read_text())
    state = manifest['benchmark_qualification_state']
    assert state['base_sha'] == '48d03262e765fe276888c4781fd2819b16888bad'
    assert state['status'] == 'new-runner-qualification-physical-validation-pending'
    assert (state['default_invocations'], state['default_rows']) == (21, 1692)
    assert state['allowed_host_only_not_applicable_rows'] == 16
    assert state['learning_case_contracts'] == 44
    assert 'exclusivity unverified' in state['performance_qualification']
    for path, digest in state['source_dependencies_sha256'].items():
        assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
