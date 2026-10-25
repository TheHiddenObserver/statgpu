"""Pure, conservative qualification shared by all seven candidate suites.

This is a sidecar assessor, not a replacement numerical oracle or a GPU profiler.
Raw artifacts are never mutated. Correctness dispositions and performance
qualification are independent: unknown process ownership cannot fail correctness.
No CLI flag or producer's ``performance_eligible`` can establish exclusivity.
"""
from __future__ import annotations
from collections import Counter
import math
import re

SUITES = {
    'gaussian-refresh': 'Ingress observation is untimed; plain public estimators are timed. Compute dtype is not inferred from ingress/output.',
    'classification-metrics': 'Output placement and declared internal promotion are not whole-computation device/dtype proof.',
    'model-families': 'Finite source-specific numerical observations do not prove every internal operation.',
    'statistical-families': 'Distribution RNG/host diagnostic boundaries and invariant-only checks remain explicit.',
    'learning-families': 'Algorithm-stage evidence is scoped; host/hybrid and unresolved operations are not native certification.',
    'workflow-families': 'Formula construction and diagnostic accessors include CPU work even with CUDA fitting.',
    'six-family-cv': 'Diagnostic RegionProfiler times are distinct from clean public-fit E2E samples.',
}
PASS = {'pass', 'passed', 'success', 'complete'}
FAIL = {'fail', 'failed', 'failure', 'error', 'incorrect', 'timeout'}
PARTIAL = {'partial', 'skipped', 'skip', 'unavailable', 'blocked', 'not_run', 'pending'}
HOST_ONLY = {'anova/partial_eta_squared', 'panel-primitives/hausman_test',
             'panel-primitives/pooling_f_test', 'panel-primitives/breusch_pagan_lm_test'}


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _status(node):
    value = node.get('status') if isinstance(node, dict) else None
    return value if isinstance(value, str) else None


def _backend(row):
    value = row.get('backend', row.get('requested_backend'))
    if isinstance(value, dict):
        return value.get('requested', value.get('executed'))
    return value


def _case(row):
    value = row.get('case_id', row.get('case', row.get('operation', row.get('model_id'))))
    if isinstance(value, dict):
        return value.get('model_name', row.get('model_id'))
    if value is None:
        problem = row.get('problem', {})
        parameters = problem.get('parameters', {}) if isinstance(problem, dict) else {}
        return parameters.get('model') if isinstance(parameters, dict) else None
    return value


def _finite_duration(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _timing_inventory(row):
    records = []
    for root in ('timing', 'clean_timing', 'clean_public_fit'):
        for item in _walk(row.get(root, {})):
            keys = [key for key in ('raw_seconds', 'samples_seconds', 'seconds') if key in item]
            if not keys:
                continue
            samples = item[keys[0]]
            valid = (isinstance(samples, list) and bool(samples)
                     and all(_finite_duration(x) for x in samples))
            declared = item.get('repeats', row.get(root, {}).get('repeats') if isinstance(row.get(root), dict) else None)
            if declared is not None:
                valid = valid and type(declared) is int and declared == len(samples)
            records.append({'lane': root, 'scope': item.get('scope'),
                            'sample_count': len(samples) if isinstance(samples, list) else None,
                            'samples_well_formed': bool(valid)})
    return records


def evaluate_artifact(document, suite, backend, exit_code=0, expected_source=None,
                      expected_cases=None):
    """Validate candidate structure and retain limited, non-promotional claims.

    ``expected_cases`` is the caller's exact selected identity set, not inferred
    from successful rows. Expanded n/dtype/weight multiplicities must additionally
    be checked by the campaign inventory. This function performs no workload.
    """
    if suite not in SUITES or backend not in {'numpy', 'cupy', 'torch'}:
        raise ValueError('Unknown reviewed suite/backend')
    errors, gaps, rows_out = [], [], []
    if not isinstance(document, dict):
        document = {}; errors.append('Artifact must be an object')
    expected_schema = 'statgpu-scoped-cv-candidate-v2' if suite == 'six-family-cv' else 'statgpu-benchmark-v2'
    if document.get('schema_version') != expected_schema:
        errors.append('Unexpected candidate schema')
    source, after = document.get('source'), document.get('source_after')
    if not isinstance(source, dict) or not isinstance(after, dict):
        errors.append('Missing source-before/source-after'); source = source or {}; after = after or {}
    sha = source.get('git_commit') if isinstance(source, dict) else None
    if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{40}', sha):
        errors.append('Missing exact source SHA')
    if expected_source is not None and sha != expected_source:
        errors.append('Unexpected source SHA')
    validation = document.get('validation', {})
    if not isinstance(validation, dict):
        validation = {}; errors.append('Validation must be an object')
    if source != after or validation.get('source_changed') is True:
        errors.append('Source identity changed')
    if isinstance(source, dict) and source.get('working_tree') != 'clean':
        gaps.append('DIRTY_OR_UNKNOWN_SOURCE')
    fingerprints = ('worktree_fingerprint', 'source_content_sha256', 'content_sha256')
    if not isinstance(source, dict) or not any(isinstance(source.get(k), str) and re.fullmatch('[0-9a-f]{64}', source[k]) for k in fingerprints):
        errors.append('Missing content fingerprint')
    if type(exit_code) is not int or exit_code != 0:
        errors.append('Nonzero or missing process exit')
    state = validation.get('status')
    if not isinstance(state, str): state = None
    if state not in PASS | FAIL | PARTIAL:
        errors.append('Missing recognized validation status')
    elif state in FAIL:
        errors.append('Runner validation failed')
    elif state in PARTIAL:
        gaps.append('INCOMPLETE_RUNNER_VALIDATION')
    rows = document.get('results', document.get('rows'))
    if not isinstance(rows, list) or not rows:
        errors.append('Missing nonempty result rows'); rows = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f'Row {index}: not an object'); continue
        case, row_backend, status = _case(row), _backend(row), row.get('status')
        if not isinstance(status, str): status = 'invalid'
        if isinstance(case, str) and case:
            seen.add(case)
        else:
            errors.append(f'Row {index}: missing case identity')
        if row_backend != backend:
            errors.append(f'Row {index}: wrong or missing backend')
        if 'requested_backend' in row and row['requested_backend'] != backend:
            errors.append(f'Row {index}: contradictory requested_backend')
        declared = row.get('backend')
        if isinstance(declared, dict):
            for field in ('requested', 'executed'):
                if field in declared and declared[field] != backend:
                    errors.append(f'Row {index}: contradictory {field} backend')
        na = status == 'not-applicable'
        if na and not (suite == 'statistical-families' and backend != 'numpy' and case in HOST_ONLY):
            errors.append(f'Row {index}: unauthorized not-applicable')
        elif status in FAIL:
            errors.append(f'Row {index}: numerical/execution failure')
        elif status in PARTIAL:
            gaps.append(f'ROW_{index}_INCOMPLETE')
        elif status not in PASS and not na:
            errors.append(f'Row {index}: unknown disposition')
        # The CV adapter has two named oracle dispositions instead of one status.
        correct = row.get('correctness')
        if status in PASS:
            if not isinstance(correct, dict) or not correct:
                errors.append(f'Row {index}: missing correctness object')
            elif suite == 'six-family-cv':
                clean = row.get('clean_public_fit')
                if not isinstance(clean, dict) or _status(clean) != 'pass' or clean.get('instrumented') is not False:
                    errors.append(f'Row {index}: missing successful clean CV lane')
                direct, external = correct.get('direct_refit'), correct.get('external')
                if not isinstance(direct, dict) or _status(direct) != 'pass':
                    errors.append(f'Row {index}: missing CV direct-refit oracle')
                if not isinstance(external, dict) or _status(external) not in {'pass', 'not_applicable'}:
                    errors.append(f'Row {index}: missing CV external disposition')
                if _status(external) == 'not_applicable' and case not in {'PenalizedGLM_CV', 'CoxPHCV'}:
                    errors.append(f'Row {index}: unjustified CV external not-applicable')
                payload = row.get('case')
                runs = payload.get('runs') if isinstance(payload, dict) else None
                if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) or _status(run) != 'success' for run in runs):
                    errors.append(f'Row {index}: missing or unsuccessful CV measured/reference run')
                else:
                    native = [run for run in runs if run.get('framework') == 'statgpu']
                    if len(native) != 1 or native[0].get('backend') != backend:
                        errors.append(f'Row {index}: wrong CV measured backend disposition')
                    references = [run for run in runs if run.get('framework') == 'sklearn']
                    if case in {'RidgeCV', 'LassoCV', 'ElasticNetCV', 'LogisticRegressionCV'} and len(references) != 1:
                        errors.append(f'Row {index}: missing CV sklearn reference run')
            elif _status(correct) not in {'pass', 'passed'}:
                errors.append(f'Row {index}: missing successful correctness disposition')
        # Do not let a pass parent mask failed correctness, convergence or trace.
        for field in ('correctness', 'convergence', 'execution_trace', 'execution_provenance',
                      'backend_provenance', 'execution_qualification', 'clean_public_fit'):
            if any(isinstance(x.get('status'), str) and x['status'] in FAIL for x in _walk(row.get(field))):
                errors.append(f'Row {index}: failed {field}')
            if any(isinstance(x.get('status'), str) and x['status'] in PARTIAL for x in _walk(row.get(field))):
                gaps.append(f'ROW_{index}_INCOMPLETE_{field.upper()}')
            for node in _walk(row.get(field)):
                if 'status' not in node: continue
                nested = node['status']
                if not isinstance(nested, str) or nested not in PASS | FAIL | PARTIAL | {'observed', 'completed', 'not_applicable', 'not-applicable', 'available', 'incomplete', 'unverifiable'}:
                    errors.append(f'Row {index}: unknown {field} disposition')
                elif nested in {'incomplete', 'unverifiable'}:
                    gaps.append(f'ROW_{index}_UNVERIFIED_{field.upper()}')
        timings = _timing_inventory(row)
        if suite == 'six-family-cv' and status in PASS and not any(record['lane'] == 'clean_public_fit' for record in timings):
            errors.append(f'Row {index}: missing clean CV samples')
        if status in PASS and not timings:
            errors.append(f'Row {index}: missing declared timing samples')
        if any(not x['samples_well_formed'] for x in timings):
            errors.append(f'Row {index}: invalid timing samples')
        evidence = [key for key in ('execution_trace', 'execution_provenance', 'backend_provenance', 'execution_qualification') if row.get(key)]
        rows_out.append({'row_index': index, 'case': case, 'backend': row_backend,
                         'reported_disposition': status, 'timing_records': timings,
                         'evidence_fields_present': evidence,
                         'device_claim': 'recorded-scoped-evidence-requires-review' if evidence else 'not-established-by-this-artifact',
                         'compute_dtype_claim': 'not-inferred-from-input-or-output',
                         'performance_eligible': False})
    if expected_cases is not None and seen != set(expected_cases):
        errors.append('Selected case identity set mismatch')
    return {'schema_version': 'statgpu-candidate-qualification-v1', 'suite': suite,
            'backend': backend, 'source_sha': sha,
            'correctness': {'status': 'failed' if errors else 'partial' if gaps else 'complete',
                            'errors': errors, 'uncovered': gaps,
                            'scope': 'stored runner assertions and exact selected identities; not a rerun of numerical oracles'},
            'row_status_counts': dict(Counter(r['reported_disposition'] for r in rows_out)),
            'rows': rows_out, 'full_api_coverage': False, 'whole_pipeline_native_gpu': False,
            'performance': {'qualification': 'diagnostic_only', 'exclusive_gpu_ownership': 'UNVERIFIED_EXCLUSIVITY',
                            'eligible': False, 'comparisons': [], 'canonical_dashboard_ready': False,
                            'reasons': ['UNVERIFIED_EXCLUSIVITY', 'NO_CANONICAL_PROMOTION_REVIEW',
                                        'PEAK_MEMORY_NOT_ESTABLISHED_BY_CANDIDATE_TIMING', SUITES[suite]]}}
