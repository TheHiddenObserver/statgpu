"""Schema compatibility and non-fit evidence contract (synthetic, not runtime evidence)."""
from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / 'dev/benchmarks/benchmark_frontend_schema.json').read_text())


def bundle():
    data = json.loads((ROOT / 'frontend/public/data/benchmark_data.json').read_text())
    data['runs'] = [deepcopy(data['runs'][0])]
    return data


def operation_bundle():
    data = bundle()
    data['schema_version'] = '1.2.0'
    run = data['runs'][0]
    run['metrics'] = {'operation_timing': {
        'duration_ms': 2.5, 'scope': 'classification_metric', 'statistic': 'median_of_recorded_samples',
        'unit': 'ms', 'sample_count': 3, 'quality': 'measured', 'source_file': run['source']['file'],
    }}
    run['scale'] = {'scale_key': 'n20_p0', 'n_samples': 20, 'n_features': 0,
                    'feature_dimension_kind': 'not_applicable', 'label': '20 samples (feature dimension not applicable)'}
    run['evidence'] = {
        'qualification': 'diagnostic_only', 'correctness_status': 'pass',
        'validation_status': 'validated_diagnostic', 'performance_eligible': False,
        'canonical_dashboard_ready': False, 'machine_verified_exclusivity': False,
        'raw_performance_eligible': True, 'raw_pointer': '/rows/0/seconds',
        'source_commit': 'a' * 40,
    }
    return data


def validate(data):
    return list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(data))


def test_existing_v11_bundle_remains_valid():
    assert not validate(bundle())


def test_v12_operation_preserves_p0_and_raw_eligibility():
    data = operation_bundle()
    assert not validate(data)
    run = data['runs'][0]
    assert run['scale']['n_features'] == 0
    assert run['evidence']['raw_performance_eligible'] is True
    assert run['evidence']['performance_eligible'] is False
    assert 'timing' not in run['metrics']


@pytest.mark.parametrize('version', ['1.0.0', '1.3.0', 'statgpu-benchmark-update-preview-v1'])
def test_unknown_normalized_schema_version_is_rejected(version):
    data = bundle(); data['schema_version'] = version
    assert validate(data)


@pytest.mark.parametrize('bad', [0, -1, None, True, '2.5'])
def test_nonpositive_or_non_numeric_operation_duration_rejected(bad):
    data = operation_bundle(); data['runs'][0]['metrics']['operation_timing']['duration_ms'] = bad
    assert validate(data)


@pytest.mark.parametrize('bad', [None, ''])
def test_absent_scope_cannot_be_labelled_fit(bad):
    data = operation_bundle(); data['runs'][0]['metrics']['operation_timing']['scope'] = bad
    assert validate(data)


def test_operation_evidence_required_and_fit_field_mutually_exclusive():
    data = operation_bundle(); del data['runs'][0]['evidence']
    assert validate(data)
    data = operation_bundle(); data['runs'][0]['metrics']['timing'] = bundle()['runs'][0]['metrics']['timing']
    assert validate(data)


def test_p0_requires_explicit_not_applicable_classification():
    data = operation_bundle(); del data['runs'][0]['scale']['feature_dimension_kind']
    assert validate(data)
    data = operation_bundle(); data['runs'][0]['scale']['n_features'] = 1
    assert validate(data)


def test_new_semantics_cannot_be_misrepresented_as_v11():
    data = operation_bundle(); data['schema_version'] = '1.1.0'
    assert validate(data)


def test_missing_timing_is_valid_and_does_not_invent_measurement():
    data = operation_bundle(); data['runs'][0]['metrics'] = {}
    assert not validate(data)


@pytest.mark.parametrize('change', [
    {'qualification': 'diagnostic_only'}, {'qualification': 'operator_attested'},
    {'machine_verified_exclusivity': False}, {'raw_performance_eligible': False},
    {'source_commit': None}, {'source_commit': 'wrong'}, {'source_fingerprint': None},
    {'validation_status': 'validated_diagnostic'}, {'evidence_errors': ['failed']},
])
def test_positive_eligibility_requires_consistent_machine_evidence(change):
    data = operation_bundle()
    evidence = data['runs'][0]['evidence']
    evidence.update(qualification='machine_verified', validation_status='pass',
                    performance_eligible=True, canonical_dashboard_ready=True,
                    machine_verified_exclusivity=True, source_fingerprint='b' * 64)
    assert not validate(data)
    evidence.update(change)
    assert validate(data)


def test_diagnostic_records_cannot_carry_a_speedup_even_with_reported_flag():
    data = operation_bundle()
    data['runs'][0]['metrics']['speedup'] = {
        'value': 2, 'reference_backend': 'numpy', 'reference_framework': 'statgpu',
        'reported_semantics': 'reported_by_runner', 'quality': 'reported', 'source_file': 'raw.json',
    }
    assert validate(data)
