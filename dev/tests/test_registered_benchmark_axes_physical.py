"""Real single-CUDA gates for newly configurable runner contracts.

These are small correctness fixtures, not performance evidence. Unavailable
hardware skips locally; a physical acceptance manifest must require their calls.
"""
from types import SimpleNamespace

import pytest

from dev.benchmarks import benchmark_model_coverage as models
from dev.benchmarks import benchmark_learning_coverage as learning
from dev.benchmarks import benchmark_cv_coverage as cv


def runtime(backend):
    try:
        return models.Runtime(backend, 0)
    except models.BackendUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('case,solver', [('glm:gaussian', 'lbfgs'), ('inference:hc3', 'default')])
def test_physical_configured_weighted_model(backend, dtype, case, solver):
    rt = runtime(backend)
    row = models.run_case(case, rt, 48, 3, 19, 0, 1,
                         dtype=dtype, weights='analytic', solver=solver)
    assert row['status'] == 'passed', row
    assert row['input_provenance']['arguments'][0]['dtype'] == dtype
    weight = row['input_provenance']['fit_keywords']['sample_weight']
    assert weight['backend'] == backend and weight['dtype'] == dtype
    assert weight['device'] == rt.device
    assert row['backend_provenance']['compute_dtypes']
    if case.startswith('inference:'):
        assert row['backend_provenance']['compute_dtypes'] == ['float64']
        assert row['execution_trace']['status'] == 'observed'


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('solver', ['full', 'covariance', 'randomized'])
def test_physical_configured_pca(backend, dtype, solver):
    row = learning.run_case('pca', runtime(backend), 48, 3, 13, 0, 1,
                            dtype=dtype, solver=solver)
    assert row['status'] == 'pass', row
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace']
    assert all(trace['status'] == 'observed' for trace in row['execution_trace'])
    assert row['backend'] == backend
    assert row['execution_qualification']['array_device'] == 'cuda:0'
    assert row['execution_qualification']['observations']['requested_backend_array_events'] > 0


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('covariance_type', ['diag', 'full', 'tied', 'spherical'])
def test_physical_configured_gaussian_mixture(backend, dtype, covariance_type):
    row = learning.run_case('gaussian_mixture', runtime(backend), 48, 3, 13, 0, 1,
                            dtype=dtype, covariance_type=covariance_type)
    assert row['status'] == 'pass', row
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace']
    assert all(trace['status'] == 'observed' for trace in row['execution_trace'])
    assert row['backend'] == backend
    assert row['execution_qualification']['array_device'] == 'cuda:0'
    assert row['execution_qualification']['observations']['requested_backend_array_events'] > 0


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
def test_physical_configured_cv_folds_and_grid(backend):
    runtime(backend)  # Only missing physical hardware may skip, not fit errors.
    args = SimpleNamespace(backends=[backend], sizes=[60], features=3, seed=20260807,
                           repeats=1, warmup=0, smoke=False, require_backends=True,
                           folds=2, grid_size=4, dtype='float64', weights='none', sparse='dense')
    result = cv.run(args)
    assert result['validation']['status'] == 'complete', result
    assert len(result['results']) == 6
    assert all(row['status'] == 'pass' for row in result['results'])
    assert all(row['clean_public_fit']['instrumented'] is False for row in result['results'])
    assert all(row['execution_provenance'] for row in result['results'])
    assert all(row['backend'] == backend for row in result['results'])
    assert all(row['case']['cv']['fold_count'] == 2 for row in result['results'])
    assert all(row['case']['grid']['candidate_count'] == 4 for row in result['results'])


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('covariance,weights', [
    ('nonrobust', 'none'), ('nonrobust', 'analytic'),
    ('hc0', 'none'), ('hc0', 'analytic'), ('hc3', 'none'),
])
def test_physical_inference_arithmetic_precision(backend, dtype, covariance, weights):
    # HC3 analytic weights are already mandatory in the existing weighted test.
    rt = runtime(backend)
    row = models.run_case('inference:' + covariance, rt, 48, 3, 19, 0, 1,
                          dtype=dtype, weights=weights)
    assert row['status'] == 'passed', row
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace']['status'] == 'observed'
    expected = 'float64' if backend == 'torch' or weights == 'analytic' else dtype
    assert row['backend_provenance']['compute_dtypes'] == [expected]
    assert row['input_provenance']['arguments'][0]['dtype'] == dtype
    assert row['execution_trace']['device'] == rt.device
