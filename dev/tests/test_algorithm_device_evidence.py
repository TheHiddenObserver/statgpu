"""Algorithm-stage controls; Torch CPU is real arithmetic, not CUDA evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from dev.benchmarks import benchmark_learning_coverage as learning
from dev.benchmarks import benchmark_model_coverage as models
from dev.benchmarks.learning_execution_trace import (
    NumericalTrace, TraceIncomplete, estimator_stages, model_stages, stage,
)

CASES = sorted(learning.ESTIMATOR_TRACE_CASES)
MODEL_CASES = ['cox:breslow', 'cox:efron', *['panel:' + p for p in models.PANELS]]


def torch_cpu_runtime():
    torch = pytest.importorskip('torch')
    return SimpleNamespace(name='torch', module=torch, device='cpu', context=nullcontext,
                           sync=lambda device: None, array=lambda x: torch.as_tensor(x))


def inject_torch_cpu(model, monkeypatch):
    """Explicit test injection only, never emulate torch CUDA availability."""
    from statgpu.backends import TorchBackend
    backend = TorchBackend(device='cpu')
    model.set_params(device='cpu')
    monkeypatch.setattr(model, '_get_backend', lambda **kwargs: backend)
    if hasattr(model, '_get_xp'):
        monkeypatch.setattr(model, '_get_xp', lambda: backend.xp)
    if hasattr(model, 'backend'):
        model.backend = 'torch'
    if hasattr(model, '_use_gpu_path'):
        monkeypatch.setattr(model, '_use_gpu_path', lambda: True)
        monkeypatch.setattr(model, '_to_array', lambda X, **kwargs: backend.asarray(X))
    return model


@pytest.mark.parametrize('case', CASES)
def test_algorithm_stages_real_torch_cpu_and_independent_oracle(case, monkeypatch):
    rt = torch_cpu_runtime()
    X, y, labels = learning.data(48, 4, 42)
    if case in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear', 'spline', 'gam'):
        X = X[:, :1]
    if case.startswith('kernel_regression') or case == 'gam':
        y = np.sin(X[:, 0]) + .2 * X[:, 0]
    model = inject_torch_cpu(learning.estimator(case, 'cpu', 42, len(X)), monkeypatch)
    dx, dy = rt.array(X), rt.array(y)
    fit_trace = NumericalTrace(rt, estimator_stages(case, 'torch', 'fit'))
    with fit_trace:
        model.fit(dx, dy) if case.startswith('kernel_regression') or case == 'gam' else model.fit(dx)
    fit_report = fit_trace.report()
    operation_trace = NumericalTrace(rt, estimator_stages(case, 'torch', 'operation'))
    with operation_trace:
        out = learning.operation(case, model, dx)
    operation_trace.report()
    assert learning.validate(case, model, X, y, out, labels)['status'] == 'pass'
    proof = learning.verify_device(model, out, rt, case)
    assert fit_report['device'] == 'cpu'
    assert proof['observed_array_device'] == 'cpu'
    if case in ('gam', 'agglomerative'):
        assert isinstance(out, np.ndarray)
        assert proof['output']['role'] == 'host-report'
    if case in ('kernel_pca', 'nystroem'):
        assert any(v['role'] == 'public-host-report' for v in proof['fitted_fields'].values())


@pytest.mark.parametrize('case', MODEL_CASES)
def test_current_model_numerical_paths_real_torch_cpu(case, monkeypatch):
    rt = torch_cpu_runtime()
    factory, args, kw, target, contract, inference = models.make_case(case, 48, 3, 20261020, 'numpy')
    model = inject_torch_cpu(factory(), monkeypatch)
    observer = NumericalTrace(rt, model_stages(case, 'torch'))
    with observer:
        model.fit(*(rt.array(a) for a in args), **kw)
    report = observer.report()
    assert report['device'] == 'cpu'
    assert models.check_fit(model, target, contract, inference)
    if case.startswith('cox:'):
        controls = [e['host_control'] for r in report['stages'] for e in r['operations'] if 'host_control' in e]
        assert 'counts' in controls and 'first_indices' in controls


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', CASES)
def test_learning_algorithm_stages_physical_gpu_when_available(backend, case):
    try:
        rt = learning.Runtime(backend, 0)
    except learning.BackendUnavailable as exc:
        pytest.skip(str(exc))
    row = learning.run_case(case, rt, 48, 4, 42, 0, 1)
    assert row['status'] == 'pass', row
    assert row['correctness']['status'] == 'pass'
    assert all(t['status'] == 'observed' for t in row['execution_trace'])
    assert row['performance_eligible'] is False


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', MODEL_CASES)
def test_model_algorithm_stages_physical_gpu_when_available(backend, case):
    try:
        rt = models.Runtime(backend, 0)
    except models.BackendUnavailable as exc:
        pytest.skip(str(exc))
    row = models.run_case(case, rt, 48, 3, 20261020, 0, 1)
    assert row['status'] == 'passed', row
    assert row['execution_trace']['status'] == 'observed'


def test_native_operation_plus_host_numerics_is_not_native_stage():
    rt = torch_cpu_runtime()
    def mixed(x):
        result = rt.module.exp(x)
        np.sum(x.numpy())
        return result
    observer = NumericalTrace(rt, [stage(mixed, 'strict', ('exp',), reject_host_numerics=True)])
    with observer:
        mixed(rt.module.ones(4))
    with pytest.raises(TraceIncomplete, match='Host/wrong-backend') as error:
        observer.report()
    assert error.value.execution_trace['stages'][0]['operations']


def test_named_host_metadata_does_not_exempt_other_array_numerics():
    rt = torch_cpu_runtime()
    def mixed(x, indices):
        np.sum(indices)
        return rt.module.exp(x)
    # Only searchsorted on that exact metadata parameter was approved.
    observer = NumericalTrace(rt, [stage(mixed, 'strict', ('exp',), reject_host_numerics=True,
                                         host_controls={'searchsorted': ('indices',)})])
    with observer:
        mixed(rt.module.ones(4), np.arange(4))
    with pytest.raises(TraceIncomplete, match='Host/wrong-backend'):
        observer.report()


def test_partial_failure_retains_native_events_and_complete_exception_chain():
    rt = torch_cpu_runtime()
    def fails(x):
        rt.module.exp(x)
        try:
            raise ValueError('inner numerical condition')
        except ValueError as exc:
            raise RuntimeError('outer operation failure') from exc
    observer = NumericalTrace(rt, [stage(fails, 'partial', ('exp',))])
    with pytest.raises(RuntimeError):
        with observer:
            fails(rt.module.ones(4))
    with pytest.raises(TraceIncomplete) as error:
        observer.report()
    report = error.value.execution_trace
    assert report['status'] == 'incomplete'
    assert report['stages'][0]['completed'] is False
    assert report['stages'][0]['operations'][0]['operation'] == 'exp'


def test_learning_incomplete_evidence_still_reports_correctness(monkeypatch):
    def never_called(x):
        return np.exp(x)
    monkeypatch.setattr(learning, 'estimator_stages',
                        lambda *args: [stage(never_called, 'missing-stage', ('exp',))])
    row = learning.run_case('kde', learning.Runtime('numpy'), 32, 4, 42, 0, 1)
    assert row['status'] == 'blocked'
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace'][0]['status'] == 'incomplete'
    assert 'timing' not in row


def test_model_incomplete_evidence_still_reports_correctness(monkeypatch):
    def never_called(x):
        return np.exp(x)
    monkeypatch.setattr(models, 'model_stages',
                        lambda *args: [stage(never_called, 'missing-stage', ('exp',))])
    row = models.run_case('panel:within', models.Runtime('numpy'), 48, 3, 20261020, 0, 1)
    assert row['status'] == 'blocked'
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace']['status'] == 'incomplete'
    assert 'timing' not in row


def test_model_convergence_warning_keeps_coefficients_and_reference_error(monkeypatch):
    original = models.make_case
    def warned(*args, **kwargs):
        factory, args, kw, target, contract, inference = original(*args, **kwargs)
        def new_factory():
            model = factory()
            fit = model.fit
            def warning_fit(*args, **kwargs):
                result = fit(*args, **kwargs)
                warnings.warn('solver did not converge', RuntimeWarning)
                return result
            model.fit = warning_fit
            return model
        return new_factory, args, kw, target, contract, inference
    monkeypatch.setattr(models, 'make_case', warned)
    row = models.run_case('glm:gaussian', models.Runtime('numpy'), 48, 3, 7, 0, 1)
    assert row['status'] == 'failed'
    assert row['correctness']['status'] == 'pass'
    assert row['diagnostics']['coef_max_abs_error'] >= 0
    assert row['diagnostics']['parameters']
    assert row['convergence']['status'] == 'failed'
    assert 'timing' not in row


@pytest.mark.parametrize('case', CASES)
def test_case_specific_ownership_rejects_no_array(case):
    rt = torch_cpu_runtime()
    with pytest.raises(learning.Blocked, match='ownership'):
        learning.verify_device(SimpleNamespace(), 1.0, rt, case)


def test_nystroem_torch_landmark_svd_cannot_use_numpy(monkeypatch):
    rt = torch_cpu_runtime()
    model = inject_torch_cpu(learning.estimator('nystroem', 'cpu', 7, 32), monkeypatch)
    X, _, _ = learning.data(32, 4, 7)
    def forbidden(*args, **kwargs):
        raise AssertionError('host SVD executed')
    monkeypatch.setattr(np.linalg, 'svd', forbidden)
    model.fit(rt.array(X))
    out = model.transform(rt.array(X))
    assert out.device.type == 'cpu'
    assert isinstance(model.normalization_, np.ndarray)


@pytest.mark.parametrize('method', ['quantile', 'uniform'])
def test_gam_native_knots_preserve_rule_and_duplicate_removal(method, monkeypatch):
    rt = torch_cpu_runtime()
    model = learning.estimator('gam', 'cpu', 42, 12)
    model.knot_method = method
    values = np.array([0., 0., 0., 1., 1., 2., 3., 4., 4., 4., 4., 4.])
    expected = (np.percentile(values, [100/3, 200/3]) if method == 'quantile'
                else np.linspace(values.min(), values.max(), 4)[1:-1])
    expected = np.unique(expected)
    def forbidden(*args, **kwargs):
        raise AssertionError('host quantile executed')
    monkeypatch.setattr(np, 'quantile', forbidden)
    monkeypatch.setattr(np, 'percentile', forbidden)
    actual = model._create_knots(rt.array(values), 6, rt.module)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14, rtol=1e-14)


def test_kpca_eigensolver_failure_does_not_retry_on_host(monkeypatch):
    rt = torch_cpu_runtime()
    model = inject_torch_cpu(learning.estimator('kernel_pca', 'cpu', 7, 32), monkeypatch)
    def failed(*args, **kwargs):
        raise RuntimeError('native eigensolver failed')
    def forbidden(*args, **kwargs):
        raise AssertionError('host eigensolver executed')
    monkeypatch.setattr(rt.module.linalg, 'eigh', failed)
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    with pytest.raises(RuntimeError, match='native eigensolver failed'):
        model.fit(rt.array(learning.data(32, 4, 7)[0]))


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', ['kernel_pca', 'nystroem'])
def test_explicit_gpu_fit_and_transform_move_host_inputs_when_available(backend, case):
    try:
        rt = learning.Runtime(backend, 0)
    except learning.BackendUnavailable as exc:
        pytest.skip(str(exc))
    X, _, _ = learning.data(32, 4, 7)
    with rt.context():
        model = learning.estimator(case, {'cupy': 'cuda', 'torch': 'torch'}[backend], 7, len(X))
        model.fit(X)
        out = model.transform(X)
    assert learning.array_identity(out)['device'] == rt.device
    assert learning.array_identity(out)['backend'] == backend


def test_model_fit_failure_retains_partial_trace_and_chained_exception(monkeypatch):
    class Failing:
        def fit(self, X):
            np.exp(X)
            try:
                raise ValueError('inner cause')
            except ValueError as exc:
                raise RuntimeError('outer failure') from exc
    contract = {'parameters': {}, 'data_id': 'fixture', 'reference': 'fixture'}
    monkeypatch.setattr(models, 'make_case',
                        lambda *args: (Failing, (np.ones(4),), {}, np.ones(4), contract, {}))
    monkeypatch.setattr(models, 'model_stages',
                        lambda *args: [stage(Failing.fit, 'failing-fit', ('exp',))])
    row = models.run_case('fixture', models.Runtime('numpy'), 48, 3, 7, 0, 1)
    assert row['status'] == 'failed'
    assert 'inner cause' in row['exception']['traceback']
    assert 'outer failure' in row['exception']['traceback']
    assert row['execution_trace']['stages'][0]['operations']
    assert row['execution_trace']['stages'][0]['completed'] is False
    assert 'timing' not in row


def test_learning_oracle_failure_keeps_preflight_trace(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('independent oracle rejected result')
    monkeypatch.setattr(learning, 'validate', fail)
    with pytest.raises(AssertionError) as error:
        learning.run_case('kde', learning.Runtime('numpy'), 32, 4, 42, 0, 1)
    assert error.value.execution_trace
    assert all(t['status'] == 'observed' for t in error.value.execution_trace)


def test_large_c_reference_is_labeled_approximation():
    contract = models.make_case('glm:binomial', 48, 3, 20261020, 'numpy')[4]
    assert 'approximation' in contract['objective_scaling']
    assert 'not a strictly unpenalized fit' in contract['objective_scaling']


def test_float64_stage_rejects_real_float32_arithmetic():
    rt = torch_cpu_runtime()
    def narrowed(x):
        return rt.module.exp(x.to(rt.module.float32))
    observer = NumericalTrace(rt, [stage(narrowed, 'float64-stage', ('exp',), expected_dtype='float64')])
    with observer:
        narrowed(rt.module.ones(4, dtype=rt.module.float64))
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        observer.report()


def test_runtime_float_dtype_contract_cannot_be_ignored():
    rt = torch_cpu_runtime()
    rt.dtype = 'float64'
    def narrowed(x):
        return rt.module.exp(x.to(rt.module.float32))
    observer = NumericalTrace(rt, [stage(narrowed, 'runtime-dtype', ('exp',))])
    with observer:
        narrowed(rt.module.ones(4, dtype=rt.module.float64))
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        observer.report()


def test_float64_stage_allows_integer_indices_and_boolean_masks():
    rt = torch_cpu_runtime()
    def controlled(x):
        rt.module.argsort(x)
        return rt.module.where(x > 0, x, 0.)
    observer = NumericalTrace(rt, [stage(controlled, 'controls', ('argsort', 'where'), expected_dtype='float64')])
    with observer:
        controlled(rt.module.ones(4, dtype=rt.module.float64))
    assert observer.report()['status'] == 'observed'


def test_gam_basis_contract_rejects_native_token_plus_host_solve():
    rt = torch_cpu_runtime()
    def host_solve(x):
        output = rt.module.where(x > 0, x, 0.)
        np.linalg.solve(np.eye(len(x)), x.numpy())
        return output
    spec = estimator_stages('gam', 'torch', 'operation')[0]
    spec = dict(spec, code=host_solve.__code__)
    observer = NumericalTrace(rt, [spec])
    with observer:
        host_solve(rt.module.ones(4, dtype=rt.module.float64))
    with pytest.raises(TraceIncomplete, match='Host/wrong-backend'):
        observer.report()


def test_field_ownership_rejects_float32_native_state():
    X = learning.data(32, 4, 7)[0][:, :1]
    model = learning.estimator('kde', 'cpu', 7, len(X)).fit(X)
    output = model.predict(X)
    model.samples_ = model.samples_.astype(np.float32)
    with pytest.raises(learning.Blocked, match='Fitted field dtype mismatch'):
        learning.verify_device(model, output, learning.Runtime('numpy'), 'kde')


def test_field_ownership_rejects_float32_host_report():
    X = learning.data(32, 4, 7)[0]
    model = learning.estimator('kernel_pca', 'cpu', 7, len(X)).fit(X)
    output = model.transform(X)
    model.alphas_ = model.alphas_.astype(np.float32)
    with pytest.raises(learning.Blocked, match='Fitted field dtype mismatch'):
        learning.verify_device(model, output, learning.Runtime('numpy'), 'kernel_pca')


def test_field_ownership_rejects_float32_output():
    X = learning.data(32, 4, 7)[0][:, :1]
    model = learning.estimator('kde', 'cpu', 7, len(X)).fit(X)
    output = model.predict(X).astype(np.float32)
    with pytest.raises(learning.Blocked, match='Output dtype mismatch'):
        learning.verify_device(model, output, learning.Runtime('numpy'), 'kde')


def test_shared_estimator_guard_cannot_impersonate_kernel_pca_stage():
    from statgpu._base import BaseEstimator
    from statgpu.nonparametric.kernel_methods import KernelPCA
    class Unrelated(BaseEstimator):
        def fit(self, X, y=None):
            np.mean(X)
            np.linalg.eigh(X.T @ X)
            np.sqrt(np.abs(X))
            return self
        def predict(self, X):
            return X
    assert Unrelated.fit.__code__ is KernelPCA.fit.__code__  # Shared finite guard.
    spec = estimator_stages('kernel_pca', 'numpy', 'fit')[0]
    assert spec['code'] is not Unrelated.fit.__code__
    observer = NumericalTrace(learning.Runtime('numpy'), [spec])
    with observer:
        Unrelated(device='cpu').fit(np.eye(4))
    with pytest.raises(TraceIncomplete, match='Missing or incomplete numerical stage'):
        observer.report()


def test_source_stage_wrapper_cycle_fails_closed():
    def cyclic(x):
        return np.exp(x)
    cyclic.__wrapped__ = cyclic
    with pytest.raises(TraceIncomplete, match='Cannot resolve source stage'):
        stage(cyclic, 'invalid-wrapper', ('exp',))
