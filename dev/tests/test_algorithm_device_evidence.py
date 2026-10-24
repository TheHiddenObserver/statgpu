"""Algorithm-stage controls; Torch CPU is real arithmetic, not CUDA evidence."""
from contextlib import nullcontext
from types import SimpleNamespace
import warnings
import json

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
                           sync=lambda device: None,
                           array=lambda x: torch.as_tensor(x, device='cpu'))


def inject_torch_cpu(model, monkeypatch):
    """Isolate internal Torch CPU controls from the machine's real CUDA.

    This is not a public ``device='torch'`` request (which means CUDA).
    GAM's internal Torch ingress also consults CUDA availability, so replacing
    only its backend/xp leaves this CPU-only test dependent on the host GPU.
    Inject the ingress/backend for nested estimators as well: TSNE constructs
    PCA, GMM constructs KMeans, and KernelRidgeCV refits KernelRidge. Patching
    only the outer object leaves those children on the public NumPy CPU path.
    Numerical implementations and native primitives are not replaced. The
    caller's function-scoped monkeypatch restores all state afterward;
    physical GPU tests below never use this helper.
    """
    from statgpu.backends import TorchBackend
    from statgpu._base import BaseEstimator
    backend = TorchBackend(device='cpu')
    monkeypatch.setattr(BaseEstimator, '_get_backend', lambda self, **kwargs: backend)
    monkeypatch.setattr(BaseEstimator, '_to_array', lambda self, X, **kwargs: backend.asarray(X))
    monkeypatch.setattr(backend.xp.cuda, 'is_available', lambda: False)
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


@pytest.mark.parametrize('cuda_available', [False, True])
def test_torch_cpu_gam_injection_isolates_available_cuda(cuda_available, monkeypatch):
    """Availability alone must not turn a CPU-control test into CUDA work."""
    torch = pytest.importorskip('torch')
    original_available = torch.cuda.is_available
    original_empty = torch.empty
    X, y, labels = learning.data(48, 1, 42)
    y = np.sin(X[:, 0]) + .2 * X[:, 0]

    def reject_cuda_allocation(*args, **kwargs):
        if str(kwargs.get('device', '')).startswith('cuda'):
            raise AssertionError('CPU-control test attempted a CUDA allocation')
        return original_empty(*args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(torch.cuda, 'is_available', lambda: cuda_available)
        local.setattr(torch, 'empty', reject_cuda_allocation)
        assert torch.cuda.is_available() is cuda_available
        model = inject_torch_cpu(learning.estimator('gam', 'cpu', 42, len(X)), local)
        rt = torch_cpu_runtime()
        dx, dy = rt.array(X), rt.array(y)
        assert dx.device.type == dy.device.type == 'cpu'
        trace = NumericalTrace(rt, estimator_stages('gam', 'torch', 'fit'))
        with trace:
            model.fit(dx, dy)
        assert trace.report()['device'] == 'cpu'
        assert model.coef_.device.type == 'cpu'
        assert learning.validate('gam', model, X, y, model.predict(dx), labels)['status'] == 'pass'
    assert torch.cuda.is_available is original_available


def test_public_gam_explicit_cpu_ignores_cuda_availability(monkeypatch):
    """Public CPU dispatch is NumPy, unlike the injected internal controls."""
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def reject_torch_allocation(*args, **kwargs):
        raise AssertionError('Public NumPy CPU request entered Torch allocation')

    monkeypatch.setattr(torch, 'empty', reject_torch_allocation)
    X, y, labels = learning.data(48, 1, 42)
    y = np.sin(X[:, 0]) + .2 * X[:, 0]
    model = learning.estimator('gam', 'cpu', 42, len(X)).fit(X, y)
    assert model._xp is np
    assert isinstance(model.coef_, np.ndarray)
    assert learning.validate('gam', model, X, y, model.predict(X), labels)['status'] == 'pass'


@pytest.mark.parametrize('case', CASES)
def test_algorithm_stages_real_torch_cpu_and_independent_oracle(case, monkeypatch):
    rt = torch_cpu_runtime()
    # Exercise CPU isolation even on a machine without an actual CUDA device.
    monkeypatch.setattr(rt.module.cuda, 'is_available', lambda: True)
    X, y, labels = learning.data(48, 4, 42)
    if case in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear', 'spline', 'gam'):
        X = X[:, :1]
    if case.startswith('kernel_regression') or case == 'gam':
        y = np.sin(X[:, 0]) + .2 * X[:, 0]
    if case in ('nmf', 'minibatch_nmf'):
        X = np.abs(X)  # Match the runner's nonnegative factorization fixture.
    model = inject_torch_cpu(learning.estimator(case, 'cpu', 42, len(X)), monkeypatch)
    assert not rt.module.cuda.is_available()
    dx, dy = rt.array(X), rt.array(y)
    fit_specs = estimator_stages(case, 'torch', 'fit')
    if case == 'dbscan':
        # A CPU tensor truthfully has is_cuda=False, so public DBSCAN fit uses
        # its tensor-distance/host-label fallback, not the CUDA-only dispatch.
        # Observe that real branch here. The unchanged CUDA branch is covered
        # by the physical tests and a separate direct real-Torch-CPU primitive
        # test in test_learning_compute_qualification.py.
        from statgpu.unsupervised import DBSCAN
        fit_specs = [stage(DBSCAN._neighbor_graph_sparse, 'dbscan-cpu-tensor-distances',
                           ('sum', 'matmul'), expected_dtype='float32',
                           reject_host_numerics=True, intermediates=('X_f32', 'distances')),
                     stage(DBSCAN._fit_gpu_fallback, 'dbscan-cpu-host-label-fallback',
                           expected_backend='numpy', placement='host-connected-components')]
    fit_trace = NumericalTrace(rt, fit_specs)
    supervised = case.startswith('kernel_regression') or case in ('gam', 'kernel_ridge', 'kernel_ridge_cv')
    with fit_trace:
        model.fit(dx, dy) if supervised else model.fit(dx)
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
    monkeypatch.setattr(rt.module.cuda, 'is_available', lambda: True)
    factory, args, kw, target, contract, inference = models.make_case(case, 48, 3, 20261020, 'numpy')
    model = inject_torch_cpu(factory(), monkeypatch)
    assert not rt.module.cuda.is_available()
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
    if row['status'] != 'pass':
        # Keep full operation identities/partial traces in JUnit, not a truncated dict repr.
        pytest.fail(json.dumps(row, indent=2, sort_keys=True), pytrace=False)
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


def _cupy_dbscan_source_trace_on_numpy(monkeypatch, geometry_dtype=np.float32):
    """Exercise the actual nested CuPy source with CPU primitives, not CUDA proof."""
    import sys
    from statgpu.unsupervised import DBSCAN
    # Only the module boundary is substituted. Both DBSCAN method code objects
    # and the exact registered stage contracts remain unchanged.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    monkeypatch.setattr(np, 'asnumpy', np.asarray, raising=False)
    class Backend:
        float32 = geometry_dtype
        def __getattr__(self, name):
            return getattr(np, name)
    X = np.array([[0., 0.], [.05, 0.], [0., .05],
                  [2., 2.], [2.05, 2.], [2., 2.05]], dtype=np.float64)
    model = DBSCAN(eps=.2, min_samples=2, device='cpu')
    trace = NumericalTrace(SimpleNamespace(name='numpy', module=np, device='cpu'),
                           estimator_stages('dbscan', 'cupy', 'fit'))
    with trace:
        labels, core = model._fit_gpu_cupy(Backend(), X, len(X), X.shape[1])
    assert labels.dtype == core.dtype == np.int64
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    return trace


def test_cupy_dbscan_nested_float32_geometry_integer_counts_contract(monkeypatch):
    trace = _cupy_dbscan_source_trace_on_numpy(monkeypatch)
    report = trace.report()
    assert report['status'] == 'observed'
    assert len(trace.records) == 2
    for record in trace.records:
        sums = [event for event in record['operations'] if event['operation'] == 'sum']
        assert sums and all(item['dtype'] == 'float32'
                            for event in sums for item in event['inputs'] + event['outputs'])
    outer = next(record for record in trace.records if 'counts-host' in record['stage'])
    assert {item['dtype'] for item in outer['intermediates']['counts']} == {'int64'}
    assert {item['dtype'] for item in outer['intermediates']['labels_core']} == {'int64'}


@pytest.mark.parametrize('dtype', [np.float16, np.float64])
def test_cupy_dbscan_geometry_contract_rejects_unexpected_precision(monkeypatch, dtype):
    trace = _cupy_dbscan_source_trace_on_numpy(monkeypatch, dtype)
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        trace.report()


@pytest.mark.parametrize('field,value', [('backend', 'cupy'), ('device', 'cuda:0')])
def test_cupy_dbscan_nested_geometry_still_rejects_wrong_placement(monkeypatch, field, value):
    trace = _cupy_dbscan_source_trace_on_numpy(monkeypatch)
    outer = next(record for record in trace.records if 'counts-host' in record['stage'])
    event = next(event for event in outer['operations'] if event['operation'] == 'sum')
    event['outputs'] = [dict(item, **{field: value}) for item in event['outputs']]
    with pytest.raises(TraceIncomplete, match='wrong backend/device'):
        trace.report()


def test_unrelated_float64_stage_still_rejects_float32_numpy_arithmetic():
    def target(x):
        return np.sum(x.astype(np.float32))
    trace = NumericalTrace(SimpleNamespace(name='numpy', module=np, device='cpu'),
                           [stage(target, 'strict-float64', ('sum',), expected_dtype='float64')])
    with trace:
        target(np.ones(4, dtype=np.float64))
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        trace.report()


def test_physical_learning_failure_keeps_full_trace_diagnostics(monkeypatch):
    row = {'status': 'blocked', 'reason': 'wrong dtype',
           'execution_trace': [{'operations': [{'operation': 'sum',
                'inputs': [{'backend': 'cupy', 'device': 'cuda:0', 'dtype': 'float32'}],
                'outputs': [{'backend': 'cupy', 'device': 'cuda:0', 'dtype': 'float32'}]}],
                'padding': list(range(200)), 'last_field': 'must-not-be-truncated'}]}
    monkeypatch.setattr(learning, 'Runtime', lambda *args: object())
    monkeypatch.setattr(learning, 'run_case', lambda *args: row)
    with pytest.raises(pytest.fail.Exception) as error:
        test_learning_algorithm_stages_physical_gpu_when_available('cupy', 'dbscan')
    assert json.loads(str(error.value)) == row
