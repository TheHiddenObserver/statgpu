"""Bounded CPU and synthetic contracts, never physical CUDA execution evidence."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import dis
import sys

import numpy as np
import pytest

from dev.benchmarks import benchmark_learning_coverage as b
from dev.benchmarks.learning_execution_trace import (
    NumericalTrace, TraceIncomplete, estimator_stages, learning_stages, stage,
)

FORMER_OWNERSHIP_ONLY = (
    'kernel_ridge', 'kernel_ridge_cv', 'pca', 'incremental_pca', 'truncated_svd',
    'kmeans', 'minibatch_kmeans', 'dbscan', 'gaussian_mixture', 'nmf',
    'minibatch_nmf', 'tsne', 'umap', 'umap_nndescent', 'bspline_basis',
    'natural_spline', 'cyclic_spline', 'thin_plate_spline', 'smoothing_helpers',
    'pairwise_kernels', 'chi2_kernel',
)


def specs_for(case, backend='torch'):
    return (learning_stages(case, backend) if case in b.TRACE_CASES
            else estimator_stages(case, backend, 'fit'))


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'torch'])
@pytest.mark.parametrize('case', b.CASES)
def test_all_44_cases_have_explicit_source_contracts(case, backend):
    specs = specs_for(case, backend)
    assert specs
    assert all(spec['code'] and spec['label'] for spec in specs)
    if case in FORMER_OWNERSHIP_ONLY:
        assert any(spec['operations'] for spec in specs) or (
            case == 'dbscan' and backend == 'numpy')
    if case in b.ESTIMATOR_TRACE_CASES:
        operation = estimator_stages(case, backend, 'operation')
        assert operation or case in ('dbscan', 'agglomerative', 'tsne', 'umap', 'umap_nndescent')


def synthetic_protocol(case):
    """Metadata fixture only: no fake arrays are presented as GPU evidence."""
    runtime = SimpleNamespace(name='torch', device='cuda:3', dtype='float64')
    specs = specs_for(case)
    trace = NumericalTrace(runtime, specs)
    for spec in specs:
        backend = spec['expected_backend'] or runtime.name
        dtype = spec['expected_dtype'] or 'float64'
        if isinstance(dtype, tuple):
            dtype = dtype[0]
        array = {'kind': 'array', 'backend': backend,
                 'device': 'cpu' if backend == 'numpy' else 'cuda:3',
                 'dtype': dtype, 'shape': [8, 4]}
        operations = []
        for name in spec['operations']:
            for _ in range(spec['operation_counts'].get(name, 1)):
                operations.append({'operation': name, 'backend': backend,
                    'inputs': [dict(array)], 'outputs': [dict(array)], 'status': 'completed'})
        record = {'stage': spec['label'], 'completed': True, 'operations': operations,
                  'intermediates': {field: [dict(array)] for field in spec['intermediates']},
                  'executed_assignments': {site['local']: site['expected_count']
                                           for site in spec['assignment_sites'].values()}}
        trace.records.extend(deepcopy(record) for _ in range(max(spec['minimum_calls'], 1)))
    trace.report()
    return trace


@pytest.mark.parametrize('case', FORMER_OWNERSHIP_ONLY)
@pytest.mark.parametrize('corruption', ['cpu-substitution', 'wrong-device', 'scalar-only',
                                      'partial-stage', 'wrong-dtype', 'missing-stage'])
def test_every_new_contract_rejects_invalid_numerical_evidence(case, corruption):
    trace = synthetic_protocol(case)
    candidates = [record for record in trace.records if record['operations']
                  and any(event['backend'] == 'torch' for event in record['operations'])]
    target = candidates[0]
    if corruption == 'missing-stage':
        trace.records[:] = [record for record in trace.records if record['stage'] != target['stage']]
    elif corruption == 'partial-stage':
        target['operations'].clear()
    else:
        for event in target['operations']:
            if event['backend'] != 'torch':
                continue
            if corruption == 'cpu-substitution':
                for value in event['inputs'] + event['outputs']:
                    value.update(backend='numpy', device='cpu')
            elif corruption == 'wrong-device':
                event['inputs'][0]['device'] = 'cuda:0'
            elif corruption == 'scalar-only':
                event['inputs'] = []
                event['outputs'] = []
            elif corruption == 'wrong-dtype':
                event['inputs'][0]['dtype'] = 'float16'
    with pytest.raises(TraceIncomplete):
        trace.report()


@pytest.mark.parametrize('case', FORMER_OWNERSHIP_ONLY)
def test_new_contracts_require_all_source_local_intermediates(case):
    trace = synthetic_protocol(case)
    target = next((r for r in trace.records if r['intermediates']), None)
    if target is None:
        # chi2's exp/sum primitive operands are themselves its intermediates.
        assert case == 'chi2_kernel'
        return
    target['intermediates'][next(iter(target['intermediates']))] = []
    with pytest.raises(TraceIncomplete, match='intermediate'):
        trace.report()


def test_partial_cv_folds_and_source_loop_iterations_cannot_pass():
    trace = synthetic_protocol('kernel_ridge_cv')
    target = next(record for record in trace.records
                  if record['stage'] == 'kernel-ridge-folds-score-selection-refit')
    index = next(i for i, event in enumerate(target['operations']) if event['operation'] == 'linalg.eigh')
    target['operations'].pop(index)
    with pytest.raises(TraceIncomplete, match='count'):
        trace.report()
    for case, label in [('tsne', 'tsne-gradient-optimization'), ('umap', 'umap-layout-gradient')]:
        trace = synthetic_protocol(case)
        target = next(record for record in trace.records if record['stage'] == label)
        target['executed_assignments']['grad'] -= 1
        with pytest.raises(TraceIncomplete, match='loop execution'):
            trace.report()


def test_missing_contract_is_not_established_by_native_output():
    runtime = SimpleNamespace(name='torch', device='cuda:3')
    row = b.classify_placement({'case': 'pca', 'status': 'pass', 'performance_eligible': True,
                               'execution': {'observed_array_device': 'cuda:3'}}, runtime)
    assert row['execution_qualification']['compute_placement'] == 'unverified'
    assert not row['performance_eligible']


def test_hybrid_classification_comes_from_events_not_case_whitelist():
    runtime = SimpleNamespace(name='torch', device='cuda:3')
    event = {'backend': 'torch', 'operation': 'sum',
             'inputs': [{'kind': 'array', 'backend': 'torch', 'device': 'cuda:3'}]}
    row = {'case': 'unregistered-case-name', 'status': 'pass', 'execution_trace': [
        {'status': 'observed', 'stages': [{'stage': 'generic-stage', 'operations': [event]}]}]}
    native = b.classify_placement(deepcopy(row), runtime)
    assert native['execution_qualification']['compute_placement'].startswith('observed-native')
    row['execution_trace'][0]['stages'][0]['operations'].append({
        'backend': 'numpy', 'operation': 'sum', 'inputs': [{'kind': 'array', 'backend': 'numpy', 'device': 'cpu'}]})
    hybrid = b.classify_placement(row, runtime)
    assert hybrid['execution_qualification']['compute_placement'] == 'mixed-host-and-observed-native-arithmetic'


@pytest.mark.parametrize('case', ['bandwidth_nrd', 'bandwidth_ucv', 'bandwidth_cv',
    'kde_bootstrap', 'kde_normal_ci', 'knockoff_fixed', 'knockoff_model',
    'smoothing_helpers', 'pairwise_kernels', 'cyclic_spline', 'pca'])
def test_learning_timings_exclude_all_observation(case, monkeypatch):
    original = b.timed
    boundaries = []
    def clean_timed(call, runtime):
        def guarded():
            boundaries.append((sys.getprofile(), sys.gettrace()))
            return call()
        return original(guarded, runtime)
    monkeypatch.setattr(b, 'timed', clean_timed)
    row = b.run_case(case, b.Runtime('numpy'), 32, 4, 42, 1, 2)
    assert row['status'] == 'pass', row.get('reason')
    assert boundaries and all(profile is None and trace is None for profile, trace in boundaries)
    assert row['timing']['instrumentation_in_timed_region'] is False
    assert 'untimed preflight' in row['timing']['instrumentation']
    assert len(row['execution_trace']) == (2 if case == 'pca' else 1)
    assert not row['performance_eligible']


@pytest.mark.parametrize('case', ['bandwidth_nrd', 'kde_bootstrap', 'knockoff_fixed'])
def test_clean_timed_results_still_pass_independent_oracles(case, monkeypatch):
    original = b.timed
    def corrupt(call, runtime):
        result, elapsed = original(call, runtime)
        if case == 'bandwidth_nrd':
            result.bandwidth_factor_ *= 2
        elif case == 'kde_bootstrap':
            result.lower[0] += 1
        else:
            result.selected_features = np.arange(4)
        return result, elapsed
    monkeypatch.setattr(b, 'timed', corrupt)
    with pytest.raises(AssertionError):
        b.run_case(case, b.Runtime('numpy'), 32, 4, 42, 0, 1)
    assert sys.getprofile() is None and sys.gettrace() is None


def torch_cpu():
    torch = pytest.importorskip('torch')
    return SimpleNamespace(name='torch', module=torch, device='cpu', dtype='float64',
        context=nullcontext, array=lambda x: torch.as_tensor(x, dtype=torch.float64), sync=lambda device: None)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_operator_only_kernels_observe_actual_native_slot_execution(backend):
    from statgpu.nonparametric.kernel_methods import _kernels
    runtime = b.Runtime('numpy') if backend == 'numpy' else torch_cpu()
    X = runtime.array(np.arange(12.).reshape(4, 3))
    specs = learning_stages('pairwise_kernels', backend)[:2]
    originals = (np.matmul, sys.gettrace(), sys.getprofile())
    trace = NumericalTrace(runtime, specs)
    with trace:
        linear = _kernels.linear_kernel(X, X, xp=runtime.module)
        poly = _kernels.polynomial_kernel(X, X, xp=runtime.module)
    report = trace.report()
    assert len(report['stages']) == 2
    assert all(record['operator_evidence'] for record in report['stages'])
    assert all(event['evidence_kind'].startswith('executed-source-opcode')
               for record in report['stages'] for event in record['operator_evidence'])
    assert (np.matmul, sys.gettrace(), sys.getprofile()) == originals
    expected = b.host(X) @ b.host(X).T
    np.testing.assert_allclose(b.host(linear), expected)
    np.testing.assert_allclose(b.host(poly), (expected / 3 + 1) ** 3)
    assert report['device'] == 'cpu'


def test_python_39_bytecode_without_positions_is_resolved(monkeypatch):
    from statgpu.nonparametric.kernel_methods._kernels import linear_kernel
    original = dis.get_instructions
    def legacy(fn):
        for instruction in original(fn):
            yield SimpleNamespace(offset=instruction.offset, starts_line=instruction.starts_line,
                opname='BINARY_MATRIX_MULTIPLY' if instruction.opname == 'BINARY_OP' and instruction.argrepr == '@' else instruction.opname,
                argrepr=instruction.argrepr)
    monkeypatch.setattr(dis, 'get_instructions', legacy)
    spec = stage(linear_kernel, 'legacy-slot', ('operator.matmul',), operator_sites=(('X @ Y.T', '$return'),))
    assert len(spec['operator_sites']) == 1


def test_operator_source_contract_refuses_unverifiable_expression():
    from statgpu.nonparametric.kernel_methods._kernels import linear_kernel
    with pytest.raises(TraceIncomplete, match='unique numerical operator'):
        stage(linear_kernel, 'wrong-site', ('operator.matmul',), operator_sites=(('Y @ X.T', '$return'),))


def test_operator_observation_does_not_replace_existing_trace():
    from statgpu.nonparametric.kernel_methods._kernels import linear_kernel
    prior = lambda *args: None
    spec = stage(linear_kernel, 'slot', ('operator.matmul',), operator_sites=(('X @ Y.T', '$return'),))
    sys.settrace(prior)
    try:
        with pytest.raises(TraceIncomplete, match='Existing trace'):
            with NumericalTrace(b.Runtime('numpy'), [spec]):
                pass
        assert sys.gettrace() is prior
    finally:
        sys.settrace(None)


def test_operator_cpu_recast_replacement_slot_is_rejected(monkeypatch):
    from statgpu.nonparametric.kernel_methods._kernels import linear_kernel
    runtime = torch_cpu()
    tensor = runtime.module.Tensor
    def cpu_recast(self, other):
        return runtime.module.as_tensor(self.numpy() @ other.numpy())
    monkeypatch.setattr(tensor, '__matmul__', cpu_recast)
    spec = stage(linear_kernel, 'slot', ('operator.matmul',), operator_sites=(('X @ Y.T', '$return'),))
    trace = NumericalTrace(runtime, [spec])
    with trace:
        linear_kernel(runtime.array(np.ones((3, 2))), runtime.array(np.ones((4, 2))))
    with pytest.raises(TraceIncomplete, match='Failed numerical primitive'):
        trace.report()


def test_numpy_operator_subclass_is_not_native_primitive_proof():
    from statgpu.nonparametric.kernel_methods._kernels import linear_kernel
    class Substitute(np.ndarray):
        def __matmul__(self, other):
            return np.zeros((self.shape[0], other.shape[1]))
    X = np.ones((3, 2)).view(Substitute)
    spec = stage(linear_kernel, 'slot', ('operator.matmul',), operator_sites=(('X @ Y.T', '$return'),))
    trace = NumericalTrace(b.Runtime('numpy'), [spec])
    with trace:
        linear_kernel(X, X)
    with pytest.raises(TraceIncomplete):
        trace.report()


def test_numpy_ufunc_reduction_protocol_is_unchanged():
    def kernel(X):
        return np.min(X) + np.max(X)
    original = np.minimum
    trace = NumericalTrace(b.Runtime('numpy'), [stage(kernel, 'reductions', ('min', 'max'))])
    with trace:
        assert kernel(np.arange(4.)) == 3
        np.testing.assert_array_equal(np.minimum.accumulate([3, 2, 4]), [3, 2, 2])
    trace.report()
    assert np.minimum is original


def test_bootstrap_tracing_preserves_real_seeded_rng_and_result():
    from statgpu.nonparametric import kde_bootstrap_confidence_interval
    runtime = torch_cpu()
    X = runtime.array(np.random.default_rng(7).normal(size=(24, 1)))
    Q = runtime.array(np.linspace(-1, 1, 5))
    def call():
        return kde_bootstrap_confidence_interval(X, Q, bandwidth=.4, n_resamples=20,
            random_state=7, backend='torch', return_bootstrap_samples=True)
    before = call()
    original = runtime.module.rand
    traces = []
    observed = b.traced_call('kde_bootstrap', call, runtime, traces)
    after = call()
    for field in ('estimate', 'lower', 'upper', 'bootstrap_samples'):
        np.testing.assert_array_equal(getattr(before, field), getattr(observed, field))
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    assert runtime.module.rand is original
    assert traces[0]['device'] == 'cpu'
    assert len([record for record in traces[0]['stages'] if record['stage'] == 'native-resampling']) == 20


@pytest.mark.parametrize('case', ['pca', 'incremental_pca', 'truncated_svd', 'tsne'])
def test_real_torch_sign_control_dtype_is_retained_not_relabelled(case, monkeypatch):
    from statgpu._base import BaseEstimator
    from statgpu.backends import TorchBackend
    runtime = torch_cpu()
    backend = TorchBackend('cpu')
    # Only inject a CPU backend to exercise unchanged numerical implementations;
    # this is not the public device='torch' CUDA dispatch or physical evidence.
    monkeypatch.setattr(BaseEstimator, '_get_backend', lambda self: backend)
    row = b.run_case(case, runtime, 24, 4, 42, 0, 1)
    assert row['status'] == 'pass', row.get('reason')
    controls = [event for trace in row['execution_trace'] for record in trace['stages']
                for event in record['operations'] if event.get('floating_dtype_controls')]
    assert controls
    assert all(event['inputs'][0]['dtype'] == 'bool' for event in controls)
    assert all(event['outputs'][0]['dtype'] == 'float32' for event in controls)
    assert all(event['floating_dtype_controls'][0]['values'] == [-1.0, 1.0] for event in controls)
    assert row['execution']['observed_array_device'] == 'cpu'
    assert not row['performance_eligible']


def test_sign_control_does_not_allow_float32_data_or_arbitrary_constants():
    runtime = torch_cpu()
    torch = runtime.module
    from statgpu.backends import TorchBackend
    from statgpu.unsupervised._utils import svd_flip_components
    spec = next(s for s in estimator_stages('pca', 'torch', 'fit')
                if s['label'] == 'decomposition-sign-convention')
    trace = NumericalTrace(runtime, [spec])
    with trace:
        svd_flip_components(TorchBackend('cpu'), torch.tensor([[1., -2.]], dtype=torch.float32))
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        trace.report()

    def other_constants(x):
        return torch.where(x > 0, -1., 2.)
    spec = stage(other_constants, 'constant-control', ('where',), expected_dtype='float64',
                 control_outputs={'where': {'values': (-1., 1.), 'dtype': 'float32'}})
    trace = NumericalTrace(runtime, [spec])
    with trace:
        other_constants(torch.ones(3, dtype=torch.float64))
    assert not trace.records[0]['operations'][0].get('floating_dtype_controls')
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        trace.report()


def test_real_torch_dbscan_distance_and_propagation_primitives_on_cpu():
    from statgpu.backends import TorchBackend
    from statgpu.unsupervised import DBSCAN
    runtime = torch_cpu()
    X, _, labels = b.data(24, 4, 42)
    model = DBSCAN(eps=1., min_samples=3, device='cpu')
    trace = NumericalTrace(runtime, estimator_stages('dbscan', 'torch', 'fit'))
    with trace:
        result, core = model._fit_gpu(TorchBackend('cpu'), runtime.array(X), 24, 4)
    report = trace.report()
    operations = {event['operation'] for record in report['stages'] for event in record['operations']}
    assert {'Tensor.__matmul__', 'Tensor.sum', 'Tensor.clamp_', 'Tensor.scatter_reduce_'} <= operations
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(result, labels) == 1
    assert len(core) == 24
    assert report['device'] == 'cpu'
    assert runtime.module.Tensor.__matmul__.__class__.__name__ == 'method_descriptor'


def test_cpu_oracle_eligibility_is_not_performance_promotion():
    pca = b.run_case('pca', b.Runtime('numpy'), 24, 4, 42, 0, 1)
    nmf = b.run_case('nmf', b.Runtime('numpy'), 24, 4, 42, 0, 1)
    assert pca['oracle_comparison_eligible']
    assert not nmf['oracle_comparison_eligible']
    assert not pca['performance_eligible'] and not nmf['performance_eligible']


@pytest.mark.parametrize('backend', ['cupy', 'torch'])
@pytest.mark.parametrize('case', b.CASES)
def test_physical_gpu_complete_learning_compute_contract(case, backend):
    """P100/other physical lane: all 44 configured cases, not CPU simulation.

    A missing runtime skips explicitly. An available runtime must satisfy every
    numerical-stage and concrete-device contract; blocked rows are failures, not
    passes. This small fixture is not exclusivity or canonical speedup evidence.
    """
    try:
        runtime = b.Runtime(backend, 0)
    except b.BackendUnavailable as exc:
        pytest.skip(str(exc))
    row = b.run_case(case, runtime, 48, 4, 42, 0, 1)
    assert row['status'] == 'pass', row.get('reason')
    assert row['correctness']['status'] == 'pass'
    assert row['execution_trace']
    assert all(trace['status'] == 'observed' and trace['device'] == 'cuda:0'
               for trace in row['execution_trace'])
    assert row['timing']['instrumentation_in_timed_region'] is False
    assert row['timing']['synchronized_device'] == 'cuda:0'
    assert not row['performance_eligible']
    assert row['execution_qualification']['observations']['requested_backend_array_events'] > 0


def test_unresolvable_operator_contract_blocks_instead_of_claiming_ownership(monkeypatch):
    original = b.learning_stages
    def unavailable(case, backend):
        if case == 'pairwise_kernels':
            raise TraceIncomplete('unsupported interpreter opcode mapping')
        return original(case, backend)
    monkeypatch.setattr(b, 'learning_stages', unavailable)
    row = b.run_case('pairwise_kernels', b.Runtime('numpy'), 24, 4, 42, 0, 1)
    assert row['status'] == 'blocked'
    assert row['blocked_kind'] == 'incomplete-execution-evidence'
    assert row['correctness']['status'] == 'pass'
    assert not row['performance_eligible'] and 'timing' not in row


@pytest.mark.parametrize('case', ['tsne', 'umap'])
def test_actual_truncated_embedding_iterations_are_not_hidden_by_child_operations(case, monkeypatch):
    if case == 'tsne':
        from statgpu.unsupervised import TSNE
        validate = TSNE._validate_params
        def shorter(self, n):
            validate(self, n)
            self._max_iter = 124
        monkeypatch.setattr(TSNE, '_validate_params', shorter)
    else:
        from statgpu.unsupervised import UMAP
        monkeypatch.setattr(UMAP, '_epochs', lambda self, n: 99)
    row = b.run_case(case, b.Runtime('numpy'), 24, 4, 42, 0, 1)
    assert row['correctness']['status'] == 'pass'
    assert row['status'] == 'blocked' and 'timing' not in row
    assert 'Incomplete numerical loop execution' in row['reason']
    record = next(record for trace in row['execution_trace'] for record in trace['stages']
                  if record.get('executed_assignments'))
    assert record['executed_assignments']['grad'] == (124 if case == 'tsne' else 99)


@pytest.mark.parametrize('cuda_branch', [False, True])
def test_krr_cv_real_torch_branch_controls_keep_float32_metadata(cuda_branch, monkeypatch):
    from statgpu._base import BaseEstimator
    from statgpu.backends import TorchBackend
    runtime = torch_cpu()
    backend = TorchBackend('cpu')
    monkeypatch.setattr(BaseEstimator, '_get_backend', lambda self: backend)
    monkeypatch.setattr(BaseEstimator, '_to_array', lambda self, X, **kwargs: backend.asarray(X))
    if cuda_branch:
        # Select the Torch-specific batched branch only. Tensor.device and all
        # numerical operations remain genuinely CPU; this is not CUDA evidence.
        monkeypatch.setattr(runtime.module.Tensor, 'is_cuda', property(lambda self: True))
    row = b.run_case('kernel_ridge_cv', runtime, 24, 4, 42, 0, 1)
    assert row['status'] == 'pass', row.get('reason')
    controls = [event for trace in row['execution_trace'] for record in trace['stages']
                for event in record['operations'] if event.get('floating_dtype_controls')]
    assert len(controls) == 3
    assert all(event['inputs'][0]['dtype'] == 'bool' for event in controls)
    assert all(event['outputs'][0]['dtype'] == 'float32' for event in controls)
    assert all(event['floating_dtype_controls'][0]['values'] == [1.0, 0.0] for event in controls)
    assert all('_krr_cv.py' in event['floating_dtype_controls'][0]['source'] for event in controls)
    assert row['correctness']['status'] == 'pass' and not row['performance_eligible']


def test_branch_dtype_control_cannot_exempt_a_nested_foreign_callsite():
    runtime = torch_cpu()
    torch = runtime.module
    def nested(x):
        return torch.where(x > 0, -1., 1.)
    def outer(x):
        return nested(x)
    spec = stage(outer, 'outer', ('where',), expected_dtype='float64',
                 control_outputs={'where': {'values': (-1., 1.), 'dtype': 'float32'}})
    trace = NumericalTrace(runtime, [spec])
    with trace:
        outer(torch.ones(3, dtype=torch.float64))
    assert not trace.records[0]['operations'][0].get('floating_dtype_controls')
    with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
        trace.report()


@pytest.mark.parametrize('copy_constant', [False, True, 'inplace'])
def test_float32_branch_control_input_requires_exact_returned_object(copy_constant):
    runtime = torch_cpu()
    torch = runtime.module
    def branch(x):
        constants = torch.where(x > 0, 1., 0.)
        if copy_constant == 'inplace':
            constants += torch.tensor([.123, .456, .789], dtype=torch.float32)
        elif copy_constant:
            constants = constants.clone()
        return torch.where(x > 0, x, constants)
    spec = stage(branch, 'branch', ('where',), expected_dtype='float64',
                 control_outputs={'where': {'values': (1., 0.), 'dtype': 'float32',
                                            'allow_control_inputs': True}})
    trace = NumericalTrace(runtime, [spec])
    with trace:
        branch(torch.ones(3, dtype=torch.float64))
    assert not trace._constant_arrays
    if copy_constant:
        with pytest.raises(TraceIncomplete, match='Wrong floating dtype'):
            trace.report()
    else:
        report = trace.report()
        outer = report['stages'][0]['operations'][-1]
        assert outer['inputs'][-1]['dtype'] == 'float32'
        assert outer['floating_dtype_input_controls'][0]['input_index'] == 2
