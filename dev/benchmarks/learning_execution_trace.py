"""Scoped diagnostic observation of selected numerical operations, not a GPU profiler.

Only source-identified stage invocations and calls to non-allocation numerical
primitives count. Each invocation must meet its own requirement. Observations do
not prove that unobserved operators or the whole algorithm ran on the GPU.
"""
from contextlib import ExitStack
from functools import wraps
import sys
import inspect
import types
from pathlib import Path
from unittest.mock import patch

from dev.benchmarks.benchmark_release_refresh import array_identity


class TraceIncomplete(RuntimeError):
    pass


def identities(value):
    if isinstance(value, dict):
        return [item for v in value.values() for item in identities(v)]
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in identities(v)]
    import numpy as np
    if isinstance(value, np.generic):
        # A NumPy scalar can be a legitimate scalar argument to a native
        # primitive (e.g. CuPy slogdet's where fill values). Retain its host
        # identity, but never confuse it with a host ndarray or array evidence.
        return [{'backend': 'numpy', 'device': 'cpu', 'dtype': str(value.dtype),
                 'kind': 'numpy-scalar'}]
    try:
        return [dict(array_identity(value), kind='array', shape=list(value.shape))]
    except (RuntimeError, AttributeError, TypeError):
        return []


def stage(fn, label, operations=(), minimum_calls=1, placement='requested-backend',
          reject_host_numerics=False, host_controls=None, expected_dtype=None):
    # Estimator validation decorators share one wrapper code object. Resolve
    # their implementation, or unrelated decorated methods could satisfy a
    # method-specific stage. This never unwraps native primitive provenance.
    try:
        source = inspect.unwrap(fn)
        code = source.__code__
    except (ValueError, AttributeError, TypeError) as exc:
        raise TraceIncomplete('Cannot resolve source stage: ' + label) from exc
    return {'code': code, 'label': label, 'operations': tuple(operations),
            'minimum_calls': minimum_calls, 'placement': placement,
            'reject_host_numerics': reject_host_numerics,
            'host_controls': host_controls or {}, 'expected_dtype': expected_dtype}


class NumericalTrace:
    """Process-local observer; not thread-safe and rejects an existing profiler.

    Native operations are called unchanged, with original arguments. No arrays
    are transferred, values read, or operations synchronized by the observer.
    """
    OPS = ('exp', 'sum', 'mean', 'abs', 'sqrt', 'quantile', 'std', 'var',
           'dot', 'matmul', 'correlate', 'bincount', 'cumsum', 'searchsorted', 'argsort', 'rand',
           'where', 'min', 'max', 'log', 'argmin', 'diff')
    LINALG = ('inv', 'solve', 'eigh', 'eigvalsh', 'qr', 'svd', 'det',
              'solve_ex', 'cholesky', 'solve_triangular')

    def __init__(self, runtime, stages):
        self.runtime = runtime
        self.stages = stages
        self.by_code = {s['code']: s for s in stages}
        self.active = {}
        self.frames = {}
        self.records = []
        self._stack = ExitStack()

    def _profile(self, frame, event, arg):
        spec = self.by_code.get(frame.f_code)
        if spec is None:
            return
        if event == 'call':
            record = {'stage': spec['label'], 'source': frame.f_code.co_filename,
                      'line': frame.f_code.co_firstlineno, 'operations': [],
                      'completed': False, 'placement': spec['placement'],
                      'expected_floating_dtype': spec['expected_dtype'] or getattr(self.runtime, 'dtype', None)}
            self.active[id(frame)] = record
            self.frames[id(frame)] = frame
            self.records.append(record)
        elif event == 'return':
            record = self.active.pop(id(frame), None)
            self.frames.pop(id(frame), None)
            if record is not None:
                record['completed'] = arg is not None

    def _wrap(self, fn, backend, name):
        @wraps(fn)
        def observed(*args, **kwargs):
            records = tuple(self.active.items())
            if not records:
                return fn(*args, **kwargs)
            inputs = identities((args, kwargs))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                for _, record in records:
                    record['operations'].append({'operation': name, 'backend': backend,
                        'inputs': inputs, 'outputs': [], 'status': 'failed',
                        'exception': exception_chain(exc)})
                raise
            outputs = identities(result)
            # Resolve metadata allowances once against exact source-local
            # objects; propagate that narrow classification to ancestor stages.
            controls = []
            for frame_id, record in records:
                frame = self.frames[frame_id]
                spec = self.by_code[frame.f_code]
                for field in spec['host_controls'].get(name, ()):
                    if backend == 'numpy' and args and args[0] is frame.f_locals.get(field):
                        controls.append({'stage': spec['label'], 'field': field})
            # Allocation/copy and scalar-only calls never establish arithmetic.
            for _, record in records:
                event = {'operation': name, 'backend': backend,
                         'inputs': inputs, 'outputs': outputs, 'status': 'completed'}
                if controls:
                    event['host_control'] = controls[0]['field']
                    event['host_control_sources'] = controls
                record['operations'].append(event)
            return result
        return observed

    @staticmethod
    def _numpy_generated_wrapper(fn, module):
        # NumPy <=1.24 uses a generated Python __array_function__ wrapper.
        # Recognize the actual decorator protocol, not merely its filename or
        # functools __wrapped__ attribute, and verify closure implementations.
        code=getattr(fn, '__code__', None)
        if code is None or code.co_filename != '<__array_function__ internals>':
            return False
        from numpy.core import overrides
        if fn.__globals__ is not vars(overrides):
            return False
        def dispatcher(x):return (x,)
        def implementation(x):return x
        template=overrides.array_function_dispatch(dispatcher)(implementation)
        reference=getattr(template, '__code__', None)
        fields=('co_code','co_consts','co_names','co_freevars','co_argcount','co_kwonlyargcount','co_flags')
        if reference is None or any(getattr(code,k)!=getattr(reference,k) for k in fields):
            return False
        closure=dict(zip(code.co_freevars, [c.cell_contents for c in fn.__closure__ or ()]))
        if closure.get('public_api') is not fn or closure.get('implementation') is not getattr(fn,'_implementation',None):
            return False
        for name in ('implementation','dispatcher'):
            candidate=closure.get(name)
            if candidate is None:
                return False
            NumericalTrace._verify_native_callable(candidate,module,'numpy',name)
        return True

    @staticmethod
    def _verify_native_callable(fn, module, backend, name):
        # Do not label an arbitrary Python replacement as native merely because
        # it occupies torch.exp/cupy.sum. Python library implementations must
        # originate inside the installed library; C primitives identify their
        # library via callable/type module. No unwrap: wrappers are not native.
        if backend=='numpy' and NumericalTrace._numpy_generated_wrapper(fn,module):
            return
        code = getattr(fn, '__code__', None)
        if code is None:
            code = getattr(getattr(type(fn), '__call__', None), '__code__', None)
        if code is not None:
            root = Path(module.__file__).resolve().parent
            source = Path(code.co_filename).resolve()
            if root not in source.parents:
                raise TraceIncomplete('Untrusted replacement of native primitive: ' + backend + '.' + name)
        else:
            kind=type(fn)
            recognized=(kind is types.BuiltinFunctionType or
                        (backend=='numpy' and kind.__module__=='numpy' and kind.__name__ in ('ufunc','_ArrayFunctionDispatcher')) or
                        (backend=='cupy' and kind.__module__.startswith('cupy.') and kind.__name__=='ufunc'))
            if not recognized:
                raise TraceIncomplete('Unverifiable native callable type: ' + backend + '.' + name)
            origin = getattr(fn, '__module__', '') or type(fn).__module__
            if not (origin == backend or origin.startswith(backend + '.')):
                raise TraceIncomplete('Unverifiable native primitive: ' + backend + '.' + name)

    def __enter__(self):
        if sys.getprofile() is not None:
            raise TraceIncomplete('Existing profiler prevents isolated numerical trace')
        import numpy as np
        modules = [('numpy', np)]
        if self.runtime.name != 'numpy':
            modules.append((self.runtime.name, self.runtime.module))
        try:
            for backend, module in modules:
                for name in dict.fromkeys(self.OPS):
                    fn = getattr(module, name, None)
                    if callable(fn):
                        self._verify_native_callable(fn, module, backend, name)
                        self._stack.enter_context(patch.object(module, name, self._wrap(fn, backend, name)))
                for name in self.LINALG:
                    fn = getattr(module.linalg, name, None)
                    if callable(fn):
                        self._verify_native_callable(fn, module, backend, 'linalg.' + name)
                        self._stack.enter_context(patch.object(module.linalg, name, self._wrap(fn, backend, 'linalg.' + name)))
            if self.runtime.name == 'cupy' and any('RandomState.random_sample' in spec['operations'] for spec in self.stages):
                module = self.runtime.module
                fn = module.random.RandomState.random_sample
                self._verify_native_callable(fn, module, 'cupy', 'RandomState.random_sample')
                self._stack.enter_context(patch.object(module.random.RandomState, 'random_sample',
                    self._wrap(fn, 'cupy', 'RandomState.random_sample')))
            sys.setprofile(self._profile)
            return self
        except BaseException:
            sys.setprofile(None)
            self._stack.close()
            raise

    def __exit__(self, *exc):
        sys.setprofile(None)
        self._stack.close()
        return False

    def _validate(self):
        required_backend = self.runtime.name
        required_device = self.runtime.device
        for spec in self.stages:
            records = [r for r in self.records if r['stage'] == spec['label']]
            if len(records) < spec['minimum_calls'] or any(not r['completed'] for r in records):
                raise TraceIncomplete('Missing or incomplete numerical stage: ' + spec['label'])
            for record in records:
                observed = set()
                for event in record['operations']:
                    if event.get('status') == 'failed':
                        raise TraceIncomplete('Failed numerical primitive in stage ' + spec['label']
                                              + ': ' + event['operation'])
                    if event['backend'] != required_backend:
                        if (spec['reject_host_numerics'] and not event.get('host_control')
                                and any(a.get('kind') == 'array' for a in event['inputs'])):
                            raise TraceIncomplete('Host/wrong-backend array arithmetic in stage '
                                                  + spec['label'] + ': ' + event['operation'])
                        continue
                    matches = lambda item: (item['backend'] == required_backend and item['device'] == required_device
                        and (item.get('kind') == 'array'
                            or (required_backend == 'numpy' and item.get('kind') == 'numpy-scalar'
                                and 'shape' not in item)))
                    scalar_control = lambda item: (item.get('kind') == 'numpy-scalar'
                        and item['backend'] == 'numpy' and item['device'] == 'cpu'
                        and 'shape' not in item)
                    array_inputs = [item for item in event['inputs'] if not scalar_control(item)]
                    # Only INPUT scalar controls are exempt from placement.
                    # In particular, NumPy scalar outputs and 0D host ndarrays
                    # must still reject a purported GPU-native primitive.
                    if any(not matches(a) for a in array_inputs + event['outputs']):
                        raise TraceIncomplete('Native numerical primitive used wrong backend/device in stage '
                            + spec['label'] + ': operation=' + event['operation']
                            + '; expected=' + required_backend + '/' + required_device
                            + '; inputs=' + repr(event['inputs']) + '; outputs=' + repr(event['outputs']))
                    expected_dtype = spec.get('expected_dtype') or getattr(self.runtime, 'dtype', None)
                    if expected_dtype is not None:
                        import numpy as np
                        expected_dtype = str(expected_dtype).replace('torch.', '')
                        for item in array_inputs + event['outputs']:
                            dtype = str(item.get('dtype', '')).replace('torch.', '')
                            try:
                                control_dtype = np.dtype(dtype).kind in 'biu'
                            except TypeError:
                                control_dtype = False
                            if not control_dtype and dtype != expected_dtype:
                                raise TraceIncomplete('Wrong floating dtype in stage ' + spec['label']
                                    + ': operation=' + event['operation'] + '; expected=' + expected_dtype
                                    + '; observed=' + dtype)
                    if (array_inputs or event['operation'] in ('rand', 'RandomState.random_sample')) and event['outputs']:
                        observed.add(event['operation'])
                if not set(spec['operations']).issubset(observed):
                    raise TraceIncomplete('No matching native arithmetic for stage ' + spec['label'] + ': expected ' + ','.join(spec['operations']))

    def snapshot(self, status='incomplete', reason=None):
        report = {'status': status, 'backend': self.runtime.name, 'device': self.runtime.device,
                'stages': self.records,
                'scope': 'selected native numerical primitive calls within source-identified stage invocations; not whole-algorithm GPU residency',
                'instrumentation': 'Python stage profiler and native function wrappers; see runner for timed versus untimed scope',
                'scalar_controls': 'NumPy scalar inputs retain host identities but do not establish array execution; outputs and all ndarrays remain placement-checked',
                'unobserved': 'array operator overloads, fused kernels, library-internal operations and unregistered functions are not exhaustively traced'}
        if reason is not None:
            report['reason'] = str(reason)
        return report

    def report(self):
        try:
            self._validate()
        except TraceIncomplete as exc:
            exc.execution_trace = self.snapshot(reason=exc)
            raise
        return self.snapshot(status='observed')


def exception_chain(exc):
    """JSON-safe complete cause/context chain, without dropping partial evidence."""
    import traceback
    return {'type': type(exc).__name__, 'message': str(exc),
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}



def learning_stages(case, backend):
    from statgpu.nonparametric.kernel_smoothing import _kde as kde
    from statgpu.nonparametric.kernel_smoothing import _bandwidth_selection as bw
    from statgpu.nonparametric.kernel_smoothing import _kernel_common as common
    from statgpu.feature_selection import _knockoff_utils as ku
    specs = []
    if case.startswith('knockoff'):
        fn = ku._build_fixed_x_knockoffs if case == 'knockoff_fixed' else ku._build_model_x_knockoffs
        specs = [stage(fn, 'knockoff-construction', ('linalg.solve', 'linalg.eigh')),
                 stage(ku._corr_diff_statistics, 'knockoff-statistic', ('mean', 'abs')),
                 stage(ku._knockoff_threshold_and_path, 'host-threshold', placement='host-report-and-selection')]
    elif case == 'bandwidth_helpers':
        specs = [stage(bw.select_bandwidth, 'host-scalar-rule', minimum_calls=2, placement='host-scalar-no-GPU-work-required')]
    elif case.startswith('bandwidth_'):
        method = case[len('bandwidth_'):]
        if method == 'cv':
            specs = [stage(bw._kernel_regression_cv_score, 'cv-score', ('exp', 'sum')),
                     stage(bw._golden_section_minimize, 'host-search', placement='host-control-with-native-objective')]
        else:
            specs = [stage(kde.KernelDensityEstimator.fit, 'kde-fit', ('sum',)),
                     stage(bw.select_bandwidth, 'bandwidth-selection', placement='host-selector')]
            if backend != 'numpy':
                from statgpu.nonparametric.kernel_smoothing import _native_bandwidth as native_bw
                specs.append(stage(common._weighted_covariance, 'native-covariance', ('sum',)))
                specs[1] = stage(native_bw.select_bandwidth_native, 'native-bandwidth-selection', ('sum',))
                if method in ('nrd', 'nrd0'):
                    specs.append(stage(native_bw._nrd_factor, 'native-nrd', ('std', 'quantile')))
                elif method not in ('scott', 'silverman'):
                    specs.extend([stage(native_bw._r_factor, 'native-r-selector', ('std', 'quantile', 'bincount')),
                                  stage(native_bw._binned_functional, 'native-selector-objective', ('exp', 'sum')),
                                  stage(native_bw._weighted_quantile_resample_1d, 'native-weighted-selector-resampling', ('argsort', 'cumsum', 'searchsorted'), minimum_calls=0)])
                specs.append(stage(native_bw._project_to_principal_axis, 'native-selector-projection', ('linalg.eigh', 'sum'), minimum_calls=0))
            elif method in ('nrd', 'nrd0'):
                specs.append(stage(common._bandwidth_factor_1d_nrd, 'host-nrd', placement='host-numerical-selector'))
            elif method not in ('scott', 'silverman'):
                specs.append(stage(bw._bandwidth_factor_1d_r_selectors, 'host-r-selector', placement='host-numerical-selector'))
    elif case in ('kde_bootstrap', 'kde_normal_ci'):
        count = 21 if case == 'kde_bootstrap' and backend != 'numpy' else 1
        specs = [stage(kde.KernelDensityEstimator.fit, 'kde-fit', ('sum',), minimum_calls=count),
                 stage(kde.KernelDensityEstimator._evaluate_density, 'kde-density', ('exp',), minimum_calls=count),
                 stage(kde.kde_confidence_interval, 'host-interval-assembly', placement='mixed-density-and-host-report')]
        if case == 'kde_bootstrap' and backend != 'numpy':
            from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
            specs[-1] = stage(kde.kde_confidence_interval, 'public-interval-assembly', placement='host-api-report-and-scalar-control')
            specs.extend([
                stage(common._weighted_covariance, 'native-covariance', ('sum',), minimum_calls=count),
                stage(native._probability_cdf, 'native-probability-cdf', ('cumsum',)),
                stage(native._draw_indices, 'native-resampling', ('searchsorted', 'rand' if backend == 'torch' else 'RandomState.random_sample'), minimum_calls=20),
                stage(native._percentiles, 'native-interval-quantiles', ('quantile',)),
            ])
    else:
        raise ValueError('No stage contract for ' + case)
    return specs


def estimator_stages(case, backend, phase):
    """Algorithm-specific contracts for fitted estimators, never field ownership."""
    from statgpu.nonparametric.kernel_smoothing import _kde, _kernel_regression, _kernel_common
    from statgpu.nonparametric.kernel_methods import _kpca, _nystroem, _kernels
    from statgpu.nonparametric.splines import _transformer, _bspline_basis, _penalized
    from statgpu.semiparametric._gam import GAM
    from statgpu.unsupervised._agglomerative import AgglomerativeClustering
    from statgpu.unsupervised._utils import squared_euclidean_distances

    def native(fn, label, operations):
        return stage(fn, label, operations, reject_host_numerics=True, expected_dtype='float64')

    if case in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear'):
        cls = _kde.KernelDensityEstimator if case == 'kde' else _kernel_regression.KernelRegression
        if phase == 'fit':
            fit = cls._fit_samples if case == 'kde' else cls.fit
            return [native(fit, case + '-fit', ('sum',)),
                    native(_kernel_common._weighted_covariance, 'weighted-covariance', ('sum',))]
        fn = (cls._evaluate_density if case == 'kde' else
              cls._evaluate_nadaraya_watson if case.endswith('_nw') else cls._evaluate_local_linear)
        return [native(fn, case + '-evaluation', ('exp',) if case == 'kde' else ('exp', 'sum'))]
    if case == 'spline':
        return ([native(_transformer.SplineTransformer._build_knots, 'spline-knot-range', ('min', 'max'))]
                if phase == 'fit' else
                [native(_transformer.SplineTransformer._basis_degree, 'spline-de-boor-recurrence', ('where',))])
    if case == 'kernel_pca':
        return ([native(_kpca.KernelPCA.fit, 'kernel-pca-eigensystem', ('mean', 'linalg.eigh', 'sqrt')),
                 native(_kernels.rbf_kernel, 'kernel-pca-training-gram', ('sum',))]
                if phase == 'fit' else
                [native(_kpca.KernelPCA.transform, 'kernel-pca-query-centering', ('mean',)),
                 native(_kernels.rbf_kernel, 'kernel-pca-query-gram', ('sum',))])
    if case == 'nystroem':
        return ([native(_nystroem.Nystroem.fit, 'nystroem-landmark-svd', ('linalg.svd', 'sqrt')),
                 native(_kernels.rbf_kernel, 'nystroem-landmark-gram', ('sum',))]
                if phase == 'fit' else
                [native(_kernels.rbf_kernel, 'nystroem-query-gram', ('sum',))])
    if case == 'gam':
        basis = stage(_bspline_basis.bspline_basis, 'gam-de-boor-recurrence', ('where',),
                      placement='native-basis-with-host-knot-validation-and-scalar-loop-controls',
                      reject_host_numerics=True, expected_dtype='float64',
                      host_controls={'diff': ('knots_np',)})
        if phase == 'fit':
            return [native(GAM._create_knots, 'gam-knot-quantiles', ('quantile',)), basis,
                    native(_penalized.penalized_ls, 'gam-penalized-system', ('linalg.solve',))]
        return [basis]
    if case == 'agglomerative':
        if phase != 'fit':
            return []  # This operation accesses stored host labels, not numerics.
        if backend == 'numpy':
            return [stage(AgglomerativeClustering.fit, 'host-scipy-linkage',
                          placement='host-scipy-linkage-not-native-primitive-observed')]
        return [native(squared_euclidean_distances, 'agglomerative-distances', ('sum', 'matmul')),
                native(AgglomerativeClustering._fit_gpu, 'agglomerative-merge-selection', ('argmin',)),
                stage(AgglomerativeClustering._labels_from_children, 'agglomerative-host-label-assembly',
                      placement='host-index-bookkeeping-and-report')]
    raise ValueError('No estimator stage contract for ' + case)


def model_stages(case, backend):
    """Current Cox and panel numerical dispatch, excluding public report copies."""
    from statgpu.survival import _cox_counting
    from statgpu.losses._cox_ph import CoxPartialLikelihoodLoss
    from statgpu.panel import _linalg, _reductions
    if case in ('cox:breslow', 'cox:efron'):
        return [stage(CoxPartialLikelihoodLoss._full_objective_from_eta_backend,
                      'cox-risk-set-likelihood-score-information', ('exp', 'sum', 'cumsum'),
                      reject_host_numerics=True, expected_dtype='float64',
                      placement='native-risk-moments-with-host-event-index-bookkeeping',
                      host_controls={'cumsum': ('counts',), 'searchsorted': ('first_indices',)}),
                stage(_cox_counting._solve, 'cox-newton-system', ('linalg.solve',),
                      reject_host_numerics=True, expected_dtype='float64')]
    if case == 'panel:fama_macbeth':
        solve = 'linalg.solve_ex' if backend == 'torch' else 'linalg.solve'
        return [stage(_reductions._validate_group_reduction_inputs, 'panel-host-group-counts',
                      minimum_calls=0, placement='host-index-bookkeeping',
                      host_controls={'bincount': ('codes_np',)}),
                stage(_linalg.panel_lstsq_gram_certified_batched, 'panel-certified-period-systems',
                      ('matmul', 'linalg.eigvalsh', solve), reject_host_numerics=True, expected_dtype='float64'),
                stage(_linalg._panel_svd, 'panel-svd-fallback', ('linalg.svd',),
                      minimum_calls=0, reject_host_numerics=True, expected_dtype='float64')]
    if case.startswith('panel:'):
        return [stage(_linalg._panel_svd, 'panel-least-squares-svd', ('linalg.svd',),
                      reject_host_numerics=True, expected_dtype='float64')]
    return []
