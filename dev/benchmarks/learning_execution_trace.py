"""Scoped diagnostic observation of selected numerical operations, not a GPU profiler.

Only source-identified stage invocations and calls to non-allocation numerical
primitives count. Each invocation must meet its own requirement. Observations do
not prove that unobserved operators or the whole algorithm ran on the GPU.
"""
from contextlib import ExitStack
from functools import wraps
import sys
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


def stage(fn, label, operations=(), minimum_calls=1, placement='requested-backend'):
    return {'code': fn.__code__, 'label': label, 'operations': tuple(operations),
            'minimum_calls': minimum_calls, 'placement': placement}


class NumericalTrace:
    """Process-local observer; not thread-safe and rejects an existing profiler.

    Native operations are called unchanged, with original arguments. No arrays
    are transferred, values read, or operations synchronized by the observer.
    """
    OPS = ('exp', 'sum', 'mean', 'abs', 'sqrt', 'quantile', 'std', 'var',
           'dot', 'matmul', 'correlate', 'bincount', 'cumsum', 'searchsorted', 'argsort', 'rand')
    LINALG = ('inv', 'solve', 'eigh', 'eigvalsh', 'qr', 'svd', 'det')

    def __init__(self, runtime, stages):
        self.runtime = runtime
        self.stages = stages
        self.by_code = {s['code']: s for s in stages}
        self.active = {}
        self.records = []
        self._stack = ExitStack()

    def _profile(self, frame, event, arg):
        spec = self.by_code.get(frame.f_code)
        if spec is None:
            return
        if event == 'call':
            record = {'stage': spec['label'], 'source': frame.f_code.co_filename,
                      'line': frame.f_code.co_firstlineno, 'operations': [],
                      'completed': False, 'placement': spec['placement']}
            self.active[id(frame)] = record
            self.records.append(record)
        elif event == 'return':
            record = self.active.pop(id(frame), None)
            if record is not None:
                record['completed'] = arg is not None

    def _wrap(self, fn, backend, name):
        @wraps(fn)
        def observed(*args, **kwargs):
            records = tuple(self.active.values())
            if not records:
                return fn(*args, **kwargs)
            inputs = identities((args, kwargs))
            result = fn(*args, **kwargs)
            outputs = identities(result)
            # NumPy/CuPy out= ufuncs return an array; an in-place None result
            # is intentionally not accepted as numerical execution evidence.
            event = {'operation': name, 'backend': backend,
                     'inputs': inputs, 'outputs': outputs}
            for record in records:
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

    def report(self):
        required_backend = self.runtime.name
        required_device = self.runtime.device
        for spec in self.stages:
            records = [r for r in self.records if r['stage'] == spec['label']]
            if len(records) < spec['minimum_calls'] or any(not r['completed'] for r in records):
                raise TraceIncomplete('Missing or incomplete numerical stage: ' + spec['label'])
            for record in records:
                observed = set()
                for event in record['operations']:
                    if event['backend'] != required_backend:
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
                    if (array_inputs or event['operation'] in ('rand', 'RandomState.random_sample')) and event['outputs']:
                        observed.add(event['operation'])
                if not set(spec['operations']).issubset(observed):
                    raise TraceIncomplete('No matching native arithmetic for stage ' + spec['label'] + ': expected ' + ','.join(spec['operations']))
        return {'status': 'observed', 'backend': required_backend, 'device': required_device,
                'stages': self.records,
                'scope': 'selected native numerical primitive calls within source-identified stage invocations; not whole-algorithm GPU residency',
                'instrumentation': 'Python stage profiler and native function wrappers included in measured diagnostic call',
                'scalar_controls': 'NumPy scalar inputs retain host identities but do not establish array execution; outputs and all ndarrays remain placement-checked',
                'unobserved': 'array operator overloads, fused kernels, library-internal operations and unregistered functions are not exhaustively traced'}


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
