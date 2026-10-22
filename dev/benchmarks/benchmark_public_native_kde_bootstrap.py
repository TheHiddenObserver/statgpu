#!/usr/bin/env python3
"""Correctness-gated FULL public KDE bootstrap (NumPy, CuPy, Torch).

Includes native weighted RNG, original fit, every refit, quantiles, required
H2D transfers and public NumPy returns. Captured realized indices, guards and
operation traces run separately, never in measured calls. A CPU-only run is
an honest harness check, not physical-GPU evidence. See the acceptance document.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import statgpu
from dev.benchmarks.benchmark_release_refresh import BackendUnavailable, Runtime, array_identity, environment
from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, stage
from dev.benchmarks.benchmark_native_kde_bootstrap import array_hash, data_identity
from dev.benchmarks.benchmark_native_kde_rng import capture_public, validate_indices

BACKENDS = ('numpy', 'cupy', 'torch')
FIELDS = ('estimate', 'lower', 'upper', 'draws')
SELECTORS = ('scott', 'silverman', 'nrd', 'nrd0', 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi')
KERNELS = ('gaussian', 'rectangular', 'triangular', 'epanechnikov', 'biweight', 'triweight', 'cosine', 'optcosine')
ATOL, RTOL = 3e-8, 3e-5
MAX_WORK = 1_000_000_000
MAX_TOTAL_WORK = 20_000_000_000
MAX_HOST_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class Problem:
    n: int = 256
    queries: int = 64
    resamples: int = 30
    dimension: int = 1
    seed: int = 20261022
    weights: str = 'unequal'
    bandwidth: str | float = .4
    kernel: str = 'gaussian'
    confidence_level: float = .95


PRESETS = {'tiny': Problem(32, 7, 5), 'small': Problem(), 'gpu': Problem(4096, 512, 100)}


class PublicRuntime(Runtime):
    """Optional explicit Torch-CPU diagnostics; never implicit CUDA fallback."""
    def __init__(self, name, ordinal=0, torch_cpu=False):
        if name == 'torch' and torch_cpu:
            self.name, self.ordinal, self.device, self.hardware = name, None, 'cpu', {}
            try:
                self.module = importlib.import_module(name)
            except ImportError as exc:
                raise BackendUnavailable('Torch import unavailable: ' + str(exc)) from exc
        else:
            super().__init__(name, ordinal)

    def context(self):
        return nullcontext() if self.device == 'cpu' else super().context()

    def sync(self, actual_device):
        if actual_device != self.device:
            raise RuntimeError('executed device differs from synchronized device')
        if self.device != 'cpu':
            super().sync(actual_device)


def guard(problem, warmup, repeats, batch_size, workspace_bytes):
    for key in ('n', 'queries', 'resamples', 'dimension'):
        value = getattr(problem, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + ' must be a positive integer')
    if problem.n < 2:
        raise ValueError('n must be at least two')
    for key, value, minimum in (('warmup', warmup, 0), ('repeats', repeats, 1), ('batch_size', batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(key + ' has invalid integer count')
    if isinstance(problem.seed, bool) or not isinstance(problem.seed, int) or problem.seed < 0:
        raise ValueError('seed must be a nonnegative integer (not restricted to uint32)')
    if problem.weights not in ('none', 'unequal', 'sparse') or problem.kernel not in KERNELS:
        raise ValueError('unknown weights/kernel')
    if problem.kernel in ('cosine', 'optcosine') and problem.dimension != 1:
        raise ValueError('cosine/optcosine require dimension=1')
    if isinstance(problem.bandwidth, str):
        if problem.bandwidth not in SELECTORS:
            raise ValueError('unknown bandwidth selector')
    elif not np.isfinite(problem.bandwidth) or problem.bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive')
    if not np.isfinite(problem.confidence_level) or not 0 < problem.confidence_level < 1:
        raise ValueError('confidence_level must be in (0,1)')
    if workspace_bytes is not None and (isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int) or not 0 < workspace_bytes <= MAX_HOST_BYTES):
        raise ValueError('workspace_bytes must be positive and at most 512 MiB')
    # Every original/refit covariance is O(n*p**2), inversion/eigh O(p**3).
    # Guard these BEFORE allocating generated data, oracle arrays or a backend.
    kernel_work = problem.n * problem.queries * problem.dimension
    matrix_work = problem.n * problem.dimension**2 + problem.dimension**3
    work = (kernel_work + matrix_work) * (problem.resamples+1)
    total = work * 3 * (warmup + repeats + 5)
    selector_bytes = 8*(20*8192+20*2048) if isinstance(problem.bandwidth,str) and problem.bandwidth not in ('scott','silverman') else 0
    host_bytes = 8 * (4*problem.n*problem.resamples + 8*problem.queries*problem.n*problem.dimension
        + 12*problem.resamples*problem.queries + 20*problem.n*problem.dimension
        + 32*problem.dimension**2) + selector_bytes
    if work > MAX_WORK or total > MAX_TOTAL_WORK or host_bytes > MAX_HOST_BYTES:
        raise ValueError('fixed workload guard exceeded; no automatic downscaling')
    return {'kernel_work_per_fit': kernel_work, 'matrix_work_per_fit': matrix_work,
        'numerical_work_per_call': work, 'total_numerical_work_upper_bound': total,
        'managed_host_buffer_estimate_bytes': host_bytes,
        'scope': 'conservative harness buffers; not total process/GPU memory or selector allocation cap'}


def make_data(problem):
    rng = np.random.default_rng(problem.seed)
    x = rng.normal(size=(problem.n, problem.dimension))
    q = rng.normal(scale=1.2, size=(problem.queries, problem.dimension))
    if problem.dimension == 1:
        x, q = x[:, 0], q[:, 0]
    w = None if problem.weights == 'none' else np.linspace(.5, 1.5, problem.n)
    if problem.weights == 'sparse':
        w[::3] = 0.
    # Inputs stay writable to avoid Torch's warning for nonwritable NumPy views.
    # Identity hashes before and after execution detect mutation instead.
    return dict(samples=x, points=q, weights=w)


def gaussian_reference(samples, points, weights, indices, bandwidth, confidence_level):
    """Independent weighted covariance/jitter and Gaussian mixture algebra.

    Uses explicit centered distances, independent of production's quadratic
    expansion. Fixed numeric covariance factor only; selector oracle below uses
    trusted existing NumPy fits, with CPU shortcut semantics explicitly aligned.
    """
    x = np.asarray(samples).reshape(len(samples), -1)
    q = np.asarray(points).reshape(-1, x.shape[1])
    w = np.full(len(x), 1./len(x)) if weights is None else weights/np.sum(weights)
    def density(values, probabilities):
        probabilities = probabilities / probabilities.sum()
        centered = values - np.sum(values*probabilities[:, None], axis=0)
        denominator = 1. - np.sum(probabilities**2)
        if denominator <= 1e-15:
            raise ValueError('effective degrees of freedom too small')
        cov = (centered.T * probabilities) @ centered / denominator
        cov = .5*(cov+cov.T)
        cov += max(float(np.trace(cov))/x.shape[1]*1e-12, 1e-12)*np.eye(x.shape[1])
        cov *= float(bandwidth)**2
        delta = q[:, None, :] - values[None, :, :]
        quad = np.einsum('qni,ij,qnj->qn', delta, np.linalg.inv(cov), delta)
        return np.exp(-.5*quad) @ probabilities / np.sqrt((2*np.pi)**x.shape[1]*np.linalg.det(cov))
    draws = np.stack([density(x[row], w[row]) for row in indices])
    bounds = np.quantile(draws, [(1-confidence_level)/2, (1+confidence_level)/2], axis=0)
    return dict(estimate=density(x,w), lower=bounds[0], upper=bounds[1], draws=draws)


def capture_selector_models(call, runtime):
    """Capture only diagnostics from actual fitted selectors, outside timing."""
    from statgpu.nonparametric.kernel_smoothing import _kde as kde
    original = kde.select_bandwidth
    records = []
    def selected(*args, **kwargs):
        info = original(*args, **kwargs)
        details = info.details
        vector = details.get('projection_vector')
        records.append({'factor':float(info.factor),
            'projection_vector':None if vector is None else runtime.host(vector).copy()})
        return info
    with patch.object(kde, 'select_bandwidth', selected):
        value = call()
    return value, records


def native_selector_reference(samples, weights, bandwidth, record):
    """NumPy scalar rule with stable ties and the actually executed PCA axis.

    The captured axis is checked as a principal unit eigenvector, not accepted
    as a fitted factor. Independent centered projection/variance/interpolation
    feeds the existing NumPy scalar selector, preserving native tie/orientation
    policy without manufacturing legacy-CPU eigenvector sign equivalence.
    """
    from statgpu.nonparametric.kernel_smoothing import _bandwidth_selection as legacy
    from statgpu.nonparametric.kernel_smoothing._kernel_common import _bandwidth_factor_1d_nrd
    x = np.asarray(samples,dtype=np.float64).reshape(len(samples),-1)
    w = np.full(len(x),1./len(x)) if weights is None else np.asarray(weights,dtype=np.float64)/np.sum(weights)
    neff = 1./np.sum(w*w)
    dimension = x.shape[1]
    if bandwidth == 'scott':
        factor = neff**(-1./(dimension+4.))
    elif bandwidth == 'silverman':
        factor = (neff*(dimension+2.)/4.)**(-1./(dimension+4.))
    else:
        centered = x-np.sum(x*w[:,None],axis=0)
        covariance = (centered.T*w)@centered
        covariance = .5*(covariance+covariance.T)
        if dimension > 1:
            vector = record.get('projection_vector')
            if vector is None or vector.shape != (dimension,):
                raise AssertionError('actual native projection vector was not captured')
            np.testing.assert_allclose(vector@vector,1.,rtol=1e-10,atol=1e-12)
            largest = np.linalg.eigvalsh(covariance)[-1]
            if largest > 0:
                np.testing.assert_allclose(covariance@vector,largest*vector,rtol=3e-8,atol=2e-10)
            if vector[np.argmax(np.abs(vector))] < 0:
                raise AssertionError('native principal-axis sign policy violated')
            work = centered@vector
            mean = np.sum(w*work)
            variance = float(np.sum(w*(work-mean)**2)/(1.-np.sum(w*w)))
            if not np.isfinite(variance) or variance <= 0:
                variance = max(float(np.var(work,ddof=1)),np.finfo(float).tiny)
        else:
            work = x[:,0]
            variance = float(covariance[0,0]/(1.-np.sum(w*w)))
            variance += max(variance*1e-12,1e-12)
        if bandwidth in ('nrd','nrd0'):
            factor = _bandwidth_factor_1d_nrd(bandwidth,n_eff=neff,samples_2d=work[:,None],
                data_cov=np.asarray([[variance]]),xp=np)
        else:
            # Native interpolation intentionally stable-sorts duplicate values.
            if np.max(w)-np.min(w) > 1e-12:
                if dimension > 1:
                    w = w/w.sum()
                count = round(max(128.,min(8192.,8./np.sum(w*w))))
                order = np.argsort(work,kind='stable')
                cdf = np.cumsum(w[order]); cdf[-1] = 1.
                work = np.interp((np.arange(count)+.5)/count,cdf,work[order])
            factor = legacy._bandwidth_factor_1d_r_selectors(bandwidth,samples_2d=work[:,None],
                weights_1d=np.full(len(work),1./len(work)),data_cov=np.asarray([[variance]]))
    np.testing.assert_allclose(record['factor'],factor,rtol=RTOL,atol=ATOL)
    return float(factor)


def reference_for_indices(data, problem, indices, backend, selector_records=None):
    if not isinstance(problem.bandwidth, str) and problem.kernel == 'gaussian':
        return gaussian_reference(data['samples'], data['points'], data['weights'], indices,
            problem.bandwidth, problem.confidence_level), 'analytic'
    from statgpu.nonparametric import fit_kde
    x, q, w = (data[key] for key in ('samples', 'points', 'weights'))
    probabilities = np.full(problem.n, 1./problem.n) if w is None else w/w.sum()
    if backend != 'numpy' and isinstance(problem.bandwidth,str):
        if selector_records is None or len(selector_records) != problem.resamples+1:
            raise AssertionError('missing actual original/per-draw selector diagnostics')
        factor = native_selector_reference(x,w,problem.bandwidth,selector_records[0])
        model = fit_kde(x,weights=w,bandwidth=factor,kernel=problem.kernel,backend='numpy')
        draws = []
        for row, record in zip(indices,selector_records[1:]):
            factor = native_selector_reference(x[row],probabilities[row],problem.bandwidth,record)
            draws.append(fit_kde(x[row],weights=probabilities[row],bandwidth=factor,kernel=problem.kernel,backend='numpy').pdf(q))
        draws = np.stack(draws)
        bounds = np.quantile(draws,[(1-problem.confidence_level)/2,(1+problem.confidence_level)/2],axis=0)
        return dict(estimate=model.pdf(q),lower=bounds[0],upper=bounds[1],draws=draws),'trusted-statgpu-aligned-native-policy'
    model = fit_kde(x, weights=w, bandwidth=problem.bandwidth, kernel=problem.kernel, backend='numpy')
    bandwidth = (model.bandwidth_factor_ if backend == 'numpy' and problem.dimension == 1
        and problem.kernel == 'gaussian' else problem.bandwidth)
    draws = np.stack([fit_kde(x[row], weights=probabilities[row], bandwidth=bandwidth,
        kernel=problem.kernel, backend='numpy').pdf(q) for row in indices])
    bounds = np.quantile(draws, [(1-problem.confidence_level)/2, (1+problem.confidence_level)/2], axis=0)
    return dict(estimate=model.pdf(q), lower=bounds[0], upper=bounds[1], draws=draws), 'trusted-statgpu'


def check(value, reference, problem):
    if set(value) != set(FIELDS):
        raise AssertionError('unexpected public output fields')
    metrics = {}
    for key in FIELDS:
        actual = value[key]
        shape = (problem.resamples, problem.queries) if key == 'draws' else (problem.queries,)
        if not isinstance(actual, np.ndarray) or actual.dtype != np.float64 or actual.shape != shape:
            raise AssertionError('invalid public NumPy float64 output ' + key)
        if not np.isfinite(actual).all():
            raise AssertionError('nonfinite ' + key)
        np.testing.assert_allclose(actual, reference[key], rtol=RTOL, atol=ATOL)
        metrics[key + '_max_abs_diff'] = float(np.max(np.abs(actual-reference[key])))
    if np.any(value['draws'] < 0) or np.any(value['lower'] > value['upper']):
        raise AssertionError('negative density or reversed percentile interval')
    return metrics


def make_call(runtime, data, problem, batch_size, workspace_bytes):
    from statgpu.nonparametric import kde_bootstrap_confidence_interval
    def call():
        with runtime.context():
            x, q, w = (runtime.array(data[key]) for key in ('samples', 'points', 'weights'))
            for array in (x, q, w):
                if array is not None and array_identity(array)['device'] != runtime.device:
                    raise AssertionError('wrong concrete ingress device')
            result = kde_bootstrap_confidence_interval(x, q, weights=w, bandwidth=problem.bandwidth,
                kernel=problem.kernel, backend=runtime.name, n_resamples=problem.resamples,
                confidence_level=problem.confidence_level, random_state=problem.seed,
                return_bootstrap_samples=True, batch_size=batch_size, workspace_bytes=workspace_bytes)
            if result.backend != runtime.name:
                raise AssertionError('explicit public backend changed')
            return dict(estimate=result.estimate, lower=result.lower, upper=result.upper, draws=result.bootstrap_samples)
    return call


def capture_realized(call, runtime):
    if runtime.name == 'numpy':
        return capture_public(call)
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    original = native._draw_indices
    rows = []
    def capture(*args, **kwargs):
        row = original(*args, **kwargs)
        identity = array_identity(row)
        if identity['backend'] != runtime.name or identity['device'] != runtime.device:
            raise AssertionError('native sampled indices on wrong backend/device')
        rows.append(runtime.host(row).copy())
        return row
    with patch.object(native, '_draw_indices', capture):
        result = call()
    if not rows:
        raise AssertionError('public native RNG was not observed')
    return result, np.stack(rows)


def diagnose(runtime, data, problem, batch_size, workspace_bytes):
    call = make_call(runtime, data, problem, batch_size, workspace_bytes)
    (value, indices), selector_records = capture_selector_models(lambda: capture_realized(call,runtime),runtime)
    repeated, repeated_indices = capture_realized(call, runtime)
    np.testing.assert_array_equal(indices, repeated_indices)
    rng_report = validate_indices(indices, data, problem)
    rng_report['same_seed_indices_reproduced'] = True
    reference, reference_type = reference_for_indices(data, problem, indices, runtime.name, selector_records)
    metrics = check(value, reference, problem)
    check(repeated, reference, problem)
    check(call(), reference, problem)
    return reference, {'status': 'pass', 'reference': {'name': 'independent weighted Gaussian algebra' if reference_type == 'analytic'
        else 'NumPy scalar selector and refits on actual rows; native principal axis/stable ties aligned; historical CPU policy preserved',
        'reference_type': 'trusted-statgpu' if reference_type.startswith('trusted-statgpu') else reference_type, 'selector_alignment': reference_type, 'version': np.__version__ if reference_type == 'analytic' else statgpu.__version__},
        'metrics': metrics, 'thresholds': {'rtol': RTOL, 'atol': ATOL,
        'source': 'public bootstrap fixture: float64 arithmetic plus selector optimizer/PCA reduction allowance'}}, rng_report


def audit_native_export_sources():
    """Finite source audit, not a proof about arbitrary indirect Python code.

    Complements dynamic hooks for extension methods that cannot be replaced or
    observed on a particular CuPy build. A newly introduced direct full-array
    exporter in these numerical functions requires review rather than a pass.
    """
    import ast
    from contextvars import ContextVar
    import inspect
    import textwrap
    import types
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as bootstrap, _native_bandwidth as bandwidth
    from statgpu.nonparametric.kernel_smoothing import _kde as kde, _kernel_common as common
    functions = []
    for module in (bootstrap,bandwidth):
        declared = ast.parse(Path(module.__file__).read_text())
        functions.extend(getattr(module,node.name) for node in declared.body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)))
    functions.extend(getattr(kde.KernelDensityEstimator,name) for name in
        ('fit','_fit_samples','_prepare_samples','pdf','_evaluate_density'))
    functions.extend(getattr(common,name) for name in
        ('_as_samples_2d','_as_points_2d','_normalize_weights','_weighted_covariance','_stable_inv_and_det'))
    forbidden = {'get','cpu','numpy','tolist','__array__','asnumpy','_to_numpy'}
    def names(code):
        result = set(code.co_names)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):
                result.update(names(item))
        return result
    def local_bindings(code):
        result = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_freevars)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):
                result.update(local_bindings(item))
        return result
    checked = []
    for function in functions:
        # BaseEstimator's known validation decorator uses kwargs.get; audit the
        # wrapped KDE numerical function, not that shared routing wrapper.
        if Path(function.__code__.co_filename).resolve() == ROOT/'statgpu/_base.py' and hasattr(function,'__wrapped__'):
            function = function.__wrapped__
        exports = names(function.__code__) & forbidden
        if 'get' in exports:
            # The ambiguity-warning scope reads a ContextVar, not an array.
            # Permit only verified ContextVar receivers; arbitrary array.get
            # and dynamic/unknown receivers still fail this finite audit.
            try:
                syntax = ast.parse(textwrap.dedent(inspect.getsource(function)))
                gets = [node for node in ast.walk(syntax)
                    if isinstance(node,ast.Attribute) and node.attr == 'get']
                rebound_names = {node.id for node in ast.walk(syntax)
                    if isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del))}
                local_names = local_bindings(function.__code__) | rebound_names
                if gets and all(isinstance(node.value,ast.Name)
                    and node.value.id not in local_names
                    and isinstance(function.__globals__.get(node.value.id),ContextVar) for node in gets):
                    exports.remove('get')
            except (OSError,TypeError,SyntaxError):
                pass
        if exports:
            raise TraceIncomplete('array may be exported to host by native source '+function.__qualname__+': '+','.join(sorted(exports)))
        checked.append(function.__module__+'.'+function.__qualname__)
    return {'status':'pass','functions':checked,
        'scope':'direct full-array export names in selected production numerical functions; not exhaustive transitive/data-flow proof'}


class PublicTrace(NumericalTrace):
    OPS = tuple(dict.fromkeys(name for name in NumericalTrace.OPS +
        ('cumsum', 'searchsorted', 'cos', 'maximum', 'where') if name != 'rand'))

    def __init__(self, runtime, stages, problem):
        super().__init__(runtime, stages)
        self.problem = problem
        self.rng_outputs = []
        self.host_guards = []

    def _profile(self, frame, event, arg):
        if self.active and self.runtime.name == 'cupy':
            if event == 'c_call':
                owner = getattr(arg,'__self__',None)
                name = getattr(arg,'__name__','')
            elif event == 'call':
                owner = frame.f_locals.get('self')
                name = frame.f_code.co_name
            else:
                owner, name = None, ''
            if isinstance(owner,self.runtime.module.ndarray) and owner.ndim > 0 and name in ('get','tolist','__array__'):
                raise TraceIncomplete('CuPy array exported to host through ndarray.'+name)
        super()._profile(frame,event,arg)

    def _wrap(self, fn, backend, name):
        wrapped = super()._wrap(fn, backend, name)
        def guarded(*args, **kwargs):
            if self.active and backend == 'numpy' and self.runtime.name != 'numpy':
                from dev.benchmarks.learning_execution_trace import identities
                if any(identity['backend'] == 'numpy' for identity in identities((args, kwargs))):
                    raise TraceIncomplete('host array numerical operation inside native stages: ' + name)
            return wrapped(*args, **kwargs)
        return guarded

    def __enter__(self):
        xp = self.runtime.module
        owner, method = (xp, 'rand') if self.runtime.name == 'torch' else (xp.random.RandomState, 'random_sample')
        # Save and verify before the base observer installs its own RNG hook.
        original = getattr(owner, method)
        self._verify_native_callable(original, xp, self.runtime.name, method)
        super().__enter__()
        try:
            def rng(*args, **kwargs):
                value = original(*args, **kwargs)
                if not self.active:
                    raise TraceIncomplete('RNG executed outside source-identified numerical stage')
                identity = array_identity(value)
                if (identity['backend'] != self.runtime.name or identity['device'] != self.runtime.device
                        or identity['dtype'] != 'float64' or tuple(value.shape) != (self.problem.n,)):
                    raise TraceIncomplete('native RNG returned wrong backend/device/dtype/shape')
                self.rng_outputs.append(dict(identity, shape=list(value.shape)))
                return value
            self._stack.enter_context(patch.object(owner, method, rng))
            if self.runtime.name == 'cupy':
                original_host = xp.asnumpy
                def asnumpy(value, *a, **k):
                    if self.active and getattr(value, 'ndim', 0) > 0:
                        raise TraceIncomplete('device array exported to host inside native numerical stage')
                    return original_host(value, *a, **k)
                self._stack.enter_context(patch.object(xp, 'asnumpy', asnumpy))
                self.host_guards.append('cupy.asnumpy')
                # Extension types can be immutable. Check the direct-get profile
                # hook on this installed CuPy before claiming it protects a run.
                with self.runtime.context():
                    probe = xp.zeros(1,dtype=xp.float64)
                self.active[-1] = {'operations':[]}
                caught = False
                try:
                    probe.get()
                except TraceIncomplete:
                    caught = True
                finally:
                    self.active.pop(-1,None)
                self.direct_get_profile_observable = caught
                # A profiler exception disables sys.setprofile in CPython.
                sys.setprofile(self._profile)
                if caught:
                    self.host_guards.append('cupy.ndarray.get/tolist/__array__ verified profile hook')
            else:
                for name in ('cpu', 'numpy', 'tolist', '__array__'):
                    original_host = getattr(xp.Tensor, name)
                    def host(tensor, *a, _original=original_host, **k):
                        if self.active and tensor.ndim > 0:
                            raise TraceIncomplete('Torch array exported to host inside native numerical stage')
                        return _original(tensor, *a, **k)
                    self._stack.enter_context(patch.object(xp.Tensor, name, host))
                    self.host_guards.append('torch.Tensor.'+name)
                original_to = xp.Tensor.to
                def to(tensor,*args,**kwargs):
                    target = kwargs.get('device')
                    if args and isinstance(args[0],(str,xp.device)):
                        target = args[0]
                    elif args and isinstance(args[0],xp.Tensor):
                        target = args[0].device
                    if self.active and tensor.ndim > 0 and target is not None and xp.device(target).type == 'cpu':
                        raise TraceIncomplete('Torch array exported to host with Tensor.to')
                    return original_to(tensor,*args,**kwargs)
                self._stack.enter_context(patch.object(xp.Tensor,'to',to))
                self.host_guards.append('torch.Tensor.to(cpu)')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def report(self):
        result = super().report()
        if len(self.rng_outputs) != self.problem.resamples:
            raise TraceIncomplete('one actual native RNG call per replicate was not observed')
        result['rng_generation'] = {'status': 'observed', 'backend': self.runtime.name,
            'device': self.runtime.device, 'outputs': self.rng_outputs,
            'scope': 'genuine native RNG calls; not a CUDA kernel profiler'}
        result['host_conversion_guard'] = {'status': 'pass', 'hooks': self.host_guards,
            'source_audit': self.source_audit,
            'direct_get_profile_observable':getattr(self,'direct_get_profile_observable',None),
            'scope': 'finite listed export hooks in native stages plus direct-export source audit, not exhaustive transitive proof; scalar control and final public NumPy returns allowed'}
        return result


def trace_public(runtime, data, problem, batch_size, workspace_bytes):
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native, _kde as kde, _kernel_common as common
    specs = [stage(native.native_bootstrap, 'native-bootstrap'),
        stage(native._probability_cdf, 'probability-cdf', ('cumsum',)),
        stage(native._draw_indices, 'native-weighted-rng', ('searchsorted',), minimum_calls=problem.resamples),
        stage(native._percentiles, 'native-percentile', ('quantile',)),
        stage(kde.KernelDensityEstimator.fit, 'native-fit', ('sum', 'linalg.inv'), minimum_calls=problem.resamples+1),
        stage(common._weighted_covariance, 'weighted-covariance', ('sum',), minimum_calls=problem.resamples+1),
        stage(kde.KernelDensityEstimator._evaluate_density, 'native-density',
            ('exp',) if problem.kernel == 'gaussian' else (), minimum_calls=problem.resamples+1)]
    source_audit = audit_native_export_sources()
    with PublicTrace(runtime, specs, problem) as trace:
        trace.source_audit = source_audit
        result = make_call(runtime, data, problem, batch_size, workspace_bytes)()
    runtime.sync(runtime.device)
    report = trace.report()
    if problem.kernel != 'gaussian':
        report['density_trace_limitation'] = 'kernel operator overloads are not exhaustively observed; fit/covariance/RNG/quantile primitives are observed'
    return result, report


def fingerprint():
    names = [str(path.relative_to(ROOT)) for path in sorted((ROOT/'statgpu').rglob('*.py'))]
    names += ['dev/benchmarks/benchmark_public_native_kde_bootstrap.py',
        'dev/tests/test_public_native_kde_bootstrap.py',
        'dev/benchmarks/PUBLIC_NATIVE_KDE_BOOTSTRAP_ACCEPTANCE.md',
        'dev/benchmarks/benchmark_release_refresh.py', 'dev/benchmarks/learning_execution_trace.py',
        'dev/benchmarks/benchmark_native_kde_bootstrap.py', 'dev/benchmarks/benchmark_native_kde_rng.py']
    hashes, missing = {}, []
    for name in names:
        if (ROOT/name).is_file():
            hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        else:
            missing.append(name)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    tracked = subprocess.run(['git','ls-files','--error-unmatch',*names], cwd=ROOT, capture_output=True).returncode == 0
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    return {'git_commit': git('rev-parse','HEAD'), 'working_tree': 'dirty' if git('status','--porcelain') or not tracked else 'clean',
        'all_sources_tracked': tracked, 'statgpu_version': statgpu.__version__, 'content_hashes': hashes,
        'worktree_fingerprint': hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'missing_sources': missing, 'exact_import_paths': Path(statgpu.__file__).resolve() == ROOT/'statgpu/__init__.py'
        and Path(native.__file__).resolve() == ROOT/'statgpu/nonparametric/kernel_smoothing/_native_bootstrap.py'}


def _valid_rng_report(report, problem):
    finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
    digest = report.get('indices_sha256', '')
    error, epsilon = report.get('cdf_max_abs_error'), report.get('dkw_epsilon')
    expected = float(np.sqrt(np.log(2./1e-6)/(2.*problem['resamples']*problem['n'])))
    return (report.get('status') == 'pass' and report.get('same_seed_indices_reproduced') is True
        and report.get('seed') == problem['seed'] and report.get('index_shape') == [problem['resamples'],problem['n']]
        and isinstance(digest,str) and len(digest) == 64 and all(char in '0123456789abcdef' for char in digest)
        and finite(error) and finite(epsilon) and 0 <= error <= epsilon and epsilon == expected
        and report.get('dkw_alpha') == 1e-6)


def _valid_trace_evidence(trace, name, device, problem):
    rng = trace.get('rng_generation',{})
    outputs = rng.get('outputs',[])
    matching = lambda item: item.get('backend') == name and item.get('device') == device
    if (trace.get('status') != 'observed' or not matching(trace)
        or trace.get('host_conversion_guard',{}).get('status') != 'pass'
        or not trace.get('host_conversion_guard',{}).get('hooks')
        or trace.get('host_conversion_guard',{}).get('source_audit',{}).get('status') != 'pass'
        or rng.get('status') != 'observed' or not matching(rng)
        or len(outputs) != problem['resamples']
        or any(not matching(item) or item.get('dtype') != 'float64'
            or item.get('shape') != [problem['n']] for item in outputs)):
        return False
    stages = trace.get('stages',[])
    specs = {'native-bootstrap': (1,()), 'probability-cdf': (1,('cumsum',)),
        'native-weighted-rng': (problem['resamples'],('searchsorted',)),
        'native-percentile': (1,('quantile',)),
        'native-fit': (problem['resamples']+1,('sum','linalg.inv')),
        'weighted-covariance': (problem['resamples']+1,('sum',)),
        'native-density': (problem['resamples']+1,('exp',) if problem['kernel'] == 'gaussian' else ())}
    for label, (minimum, required) in specs.items():
        records = [record for record in stages if record.get('stage') == label]
        if len(records) < minimum or any(record.get('completed') is not True for record in records):
            return False
        for record in records:
            observed = set()
            for event in record.get('operations',[]):
                if event.get('backend') != name:
                    continue
                operands = event.get('inputs',[]) + event.get('outputs',[])
                if any(not matching(item) for item in operands):
                    return False
                if event.get('inputs') and event.get('outputs'):
                    observed.add(event.get('operation'))
            if not set(required).issubset(observed):
                return False
    return True


def finalize(result):
    """Fail closed: absent GPU/source/operation/correctness evidence gives no ratios."""
    reasons = []
    before, after = result['source'], result['source_after']
    if before != after:
        reasons.append('source identity changed during execution')
    for source in (before, after):
        if source.get('missing_sources') or not source.get('exact_import_paths') or not source.get('git_commit'):
            reasons.append('source provenance incomplete or imported outside checkout')
        if source.get('working_tree') != 'clean' or not source.get('all_sources_tracked'):
            reasons.append('canonical evidence requires clean fully tracked source; dirty fingerprint is diagnostic only')
    if result['data_identity'] != result['data_identity_after']:
        reasons.append('input data changed during execution')
    if result['timing']['warmup'] != 2 or result['timing']['repeats'] != 7:
        reasons.append('canonical timing requires two warmups and seven repeats')
    rows = result['rows']
    if len(rows) != 3 or {row['backend'] for row in rows} != set(BACKENDS):
        reasons.append('requires exactly one NumPy/CuPy/Torch row')
    for row in rows:
        name = row['backend']
        if row.get('status') != 'pass':
            reasons.append(name + ': ' + row.get('reason','did not pass'))
            continue
        if row.get('correctness',{}).get('status') != 'pass' or not _valid_rng_report(row.get('rng_validation',{}), result['problem']):
            reasons.append(name + ': missing actual-draw correctness or reproducibility evidence')
        timing = row.get('timing',{})
        samples = timing.get('samples_seconds',[])
        if (timing.get('instrumented') is not False or len(samples) != result['timing']['repeats']
            or not samples or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not np.isfinite(t) or t <= 0 for t in samples)
            or timing.get('median_seconds') != statistics.median(samples)
            or timing.get('synchronized_device') != row.get('device')):
            reasons.append(name + ': invalid synchronized timing')
        if name != 'numpy':
            trace = row.get('operation_evidence',{})
            rng = trace.get('rng_generation',{})
            if not row.get('device','').startswith('cuda:'):
                reasons.append(name + ': physical CUDA not executed')
            if not _valid_trace_evidence(trace, name, row.get('device'), result['problem']):
                reasons.append(name + ': missing native arithmetic/RNG/host-transfer guards')
    selector = result['problem']['bandwidth']
    comparable = not (isinstance(selector,str) and
        ((result['problem']['dimension'] == 1 and result['problem']['kernel'] == 'gaussian')
         or selector in ('ucv','bcv','sj','sj-ste','sj-dpi')))
    eligible = not reasons
    result['validation'] = {'tier': 'remote-full' if eligible else 'local-minimal',
        'status': 'complete' if eligible else 'partial', 'physical_gpu_claim': eligible,
        'uncovered_reasons': sorted(set(reasons))}
    result['performance'] = {'eligible': eligible and comparable, 'comparisons': [],
        'pairing': 'same problem, independently validated backend RNG realizations, NOT identical draws',
        'selector_policy_comparable': comparable,
        'selector_caveat': 'NumPy 1D Gaussian fixes the original factor; native refits selectors. Native R-style rules use stable ties/canonical PCA sign; legacy CPU rules are preserved, not assumed equivalent'}
    if eligible and comparable:
        cpu = next(row for row in rows if row['backend'] == 'numpy')
        for row in rows:
            if row['backend'] != 'numpy':
                result['performance']['comparisons'].append({'baseline':'numpy', 'candidate':row['backend'],
                    'baseline_over_candidate':cpu['timing']['median_seconds']/row['timing']['median_seconds'],
                    'scope':result['timing']['scope'], 'environment_id':result['environment']['environment_id'],
                    'synchronized_device':row['device']})
    return result


def run_benchmark(problem, *, warmup=2, repeats=7, device=0, batch_size=128,
                  workspace_bytes=None, cpu_only=False, torch_cpu=False):
    guard_report = guard(problem, warmup, repeats, batch_size, workspace_bytes)
    data = make_data(problem)
    result = {'schema_version':'statgpu-benchmark-v2', 'benchmark':{
        'name':'public-native-kde-bootstrap', 'method':'pointwise weighted percentile bootstrap',
        'script':'dev/benchmarks/benchmark_public_native_kde_bootstrap.py',
        'public_api':'kde_bootstrap_confidence_interval'}, 'source':fingerprint(),
        'environment':environment(), 'problem':asdict(problem), 'guard':guard_report,
        'data_identity':data_identity(data), 'timing':{'scope':'end-to-end public RNG-inclusive bootstrap with input transfer and NumPy returns',
        'warmup':warmup,'repeats':repeats,'statistic':'median','transfer_policy':'included',
        'batch_size':batch_size,'workspace_bytes':workspace_bytes}, 'rows':[]}
    for name in BACKENDS:
        row = {'backend':name, 'status':'failed'}
        result['rows'].append(row)
        if cpu_only and name != 'numpy' and not (torch_cpu and name == 'torch'):
            row.update(status='unavailable',reason='CPU-only requested; no physical GPU run')
            continue
        try:
            runtime = PublicRuntime(name,device,torch_cpu=torch_cpu)
            row.update(device=runtime.device,hardware=runtime.hardware)
            reference, correctness, rng_report = diagnose(runtime,data,problem,batch_size,workspace_bytes)
            row.update(correctness=correctness,rng_validation=rng_report,
                selector_policy={'refit':'original factor fixed for NumPy 1D Gaussian' if name == 'numpy' and problem.dimension == 1 and problem.kernel == 'gaussian' else 'per-replicate string-selector refit',
                    'weighted_ties':'legacy NumPy ordering' if name == 'numpy' else 'stable ordering',
                    'projection_sign':'legacy NumPy eigensolver orientation' if name == 'numpy' else 'largest-magnitude component nonnegative'})
            if name != 'numpy':
                traced, evidence = trace_public(runtime,data,problem,batch_size,workspace_bytes)
                check(traced,reference,problem)
                row['operation_evidence'] = evidence
            call = make_call(runtime,data,problem,batch_size,workspace_bytes)
            for _ in range(warmup):
                check(call(),reference,problem)
                runtime.sync(runtime.device)
            times = []
            for _ in range(repeats):
                runtime.sync(runtime.device)
                start = time.perf_counter()
                value = call()
                runtime.sync(runtime.device)
                times.append(time.perf_counter()-start)
                check(value,reference,problem)
            row.update(status='pass',timing={'samples_seconds':times,'median_seconds':statistics.median(times),
                'synchronized_device':runtime.device,'instrumented':False})
        except BackendUnavailable as exc:
            row.update(status='unavailable',reason=str(exc))
        except Exception as exc:
            row.update(status='failed',reason=type(exc).__name__+': '+str(exc))
    # Hardware/software/device participates in the comparison group identity.
    result['environment']['executed_devices'] = {row['backend']:dict(device=row.get('device'),hardware=row.get('hardware',{})) for row in result['rows']}
    result['environment']['environment_id'] = hashlib.sha256(json.dumps(result['environment'],sort_keys=True).encode()).hexdigest()[:20]
    result.update(source_after=fingerprint(),data_identity_after=data_identity(data))
    return finalize(result)


def parse_bandwidth(value):
    return value.lower() if value.lower() in SELECTORS else float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=PRESETS,default='small')
    for name in ('n','queries','resamples','dimension','seed'):
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--weights',choices=('none','unequal','sparse'))
    parser.add_argument('--kernel',choices=KERNELS)
    parser.add_argument('--bandwidth',type=parse_bandwidth)
    parser.add_argument('--confidence-level',type=float)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--workspace-bytes',type=int)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--torch-cpu',action='store_true',help='explicit diagnostic Torch CPU tier; never physical evidence')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    if args.device < 0:
        parser.error('device must be nonnegative')
    overrides = {key:getattr(args,key) for key in asdict(PRESETS[args.preset]) if getattr(args,key,None) is not None}
    result = run_benchmark(replace(PRESETS[args.preset],**overrides),warmup=args.warmup,repeats=args.repeats,
        device=args.device,batch_size=args.batch_size,workspace_bytes=args.workspace_bytes,
        cpu_only=args.cpu_only,torch_cpu=args.torch_cpu)
    result['benchmark']['command'] = shlex.join([sys.executable,'-m','dev.benchmarks.benchmark_public_native_kde_bootstrap',*(sys.argv[1:] if argv is None else argv)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status':result['validation']['status'],'output':str(args.output),
        'performance_eligible':result['performance']['eligible']}))
    return 0 if result['validation']['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
