#!/usr/bin/env python3
"""Controlled-index KDE experiment, NOT timing of the public bootstrap API.

Example (requires a real CUDA device for a complete result)::

    python -m dev.benchmarks.benchmark_native_kde_bootstrap \
        --preset small --weights unequal --output results/native-kde-small.json

Host data and weighted resampling indices are generated once, outside every
measurement. Each variant includes the original estimate, all fixed-index
refits, linear pointwise quantiles, required input transfers and host results.
CPU-only execution is a partial harness check (exit 2), never GPU evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
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
from dev.benchmarks.benchmark_release_refresh import (
    BackendUnavailable, Runtime, array_identity, environment,
)
from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, stage

VARIANTS = ('numpy-controlled', 'cupy-current-controlled', 'cupy-native-controlled')
FIELDS = ('estimate', 'lower', 'upper', 'draws')
# Fixed safety ceilings, not knobs for silently increasing an experiment budget.
MAX_WORK = 1_000_000_000
MAX_TOTAL_WORK = 15_000_000_000
MAX_HOST_BYTES = 512 * 1024**2
MAX_WORKSPACE_BYTES = 512 * 1024**2
ATOL, RTOL = 2e-10, 2e-10


@dataclass(frozen=True)
class Problem:
    n: int = 256
    queries: int = 64
    resamples: int = 30
    seed: int = 20261021
    weights: str = 'unequal'
    bandwidth: float = .4
    confidence_level: float = .95


PRESETS = {'tiny': Problem(32, 7, 5), 'small': Problem(),
           'gpu': Problem(4096, 512, 100)}


class Blocked(RuntimeError):
    pass


def guard(problem, warmup, repeats, workspace_bytes, query_chunk=None):
    """Reject, never change, requests outside the fixed resource envelope."""
    for name in ('n', 'queries', 'resamples'):
        value = getattr(problem, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if problem.n < 2:
        raise ValueError('n must be at least two')
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup must be nonnegative and repeats positive')
    if not np.isfinite(problem.bandwidth) or problem.bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive')
    if not np.isfinite(problem.confidence_level) or not 0 < problem.confidence_level < 1:
        raise ValueError('confidence_level must be finite and in (0, 1)')
    if problem.weights not in ('none', 'unequal', 'sparse'):
        raise ValueError('unknown weights mode')
    if not 0 < workspace_bytes <= MAX_WORKSPACE_BYTES:
        raise Blocked('workspace budget exceeds fixed guard or is nonpositive; no downscaling')
    if query_chunk is not None and (not isinstance(query_chunk, int) or query_chunk < 1):
        raise ValueError('query_chunk must be a positive integer')
    work = problem.n * problem.queries * (problem.resamples + 1)
    total = work * (3 * (warmup + repeats + 1) + 4)
    # Include index list/stack/astype overlap, Gaussian oracle temporaries,
    # retained host outputs and vectors. This is a managed-buffer estimate,
    # not a hard process RSS cap (allocator/library/interpreter overhead excluded).
    host_bytes = 8 * (3 * problem.n * problem.resamples + 6 * problem.n * problem.queries
                      + 5 * problem.resamples * problem.queries + 10 * problem.n)
    if work > MAX_WORK or total > MAX_TOTAL_WORK or host_bytes > MAX_HOST_BYTES:
        raise Blocked('requested workload exceeds fixed resource guard; no downscaling')
    return {'kernel_evaluations_per_call': work, 'total_kernel_evaluations_upper_bound': total,
            'host_buffer_estimate_bytes': host_bytes,
            'host_buffer_estimate_scope': 'managed buffers including index-generation overlap and oracle temporaries; not a hard process RSS cap',
            'limits': {'per_call': MAX_WORK, 'total': MAX_TOTAL_WORK,
                       'host_bytes': MAX_HOST_BYTES, 'workspace_bytes': MAX_WORKSPACE_BYTES}}


def array_hash(array):
    value = np.ascontiguousarray(array)
    header = json.dumps({'dtype': value.dtype.str, 'shape': value.shape}).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def data_identity(data):
    return {key: None if value is None else array_hash(value) for key, value in data.items()}


def make_data(problem):
    rng = np.random.default_rng(problem.seed)
    x = rng.normal(size=problem.n).astype(np.float64)
    q = np.linspace(-2.5, 2.5, problem.queries, dtype=np.float64)
    weights = None
    if problem.weights != 'none':
        weights = np.linspace(.5, 1.5, problem.n, dtype=np.float64)
        if problem.weights == 'sparse':
            weights[::3] = 0.
    probabilities = np.full(problem.n, 1. / problem.n) if weights is None else weights / weights.sum()
    # Identical realized rows, not merely equal RNG seeds across backends.
    indices = np.stack([rng.choice(problem.n, problem.n, replace=True, p=probabilities)
                        for _ in range(problem.resamples)]).astype(np.int64)
    data = {'samples': x, 'points': q, 'weights': weights, 'indices': indices}
    for value in data.values():
        if value is not None:
            value.flags.writeable = False
    return data


def _linear_quantile(values, levels, axis=0):
    # NumPy 1.20 remains supported by the package.
    try:
        return np.quantile(values, levels, axis=axis, method='linear')
    except TypeError:
        return np.quantile(values, levels, axis=axis, interpolation='linear')


def analytic_reference(data, problem):
    """Independent 1D weighted covariance/jitter and Gaussian density oracle."""
    x, q, indices = data['samples'], data['points'], data['indices']
    w = data['weights']
    w = np.full(x.size, 1. / x.size) if w is None else w / np.sum(w)

    def density(values, weights):
        weights = weights / np.sum(weights)
        center = np.sum(weights * values)
        denominator = 1. - np.sum(weights * weights)
        if denominator <= 1e-15:
            raise ValueError('effective degrees of freedom is too small')
        variance = np.sum(weights * (values - center)**2) / denominator
        covariance = (variance + max(variance * 1e-12, 1e-12)) * problem.bandwidth**2
        return np.exp(-.5 * (q[:, None] - values[None, :])**2 / covariance) @ weights / np.sqrt(2. * np.pi * covariance)

    draws = np.stack([density(x[row], w[row]) for row in indices])
    alpha = (1. - problem.confidence_level) / 2.
    lower, upper = _linear_quantile(draws, [alpha, 1. - alpha])
    return dict(estimate=density(x, w), lower=lower, upper=upper, draws=draws)


def check(value, reference, problem):
    metrics = {}
    shapes = dict(estimate=(problem.queries,), lower=(problem.queries,),
                  upper=(problem.queries,), draws=(problem.resamples, problem.queries))
    if set(value) != set(FIELDS):
        raise AssertionError('missing or unexpected output fields')
    for key in FIELDS:
        actual = value[key]
        if not isinstance(actual, np.ndarray) or actual.dtype != np.float64:
            raise AssertionError(key + ' must be a host float64 array at the reporting boundary')
        if actual.shape != shapes[key] or not np.isfinite(actual).all():
            raise AssertionError('invalid shape or nonfinite ' + key)
        np.testing.assert_allclose(actual, reference[key], atol=ATOL, rtol=RTOL)
        metrics[key + '_max_abs_diff'] = float(np.max(np.abs(actual - reference[key])))
    if np.any(value['lower'] > value['upper']) or np.any(value['draws'] < 0):
        raise AssertionError('invalid density/interval ordering')
    return metrics


def _require_device(value, runtime):
    identity = array_identity(value)
    if identity['backend'] != runtime.name or identity['device'] != runtime.device:
        raise TraceIncomplete('wrong concrete operand/result device: ' + str(identity))
    return identity


def make_call(variant, runtime, data, problem, workspace_bytes, query_chunk):
    """Return the whole controlled workflow; no numerical setup is hidden."""
    from statgpu.nonparametric import fit_kde
    if variant not in VARIANTS:
        raise ValueError('unknown variant')
    expected_backend = 'numpy' if variant == VARIANTS[0] else 'cupy'
    if runtime.name != expected_backend:
        raise ValueError('variant cannot fall back to a different backend')

    def call():
        with runtime.context():
            x = runtime.array(data['samples'])
            q = runtime.array(data['points'])
            w = runtime.array(data['weights'])
            _require_device(x, runtime)
            _require_device(q, runtime)
            if variant == VARIANTS[2]:
                from dev.benchmarks.native_kde_bootstrap import native_kde_bootstrap
                indices = runtime.array(data['indices'])
                result = native_kde_bootstrap(x, q, indices, weights=w,
                    bandwidth=problem.bandwidth, confidence_level=problem.confidence_level,
                    workspace_bytes=workspace_bytes, query_chunk=query_chunk)
                for value in result.values():
                    _require_device(value, runtime)
                return {key: runtime.host(result[key]) for key in FIELDS}
            # Deliberately reproduce the current hybrid host gather/reweight and
            # per-replicate H2D/D2H boundaries, with pre-generated indices.
            model = fit_kde(x, weights=w, bandwidth=problem.bandwidth,
                            kernel='gaussian', backend=runtime.name)
            estimate_device = model.pdf(q, batch_size=query_chunk or 1024)
            _require_device(estimate_device, runtime)
            estimate = runtime.host(estimate_device)
            weights = data['weights']
            weights = np.full(problem.n, 1. / problem.n) if weights is None else weights / np.sum(weights)
            draws = np.empty((problem.resamples, problem.queries), dtype=np.float64)
            for i, row in enumerate(data['indices']):
                sampled_weights = weights[row]
                sampled_weights = sampled_weights / np.sum(sampled_weights)
                sample = runtime.array(data['samples'][row])
                sample_weights = runtime.array(sampled_weights)
                fitted = fit_kde(sample, weights=sample_weights, bandwidth=problem.bandwidth,
                                 kernel='gaussian', backend=runtime.name)
                prediction = fitted.pdf(q, batch_size=query_chunk or 1024)
                _require_device(prediction, runtime)
                draws[i] = runtime.host(prediction)
            alpha = (1. - problem.confidence_level) / 2.
            lower, upper = _linear_quantile(draws, [alpha, 1. - alpha])
            return dict(estimate=estimate, lower=lower, upper=upper, draws=draws)
    return call


class CandidateTrace(NumericalTrace):
    """Selected genuine device primitives plus bounded host-conversion guards.

    This is not a whole-program CUDA profiler. Scalar validation synchronization is
    permitted; device get/item/to-list conversions of numerical arrays are prohibited.
    The guards cover exposed conversion functions and observable C calls, not
    arbitrary extensions or library internals.
    """
    def __init__(self, runtime, stages):
        super().__init__(runtime, stages)
        self.host_conversion_attempts = []

    def _deny(self, name):
        self.host_conversion_attempts.append(name)
        raise TraceIncomplete('forbidden host numeric conversion during candidate: ' + name)

    def _profile(self, frame, event, arg):
        if event == 'c_call':
            owner = getattr(arg, '__self__', None)
            name = getattr(arg, '__name__', '')
            if isinstance(owner, self.runtime.module.ndarray) and name in ('get', 'item', 'tolist', '__array__', '__float__', '__int__'):
                if owner.ndim != 0:
                    self._deny('ndarray.' + name)
        super()._profile(frame, event, arg)

    def __enter__(self):
        super().__enter__()
        cp = self.runtime.module
        try:
            self._stack.enter_context(patch.object(cp, 'asnumpy', lambda *a, **kw: self._deny('cupy.asnumpy')))
            for name in ('asarray', 'array'):
                original = getattr(np, name)
                def guarded(value, *args, _name=name, _original=original, **kwargs):
                    def has_device(obj):
                        if isinstance(obj, cp.ndarray):
                            return True
                        if isinstance(obj, (list, tuple)):
                            return any(has_device(item) for item in obj)
                        return False
                    if has_device(value):
                        self._deny('numpy.' + _name)
                    return _original(value, *args, **kwargs)
                self._stack.enter_context(patch.object(np, name, guarded))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def report(self):
        report = super().report()
        report['host_conversion_guard'] = {
            'status': 'pass' if not self.host_conversion_attempts else 'failed',
            'attempts': self.host_conversion_attempts,
            'scope': 'cupy.asnumpy, NumPy array/asarray on device arrays, observable ndarray get/item/tolist array conversions; zero-dimensional validation scalars allowed',
            'limitation': 'not exhaustive instrumentation of extension-internal transfers; source review and device-contract tests remain required'}
        if self.host_conversion_attempts:
            raise TraceIncomplete('candidate attempted host numeric conversion')
        return report


def trace_candidate(runtime, data, problem, workspace_bytes, query_chunk):
    from dev.benchmarks import native_kde_bootstrap as candidate
    with runtime.context():
        arrays = {key: runtime.array(value) for key, value in data.items()}
        stages = [stage(candidate._density, 'weighted-covariance-and-density', ('sum', 'exp'),
                        minimum_calls=problem.resamples + 1),
                  stage(candidate._compute, 'device-linear-quantiles', ('quantile',))]
        with CandidateTrace(runtime, stages) as trace:
            result = candidate.native_kde_bootstrap(arrays['samples'], arrays['points'], arrays['indices'],
                weights=arrays['weights'], bandwidth=problem.bandwidth,
                confidence_level=problem.confidence_level, workspace_bytes=workspace_bytes,
                query_chunk=query_chunk)
            output_identity = {key: _require_device(value, runtime) for key, value in result.items()}
        runtime.sync(runtime.device)
        report = trace.report()
        report['outputs'] = output_identity
        return {key: runtime.host(result[key]) for key in FIELDS}, report


def trace_current(runtime, call, problem):
    from statgpu.nonparametric.kernel_smoothing import _kde
    stages = [stage(_kde.KernelDensityEstimator.fit, 'current-weighted-fit', ('sum',),
                    minimum_calls=problem.resamples + 1),
              stage(_kde.KernelDensityEstimator._evaluate_density, 'current-density', ('exp',),
                    minimum_calls=problem.resamples + 1)]
    with NumericalTrace(runtime, stages) as trace:
        value = call()
    runtime.sync(runtime.device)
    return value, trace.report()


def memory_diagnostic(runtime, data, problem, workspace_bytes, query_chunk):
    """Separate allocation-instrumented candidate run, never a timed sample."""
    from dev.benchmarks.native_kde_bootstrap import native_kde_bootstrap
    cp = runtime.module
    with runtime.context():
        pool = cp.cuda.MemoryPool()
        high = {'used': 0, 'reserved': 0}
        def malloc(size):
            allocation = pool.malloc(size)
            high['used'] = max(high['used'], pool.used_bytes())
            high['reserved'] = max(high['reserved'], pool.total_bytes())
            return allocation
        with cp.cuda.using_allocator(malloc):
            arrays = {key: runtime.array(value) for key, value in data.items()}
            baseline = pool.used_bytes()
            reserved_baseline = pool.total_bytes()
            high.update(used=baseline, reserved=reserved_baseline)
            runtime.sync(runtime.device)
            result = native_kde_bootstrap(arrays['samples'], arrays['points'], arrays['indices'],
                weights=arrays['weights'], bandwidth=problem.bandwidth,
                confidence_level=problem.confidence_level, workspace_bytes=workspace_bytes,
                query_chunk=query_chunk)
            runtime.sync(runtime.device)
            for value in result.values():
                _require_device(value, runtime)
            host = {key: runtime.host(result[key]) for key in FIELDS}
        report = {'status': 'measured', 'instrumented': True, 'device': runtime.device,
                  'resident_input_pool_bytes': baseline,
                  'peak_incremental_pool_used_bytes': high['used'] - baseline,
                  'peak_incremental_pool_reserved_bytes': high['reserved'] - reserved_baseline,
                  'scope': 'fresh CuPy allocator pool, sampled after each allocation; inputs excluded from incremental values',
                  'limitation': 'not total CUDA/process peak; excludes context, pinned buffers and allocations bypassing this allocator; workspace plan is not a hard allocator cap'}
        del result, arrays
        pool.free_all_blocks()
        return host, report


def measure(call, sync, validate, warmup, repeats):
    if sys.getprofile() is not None:
        raise TraceIncomplete('timing requires no active Python profiler')
    for _ in range(warmup):
        sync()
        value = call()
        sync()
        validate(value)
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        value = call()
        sync()
        samples.append(time.perf_counter() - start)
        validate(value)  # validation itself is outside the measured region
    return {'instrumented': False, 'samples_seconds': samples,
            'median_seconds': statistics.median(samples)}


def fingerprint():
    required = [
        'dev/benchmarks/benchmark_native_kde_bootstrap.py',
        'dev/benchmarks/native_kde_bootstrap.py',
        'dev/tests/test_benchmark_native_kde_bootstrap.py',
        'dev/tests/test_native_kde_bootstrap.py',
        'dev/benchmarks/learning_execution_trace.py',
        'dev/benchmarks/benchmark_release_refresh.py',
        'dev/plans/native_gpu_kde_bootstrap.md',
    ]
    paths = sorted((ROOT / 'statgpu').rglob('*.py')) + [ROOT / path for path in required]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.is_file()}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    try:
        commit, dirty = git('rev-parse', 'HEAD'), bool(git('status', '--porcelain'))
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', *hashes],
                                 cwd=ROOT, capture_output=True).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, tracked = None, True, False
    loaded = {'statgpu': str(Path(statgpu.__file__).resolve()),
              'runner': str(Path(__file__).resolve())}
    from dev.benchmarks import native_kde_bootstrap as candidate
    loaded['candidate'] = str(Path(candidate.__file__).resolve())
    expected = {'statgpu': str(ROOT / 'statgpu/__init__.py'),
                'runner': str(ROOT / required[0]), 'candidate': str(ROOT / required[1])}
    return {'git_commit': commit, 'working_tree': 'dirty' if dirty or not tracked else 'clean',
            'all_sources_tracked': tracked,
            'statgpu_version': statgpu.__version__, 'content_hashes': hashes,
            'worktree_fingerprint': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'missing_sources': missing, 'loaded_modules': loaded,
            'exact_import_paths': loaded == expected}


def _valid_timing(row, repeats):
    """Require self-consistent raw uninstrumented samples and concrete sync."""
    timing = row.get('timing', {})
    samples = timing.get('samples_seconds')
    positive_finite = lambda value: (isinstance(value, (int, float))
        and not isinstance(value, bool) and np.isfinite(value) and value > 0)
    return (timing.get('instrumented') is False
            and isinstance(samples, list) and len(samples) == repeats
            and bool(samples) and all(positive_finite(value) for value in samples)
            and positive_finite(timing.get('median_seconds'))
            and timing['median_seconds'] == statistics.median(samples)
            and bool(row.get('device'))
            and timing.get('synchronized_device') == row['device'])


def finalize(result):
    """Fail closed on missing, stale, unobserved or numerically invalid evidence."""
    before, after = result['source'], result['source_after']
    changed = (before.get('worktree_fingerprint') != after.get('worktree_fingerprint')
               or before.get('git_commit') != after.get('git_commit')
               or before.get('working_tree') != after.get('working_tree')
               or before.get('all_sources_tracked') != after.get('all_sources_tracked'))
    reasons = []
    if changed:
        reasons.append('source identity changed during execution')
    if any(source.get('missing_sources') or not source.get('exact_import_paths') or not source.get('git_commit')
           for source in (before, after)):
        reasons.append('source identity is incomplete or imported outside this checkout')
    if any(source.get('working_tree') != 'clean' or not source.get('all_sources_tracked')
           for source in (before, after)):
        reasons.append('canonical performance evidence requires a clean fully tracked source tree')
    if result['data_identity'] != result['data_identity_after']:
        reasons.append('input/index identity changed during execution')
    by_name = {row['variant']: row for row in result['rows']}
    if any(by_name.get(name, {}).get('status') != 'pass' for name in VARIANTS):
        reasons.append('all three controlled variants did not pass')
    for row in result['rows']:
        if row.get('status') != 'pass':
            continue
        if not _valid_timing(row, result['timing']['repeats']):
            reasons.append(row['variant'] + ': missing or inconsistent uninstrumented timing evidence')
        if row.get('data_identity') != result['data_identity'] or row.get('correctness', {}).get('status') != 'pass':
            reasons.append(row['variant'] + ': missing or inconsistent data/correctness evidence')
    concrete_device = result['environment'].get('gpu_device')
    if not (isinstance(concrete_device, str) and concrete_device.startswith('cuda:')
            and concrete_device[5:].isdigit()):
        reasons.append('concrete executed GPU device missing')
    if by_name.get(VARIANTS[0], {}).get('device') != 'cpu':
        reasons.append('NumPy baseline device is not CPU')
    for name in VARIANTS[1:]:
        row = by_name.get(name, {})
        evidence = row.get('operation_evidence', {})
        if (row.get('device') != concrete_device or evidence.get('backend') != 'cupy'
                or evidence.get('device') != concrete_device):
            reasons.append(name + ': GPU operation device/backend does not match environment')
    current_trace = by_name.get(VARIANTS[1], {}).get('operation_evidence', {})
    if current_trace.get('status') != 'observed':
        reasons.append('genuine current CuPy operation evidence missing')
    trace = by_name.get(VARIANTS[2], {}).get('operation_evidence', {})
    if trace.get('status') != 'observed' or trace.get('host_conversion_guard', {}).get('status') != 'pass':
        reasons.append('genuine candidate GPU operation/conversion evidence missing')
    memory = by_name.get(VARIANTS[2], {}).get('memory', {})
    if memory.get('status') != 'measured' or memory.get('device') != concrete_device:
        reasons.append('separate candidate pool memory diagnostic missing or on wrong device')
    if result['timing']['warmup'] < 2 or result['timing']['repeats'] < 7:
        reasons.append('performance protocol requires at least 2 warmups and 7 repetitions')
    eligible = not reasons
    result['source_changed'] = changed
    result['validation'] = {'status': 'complete' if eligible else 'partial',
                            'tier': 'physical-gpu' if eligible else 'diagnostic-only',
                            'physical_gpu_claim': eligible, 'uncovered_reasons': reasons}
    result['performance'] = {'eligible': eligible, 'paired_end_to_end': []}
    for row in result['rows']:
        row['performance_eligible'] = eligible
    if eligible:
        native = by_name[VARIANTS[2]]['timing']['median_seconds']
        for baseline in VARIANTS[:2]:
            result['performance']['paired_end_to_end'].append({
                'baseline': baseline, 'candidate': VARIANTS[2],
                'baseline_over_candidate': by_name[baseline]['timing']['median_seconds'] / native,
                'scope': result['timing']['scope'], 'environment_id': result['environment']['environment_id']})
    return result


def run_benchmark(problem, *, warmup=2, repeats=7, device=0,
                  workspace_bytes=64 * 1024**2, query_chunk=None, cpu_only=False):
    source = fingerprint()
    env = environment()
    result = {'schema_version': 'statgpu-benchmark-v2',
              'benchmark': {'name': 'native-kde-controlled-indices',
                            'script': 'dev/benchmarks/benchmark_native_kde_bootstrap.py',
                            'method': 'original estimate + fixed-index refits + linear pointwise percentile intervals',
                            'not_public_bootstrap_api_timing': True},
              'source': source, 'environment': env, 'problem': asdict(problem),
              'timing': {'scope': 'controlled-index-end-to-end-with-required-transfers-and-host-results',
                         'warmup': warmup, 'repeats': repeats, 'statistic': 'median',
                         'rng_policy': 'host data/weights/indices pre-generated identically; RNG excluded for all variants',
                         'transfer_policy': 'native transfers inputs and complete index matrix once; current CuPy transfers initial inputs and host-gathered rows/weights per replicate; both include final host outputs'},
              'correctness': {'reference': {'name': 'independent weighted Gaussian covariance/jitter formula plus NumPy fit_kde per replicate',
                                            'reference_type': 'analytic'},
                              'tolerance': {'atol': ATOL, 'rtol': RTOL, 'basis': 'float64 1D controlled well-conditioned Gaussian oracle'},
                              'coverage_limitation': 'conditional pointwise percentile algebra only; not interval coverage, tuning correction or RNG law certification'},
              'rows': []}
    try:
        result['resource_guard'] = guard(problem, warmup, repeats, workspace_bytes, query_chunk)
        from dev.benchmarks.native_kde_bootstrap import plan_workspace
        result['workspace'] = plan_workspace(problem.n, problem.queries, problem.resamples,
                                             workspace_bytes, query_chunk=query_chunk)
    except (ValueError, TypeError, MemoryError, Blocked) as exc:
        result.update(data_identity={}, data_identity_after={})
        result['rows'] = [{'variant': name, 'status': 'blocked', 'reason': str(exc)} for name in VARIANTS]
        result['source_after'] = fingerprint()
        env['environment_id'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        return finalize(result)
    try:
        data = make_data(problem)
        result['data_identity'] = data_identity(data)
        reference = analytic_reference(data, problem)
        check(reference, reference, problem)
    except Exception as exc:
        result.setdefault('data_identity', {})
        result['data_identity_after'] = result['data_identity']
        result['rows'] = [{'variant': name, 'status': 'failed',
                           'reason': 'workload/oracle: ' + type(exc).__name__ + ': ' + str(exc)}
                          for name in VARIANTS]
        result['source_after'] = fingerprint()
        env['environment_id'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        return finalize(result)
    result['problem']['data_id'] = hashlib.sha256(json.dumps(result['data_identity'], sort_keys=True).encode()).hexdigest()
    gpu = None
    unavailable = 'GPU variants explicitly disabled for CPU-only harness check'
    gpu_status = 'unavailable'
    if not cpu_only:
        try:
            gpu = Runtime('cupy', ordinal=device)
            env.update(gpu.hardware)
            env.update(gpu_device=gpu.device, cupy=gpu.module.__version__)
        except BackendUnavailable as exc:
            unavailable = str(exc)
        except Exception as exc:
            unavailable = type(exc).__name__ + ': ' + str(exc)
            gpu_status = 'failed'
    env['environment_id'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    calls = {}
    for variant in VARIANTS:
        row = {'variant': variant, 'status': 'unavailable', 'data_identity': result['data_identity']}
        result['rows'].append(row)
        rt = Runtime('numpy') if variant == VARIANTS[0] else gpu
        if rt is None:
            row['status'] = gpu_status
            row['reason'] = unavailable
            continue
        row['device'] = rt.device
        try:
            call = make_call(variant, rt, data, problem, workspace_bytes, query_chunk)
            value = call()
            row['correctness'] = {'status': 'pass', 'metrics': check(value, reference, problem)}
            if variant == VARIANTS[1]:
                observed, row['operation_evidence'] = trace_current(rt, call, problem)
                check(observed, reference, problem)
            if variant == VARIANTS[2]:
                observed, row['operation_evidence'] = trace_candidate(rt, data, problem, workspace_bytes, query_chunk)
                check(observed, reference, problem)
                try:
                    observed, row['memory'] = memory_diagnostic(rt, data, problem, workspace_bytes, query_chunk)
                except (AttributeError, NotImplementedError) as exc:
                    row['memory'] = {'status': 'unavailable', 'reason': str(exc)}
                else:
                    check(observed, reference, problem)
            calls[variant] = (call, rt)
            row['status'] = 'pass'
        except Exception as exc:
            row['status'] = 'failed'
            row['reason'] = type(exc).__name__ + ': ' + str(exc)
    if not any(row['status'] == 'failed' for row in result['rows']):
        for row in result['rows']:
            if row['status'] != 'pass':
                continue
            call, rt = calls[row['variant']]
            try:
                row['timing'] = measure(call, lambda: rt.sync(rt.device),
                    lambda value: check(value, reference, problem), warmup, repeats)
                row['timing']['synchronized_device'] = rt.device
            except Exception as exc:
                row['status'] = 'failed'
                row['reason'] = type(exc).__name__ + ': ' + str(exc)
                break
    result['data_identity_after'] = data_identity(data)
    result['source_after'] = fingerprint()
    return finalize(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='small')
    parser.add_argument('--n', type=int)
    parser.add_argument('--queries', type=int)
    parser.add_argument('--resamples', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--weights', choices=('none', 'unequal', 'sparse'))
    parser.add_argument('--bandwidth', type=float)
    parser.add_argument('--confidence-level', type=float)
    parser.add_argument('--workspace-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--query-chunk', type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('refusing to overwrite an existing evidence artifact')
    if args.device < 0:
        parser.error('device must be a nonnegative concrete CUDA ordinal')
    for name in ('bandwidth', 'confidence_level'):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            parser.error('--' + name.replace('_', '-') + ' must be finite')
    overrides = {name: getattr(args, name) for name in asdict(PRESETS[args.preset])
                 if getattr(args, name, None) is not None}
    problem = replace(PRESETS[args.preset], **overrides)
    result = run_benchmark(problem, warmup=args.warmup, repeats=args.repeats,
        device=args.device, workspace_bytes=args.workspace_bytes,
        query_chunk=args.query_chunk, cpu_only=args.cpu_only)
    result['benchmark']['command'] = shlex.join([sys.executable, '-m',
        'dev.benchmarks.benchmark_native_kde_bootstrap', *(sys.argv[1:] if argv is None else argv)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['validation']['status'], 'output': str(args.output),
                      'performance_eligible': result['performance']['eligible']}))
    return 0 if result['validation']['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
