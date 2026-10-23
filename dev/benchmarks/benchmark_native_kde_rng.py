#!/usr/bin/env python3
"""RNG-inclusive KDE experiment against the actual public bootstrap APIs.

Different RNG algorithms need not realize identical indices. Each seeded
workflow is validated against its own captured indices, outside timing. The
original controlled-index runner remains the paired-numerical comparison.
Public baselines are the current checkout, not a frozen pre-migration hybrid;
these ratios are not before/after migration speedups.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import statgpu
from dev.benchmarks import benchmark_native_kde_bootstrap as controlled
from dev.benchmarks.benchmark_release_refresh import BackendUnavailable, Runtime, environment
from dev.benchmarks.learning_execution_trace import NumericalTrace, TraceIncomplete, learning_stages, stage

Problem, PRESETS = controlled.Problem, controlled.PRESETS
FIELDS = controlled.FIELDS
VARIANTS = ('numpy-public-bootstrap', 'cupy-public-bootstrap', 'cupy-native-rng')
check, data_identity, array_hash = controlled.check, controlled.data_identity, controlled.array_hash


def guard(problem, warmup, repeats, workspace_bytes, query_chunk=None):
    # Extra diagnostic/replay calls beyond the controlled runner's protocol.
    result = controlled.guard(problem, warmup + 3, repeats, workspace_bytes, query_chunk)
    # Capture/replay diagnostics retain multiple realized index and draw arrays.
    host_bytes = 8 * (4 * problem.n * problem.resamples + 6 * problem.n * problem.queries
                      + 12 * problem.resamples * problem.queries + 20 * problem.n)
    if host_bytes > controlled.MAX_HOST_BYTES:
        raise controlled.Blocked('RNG diagnostic host buffers exceed fixed resource guard; no downscaling')
    result['host_buffer_estimate_bytes'] = host_bytes
    result['host_buffer_estimate_scope'] = 'managed capture/replay indices, retained outputs and analytic oracle buffers; not a hard process RSS cap'
    if isinstance(problem.seed, bool) or not isinstance(problem.seed, int) or not 0 <= problem.seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if (isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0
            or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1):
        raise ValueError('warmup/repeats must be nonnegative/positive integers')
    return result


def make_data(problem):
    # No bootstrap indices are pregenerated or hidden outside measured calls.
    rng = np.random.default_rng(problem.seed)
    data = {'samples': rng.normal(size=problem.n).astype(np.float64),
            'points': np.linspace(-2.5, 2.5, problem.queries, dtype=np.float64),
            'weights': None}
    if problem.weights != 'none':
        data['weights'] = np.linspace(.5, 1.5, problem.n, dtype=np.float64)
        if problem.weights == 'sparse':
            data['weights'][::3] = 0.
    for value in data.values():
        if value is not None:
            value.flags.writeable = False
    return data


def _native_call(runtime, data, problem, workspace_bytes, query_chunk, return_indices=False):
    from dev.benchmarks.native_kde_rng import native_kde_bootstrap_rng
    with runtime.context():
        x, q, w = (runtime.array(data[key]) for key in ('samples', 'points', 'weights'))
        result = native_kde_bootstrap_rng(x, q, weights=w, bandwidth=problem.bandwidth,
            confidence_level=problem.confidence_level, n_resamples=problem.resamples,
            seed=problem.seed, workspace_bytes=workspace_bytes, query_chunk=query_chunk,
            return_indices=return_indices)
        indices = None
        if return_indices:
            result, indices = result
            controlled._require_device(indices, runtime)
        for value in result.values():
            controlled._require_device(value, runtime)
        host = {key: runtime.host(result[key]) for key in FIELDS}
        return (host, runtime.host(indices)) if return_indices else host


def make_call(variant, runtime, data, problem, workspace_bytes, query_chunk):
    if variant not in VARIANTS:
        raise ValueError('unknown variant')
    expected = 'numpy' if variant == VARIANTS[0] else 'cupy'
    if runtime.name != expected:
        raise ValueError('explicit variant cannot fall back to a different backend')
    if variant == VARIANTS[2]:
        return lambda: _native_call(runtime, data, problem, workspace_bytes, query_chunk)
    from statgpu.nonparametric import kde_bootstrap_confidence_interval
    def call():
        with runtime.context():
            x, q, w = (runtime.array(data[key]) for key in ('samples', 'points', 'weights'))
            controlled._require_device(x, runtime)
            result = kde_bootstrap_confidence_interval(x, q, weights=w,
                bandwidth=problem.bandwidth, kernel='gaussian', backend=runtime.name,
                n_resamples=problem.resamples, confidence_level=problem.confidence_level,
                random_state=problem.seed, return_bootstrap_samples=True,
                batch_size=query_chunk or 1024)
            if result.backend != runtime.name:
                raise AssertionError('public API changed backend')
            return {'estimate': result.estimate, 'lower': result.lower,
                    'upper': result.upper, 'draws': result.bootstrap_samples}
    return call


def capture_public(call):
    """Capture each public backend's realized indices, only during diagnostics."""
    original = np.random.default_rng
    rows = []
    class Recorder:
        def __init__(self, rng):
            self.rng = rng
        def choice(self, *args, **kwargs):
            result = self.rng.choice(*args, **kwargs)
            rows.append(result.copy())
            return result
        def __getattr__(self, name):
            return getattr(self.rng, name)
    from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
    draw_indices = native._draw_indices
    def capture_native(*args, **kwargs):
        result = draw_indices(*args, **kwargs)
        # Explicit diagnostic-only host copy, never installed during timing.
        rows.append((result.detach().cpu().numpy() if type(result).__module__.split('.')[0] == 'torch' else result.get()).copy())
        return result
    with patch.object(np.random, 'default_rng', lambda *a, **kw: Recorder(original(*a, **kw))), \
            patch.object(native, '_draw_indices', capture_native):
        value = call()
    if not rows:
        raise AssertionError('public bootstrap RNG indices were not observed')
    return value, np.stack(rows).astype(np.int64, copy=False)


def trace_public_current(runtime, call, problem):
    """Observe the live public native baseline, not the old host-RNG path."""
    specs = learning_stages('kde_bootstrap', runtime.name)
    for spec in specs:
        if spec['label'] in ('kde-fit', 'kde-density', 'native-covariance'):
            spec['minimum_calls'] = problem.resamples + 1
        elif spec['label'] == 'native-resampling':
            spec['minimum_calls'] = problem.resamples
    with NumericalTrace(runtime, specs) as trace:
        value = call()
    runtime.sync(runtime.device)
    return value, trace.report()


def validate_indices(indices, data, problem):
    if (not isinstance(indices, np.ndarray) or indices.shape != (problem.resamples, problem.n)
            or indices.dtype.kind not in 'iu' or indices.min() < 0 or indices.max() >= problem.n):
        raise AssertionError('invalid realized resampling indices')
    weights = data['weights']
    probabilities = np.full(problem.n, 1. / problem.n) if weights is None else weights / weights.sum()
    if np.any(probabilities[indices] <= 0):
        raise AssertionError('sampled an excluded zero-probability observation')
    # A bounded, declared diagnostic, not a proof of all RNG properties.
    empirical = np.bincount(indices.ravel(), minlength=problem.n) / indices.size
    distance = float(np.max(np.abs(np.cumsum(empirical) - np.cumsum(probabilities))))
    epsilon = float(np.sqrt(np.log(2. / 1e-6) / (2. * indices.size)))
    if distance > epsilon:
        raise AssertionError('empirical sampling CDF fails the declared DKW diagnostic')
    return {'status': 'pass', 'indices_sha256': array_hash(indices),
            'index_shape': list(indices.shape), 'seed': problem.seed,
            'cdf_max_abs_error': distance, 'dkw_epsilon': epsilon, 'dkw_alpha': 1e-6,
            'scope': 'support plus one seeded empirical-CDF diagnostic; not a complete RNG certification'}


def diagnose(variant, runtime, data, problem, workspace_bytes, query_chunk):
    call = make_call(variant, runtime, data, problem, workspace_bytes, query_chunk)
    if variant == VARIANTS[2]:
        value, indices = _native_call(runtime, data, problem, workspace_bytes, query_chunk, True)
        repeated, repeated_indices = _native_call(runtime, data, problem, workspace_bytes, query_chunk, True)
    else:
        value, indices = capture_public(call)
        repeated, repeated_indices = capture_public(call)
    np.testing.assert_array_equal(indices, repeated_indices)
    rng_report = validate_indices(indices, data, problem)
    rng_report['same_seed_indices_reproduced'] = True
    reference = controlled.analytic_reference(dict(data, indices=indices), problem)
    metrics = check(value, reference, problem)
    check(repeated, reference, problem)
    # Uninstrumented public calls must agree with the captured diagnostic.
    check(call(), reference, problem)
    return reference, metrics, rng_report


class RNGTrace(controlled.CandidateTrace):
    OPS = controlled.CandidateTrace.OPS + ('cumsum', 'searchsorted')

    def __init__(self, runtime, stages, rng_shape):
        super().__init__(runtime, stages)
        self.rng_shape = tuple(rng_shape)

    def __enter__(self):
        self.rng_outputs = []
        super().__enter__()
        cp = self.runtime.module
        try:
            original = cp.random.RandomState.random_sample
            self._verify_native_callable(original, cp, 'cupy', 'RandomState.random_sample')
            def sampled(instance, *args, **kwargs):
                result = original(instance, *args, **kwargs)
                if not self.active:
                    raise TraceIncomplete('RNG executed outside registered candidate stage')
                identity = controlled._require_device(result, self.runtime)
                if result.dtype != cp.float64 or result.shape != self.rng_shape:
                    raise TraceIncomplete('native RNG returned wrong dtype/shape')
                self.rng_outputs.append(dict(identity, shape=list(result.shape)))
                return result
            self._stack.enter_context(patch.object(cp.random.RandomState, 'random_sample', sampled))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def report(self):
        result = super().report()
        if len(self.rng_outputs) != 1:
            raise TraceIncomplete('exactly one full-matrix native RNG generation must be observed')
        result['rng_generation'] = {'status': 'observed', 'operation': 'RandomState.random_sample',
            'outputs': self.rng_outputs, 'backend': 'cupy', 'device': self.runtime.device,
            'scope': 'genuine CuPy generator invocation and device output; not a CUDA kernel profiler'}
        return result


def trace_native(runtime, data, problem, workspace_bytes, query_chunk):
    from dev.benchmarks import native_kde_bootstrap as fixed
    from dev.benchmarks import native_kde_rng as candidate
    with runtime.context():
        arrays = {key: runtime.array(value) for key, value in data.items()}
        stages = [stage(candidate._probability_cdf, 'device-probability-cdf', ('cumsum',)),
                  stage(candidate._sample_indices, 'native-rng-and-weighted-sampling', ('searchsorted',)),
                  stage(fixed._density, 'weighted-covariance-and-density', ('sum', 'exp'), minimum_calls=problem.resamples + 1),
                  stage(fixed._compute, 'device-linear-quantiles', ('quantile',))]
        with RNGTrace(runtime, stages, (problem.resamples, problem.n)) as trace:
            result = candidate.native_kde_bootstrap_rng(arrays['samples'], arrays['points'],
                weights=arrays['weights'], bandwidth=problem.bandwidth,
                confidence_level=problem.confidence_level, n_resamples=problem.resamples,
                seed=problem.seed, workspace_bytes=workspace_bytes, query_chunk=query_chunk)
            outputs = {key: controlled._require_device(value, runtime) for key, value in result.items()}
        runtime.sync(runtime.device)
        report = trace.report()
        report['outputs'] = outputs
        return {key: runtime.host(result[key]) for key in FIELDS}, report


def memory_native(runtime, data, problem, workspace_bytes, query_chunk):
    from dev.benchmarks.native_kde_rng import native_kde_bootstrap_rng
    cp = runtime.module
    with runtime.context():
        pool = cp.cuda.MemoryPool()
        peak = {'used': 0, 'reserved': 0}
        def allocate(size):
            memory = pool.malloc(size)
            peak['used'] = max(peak['used'], pool.used_bytes())
            peak['reserved'] = max(peak['reserved'], pool.total_bytes())
            return memory
        with cp.cuda.using_allocator(allocate):
            arrays = {key: runtime.array(value) for key, value in data.items()}
            before, reserved = pool.used_bytes(), pool.total_bytes()
            peak.update(used=before, reserved=reserved)
            runtime.sync(runtime.device)
            result = native_kde_bootstrap_rng(arrays['samples'], arrays['points'], weights=arrays['weights'],
                bandwidth=problem.bandwidth, confidence_level=problem.confidence_level,
                n_resamples=problem.resamples, seed=problem.seed, workspace_bytes=workspace_bytes,
                query_chunk=query_chunk)
            runtime.sync(runtime.device)
            for value in result.values():
                controlled._require_device(value, runtime)
            host = {key: runtime.host(result[key]) for key in FIELDS}
        report = {'status': 'measured', 'instrumented': True, 'device': runtime.device,
                  'resident_input_pool_bytes': before,
                  'peak_incremental_pool_used_bytes': peak['used'] - before,
                  'peak_incremental_pool_reserved_bytes': peak['reserved'] - reserved,
                  'scope': 'fresh CuPy pool; inputs excluded, generated indices/RNG buffers included when using pool',
                  'limitation': 'not total CUDA/process peak; context, cuRAND state and non-pool allocations excluded; not a hard cap'}
        del result, arrays
        pool.free_all_blocks()
        return host, report


def fingerprint():
    result = controlled.fingerprint()
    extra = ['dev/benchmarks/benchmark_native_kde_rng.py', 'dev/benchmarks/native_kde_rng.py',
             'dev/tests/test_benchmark_native_kde_rng.py', 'dev/tests/test_native_kde_rng.py',
             'dev/plans/native_gpu_kde_rng.md', 'dev/benchmarks/NATIVE_KDE_RNG_ACCEPTANCE.md']
    hashes = result['content_hashes']
    for name in extra:
        path = ROOT / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            result['missing_sources'].append(name)
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', *extra], cwd=ROOT, capture_output=True).returncode == 0
    result['all_sources_tracked'] = result['all_sources_tracked'] and tracked
    if not tracked:
        result['working_tree'] = 'dirty'
    from dev.benchmarks import native_kde_rng as candidate
    result['loaded_modules'].update(rng_runner=str(Path(__file__).resolve()), rng_candidate=str(Path(candidate.__file__).resolve()))
    result['exact_import_paths'] = (result['exact_import_paths'] and
        result['loaded_modules']['rng_runner'] == str(ROOT / extra[0]) and
        result['loaded_modules']['rng_candidate'] == str(ROOT / extra[1]))
    result['worktree_fingerprint'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return result


def finalize(result):
    """Reuse the strict common gate on a copy, then enforce RNG-specific evidence."""
    import copy
    adapted = copy.deepcopy(result)
    for row in adapted['rows']:
        if row['variant'] in VARIANTS:
            row['variant'] = controlled.VARIANTS[VARIANTS.index(row['variant'])]
    controlled.finalize(adapted)
    reasons = [reason.replace('controlled variants', 'RNG-inclusive variants')
               for reason in adapted['validation']['uncovered_reasons']]
    for key in ('warmup', 'repeats'):
        if isinstance(result['timing'].get(key), bool) or not isinstance(result['timing'].get(key), int):
            reasons.append('timing protocol counts must be integers')
    names = [row.get('variant') for row in result['rows']]
    if len(names) != len(VARIANTS) or set(names) != set(VARIANTS):
        reasons.append('require exactly one row for each declared variant')
    for row in result['rows']:
        if row.get('status') == 'pass':
            rng = row.get('rng_validation', {})
            expected_epsilon = float(np.sqrt(np.log(2. / 1e-6) / (2. * result['problem']['resamples'] * result['problem']['n'])))
            error, epsilon = rng.get('cdf_max_abs_error'), rng.get('dkw_epsilon')
            finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
            law_ok = (finite(error) and finite(epsilon) and 0 <= error <= epsilon
                      and epsilon == expected_epsilon and rng.get('dkw_alpha') == 1e-6)
            digest = rng.get('indices_sha256', '')
            hash_ok = isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
            if (rng.get('status') != 'pass' or rng.get('same_seed_indices_reproduced') is not True
                    or not hash_ok or not law_ok or rng.get('seed') != result['problem']['seed']
                    or rng.get('index_shape') != [result['problem']['resamples'], result['problem']['n']]):
                reasons.append(row['variant'] + ': missing seeded realization correctness evidence')
    native = next((row for row in result['rows'] if row['variant'] == VARIANTS[2]), {})
    rng = native.get('operation_evidence', {}).get('rng_generation', {})
    if (rng.get('status') != 'observed' or rng.get('backend') != 'cupy'
            or rng.get('device') != result['environment'].get('gpu_device')
            or len(rng.get('outputs', [])) != 1
            or any(output.get('backend') != 'cupy' or output.get('device') != result['environment'].get('gpu_device')
                   or output.get('dtype') != 'float64'
                   or output.get('shape') != [result['problem']['resamples'], result['problem']['n']]
                   for output in rng.get('outputs', []))):
        reasons.append('genuine native RNG device-operation evidence missing')
    eligible = not reasons
    result['source_changed'] = adapted['source_changed']
    result['validation'] = dict(adapted['validation'], status='complete' if eligible else 'partial',
        tier='physical-gpu' if eligible else 'diagnostic-only', physical_gpu_claim=eligible,
        uncovered_reasons=reasons)
    result['performance'] = {'eligible': eligible, 'end_to_end_comparisons': [],
        'pairing': 'same input distribution/parameters; per-backend validated RNG realizations, NOT identical indices',
        'baseline_identity': 'public API from this checkout, not frozen pre-migration hybrid; historical ratios are not directly comparable'}
    for row in result['rows']:
        row['performance_eligible'] = eligible
    if eligible:
        rows = {row['variant']: row for row in result['rows']}
        for baseline in VARIANTS[:2]:
            result['performance']['end_to_end_comparisons'].append({'baseline': baseline,
                'candidate': VARIANTS[2], 'baseline_over_candidate': rows[baseline]['timing']['median_seconds'] / native['timing']['median_seconds'],
                'scope': result['timing']['scope'], 'environment_id': result['environment']['environment_id']})
    return result


def run_benchmark(problem, *, warmup=2, repeats=7, device=0, workspace_bytes=64*1024**2,
                  query_chunk=None, cpu_only=False):
    env = environment()
    result = {'schema_version': 'statgpu-benchmark-v2', 'source': fingerprint(), 'environment': env,
        'benchmark': {'name': 'native-kde-rng-end-to-end', 'script': 'dev/benchmarks/benchmark_native_kde_rng.py',
                      'method': 'RNG + original estimate + resampled refits + pointwise linear quantiles',
                      'public_baseline_api': 'kde_bootstrap_confidence_interval',
                      'public_baseline_identity': 'live current checkout; CuPy public baseline is native after public migration, not frozen pre-migration hybrid',
                      'comparison_limit': 'current public implementation versus historical experimental candidate; not before/after migration or cross-commit speedup evidence'},
        'problem': asdict(problem), 'rows': [],
        'timing': {'scope': 'rng-inclusive-end-to-end-with-input-transfers-and-host-results',
                   'warmup': warmup, 'repeats': repeats, 'statistic': 'median',
                   'rng_policy': 'generator creation and sampling included; same seed does NOT imply shared cross-backend indices',
                   'transfer_policy': 'host samples/points/weights at ingress, four host result arrays at egress; diagnostic index transfer excluded from all timed calls'},
        'correctness': {'reference': 'independent Gaussian formula using each variant own diagnostic realized indices',
                        'tolerance': {'atol': controlled.ATOL, 'rtol': controlled.RTOL},
                        'coverage_limitation': 'conditional percentile algebra plus seeded empirical-CDF diagnostic, not CI coverage or exhaustive RNG certification'}}
    data = None
    try:
        result['resource_guard'] = guard(problem, warmup, repeats, workspace_bytes, query_chunk)
        from dev.benchmarks.native_kde_rng import plan_rng_workspace
        result['workspace'] = plan_rng_workspace(problem.n, problem.queries, problem.resamples, workspace_bytes, query_chunk)
        data = make_data(problem)
        result['data_identity'] = data_identity(data)
        result['problem']['data_id'] = hashlib.sha256(json.dumps(result['data_identity'], sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        result.update(data_identity={}, data_identity_after={})
        result['rows'] = [{'variant': name, 'status': 'blocked', 'reason': type(exc).__name__ + ': ' + str(exc)} for name in VARIANTS]
        result['source_after'] = fingerprint()
        env['environment_id'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        return finalize(result)
    gpu, unavailable, unavailable_status = None, 'explicit CPU-only diagnostic', 'unavailable'
    if not cpu_only:
        try:
            gpu = Runtime('cupy', ordinal=device)
            env.update(gpu.hardware)
            env.update(gpu_device=gpu.device, cupy=gpu.module.__version__)
        except BackendUnavailable as exc:
            unavailable = str(exc)
        except Exception as exc:
            unavailable, unavailable_status = str(exc), 'failed'
    env['environment_id'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    calls = {}
    for variant in VARIANTS:
        row = {'variant': variant, 'status': 'unavailable', 'data_identity': result['data_identity']}
        result['rows'].append(row)
        runtime = Runtime('numpy') if variant == VARIANTS[0] else gpu
        if runtime is None:
            row.update(status=unavailable_status, reason=unavailable)
            continue
        row['device'] = runtime.device
        try:
            call = make_call(variant, runtime, data, problem, workspace_bytes, query_chunk)
            reference, metrics, row['rng_validation'] = diagnose(variant, runtime, data, problem, workspace_bytes, query_chunk)
            row['correctness'] = {'status': 'pass', 'metrics': metrics}
            if variant == VARIANTS[1]:
                value, row['operation_evidence'] = trace_public_current(runtime, call, problem)
                check(value, reference, problem)
            elif variant == VARIANTS[2]:
                value, row['operation_evidence'] = trace_native(runtime, data, problem, workspace_bytes, query_chunk)
                check(value, reference, problem)
                value, row['memory'] = memory_native(runtime, data, problem, workspace_bytes, query_chunk)
                check(value, reference, problem)
            calls[variant] = call, runtime, reference
            row['status'] = 'pass'
        except Exception as exc:
            row.update(status='failed', reason=type(exc).__name__ + ': ' + str(exc))
    if not any(row['status'] == 'failed' for row in result['rows']):
        for row in result['rows']:
            if row['status'] != 'pass':
                continue
            call, runtime, reference = calls[row['variant']]
            try:
                row['timing'] = controlled.measure(call, lambda: runtime.sync(runtime.device),
                    lambda value: check(value, reference, problem), warmup, repeats)
                row['timing']['synchronized_device'] = runtime.device
            except Exception as exc:
                row.update(status='failed', reason=type(exc).__name__ + ': ' + str(exc))
                break
    result['data_identity_after'] = data_identity(data)
    result['source_after'] = fingerprint()
    return finalize(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='small')
    for field in ('n', 'queries', 'resamples', 'seed'):
        parser.add_argument('--' + field, type=int)
    parser.add_argument('--weights', choices=('none', 'unequal', 'sparse'))
    parser.add_argument('--bandwidth', type=float)
    parser.add_argument('--confidence-level', type=float)
    parser.add_argument('--workspace-bytes', type=int, default=64*1024**2)
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
        parser.error('device must be nonnegative')
    for name in ('bandwidth', 'confidence_level'):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            parser.error('--' + name.replace('_', '-') + ' must be finite')
    overrides = {key: getattr(args, key) for key in asdict(PRESETS[args.preset]) if getattr(args, key, None) is not None}
    result = run_benchmark(replace(PRESETS[args.preset], **overrides), warmup=args.warmup, repeats=args.repeats,
        device=args.device, workspace_bytes=args.workspace_bytes, query_chunk=args.query_chunk, cpu_only=args.cpu_only)
    result['benchmark']['command'] = shlex.join([sys.executable, '-m', 'dev.benchmarks.benchmark_native_kde_rng', *(sys.argv[1:] if argv is None else argv)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['validation']['status'], 'output': str(args.output), 'performance_eligible': result['performance']['eligible']}))
    return 0 if result['validation']['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
