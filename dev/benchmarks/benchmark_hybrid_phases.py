#!/usr/bin/env python3
"""CPU/current-hybrid optimization-decision benchmark, not release acceptance.

No native-GPU prototype exists here. Use --inventory for the matrix. All measured
public calls include input conversion, computation and host result extraction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import statgpu
from dev.benchmarks.benchmark_release_refresh import Runtime, BackendUnavailable, environment, array_identity
from dev.benchmarks.learning_execution_trace import NumericalTrace, learning_stages
from dev.benchmarks.hybrid_phase_observer import PhaseObserver

CASES = ('kde_bootstrap', 'bandwidth_nrd', 'bandwidth_nrd0', 'bandwidth_ucv',
         'bandwidth_bcv', 'bandwidth_sj', 'bandwidth_sj-ste', 'bandwidth_sj-dpi',
         'bandwidth_cv', 'knockoff_fixed', 'knockoff_model')

@dataclass(frozen=True)
class Problem:
    n: int = 128
    p: int = 4
    queries: int = 32
    resamples: int = 30
    seed: int = 20261020

PRESETS = {'tiny': Problem(32, 2, 7, 5), 'small': Problem(),
           'medium': Problem(1000, 12, 128, 100),
           'large': Problem(5000, 32, 512, 200)}

class Blocked(RuntimeError):
    pass

class Unsupported(Blocked):
    pass


def host(x):
    name = type(x).__module__.split('.')[0]
    if name == 'cupy':
        import cupy
        return cupy.asnumpy(x)
    if name == 'torch':
        return x.detach().cpu().numpy()
    return np.asarray(x)


def fingerprint():
    paths = sorted((ROOT / 'statgpu').rglob('*.py'))
    paths += [ROOT / 'dev/benchmarks' / name for name in
              ('benchmark_hybrid_phases.py', 'hybrid_phase_observer.py',
               'benchmark_release_refresh.py', 'learning_execution_trace.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    try:
        commit = git('rev-parse', 'HEAD')
        dirty = bool(git('status', '--porcelain'))
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', *hashes], cwd=ROOT, capture_output=True).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, tracked = None, True, False
    return {'git_commit': commit, 'working_tree': 'dirty' if dirty or not tracked else 'clean',
            'statgpu_version': statgpu.__version__, 'content_hashes': hashes,
            'worktree_fingerprint': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'import_path': str(Path(statgpu.__file__).resolve())}


def guard(case, problem, max_work):
    if min(problem.n, problem.p, problem.queries, problem.resamples) < 1:
        raise ValueError('all workload dimensions must be positive')
    actual_p = 1 if case == 'kde_bootstrap' else problem.p
    if problem.n < max(8, 2 * actual_p + 2):
        raise ValueError('require n >= max(8, 2*p+2) for stable full-rank workloads')
    if case == 'kde_bootstrap':
        work = problem.n * problem.queries * problem.resamples
    elif case == 'bandwidth_cv':
        work = problem.n ** 2 * problem.p * 100
    elif case.startswith('knockoff'):
        work = problem.n * problem.p ** 2
    else:
        work = problem.n ** 2 * problem.p
    if work > max_work:
        raise Blocked(f'requested work estimate {work} exceeds --max-work {max_work}; no downscaling')
    return work


def make_call(case, rt, problem):
    from statgpu import nonparametric as N
    from statgpu.feature_selection import fixed_x_knockoff_filter, model_x_knockoff_filter
    rng = np.random.default_rng(problem.seed)
    # KDE is deliberately univariate; bandwidth and knockoff use configured p.
    p = 1 if case == 'kde_bootstrap' else problem.p
    X = rng.normal(size=(problem.n, p))
    Q = rng.normal(size=(problem.queries, p))
    y = np.sin(X[:, 0]) + X @ np.linspace(.1, .4, p)
    def call():
        dx = rt.array(X)
        dq = rt.array(Q) if not case.startswith('knockoff') else None
        dy = rt.array(y) if case.startswith('knockoff') or case == 'bandwidth_cv' else None
        if case == 'kde_bootstrap':
            if rt.name == 'torch':
                raise Unsupported('Torch weighted KDE bootstrap is unsupported; no CPU fallback')
            result = N.kde_bootstrap_confidence_interval(dx, dq, bandwidth=.4,
                n_resamples=problem.resamples, random_state=problem.seed,
                return_bootstrap_samples=True, backend=rt.name)
            return {'estimate': host(result.estimate), 'lower': host(result.lower),
                    'upper': host(result.upper), 'draws': host(result.bootstrap_samples)}
        if case.startswith('bandwidth_'):
            method = case[10:]
            device = {'numpy': 'cpu', 'cupy': 'cuda', 'torch': 'torch'}[rt.name]
            if method == 'cv':
                model = N.KernelRegression(bandwidth='cv', device=device).fit(dx, dy)
            else:
                model = N.KernelDensityEstimator(bandwidth=method, device=device).fit(dx)
            return {'factor': np.asarray(model.bandwidth_factor_), 'prediction': host(model.predict(dq))}
        fn = fixed_x_knockoff_filter if case == 'knockoff_fixed' else model_x_knockoff_filter
        kwargs = {'modelx_draws': 3} if case == 'knockoff_model' else {}
        result = fn(dx, dy, q=.2, method='corr_diff', backend=rt.name, random_state=problem.seed, **kwargs)
        return {'W': host(result.W), 'threshold': np.asarray(result.threshold),
                'selected': host(result.selected_features)}
    return call, (X, Q, y)


def check(case, value, reference, arrays, problem, captured=None):
    """Numerical checks, never FDR/CI-coverage or selector optimality claims."""
    metrics = {}
    if case.startswith('knockoff') and not captured:
        raise AssertionError('missing realized knockoff construction for oracle')
    if case == 'kde_bootstrap':
        shapes = {'estimate': (problem.queries,), 'lower': (problem.queries,),
                  'upper': (problem.queries,), 'draws': (problem.resamples, problem.queries)}
    elif case.startswith('bandwidth'):
        shapes = {'factor': (), 'prediction': (problem.queries,)}
    else:
        shapes = {'W': (problem.p,)}
    for key, shape in shapes.items():
        actual = np.asarray(value[key])
        if actual.shape != shape or not np.isfinite(actual).all():
            raise AssertionError(f'invalid {key}: expected finite shape {shape}, got {actual.shape}')
    if case.startswith('bandwidth') and float(value['factor']) <= 0:
        raise AssertionError('selected factor must be positive')
    if case.startswith('knockoff'):
        threshold_value = np.asarray(value['threshold'])
        if threshold_value.shape != () or np.isnan(threshold_value) or float(threshold_value) <= 0:
            raise AssertionError('invalid threshold')
    if case.startswith('knockoff'):
        if not captured:
            raise AssertionError('missing realized knockoff construction for oracle')
        realized = [tuple(map(host, draw)) for draw in captured]
        X, K, y = realized[0]
        expected = np.mean([np.abs(a.T @ (c-c.mean())) - np.abs(b.T @ (c-c.mean())) for a,b,c in realized], axis=0)
        np.testing.assert_allclose(value['W'], expected, atol=2e-7, rtol=2e-7)
        if not np.isfinite(expected).all():
            raise AssertionError('nonfinite W')
        threshold = np.inf
        for t in np.sort(np.unique(np.abs(expected[expected != 0]))):
            if (1 + np.sum(expected <= -t))/max(1, np.sum(expected >= t)) <= .2:
                threshold = t
                break
        np.testing.assert_allclose(value['threshold'], threshold, atol=2e-7, rtol=2e-7)
        np.testing.assert_array_equal(value['selected'], np.flatnonzero(expected >= threshold))
        if case == 'knockoff_fixed':
            np.testing.assert_allclose(K.T @ K, X.T @ X, atol=2e-7, rtol=2e-7)
            cross = X.T @ X - X.T @ K
            np.testing.assert_allclose(cross, np.diag(np.diag(cross)), atol=2e-7, rtol=2e-7)
        return {'status': 'pass', 'performance_eligible': False,
                'reference': 'independent realized corr_diff/threshold and Fixed-X Gram identities',
                'limitation': 'backend RNG differs; no model-X distribution, FDR, or full sampler certification; diagnostic timing only'}
    for key, expected in reference.items():
        np.testing.assert_allclose(value[key], expected, atol=2e-6, rtol=2e-6)
        metrics[key + '_max_abs_diff'] = float(np.max(np.abs(value[key] - expected)))
    if case == 'kde_bootstrap':
        from scipy.stats import gaussian_kde
        X, Q, _ = arrays
        np.testing.assert_allclose(value['estimate'], gaussian_kde(X.T, bw_method=.4)(Q.T), atol=2e-7, rtol=2e-7)
        rng = np.random.default_rng(problem.seed)
        weights = np.ones(problem.n)/problem.n
        expected = []
        for _ in range(problem.resamples):
            idx = rng.choice(problem.n, problem.n, replace=True, p=weights)
            expected.append(gaussian_kde(X[idx].T, bw_method=.4, weights=weights[idx])(Q.T))
        np.testing.assert_allclose(value['draws'], expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(value['lower'], np.quantile(expected, .025, axis=0), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(value['upper'], np.quantile(expected, .975, axis=0), atol=2e-6, rtol=2e-6)
    return {'status': 'pass', 'performance_eligible': True, 'metrics': metrics,
            'reference': 'CPU public selector/prediction parity; KDE additionally SciPy each identical resample',
            'tolerance': {'atol': 2e-6, 'rtol': 2e-6, 'basis': 'float64 public CPU parity with KDE covariance jitter allowance'},
            'limitation': 'no selector optimality or confidence-interval coverage certification'}


def specs_for(case, backend, problem):
    specs = learning_stages(case, backend)
    if case == 'kde_bootstrap' and backend != 'numpy':
        for spec in specs:
            if spec['label'] in ('kde-fit', 'kde-density'):
                spec['minimum_calls'] = problem.resamples + 1
    return specs


def transfer_codes():
    from statgpu.nonparametric.kernel_smoothing import _kde, _bandwidth_selection, _kernel_common
    from statgpu.feature_selection import _knockoff, _knockoff_utils
    codes = set()
    for module in (_kde, _bandwidth_selection, _kernel_common, _knockoff, _knockoff_utils):
        for name in ('_to_numpy', 'xp_asarray'):
            fn = getattr(module, name, None)
            if hasattr(fn, '__code__'):
                codes.add(fn.__code__)
    codes.update((Runtime.array.__code__, host.__code__))
    return codes


def run_case(case, rt, problem, warmup=1, repeats=3, max_work=500_000_000):
    work = guard(case, problem, max_work)
    if case == 'kde_bootstrap' and rt.name == 'torch':
        raise Unsupported('Torch weighted KDE bootstrap is unsupported; no CPU fallback')
    if warmup < 0 or repeats < 1:
        raise ValueError('warmup >= 0 and repeats >= 1 required')
    call, arrays = make_call(case, rt, problem)
    reference_call, _ = make_call(case, Runtime('numpy'), problem)
    specs = specs_for(case, rt.name, problem)
    with rt.context(), warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter('error', ConvergenceWarning)
        reference = reference_call()
        # Native operation identity is established in a wholly separate run.
        tracer = NumericalTrace(rt, specs)
        with tracer:
            evidence_value = call()
        evidence = tracer.report()
        observer = PhaseObserver(rt, specs, transfer_codes())
        with observer:
            diagnostic_value = call()
        correctness = check(case, diagnostic_value, reference, arrays, problem, observer.captured)
        # Evidence and diagnostic outputs must agree within one backend/seed.
        for key in evidence_value:
            np.testing.assert_allclose(evidence_value[key], diagnostic_value[key], atol=2e-6, rtol=2e-6)
        observer.captured.clear()  # Do not retain diagnostic GPU matrices during timing.
        samples = []
        for i in range(warmup + repeats):
            rt.sync(rt.device)
            start = time.perf_counter()
            value = call()
            rt.sync(rt.device)
            elapsed = time.perf_counter() - start
            # Checks are outside measured intervals. Deterministic per-backend seed.
            for key in value:
                np.testing.assert_allclose(value[key], diagnostic_value[key], atol=2e-6, rtol=2e-6)
            if i >= warmup:
                samples.append(elapsed)
    X, Q, y = arrays
    return {'case': case, 'backend': rt.name, 'status': 'pass',
            'variant': 'cpu-public' if rt.name == 'numpy' else 'current-public-hybrid',
            'native_gpu_variant': {'status': 'not-implemented', 'timed': False},
            'performance_eligible': correctness['performance_eligible'], 'correctness': correctness,
            'problem': {**asdict(problem), 'actual_p': X.shape[1], 'dtype': 'float64',
                        'data_id': hashlib.sha256(X.tobytes()+Q.tobytes()+y.tobytes()).hexdigest(),
                        'estimated_work': work, 'parameters': {'kde_factor': .4, 'knockoff_method': 'corr_diff', 'q': .2, 'modelx_draws': 3}},
            'backend_evidence': evidence, 'hardware': rt.hardware,
            'diagnostic': observer.report(),
            'timing': {'scope': 'end-to-end-with-transfer: input arrays + public operation + host output',
                       'instrumented': False, 'samples_seconds': samples, 'median_seconds': statistics.median(samples),
                       'warmup': warmup, 'repeats': repeats, 'synchronized_device': rt.device,
                       'transfer_policy': 'all public internal and input/output transfers included; data generation/reference/checks excluded'},
            'comparability': 'same float64 host data and workload; KDE numeric factor .4, same host draws, CPU fast density vs GPU refits; knockoff backend-native RNG differs'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='explicitly replace output file')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--backends', nargs='+', choices=('numpy', 'cupy', 'torch'), default=['numpy'])
    parser.add_argument('--preset', choices=PRESETS, default='tiny')
    for field in ('n', 'p', 'queries', 'resamples', 'seed'):
        parser.add_argument('--' + field, type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max-work', type=int, default=500_000_000)
    parser.add_argument('--output', type=Path, default=Path('results/hybrid-phases.json'))
    args = parser.parse_args(argv)
    if args.inventory:
        print(json.dumps({'cases': CASES, 'presets': {k: asdict(v) for k, v in PRESETS.items()},
                          'native_gpu_prototype': 'not implemented', 'torch_bootstrap': 'unsupported'}, indent=2))
        return 0
    if args.warmup < 0 or args.repeats < 1 or args.device < 0 or args.max_work < 1:
        parser.error('require warmup >= 0, repeats >= 1, device >= 0, max-work >= 1')
    problem = replace(PRESETS[args.preset], **{k: getattr(args, k) for k in asdict(Problem()) if getattr(args, k) is not None})
    if len(args.cases) != len(set(args.cases)) or len(args.backends) != len(set(args.backends)):
        parser.error('duplicate cases/backends are not allowed')
    if args.output.exists() and not args.overwrite:
        parser.error('output exists; choose a new path or explicitly --overwrite')
    for case in args.cases:
        try:
            guard(case, problem, float('inf'))
        except ValueError as exc:
            parser.error(str(exc))
    before = fingerprint()
    rows = []
    for backend in args.backends:
        for case in args.cases:
            try:
                rt = Runtime(backend, args.device)
                row = run_case(case, rt, problem, args.warmup, args.repeats, args.max_work)
            except Exception as exc:
                status = 'unsupported' if isinstance(exc, Unsupported) else 'blocked' if isinstance(exc, (Blocked, BackendUnavailable, MemoryError)) else 'failed'
                row = {'case': case, 'backend': backend, 'status': status, 'reason': f'{type(exc).__name__}: {exc}',
                       'problem': asdict(problem), 'performance_eligible': False}
            row.setdefault('requested_device', 'cpu' if backend == 'numpy' else f'cuda:{args.device}')
            rows.append(row)
    after = fingerprint()
    changed = before != after
    complete = all(r['status'] == 'pass' for r in rows) and not changed
    if changed:
        for row in rows:
            row['performance_eligible'] = False
            row['evidence_invalidated'] = 'source changed during run'
    comparisons = []
    for case in args.cases:
        cpu = next((r for r in rows if r['case'] == case and r['backend'] == 'numpy' and r['status'] == 'pass'), None)
        for row in rows:
            if cpu and row['case'] == case and row['backend'] != 'numpy' and row['status'] == 'pass' and cpu['performance_eligible'] and row['performance_eligible'] and not changed and cpu['problem']['data_id'] == row['problem']['data_id'] and cpu['timing']['scope'] == row['timing']['scope']:
                comparisons.append({'case': case, 'backend': row['backend'], 'scope': row['timing']['scope'],
                                    'cpu_seconds_div_backend_seconds': cpu['timing']['median_seconds']/row['timing']['median_seconds'],
                                    'problem_data_id': row['problem']['data_id']})
    env = environment()
    hardware = {r['backend'] + ':' + r.get('requested_device', 'unknown'): r['hardware'] for r in rows if 'hardware' in r}
    result = {'schema_version': 'statgpu-benchmark-v2',
              'benchmark': {'name': 'hybrid-phase-comparison', 'command': sys.argv if argv is None else argv,
                            'script': str(Path(__file__).relative_to(ROOT))},
              'source': before, 'source_after': after, 'source_changed': changed,
              'environment': {**env, 'environment_id': hashlib.sha256(json.dumps({'environment': env, 'hardware': hardware}, sort_keys=True).encode()).hexdigest()}, 'rows': rows, 'performance': {'paired_end_to_end': comparisons},
              'validation': {'tier': 'local-minimal' if args.preset == 'tiny' else 'local-full',
                             'status': 'complete' if complete else 'incomplete',
                             'physical_gpu_claim': not changed and any(r['status'] == 'pass' and r['backend'] != 'numpy' for r in rows),
                             'uncovered_reasons': ([r.get('reason') for r in rows if r['status'] != 'pass'] + (['source changed during run'] if changed else []))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: {result["validation"]["status"]}, {len(rows)} rows')
    return 0 if complete else 2

if __name__ == '__main__':
    raise SystemExit(main())
