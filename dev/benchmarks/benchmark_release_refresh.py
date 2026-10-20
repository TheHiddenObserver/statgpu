#!/usr/bin/env python3
"""Correctness-gated v0.3 Gaussian fit/predict benchmark pilot.

From repository root (no GPU required for the smoke):
  python -m dev.benchmarks.benchmark_release_refresh --smoke --output results/refresh-smoke.json
  python dev/benchmarks/benchmark_release_refresh.py --backends numpy cupy torch \
      --shapes 1000x20 10000x100 --dtypes float32 float64 --require-backends \
      --output results/refresh.json

This is a new evidence producer, NOT a canonical dashboard source. Inference,
CV, formula, memory peaks and external-library timings are intentionally outside
this pilot. CPU smoke checks the harness, not representative performance. Input
float32 does not imply float32 computation: ingress and output dtypes are recorded.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from statgpu import LinearRegression, Ridge, __version__


class BackendUnavailable(RuntimeError):
    """Only absence of a library/device is skippable; execution bugs are not."""


def array_identity(value):
    module = type(value).__module__.split('.')[0]
    if module == 'cupy':
        return {'backend': 'cupy', 'device': f'cuda:{int(value.device.id)}',
                'dtype': str(value.dtype)}
    if module == 'torch':
        return {'backend': 'torch', 'device': str(value.device),
                'dtype': str(value.dtype).replace('torch.', '')}
    if isinstance(value, np.ndarray):
        return {'backend': 'numpy', 'device': 'cpu', 'dtype': str(value.dtype)}
    raise RuntimeError(f'Unverifiable operational array: {type(value).__name__}')


class _ObserveFit:
    """Observe existing kernel dispatch, without changing its arrays or behavior."""
    def _observe(self, route, X, y, sample_weight=None):
        self.benchmark_fit_ingress = {'route': route, 'X': array_identity(X),
                                      'y': array_identity(y)}
        if sample_weight is not None:
            self.benchmark_fit_ingress['sample_weight'] = array_identity(sample_weight)

    def _fit_cpu(self, X, y, *args, **kwargs):
        self._observe('numpy', X, y, args[0] if args else kwargs.get('sample_weight'))
        return super()._fit_cpu(X, y, *args, **kwargs)

    def _fit_gpu(self, X, y, *args, **kwargs):
        self._observe('cupy', X, y, args[0] if args else kwargs.get('sample_weight'))
        return super()._fit_gpu(X, y, *args, **kwargs)

    def _fit_torch(self, X, y, *args, **kwargs):
        self._observe('torch', X, y, args[0] if args else kwargs.get('sample_weight'))
        return super()._fit_torch(X, y, *args, **kwargs)


class ObservedOLS(_ObserveFit, LinearRegression):
    pass


class ObservedRidge(_ObserveFit, Ridge):
    def fit(self, X=None, y=None, sample_weight=None, **kwargs):
        result = super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
        if not hasattr(self, 'benchmark_fit_ingress'):
            # Ridge's optimized CPU-exact override bypasses _fit_cpu. This
            # pilot supplies NumPy arrays here and checks its published route.
            if self._selected_backend_name != 'numpy':
                raise RuntimeError('Unobserved non-CPU Ridge fit route')
            self._observe('numpy', np.asarray(X), np.asarray(y), sample_weight)
            self.benchmark_fit_ingress['observation'] = 'optimized CPU exact public input; internally promoted to float64'
        return result


class Runtime:
    def __init__(self, name, ordinal=0):
        self.name, self.ordinal = name, ordinal
        self.module = np
        self.device = 'cpu'
        self.hardware = {}
        if name == 'numpy':
            return
        try:
            self.module = importlib.import_module(name)
        except ImportError as exc:
            raise BackendUnavailable(f'{name} import unavailable: {exc}') from exc
        if name == 'cupy':
            cp = self.module
            try:
                count = cp.cuda.runtime.getDeviceCount()
            except Exception as exc:
                raise BackendUnavailable(f'CuPy CUDA discovery unavailable: {exc}') from exc
            if count == 0:
                raise BackendUnavailable('CuPy reports zero CUDA devices')
            if ordinal >= count:
                raise ValueError(f'CUDA ordinal {ordinal} is outside {count} CuPy devices')
            props = cp.cuda.runtime.getDeviceProperties(ordinal)
            self.hardware = {'gpu': (props.get('name', b'').decode() if isinstance(props.get('name', b''), bytes) else str(props.get('name', 'unknown'))),
                             'gpu_uuid': str(props.get('uuid', 'unavailable')),
                             'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
                             'driver': cp.cuda.runtime.driverGetVersion()}
        elif name == 'torch':
            torch = self.module
            if not torch.cuda.is_available():
                raise BackendUnavailable('Torch CUDA is unavailable (Torch CPU is not a substitute)')
            if ordinal >= torch.cuda.device_count():
                raise ValueError(f'CUDA ordinal {ordinal} is outside Torch devices')
            props = torch.cuda.get_device_properties(ordinal)
            self.hardware = {'gpu': props.name, 'gpu_uuid': str(getattr(props, 'uuid', 'unavailable')),
                             'cuda_runtime': torch.version.cuda, 'driver': 'unavailable'}
        else:
            raise ValueError(f'Unknown backend {name}')
        self.device = f'cuda:{ordinal}'

    def context(self):
        if self.name == 'cupy':
            return self.module.cuda.Device(self.ordinal)
        if self.name == 'torch':
            return self.module.cuda.device(self.ordinal)
        return nullcontext()

    def array(self, value):
        if value is None:
            return None
        if self.name == 'torch':
            return self.module.as_tensor(value, device=self.device)
        return self.module.asarray(value)

    def sync(self, actual_device):
        if actual_device != self.device:
            raise RuntimeError(f'Execution device mismatch: {actual_device} != {self.device}')
        if self.name == 'cupy':
            self.module.cuda.Device(int(actual_device.split(':')[1])).synchronize()
        elif self.name == 'torch':
            self.module.cuda.synchronize(device=actual_device)

    def host(self, value):
        if self.name == 'cupy':
            return self.module.asnumpy(value)
        if self.name == 'torch':
            return value.detach().cpu().numpy()
        return np.asarray(value)


def make_data(n, p, dtype, seed, weighted):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(dtype)
    y = (0.7 + X.astype('float64') @ np.linspace(-0.4, 0.6, p)
         + rng.normal(scale=0.2, size=n)).astype(dtype)
    query = rng.normal(size=(max(16, n // 4), p)).astype(dtype)
    weights = rng.uniform(0.2, 2.0, size=n).astype(dtype) if weighted else None
    digest = hashlib.sha256()
    for array in (X, y, query, weights):
        if array is not None:
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
    return X, y, query, weights, digest.hexdigest()


def analytic_reference(X, y, query, weight, alpha):
    """Solve the augmented weighted least-squares system, not normal equations."""
    X, y, query = (np.asarray(v, dtype='float64') for v in (X, y, query))
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype='float64')
    D = np.column_stack((np.ones(len(y)), X))
    sw = np.sqrt(w / w.sum())
    design, target = D * sw[:, None], y * sw
    penalty = np.diag(np.r_[0.0, np.repeat(np.sqrt(alpha), X.shape[1])])
    params = np.linalg.lstsq(np.vstack((design, penalty)),
                             np.r_[target, np.zeros(D.shape[1])], rcond=None)[0]
    pred = params[0] + query @ params[1:]
    return params, pred


def correctness(X, y, weight, alpha, params, pred, ref_params, ref_pred, dtype):
    # Frozen fixture-specific mixed absolute/relative tolerances for these
    # well-conditioned Gaussian designs; not universal statistical tolerances.
    atol, rtol = (2e-4, 2e-4) if dtype == 'float32' else (2e-8, 2e-8)
    X, y = np.asarray(X, dtype='float64'), np.asarray(y, dtype='float64')
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype='float64')
    w = w / w.sum()
    def objective(beta):
        return float(0.5 * np.dot(w, (y - beta[0] - X @ beta[1:]) ** 2)
                     + 0.5 * alpha * np.dot(beta[1:], beta[1:]))
    measured, expected = objective(params), objective(ref_params)
    residual = params[0] + X @ params[1:] - y
    grad = np.r_[np.dot(w, residual), X.T @ (w * residual) + alpha * params[1:]]
    finite = all(np.isfinite(v).all() for v in (params, pred, grad))
    passed = finite and np.allclose(params, ref_params, atol=atol, rtol=rtol)
    passed = passed and np.allclose(pred, ref_pred, atol=atol, rtol=rtol)
    passed = passed and np.isclose(measured, expected, atol=atol, rtol=rtol)
    passed = passed and float(np.max(np.abs(grad))) <= atol + rtol
    return {'status': 'pass' if passed else 'fail',
            'reference': {'name': 'NumPy float64 augmented weighted least squares',
                          'version': np.__version__, 'reference_type': 'analytic'},
            'objective_scaling': 'sum(w * residual**2)/(2*sum(w)) + alpha*sum(coef**2)/2; intercept unpenalized',
            'penalty_scale_mapping': 'normal equation slope penalty = sum(w)*alpha; sklearn Ridge alpha = sum(w)*alpha',
            'thresholds': {'atol': atol, 'rtol': rtol,
                           'source': 'benchmark_release_refresh.py: well-conditioned Gaussian fixture v1'},
            'metrics': {'parameter_max_abs_diff': float(np.max(np.abs(params-ref_params))),
                        'prediction_max_abs_diff': float(np.max(np.abs(pred-ref_pred))),
                        'objective_abs_diff': abs(measured-expected),
                        'stationarity_inf_norm': float(np.max(np.abs(grad)))},
            'convergence': {'method': 'exact', 'stationarity_pass': bool(np.max(np.abs(grad)) <= atol+rtol)}}


def verify_route(model, runtime):
    ingress = getattr(model, 'benchmark_fit_ingress', None)
    if ingress is None or ingress['route'] != runtime.name:
        raise RuntimeError('Fit did not enter the requested numerical backend')
    for key in ('X', 'y', *(['sample_weight'] if 'sample_weight' in ingress else [])):
        if ingress[key]['backend'] != runtime.name or ingress[key]['device'] != runtime.device:
            raise RuntimeError('Fit operational arrays are not on the requested device')
    if getattr(model, '_selected_backend_name', None) != runtime.name:
        raise RuntimeError('Selected backend disagrees with observed execution')
    if getattr(model, '_selected_backend_device', None) != runtime.device:
        raise RuntimeError('Selected device disagrees with observed execution')
    return ingress


def time_call(call, runtime, device):
    runtime.sync(device)
    start = time.perf_counter()
    value = call()
    runtime.sync(device)
    return value, time.perf_counter() - start


def run_case(runtime, *, model_name, n, p, dtype, weighted, seed, alpha, warmup, repeats):
    X, y, query, weights, data_id = make_data(n, p, dtype, seed, weighted)
    actual_alpha = 0.0 if model_name == 'ols' else alpha
    ref_params, ref_pred = analytic_reference(X, y, query, weights, actual_alpha)
    row = {'problem': {'shape': {'n': n, 'p': p, 'n_predict': len(query)},
                       'dtype': dtype, 'seed': seed, 'data_id': data_id,
                       'parameters': {'model': model_name, 'alpha': actual_alpha,
                                      'weighted': weighted, 'fit_intercept': True,
                                      'compute_inference': False}},
           'backend': {'requested': runtime.name, 'requested_device': runtime.device},
           'status': 'failed'}
    def new_model():
        kw = {'device': {'numpy': 'cpu', 'cupy': 'cuda', 'torch': 'torch'}[runtime.name],
              'compute_inference': False, 'gpu_memory_cleanup': False}
        return ObservedOLS(**kw) if model_name == 'ols' else ObservedRidge(alpha=alpha, solver='exact', **kw)
    with runtime.context(), warnings.catch_warnings():
        warnings.simplefilter('error')
        # Input allocation/transfer is outside measured regions. Public fit may
        # still perform casts and output copies; those are deliberately included.
        xb, yb, qb, wb = (runtime.array(v) for v in (X, y, query, weights))
        def fit(model):
            return model.fit(xb, yb, sample_weight=wb)
        def predict(model):
            return model.predict(qb) if model_name == 'ols' else model.predict(qb, return_cpu=False)
        model = fit(new_model())
        ingress = verify_route(model, runtime)
        device = ingress['X']['device']
        prediction = predict(model)
        identity = array_identity(prediction)
        if identity['backend'] != runtime.name or identity['device'] != device:
            raise RuntimeError('Prediction placement does not match fit execution')
        runtime.sync(device)
        params = np.r_[float(model.intercept_), np.asarray(model.coef_, dtype='float64')]
        check = correctness(X, y, weights, actual_alpha, params, runtime.host(prediction),
                            ref_params, ref_pred, dtype)
        row['correctness'] = check
        row['backend'].update({'fit_ingress': ingress, 'prediction': identity,
                               'parameter_reporting_dtype': str(np.asarray(model.coef_).dtype),
                               'compute_dtype': 'not-instrumented; ingress/output dtypes are not kernel precision'})
        if check['status'] != 'pass':
            return row  # No usable timing on a failed correctness gate.
        fit_samples, predict_samples = [], []
        for index in range(warmup + repeats):
            candidate = new_model()  # Construction excluded, fresh model each fit.
            model, fit_time = time_call(lambda: fit(candidate), runtime, device)
            verify_route(model, runtime)
            prediction, predict_time = time_call(lambda: predict(model), runtime, device)
            if array_identity(prediction) != identity:
                raise RuntimeError('Prediction placement/dtype changed across repeats')
            params = np.r_[float(model.intercept_), np.asarray(model.coef_, dtype='float64')]
            check = correctness(X, y, weights, actual_alpha, params, runtime.host(prediction),
                                ref_params, ref_pred, dtype)
            if check['status'] != 'pass':
                row['correctness'] = check
                return row
            if index >= warmup:
                fit_samples.append(fit_time)
                predict_samples.append(predict_time)
        row['timing'] = {'warmup': warmup, 'repeats': repeats, 'statistic': 'median',
                         'correctness_preflight_fits': 1,
                         'cache_state': 'warmed by correctness preflight plus configured warmup; not cold-start',
                         'instrumentation': 'lightweight backend ingress observation included in fit timing',
                         'synchronized_device': device if runtime.name != 'numpy' else None,
                         'transfer_policy': 'native inputs preallocated; internal casting/report copies included; prediction returned native',
                         'records': [
                             {'scope': 'fit-only-public-api', 'raw_seconds': fit_samples,
                              'median_seconds': statistics.median(fit_samples)},
                             {'scope': 'predict-only-public-api', 'raw_seconds': predict_samples,
                              'median_seconds': statistics.median(predict_samples)}]}
        row['status'] = 'passed'
    return row


def source_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT)
    # Hash executable Python source, tracked diff, and this new untracked runner.
    # Avoid hashing output artifacts and environment/credential files.
    files = sorted(p for p in (ROOT/'statgpu').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc') + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode()+b'\0')
        digest.update(path.read_bytes())
    diff = git('diff', 'HEAD', '--binary', '--', 'statgpu', 'dev/benchmarks/benchmark_release_refresh.py')
    digest.update(diff)
    runner_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    runner_tracked = bool(git('ls-files', '--', runner_relative).strip())
    # dev/benchmarks/*.py can be ignored: ordinary git status alone is not
    # sufficient to label the executing source clean.
    tree_dirty = bool(git('status', '--porcelain').strip())
    return {'git_commit': git('rev-parse', 'HEAD').decode().strip(),
            'working_tree': 'dirty' if tree_dirty or not runner_tracked else 'clean',
            'runner_tracked': runner_tracked,
            'source_overlay': 'tracked' if runner_tracked else 'untracked-or-ignored runner',
            'worktree_fingerprint': digest.hexdigest(),
            'fingerprint_scope': 'all statgpu package files except bytecode caches + runner bytes + scoped tracked diff; excludes external artifacts/configuration',
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'statgpu_version': __version__, 'import_path': str(Path(importlib.import_module('statgpu').__file__).resolve())}


def environment():
    versions = {}
    for package in ('numpy', 'scipy', 'torch', 'cupy', 'cupy-cuda11x', 'cupy-cuda12x'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    result = {'python': platform.python_version(), 'os': platform.platform(),
              'cpu': cpu, 'packages': versions,
              'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    try:
        from threadpoolctl import threadpool_info
        result['threadpools'] = threadpool_info()
    except ImportError:
        result['threadpools'] = 'threadpoolctl unavailable'
    return result


def parse_shape(text):
    try:
        n, p = map(int, text.lower().split('x'))
        if p < 1 or n <= p + 1:
            raise ValueError
        return n, p
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected NxP with p>=1 and n>p+1') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backends', nargs='+', choices=['numpy', 'cupy', 'torch'], default=['numpy', 'cupy', 'torch'])
    parser.add_argument('--shapes', nargs='+', type=parse_shape, default=[(1000, 20), (10000, 100)])
    parser.add_argument('--dtypes', nargs='+', choices=['float32', 'float64'], default=['float64'])
    parser.add_argument('--models', nargs='+', choices=['ols', 'ridge'], default=['ols', 'ridge'])
    parser.add_argument('--weights', nargs='+', choices=['none', 'analytic'], default=['none', 'analytic'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--require-backends', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='CPU-only tiny harness check; overrides backends/shapes/warmup/repeats')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1 or args.cuda_device < 0 or not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error('Require warmup>=0, repeats>=1, cuda-device>=0, finite alpha>=0')
    if args.smoke:
        args.backends, args.shapes, args.warmup, args.repeats = ['numpy'], [(48, 4)], 0, 1
    if Path(importlib.import_module('statgpu').__file__).resolve().parent != ROOT/'statgpu':
        raise RuntimeError('Imported statgpu is not the benchmark checkout')
    source = source_identity()
    result = {'schema_version': 'statgpu-benchmark-v2',
              'benchmark': {'name': 'release-gaussian-refresh-pilot',
                            'script': 'dev/benchmarks/benchmark_release_refresh.py',
                            'command': shlex.join([sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]),
                            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
              'source': source, 'environment': environment(), 'backends': {}, 'rows': [],
              'validation': {'tier': 'local-minimal' if args.smoke else 'execution-only-not-canonical',
                             'uncovered_reasons': ['No inference/CV/formula/memory/external-library timing in this pilot'],
                             'canonical_dashboard_registered': False}}
    failures = 0
    for name in args.backends:
        try:
            runtime = Runtime(name, args.cuda_device)
        except BackendUnavailable as exc:
            result['backends'][name] = {'status': 'unavailable', 'reason': str(exc)}
            failures += int(args.require_backends)
            result['validation']['uncovered_reasons'].append(f'{name}: {exc}')
            continue
        except Exception as exc:
            result['backends'][name] = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
            failures += 1
            continue
        result['backends'][name] = {'status': 'available', 'device': runtime.device, **runtime.hardware}
        for n, p in args.shapes:
            for dtype in args.dtypes:
                for model_name in args.models:
                    for weight in args.weights:
                        identity = dict(model_name=model_name, n=n, p=p, dtype=dtype, weighted=weight=='analytic',
                                        seed=args.seed, alpha=args.alpha, warmup=args.warmup, repeats=args.repeats)
                        try:
                            row = run_case(runtime, **identity)
                        except Exception as exc:
                            row = {'status': 'failed', 'backend': {'requested': name}, 'case': identity,
                                   'error': f'{type(exc).__name__}: {exc}'}
                        result['rows'].append(row)
                        failures += row['status'] != 'passed'
    if not result['rows']:
        failures += 1
        result['validation']['uncovered_reasons'].append('No benchmark cases executed')
    env_group = {'environment': result['environment'], 'backends': result['backends']}
    result['environment']['environment_id'] = hashlib.sha256(json.dumps(env_group, sort_keys=True).encode()).hexdigest()
    after = source_identity()
    result['source_after'] = after
    if after['git_commit'] != source['git_commit'] or after['worktree_fingerprint'] != source['worktree_fingerprint']:
        failures += 1
        result['validation']['uncovered_reasons'].append('Source changed during execution; results invalid')
    result['validation']['status'] = 'failed' if failures else ('partial' if any(v['status']=='unavailable' for v in result['backends'].values()) else 'complete')
    # Non-finite metrics are serialized as null diagnostics, never nonstandard NaN.
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"{result['validation']['status']}: {len(result['rows'])} rows; {failures} failures; {args.output}")
    return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
