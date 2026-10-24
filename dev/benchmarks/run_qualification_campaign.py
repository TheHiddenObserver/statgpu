#!/usr/bin/env python3
"""Plan or explicitly execute a serial, bounded seven-suite qualification campaign.

Planning imports only the standard library, reads the registry, and launches no
process or numerical/GPU workload. All performance evidence stays diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'dev/benchmarks/full_coverage_manifest.json'
RUNNERS = {
    'gaussian-refresh': 'benchmark_release_refresh',
    'classification-metrics': 'benchmark_classification_metrics',
    'model-families': 'benchmark_model_coverage',
    'statistical-families': 'benchmark_statistical_coverage',
    'learning-families': 'benchmark_learning_coverage',
    'workflow-families': 'benchmark_workflow_coverage',
    'six-family-cv': 'benchmark_cv_coverage',
}
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
SCENARIOS = ('balanced-continuous', 'imbalanced-tied', 'constant-score')
HOST_ONLY = {'anova/partial_eta_squared', 'panel-primitives/hausman_test',
             'panel-primitives/pooling_f_test', 'panel-primitives/breusch_pagan_lm_test'}
ONE_FEATURE_LEARNING = {'kde', 'kernel_regression_nw', 'kernel_regression_local_linear',
    'spline', 'gam', 'smoothing_helpers', 'bandwidth_helpers', 'pairwise_kernels',
    'chi2_kernel', 'kde_normal_ci', 'kde_bootstrap', 'bspline_basis',
    'natural_spline', 'cyclic_spline', 'thin_plate_spline'}

# Executed only in an explicit campaign child. The parent supplies thread limits
# in its environment BEFORE this interpreter imports numerical libraries.
CHILD_BOOTSTRAP = r'''import importlib, importlib.metadata, json, os, platform, resource, sys, traceback
from pathlib import Path
sys.path.insert(0, sys.argv[1])
module_name, backend, threads, runtime_path = sys.argv[2:6]
threads = int(threads)
argv = sys.argv[6:]
report = {'python': platform.python_version(), 'executable': sys.executable,
          'requested_threads': threads, 'backend': backend,
          'thread_environment': {k: os.environ.get(k) for k in (
              'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')},
          'gpu_allocator_peak_bytes': {'status': 'not_measured'},
          'performance_status': 'UNVERIFIED_EXCLUSIVITY'}
class ThreadLimitViolation(RuntimeError):
    pass
rc = 1
try:
    import threadpoolctl
    if backend == 'torch':
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    module = importlib.import_module(module_name)
    def snapshot():
        pools = threadpoolctl.threadpool_info()
        torch_threads = None
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            torch_threads = {'intraop': torch.get_num_threads(), 'interop': torch.get_num_interop_threads()}
        invalid = [p for p in pools if type(p.get('num_threads')) is not int
                   or not 1 <= p['num_threads'] <= threads]
        if not pools:
            raise ThreadLimitViolation('No actual numerical thread pools observed')
        if invalid or (backend == 'torch' and any(v > threads or v < 1 for v in torch_threads.values())):
            raise ThreadLimitViolation('Actual thread pool exceeds requested resource bound')
        return {'pools': pools, 'torch_threads': torch_threads, 'cap_verified': True,
                'all_pools_equal_requested': all(p['num_threads'] == threads for p in pools)}
    report['before'] = snapshot()
    # Persist preflight even if a timeout later prevents finalization.
    with open(runtime_path + '.before.json', 'x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    rc = module.main(argv)
    rc = 0 if rc is None else int(rc)
    report['after'] = snapshot()
except BaseException as exc:
    report['error'] = type(exc).__name__ + ': ' + str(exc)
    report['resource_guard'] = isinstance(exc, ThreadLimitViolation)
    traceback.print_exc()
    rc = 86 if report['resource_guard'] else 1
finally:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report['process_peak_rss_bytes'] = int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    report['memory_scope'] = 'Whole child process high-water RSS, including imports/reference/runner; not per-algorithm allocation or GPU VRAM'
    report['packages'] = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'statsmodels', 'torch', 'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'threadpoolctl'):
        try: report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['packages'][name] = None
    report['exit_code'] = rc
    with open(runtime_path + '.after.json', 'x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
sys.exit(rc)
'''


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError('Nonfinite JSON number: ' + token)
        return value
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=reject, parse_float=finite_float, object_pairs_hook=unique_pairs)


def cgroup_paths(controller, root=Path('/sys/fs/cgroup')):
    """Visible cgroup v1/v2 leaves and ancestors, without host namespace access."""
    root = Path(root).resolve()
    paths = {root, root / controller}
    if controller == 'cpu':
        paths.add(root / 'cpu,cpuacct')
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if controllers and controller not in controllers.split(','):
                continue
            bases = [root] if not controllers else [root / controller, root / controllers]
            for base in bases:
                target = (base / relative.lstrip('/')).resolve()
                if target.is_relative_to(root):
                    paths.update([target, *[p for p in target.parents if p.is_relative_to(root)]])
    except (OSError, ValueError):
        pass
    return sorted(paths)


def cpu_limits(cgroup_root=Path('/sys/fs/cgroup')):
    """Account for scheduler affinity and every readable enclosing cgroup quota."""
    try:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    except OSError:
        affinity = 1  # Unknown affinity does not authorize optional larger profiles.
    quotas, checked = [], []
    for path in cgroup_paths('cpu', cgroup_root):
        try:
            quota, period = (path / 'cpu.max').read_text().split()
            if int(period) <= 0 or (quota != 'max' and int(quota) <= 0):
                raise ValueError('Invalid quota')
            checked.append(str(path / 'cpu.max'))
            if quota != 'max':
                quotas.append(int(quota) / int(period))
        except (OSError, ValueError):
            try:
                quota = int((path / 'cpu.cfs_quota_us').read_text())
                period = int((path / 'cpu.cfs_period_us').read_text())
                if quota == 0 or period <= 0:
                    raise ValueError('Invalid quota')
                checked.append(str(path / 'cpu.cfs_quota_us'))
                if quota > 0:
                    quotas.append(quota / period)
            except (OSError, ValueError):
                pass
    capacity = min([float(affinity), *quotas])
    quota_unknown = sys.platform.startswith('linux') and not checked
    return {'affinity_cpus': affinity, 'cgroup_quota_cpus': min(quotas) if quotas else None,
            'quota_files_checked': checked, 'quota_status': 'unknown' if quota_unknown else 'observed-or-not-applicable',
            'effective_cpus': None if quota_unknown else capacity,
            'thread_cap': 1 if quota_unknown else max(1, math.floor(capacity)),
            'fractional_quota_note': 'One baseline thread may be throttled when quota < 1; optional profiles never exceed the floor. Unknown Linux quota permits baseline 1 only.'}


def registry():
    entries = read_json(MANIFEST)['entries']
    selected = {e['id']: e for e in entries if e.get('id') in RUNNERS}
    if set(selected) != set(RUNNERS) or len([e for e in entries if e.get('id') in RUNNERS]) != 7:
        raise ValueError('Registry must contain the seven unique reviewed suites')
    for suite, entry in selected.items():
        if entry.get('safe_to_execute') is not True or entry.get('script') != f'dev/benchmarks/{RUNNERS[suite]}.py':
            raise ValueError('Registry runner is not reviewed: ' + suite)
        cases = entry.get('case_ids')
        if not isinstance(cases, list) or not cases or any(not isinstance(c, str) for c in cases) or len(cases) != len(set(cases)):
            raise ValueError('Invalid exact case registry: ' + suite)
    return selected


def parse_selectors(values, entries):
    result = {}
    for value in values:
        suite, separator, cases = value.partition('=')
        selected = cases.split(',')
        if not separator or suite not in entries or suite in result:
            raise ValueError('--cases requires one SUITE=exact_case,exact_case selector per suite')
        if len(selected) != len(set(selected)) or not set(selected) <= set(entries[suite]['case_ids']):
            raise ValueError('Unknown or duplicate exact case selector: ' + value)
        result[suite] = selected
    return result


def expected_rows(suite, backend, cases, n, p, seed, *, dtypes=None, weights=None, knockoff_p=20, selection_seeds=None):
    rows = []
    for case in cases:
        actual_n, actual_p = n, p
        if suite == 'model-families' and case.startswith('panel:'):
            periods = max(5, p + 2)
            actual_n = max(p + 3, n // periods) * periods
        if suite == 'learning-families':
            actual_p = knockoff_p if case.startswith('knockoff') else 1 if case in ONE_FEATURE_LEARNING or case.startswith('bandwidth') else p
        row = {'case': case, 'backend': backend, 'n': actual_n, 'p': actual_p,
               'input_dtype': (dtypes or ['float64'])[0], 'seed': seed}
        if suite == 'classification-metrics':
            row.pop('p')
        if suite == 'statistical-families' and case in HOST_ONLY and backend != 'numpy':
            # Current producer omits problem dimensions for authorized host-only NA.
            row = {'case': case, 'backend': backend, 'not_applicable': True}
        if suite == 'gaussian-refresh':
            for dtype in (dtypes or ('float32', 'float64')):
                for weighted in [weight == 'analytic' for weight in (weights or ('none', 'analytic'))]:
                    rows.append(dict(row, input_dtype=dtype, weighted=weighted))
        elif suite == 'classification-metrics':
            for dtype in (dtypes or ('float32', 'float64')):
                for scenario in SCENARIOS:
                    rows.append(dict(row, input_dtype=dtype, scenario=scenario))
        elif suite == 'learning-families' and case.startswith('knockoff') and selection_seeds:
            rows.extend(dict(row, seed=trial_seed) for trial_seed in selection_seeds)
        else:
            rows.append(row)
    return rows


# Conservative, reviewed campaign limits, not claims about estimator scalability.
# Each case remains addressable at explicit shapes; larger envelopes need a
# separately reviewed resource/approval policy rather than an override flag.
LEARNING_RESOURCE_POLICIES = {
    'linear_or_streamed': {'max_n': 2000, 'max_p': 64, 'work': 'bounded fit/query fixture'},
    'dense_factorization': {'max_n': 1024, 'max_p': 16, 'work': 'dense n-by-n factorization/reference'},
    'iterative_embedding': {'max_n': 512, 'max_p': 16, 'work': '250 TSNE iterations / 100 UMAP epochs; graph/reference storage'},
    'iterative_factorization': {'max_n': 2000, 'max_p': 32, 'work': 'at most 300 matrix-factorization iterations'},
    'bandwidth_search': {'max_n': 1024, 'max_p': 4, 'work': 'bounded pairwise bandwidth objective/search'},
    'kernel_cv': {'max_n': 512, 'max_p': 16, 'work': '3 alpha values x 3 folds plus full-data refit'},
    'bootstrap': {'max_n': 2000, 'max_p': 4, 'work': '20 bootstrap replicates x fixed query grid'},
    'selection': {'max_n': 2000, 'max_p': 64, 'work': 'q=.2, corr_diff; at most 32 independent seed trials'},
}
LEARNING_POLICY_CASES = {
    'dense_factorization': {'kernel_ridge', 'kernel_pca', 'thin_plate_spline', 'agglomerative', 'dbscan'},
    'iterative_embedding': {'tsne', 'umap', 'umap_nndescent'},
    'iterative_factorization': {'nmf', 'minibatch_nmf'},
    'bandwidth_search': {'bandwidth_ucv', 'bandwidth_bcv', 'bandwidth_sj', 'bandwidth_sj-ste', 'bandwidth_sj-dpi', 'bandwidth_cv'},
    'kernel_cv': {'kernel_ridge_cv'},
    'bootstrap': {'kde_bootstrap'},
    'selection': {'knockoff_fixed', 'knockoff_model'},
}
SCALE_SHAPES = {
    'gaussian-refresh': [(1024, 16), (8192, 64), (32768, 128)],
    'classification-metrics': [(1000, 0), (10000, 0), (100000, 0)],
    'model-families': [(500, 8), (1024, 8), (2048, 16)],
    'statistical-families': [(128, 4), (1024, 4), (2048, 8)],
    'learning-families': [(256, 4), (1024, 4), (2000, 8)],
    'workflow-families': [(256, 6), (1024, 6), (2048, 8)],
    'six-family-cv': [(240, 16), (1024, 16), (2048, 16)],
}
DEFAULT_SHAPES = {suite: values[:(3 if suite in ('gaussian-refresh', 'classification-metrics') else
                               2 if suite in ('statistical-families', 'workflow-families', 'six-family-cv') else 1)]
                  for suite, values in SCALE_SHAPES.items()}


def learning_resource_policy(case):
    name = next((name for name, cases in LEARNING_POLICY_CASES.items() if case in cases), 'linear_or_streamed')
    return dict(LEARNING_RESOURCE_POLICIES[name], policy=name)


def parse_shapes(values, suites):
    result = {}
    for value in values:
        suite, separator, text = value.partition('=')
        if not separator or suite not in suites or suite in result:
            raise ValueError('--shapes requires one selected SUITE=NxP,NxP selector per suite')
        try:
            shapes = [tuple(map(int, item.split('x'))) for item in text.split(',')]
        except ValueError as exc:
            raise ValueError('Invalid explicit shape: ' + value) from exc
        if not shapes or len(shapes) > 8 or any(len(shape) != 2 for shape in shapes) or len(shapes) != len(set(shapes)):
            raise ValueError('Shapes must be 1-8 distinct NxP pairs: ' + value)
        result[suite] = shapes
    return result


def _unique(values, name, maximum=32):
    if not values or len(values) > maximum or len(values) != len(set(values)):
        raise ValueError(name + ' must contain unique values (at most ' + str(maximum) + ')')
    return values


def _shape_reason(suite, case, n, p, args):
    if n < 16 or p < (0 if suite == 'classification-metrics' else 1):
        return 'shape requires n>=16 and a positive feature count (classification uses p=0)'
    if suite == 'classification-metrics' and (p != 0 or n > 100000):
        return 'classification fixture supports p=0 and n<=100000 in this resource envelope'
    if suite == 'gaussian-refresh' and (n > 32768 or p > 128 or n <= p + 2):
        return 'Gaussian campaign resource envelope requires n<=32768, p<=128 and n>p+2'
    if suite == 'model-families':
        if (case.startswith('quantile:') or case == 'wrapper:QuantileRegression') and n > 2000:
            return 'quantile independent dense LP oracle supports n<=2000; requested shape is not silently shrunk'
        if n > 2048 or p > 16 or n <= p + 2:
            return 'model fixture resource envelope requires n<=2048, p<=16 and n>p+2'
    if suite == 'six-family-cv' and (n < 30 or n > 2048 or p > 16 or n < 3 * p):
        return 'CV campaign supports 30<=n<=2048, p<=16 and n>=3p; 5000/10000 require separate approval and resource review'
    if suite in ('statistical-families', 'workflow-families') and (n > 2048 or p < 2 or p > 16 or n <= 2 * p):
        return 'fixture resource envelope requires n<=2048, 2<=p<=16 and n>2p'
    if suite == 'learning-families':
        policy = learning_resource_policy(case)
        case_p = args.knockoff_p if case.startswith('knockoff') else p
        if p < 2 or n < 2 * case_p:
            return 'learning fixture requires p>=2 and n>=2p (selection uses knockoff-p)'
        if n * n > args.max_pairwise_elements:
            return 'n*n exceeds the campaign max-pairwise-elements resource bound'
        if n > policy['max_n'] or case_p > policy['max_p']:
            return f"{policy['policy']} resource policy requires n<={policy['max_n']}, p<={policy['max_p']}: {policy['work']}"
    return None


def _plan_source():
    """Read exact content and Git ref without processes or numerical imports."""
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in source_files()}
    head = None
    try:
        gitdir = ROOT / '.git'
        if gitdir.is_file():
            gitdir = (ROOT / gitdir.read_text().strip().removeprefix('gitdir: ')).resolve()
        token = (gitdir / 'HEAD').read_text().strip()
        if token.startswith('ref: '):
            reference = token[5:]
            common = gitdir
            if (gitdir / 'commondir').exists():
                common = (gitdir / (gitdir / 'commondir').read_text().strip()).resolve()
            if (common / reference).exists():
                token = (common / reference).read_text().strip()
            else:
                token = next(line.split()[0] for line in (common / 'packed-refs').read_text().splitlines()
                             if line.endswith(' ' + reference))
        if len(token) == 40 and all(c in '0123456789abcdef' for c in token):
            head = token
    except (OSError, StopIteration):
        pass
    return {'git_commit': head, 'content_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'fingerprint_scope': 'campaign source_files content; no process launched; not a clean-tree assertion'}


CONFIGURATION_KEYS = {
    'model-families': {'dtype', 'weights', 'solver', 'sparse'},
    'learning-families': {'dtype', 'weights', 'solver', 'sparse', 'covariance_type'},
    'six-family-cv': {'dtype', 'weights', 'sparse', 'folds', 'grid_size'},
}
LEARNING_FLOAT32 = {'pca', 'incremental_pca', 'truncated_svd', 'kmeans', 'minibatch_kmeans', 'dbscan', 'agglomerative', 'gaussian_mixture'}


def parse_configurations(values, suites):
    profiles = {}
    for value in values:
        label, separator, raw = value.partition('=')
        suite, colon, name = label.partition(':')
        if not separator or not colon or suite not in suites or not name or not all(c.isalnum() or c in '-_' for c in name):
            raise ValueError('--config requires selected SUITE:NAME=JSON_OBJECT with a simple unique profile name')
        def unique_pairs(pairs):
            data = {}
            for key, item in pairs:
                if key in data: raise ValueError('Duplicate configuration key: ' + key)
                data[key] = item
            return data
        config = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(config, dict) or not config or set(config) - CONFIGURATION_KEYS.get(suite, set()):
            raise ValueError(suite + ': unsupported configuration capability/key; allowed: ' + ','.join(sorted(CONFIGURATION_KEYS.get(suite, set()))))
        if any(entry['name'] == name for entry in profiles.get(suite, [])):
            raise ValueError('Duplicate configuration profile name: ' + label)
        profiles.setdefault(suite, []).append({'name': name, 'values': config})
        if len(profiles[suite]) > 16:
            raise ValueError('At most 16 named configurations per suite')
    return profiles


def cv_grid(key, size):
    start, end = (.1, 10.) if key == 'Cs' else (1., .01)
    if size == 3: return [.1, 1., 10.] if key == 'Cs' else [1., .1, .01]
    return [start if i == 0 else end if i == size-1 else 10 ** (math.log10(start) + i * (math.log10(end) - math.log10(start)) / (size-1)) for i in range(size)]


def configuration_exclusion(suite, case, config):
    """Pure-stdlib mirror of direct runner capability masks; no numerical import."""
    if any(type(value) is not (int if key in ('folds', 'grid_size') else str) for key, value in config.items()):
        return 'configuration values must use exact CLI scalar types'
    dtype, weights = config.get('dtype', 'float64'), config.get('weights', 'none')
    solver, sparse = config.get('solver', 'default'), config.get('sparse', 'dense')
    if dtype not in ('float32', 'float64') or weights not in ('none', 'analytic'):
        return 'dtype/weights values are unsupported by the direct CLI'
    if sparse != 'dense': return 'sparse capability is unsupported by the reviewed fixture/oracle; no implicit densification'
    if suite == 'model-families':
        if dtype != 'float64' and not case.startswith(('glm:', 'glm_inference:', 'inference:')):
            return 'float32 capability requires glm, glm_inference or inference fixture'
        if weights != 'none' and not case.startswith(('glm:', 'inference:')):
            return 'analytic weights require glm or inference with a matched weighted oracle'
        if solver != 'default':
            accepted = set()
            if case.startswith(('glm:', 'glm_inference:')): accepted = {'auto', 'irls', 'newton', 'lbfgs', 'fista'}
            elif case.startswith('penalty:'):
                penalty = case.split(':')[1]
                accepted = {'fista', 'fista_bb', 'auto'}
                if penalty == 'l2': accepted |= {'newton', 'lbfgs'}
                if penalty in ('l1', 'l2', 'elasticnet'): accepted.add('admm')
            if solver not in accepted: return 'solver capability is not supported by this case/oracle'
    elif suite == 'learning-families':
        if dtype != 'float64' and case not in LEARNING_FLOAT32:
            return 'float32 capability has no reviewed learning fixture/oracle for this case'
        if weights != 'none': return 'analytic weights are not matched here; use the dedicated public KDE runner'
        if solver != 'default' and (case != 'pca' or solver not in ('full', 'covariance', 'randomized')):
            return 'solver capability supports PCA full/covariance/randomized only'
        covariance = config.get('covariance_type', 'default')
        if covariance != 'default' and (case != 'gaussian_mixture' or covariance not in ('diag', 'full', 'tied', 'spherical')):
            return 'covariance-type capability supports GaussianMixture only'
    elif suite == 'six-family-cv':
        if dtype != 'float64' or weights != 'none': return 'CV capability requires float64 and no weights'
        for key, maximum in (('folds', 5), ('grid_size', 9)):
            if key in config and (type(config[key]) is not int or not 2 <= config[key] <= maximum):
                return f'{key} requires an integer in 2..{maximum} under the reviewed CV resource policy'
        if config.get('folds', 3) * config.get('grid_size', 3) > 27:
            return 'CV folds*grid_size exceeds the reviewed 27 candidate-fold fit resource budget'
    return None


def configuration_variants(suite, args, profiles):
    result = []
    for profile in profiles.get(suite, [{'name': 'baseline', 'values': {}}]):
        values = profile['values']
        if ('dtype' in values and args.dtypes is not None) or ('weights' in values and args.weights is not None):
            raise ValueError('Do not combine named dtype/weights with the same global axis')
        dtypes = args.dtypes or ([values['dtype']] if 'dtype' in values else ['float64'])
        weights = args.weights or ([values['weights']] if 'weights' in values else ['none'])
        if suite not in CONFIGURATION_KEYS:
            return [{'name': 'baseline', 'values': {}}]
        # No knobs are added to unchanged default invocations.
        for dtype in dtypes:
            for weight in weights:
                selected = dict(values)
                if args.dtypes is not None: selected['dtype'] = dtype
                if args.weights is not None: selected['weights'] = weight
                result.append({'name': profile['name'], 'values': selected})
    return result


def build_plan(args):
    entries = registry()
    suites = args.suites or list(RUNNERS)
    selectors = parse_selectors(args.cases, entries)
    _unique(suites, 'suites'); _unique(args.backends, 'backends')
    if set(selectors) - set(suites):
        raise ValueError('Case selectors must belong to selected suites')
    if args.scale != 'representative' and not args.suites:
        raise ValueError('Medium/large scaling requires explicit --suites and selective --cases')
    explicit_shapes = parse_shapes(args.shapes, suites)
    profiles = parse_configurations(args.config, suites)
    if explicit_shapes and args.scale != 'representative':
        raise ValueError('Explicit --shapes cannot be combined with --scale medium/large')
    if args.seeds is not None:
        _unique(args.seeds, 'seeds')
        if any(seed < 0 or seed >= 2**32 - 1 for seed in args.seeds):
            raise ValueError('seeds must be nonnegative integers below 2**32-1')
    if args.selection_seeds is not None:
        _unique(args.selection_seeds, 'selection-seeds')
        if any(seed < 0 or seed >= 2**32 - 1 for seed in args.selection_seeds):
            raise ValueError('selection-seeds must be nonnegative integers below 2**32-1')
        if args.seeds is not None and len(args.seeds) > 1:
            raise ValueError('Multiple campaign seeds with selection-seeds would repeat selection trials; use separate case batches')
    if args.dtypes is not None: _unique(args.dtypes, 'dtypes', 2)
    if args.weights is not None: _unique(args.weights, 'weights', 2)
    if not 1 <= args.max_invocations <= 64:
        raise ValueError('max-invocations must be in 1..64; use separate bounded batches for larger matrices')
    if args.batch_count < 1 or not 0 <= args.batch_index < args.batch_count:
        raise ValueError('Require batch-count>=1 and 0<=batch-index<batch-count')
    if args.knockoff_p < 2 or args.knockoff_p > 64 or not 1 <= args.max_pairwise_elements <= 4000000:
        raise ValueError('Resource policy requires knockoff-p in 2..64 and max-pairwise-elements in 1..4000000')
    if args.resamples is not None and not 2 <= args.resamples <= 1000:
        raise ValueError('resamples must be in 2..1000 under the campaign resource policy')
    if args.targets is not None and not 1 <= args.targets <= 16:
        raise ValueError('targets must be in 1..16 under the campaign resource policy')
    if args.alpha is not None and (not math.isfinite(args.alpha) or args.alpha < 0):
        raise ValueError('alpha must be finite and nonnegative')
    if args.device is not None and args.device < 0:
        raise ValueError('device ordinal must be nonnegative')
    scoped_options = [('resamples', 'statistical-families'), ('targets', 'workflow-families'), ('alpha', 'gaussian-refresh'), ('selection_seeds', 'learning-families')]
    for option, suite in scoped_options:
        if getattr(args, option) is not None and suite not in suites:
            raise ValueError('--' + option.replace('_', '-') + ' requires selected suite ' + suite)
    if args.knockoff_p != 20 and ('learning-families' not in suites or not any(c.startswith('knockoff') for c in selectors.get('learning-families', entries['learning-families']['case_ids']))):
        raise ValueError('knockoff-p requires selected learning knockoff cases')
    if args.max_pairwise_elements != 4000000 and 'learning-families' not in suites:
        raise ValueError('max-pairwise-elements requires selected learning suite')
    if args.resamples is not None and not any(c.startswith('resampling/') for c in selectors.get('statistical-families', entries['statistical-families']['case_ids'])):
        raise ValueError('resamples requires selected resampling cases')
    if args.targets is not None and 'multitarget' not in selectors.get('workflow-families', entries['workflow-families']['case_ids']):
        raise ValueError('targets requires selected multitarget workflow')
    if args.alpha is not None and 'ridge' not in selectors.get('gaussian-refresh', entries['gaussian-refresh']['case_ids']):
        raise ValueError('alpha requires selected ridge case; OLS is unpenalized')
    if args.selection_seeds is not None and not any(c.startswith('knockoff') for c in selectors.get('learning-families', entries['learning-families']['case_ids'])):
        raise ValueError('selection-seeds requires at least one selected knockoff case')
    limits = cpu_limits()
    if 1 not in args.cpu_threads or len(args.cpu_threads) != len(set(args.cpu_threads)):
        raise ValueError('--cpu-threads must include baseline 1 without duplicates')
    if max(args.cpu_threads) > 1 and 'numpy' not in args.backends:
        raise ValueError('Optional CPU thread profiles require NumPy in --backends')
    if max(args.cpu_threads) > limits['thread_cap']:
        raise ValueError('CPU thread profile exceeds actual affinity/cgroup quota')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800:
        raise ValueError('Per-invocation timeout must be > 0 and <= 1800 seconds')
    source = _plan_source()
    jobs, exclusions = [], []
    for threads in sorted(args.cpu_threads):
        for suite in suites:
            for variant in configuration_variants(suite, args, profiles):
                cases = selectors.get(suite, entries[suite]['case_ids'])
                if suite == 'six-family-cv' and suite in selectors and cases != entries[suite]['case_ids']:
                    raise ValueError('CV runner only supports its six-family registry, not case filtering')
                if args.scale != 'representative' and args.profile != 'expanded' and suite != 'six-family-cv' and (suite not in selectors or len(cases) > 8):
                    raise ValueError('Medium/large requires 1-8 exact cases per selected suite')
                dtypes = ([variant['values'].get('dtype', 'float64')] if suite in CONFIGURATION_KEYS else args.dtypes or (['float32', 'float64'] if suite in ('gaussian-refresh', 'classification-metrics') else ['float64']))
                weights = ([variant['values'].get('weights', 'none')] if suite in CONFIGURATION_KEYS else args.weights or (['none', 'analytic'] if suite == 'gaussian-refresh' else ['none']))
                if suite not in ('gaussian-refresh', 'classification-metrics', *CONFIGURATION_KEYS) and dtypes != ['float64']:
                    raise ValueError(f'{suite}: unsupported dtype capability; reviewed fixture/oracle supports float64 only')
                if suite not in ('gaussian-refresh', *CONFIGURATION_KEYS) and weights != ['none']:
                    raise ValueError(f'{suite}: unsupported weight capability; no reviewed weighted fixture/oracle')
                if args.device not in (None, 0) and suite == 'six-family-cv':
                    raise ValueError('six-family-cv: unsupported device ordinal capability; audited runner uses device 0')
                if suite in explicit_shapes:
                    shapes = explicit_shapes[suite]
                elif args.scale != 'representative':
                    shapes = [SCALE_SHAPES[suite][('representative', 'medium', 'large').index(args.scale)]]
                else:
                    shapes = SCALE_SHAPES[suite] if args.profile == 'expanded' else DEFAULT_SHAPES[suite]
                # Direct model/learning runners take one shape; multi-size runners may
                # combine only equal-p shapes. Expanded jobs additionally bound cases.
                groups = [shapes] if suite not in ('model-families', 'learning-families') and len({p for _, p in shapes}) == 1 else [[shape] for shape in shapes]
                if suite == 'gaussian-refresh': groups = [shapes]
                if args.profile == 'expanded': groups = [[shape] for shape in shapes]
                default_warmup, default_repeats = (2, 5) if suite in ('gaussian-refresh', 'classification-metrics') else (1, 5) if suite == 'statistical-families' else (1, 3)
                warmup = default_warmup if args.warmup is None else args.warmup
                repeats = default_repeats if args.repeats is None else args.repeats
                default_seed = 20260807 if suite == 'six-family-cv' else 42 if suite == 'learning-families' else 20261020
                seeds = args.seeds or ([17, 42, 20261020] if args.profile == 'expanded' else [default_seed])
                if args.selection_seeds is not None and len(seeds) > 1:
                    raise ValueError('expanded multi-seed profile with selection-seeds repeats trials; select one --seeds value')
                for shape_group in groups:
                    allowed = []
                    for case in cases:
                        reason = configuration_exclusion(suite, case, variant['values'])
                        if reason:
                            raise ValueError(f'{suite}/{case}: unsupported configuration: ' + reason)
                        failures = [(n, p, _shape_reason(suite, case, n, p, args)) for n, p in shape_group]
                        failures = [(n, p, reason) for n, p, reason in failures if reason]
                        if failures:
                            if args.profile != 'expanded' or suite in explicit_shapes or suite in selectors:
                                raise ValueError(f'{suite}/{case}: ' + failures[0][2])
                            exclusions.extend({'suite': suite, 'case': case, 'n': n, 'p': p, 'reason': reason,
                                               'status': 'capability-excluded', 'cpu_threads': threads, 'seeds': seeds,
                                               'backends': [b for b in args.backends if threads == 1 or b == 'numpy']}
                                              for n, p, reason in failures)
                        else: allowed.append(case)
                    chunks = [allowed] if args.profile != 'expanded' or suite == 'six-family-cv' else [allowed[i:i+8] for i in range(0, len(allowed), 8)]
                    for chunk in chunks:
                        if not chunk: continue
                        for seed in seeds:
                            for backend in args.backends:
                                if threads != 1 and backend != 'numpy': continue
                                n, p = shape_group[0]
                                argv = ['--backends', backend, '--require-backends', '--seed', str(seed), '--warmup', str(warmup), '--repeats', str(repeats)]
                                declared_arguments = {'seed': seed} if (args.seeds is not None or args.profile == 'expanded') and suite == 'statistical-families' else {}
                                if suite == 'gaussian-refresh':
                                    argv += ['--models', *chunk, '--shapes', *[f'{nn}x{pp}' for nn, pp in shape_group], '--dtypes', *dtypes, '--weights', *weights]
                                    if args.alpha is not None:
                                        argv += ['--alpha', str(args.alpha)]; declared_arguments['alpha'] = args.alpha
                                elif suite == 'classification-metrics':
                                    argv += ['--operations', *chunk, '--sizes', *[str(nn) for nn, _ in shape_group], '--dtypes', *dtypes, '--scenarios', *SCENARIOS]
                                elif suite in ('model-families', 'learning-families'):
                                    argv += ['--cases', *chunk, '--n', str(n), '--p', str(p)]
                                    if suite == 'learning-families':
                                        argv += ['--knockoff-p', str(args.knockoff_p), '--max-pairwise-elements', str(args.max_pairwise_elements)]
                                        if args.knockoff_p != 20: declared_arguments['knockoff_p'] = args.knockoff_p
                                        if args.selection_seeds is not None:
                                            argv += ['--selection-seeds', *map(str, args.selection_seeds)]
                                            declared_arguments['selection_seeds'] = args.selection_seeds
                                else:
                                    argv += ['--sizes', *[str(nn) for nn, _ in shape_group], '--features', str(p)]
                                    if suite != 'six-family-cv': argv += ['--cases', *chunk]
                                    if suite == 'statistical-families':
                                        argv += ['--resamples', str(args.resamples or 100)]
                                        if args.resamples is not None: declared_arguments['resamples'] = args.resamples
                                    if suite == 'workflow-families':
                                        argv += ['--targets', str(args.targets or 4)]
                                        if args.targets is not None: declared_arguments['targets'] = args.targets
                                if args.device is not None:
                                    flag = '--cuda-device' if suite in ('gaussian-refresh', 'statistical-families', 'workflow-families') else '--device'
                                    if suite != 'six-family-cv':
                                        argv += [flag, str(args.device)]
                                        # Device observations are independently checked by qualification;
                                        # raw CLI metadata checks are added for producers that emit them.
                                        if suite in ('gaussian-refresh', 'statistical-families', 'learning-families', 'workflow-families'):
                                            declared_arguments[flag[2:].replace('-', '_')] = args.device
                                for key, value in variant['values'].items():
                                    argv += ['--' + key.replace('_', '-'), str(value)]
                                    declared_arguments[key] = value
                                expected = [row for nn, pp in shape_group for row in expected_rows(suite, backend, chunk, nn, pp, seed,
                                            dtypes=dtypes, weights=weights, knockoff_p=args.knockoff_p, selection_seeds=args.selection_seeds)]
                                if variant['values']:
                                    for row in expected: row['configuration'] = dict(variant['values'])
                                for row in expected:
                                    parameters = {}
                                    if suite == 'gaussian-refresh' and args.alpha is not None:
                                        parameters['alpha'] = 0.0 if row['case'] == 'ols' else args.alpha
                                    if suite == 'statistical-families' and args.resamples is not None and row['case'].startswith('resampling/'):
                                        parameters['resamples'] = args.resamples
                                    if suite == 'workflow-families' and args.targets is not None and row['case'] == 'multitarget':
                                        parameters['targets'] = args.targets
                                    if parameters: row['parameters'] = parameters

                                configuration = {'name': variant['name'], 'axes': variant['values'], 'seed': seed, 'selection_seeds': args.selection_seeds, 'dtypes': dtypes, 'weights': weights,
                                    'device_ordinal': args.device or 0, 'shapes': [{'n': nn, 'p': pp} for nn, pp in shape_group],
                                    'arguments': declared_arguments, 'warmup': warmup, 'repeats': repeats}
                                identity = {'suite': suite, 'backend': backend, 'cpu_threads': threads, 'cases': chunk, 'configuration': configuration}
                                effective = dict(identity, configuration={k: v for k, v in configuration.items() if k != 'name'})
                                key = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()[:16]
                                job_id = f'{len(jobs):04d}-{suite}-{backend}-t{threads}-s{seed}-{key}'
                                slots = [{'id': job_id + f'-r{i:05d}', 'row': row, 'configuration': configuration, 'source': source}
                                         for i, row in enumerate(expected)]
                                jobs.append({'id': job_id, 'configuration_id': key, **identity, 'scale': args.scale, 'profile': args.profile,
                                    'seed': seed, 'shapes': configuration['shapes'], 'module': 'dev.benchmarks.' + RUNNERS[suite],
                                    'argv': argv, 'expected_rows': expected, 'expected_slots': slots, 'expected_source': source,
                                    'expected_arguments': declared_arguments, 'expected_row_count': len(expected),
                                    'expected_repeats': repeats, 'expected_warmup': warmup,
                                    'require_warmup_metadata': args.warmup is not None or args.profile == 'expanded' or bool(variant['values']),
                                    'input_precision': sorted({r['input_dtype'] for r in expected if 'input_dtype' in r}),
                                    'compute_precision': 'Not inferred from input/output dtype; review scoped numerical observations in raw artifact',
                                    'resource_bounds': {'max_pairwise_elements': args.max_pairwise_elements if suite == 'learning-families' else None,
                                        'case_policies': {case: learning_resource_policy(case) for case in chunk} if suite == 'learning-families' else {},
                                        'selection_q': .2 if any(c.startswith('knockoff') for c in chunk) else None,
                                        'bootstrap_replicates': 20 if 'kde_bootstrap' in chunk else None,
                                        'timeout_seconds': args.timeout}})
    if len(jobs) > 10000:
        raise ValueError('Finite campaign exceeds 10000 invocations; narrow seeds/shapes/cases before batching')
    if len({j['configuration_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate campaign configuration would overwrite/repeat an evidence slot')
    all_count, all_rows = len(jobs), sum(j['expected_row_count'] for j in jobs)
    jobs = [job for index, job in enumerate(jobs) if index % args.batch_count == args.batch_index]
    if not jobs:
        raise ValueError('Selected batch has no supported invocations')
    if args.execute and len(jobs) > args.max_invocations:
        raise ValueError(f'Selected execution has {len(jobs)} invocations, exceeding max-invocations={args.max_invocations}; choose explicit bounded --batch-count/--batch-index or a reviewed maximum (<=64)')
    return {'schema_version': 'statgpu-qualification-plan-v1', 'plan_only': not args.execute,
            'serial': True, 'profile': args.profile, 'source_locked': bool(args.config or args.seeds or args.shapes or args.profile == 'expanded' or args.dtypes or args.weights or args.selection_seeds or args.resamples is not None or args.targets is not None or args.alpha is not None or args.device is not None or args.knockoff_p != 20 or args.max_pairwise_elements != 4000000), 'cpu_limits': limits, 'source': source, 'jobs': jobs,
            'expected_total_rows': sum(j['expected_row_count'] for j in jobs),
            'batch': {'index': args.batch_index, 'count': args.batch_count, 'all_invocations': all_count, 'all_expected_rows': all_rows},
            'execution_budget': {'max_invocations': args.max_invocations, 'selected_invocations': len(jobs),
                                 'within_budget': len(jobs) <= args.max_invocations,
                                 'worst_case_timeout_seconds': len(jobs) * args.timeout},
            'capability_exclusions': exclusions,
            'excluded_slot_count': sum(len(e['seeds']) * len(e['backends']) for e in exclusions),
            'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'canonical_dashboard_ready': False,
            'note': 'Default representative preserves 21 invocations/1692 registered rows. Expanded is an explicit finite supported matrix, not exhaustive APIs. Exclusions are not passes; thread profiles and seed trials remain separate.'}

def row_identity(row, suite, document, configuration=None):
    if not isinstance(row, dict):
        raise ValueError('Result row must be an object')
    problem = row.get('problem', {})
    case = row.get('model_id', row.get('case_id', row.get('case', row.get('operation'))))
    if isinstance(case, dict):
        case = case.get('model_name')
    if suite == 'gaussian-refresh':
        case = case or problem.get('parameters', {}).get('model')
    backend = row.get('backend', row.get('requested_backend'))
    if isinstance(backend, dict):
        backend = backend.get('requested', backend.get('executed'))
    if suite == 'statistical-families' and case in HOST_ONLY and backend != 'numpy' and row.get('status') == 'not-applicable':
        return {'case': case, 'backend': backend, 'not_applicable': True}
    shape = problem.get('shape', problem)
    identity = {'case': case, 'backend': backend, 'n': shape.get('n'), 'p': shape.get('p'),
                'input_dtype': problem.get('input_dtype', problem.get('dtype')), 'seed': problem.get('seed')}
    if suite == 'gaussian-refresh':
        identity['weighted'] = problem.get('parameters', {}).get('weighted')
    elif suite == 'classification-metrics':
        identity.pop('p')
        identity['scenario'] = problem.get('scenario')
    elif suite == 'six-family-cv':
        # CV input precision/seed are declared by protocol/coverage, not output.
        identity.update(n=row.get('n_samples'), p=row.get('n_features'),
                        input_dtype=document.get('coverage', {}).get('dtype'),
                        seed=(row.get('protocol', {}).get('seeds') or [None])[0]
                        if len(row.get('protocol', {}).get('seeds', [])) == 1 else None)
    if configuration:
        observed = problem if suite != 'six-family-cv' else row
        selected = {key: observed.get(key) for key in configuration}
        if suite == 'six-family-cv':
            if 'folds' in configuration:
                selected['folds'] = row.get('case', {}).get('cv', {}).get('fold_count')
            if 'grid_size' in configuration:
                grids = row.get('case', {}).get('grid', {}).get('parameters', {})
                axes = [grids[key] for key in ('alphas', 'Cs', 'alpha_grid', 'penalties') if key in grids]
                selected['grid_size'] = len(axes[0]) if len(axes) == 1 and isinstance(axes[0], list) else None
        identity['configuration'] = selected
    return identity


def _effective_parameters(row, suite, job):
    problem = row.get('problem', {})
    actual = {}
    if suite == 'gaussian-refresh' and 'alpha' in job.get('expected_arguments', {}):
        actual['alpha'] = problem.get('parameters', {}).get('alpha')
    if suite == 'statistical-families' and 'resamples' in job.get('expected_arguments', {}) and row.get('case_id', '').startswith('resampling/'):
        actual['resamples'] = problem.get('resamples')
    if suite == 'workflow-families' and 'targets' in job.get('expected_arguments', {}) and row.get('case') == 'multitarget':
        actual['targets'] = row.get('problem_extra', {}).get('n_targets')
    return actual


def timing_warmups(section):
    """Return observed warmups and timing regions lacking inherited metadata."""
    observed, missing = [], []
    sample_fields = {'raw_seconds', 'samples_seconds', 'seconds', 'repeat_samples'}
    def visit(value, inherited=None):
        if isinstance(value, dict):
            current = inherited
            if 'warmup' in value:
                current = value['warmup']
                observed.append(current)
            if sample_fields.intersection(value) and current is None:
                missing.append(value.get('scope', 'unnamed timing region'))
            for child in value.values():
                if isinstance(child, (dict, list)): visit(child, current)
        elif isinstance(value, list):
            for child in value: visit(child, inherited)
    visit(section)
    return observed, missing


def assess(output, job, exit_code, expected_source):
    from dev.benchmarks.benchmark_qualification import evaluate_artifact
    try:
        document = read_json(output)
        qualified = evaluate_artifact(document, job['suite'], job['backend'], exit_code=exit_code,
                                      expected_source=expected_source, expected_cases=job['cases'])
        raw_rows = document.get('results', document.get('rows', []))
        axes = job.get('configuration', {}).get('axes', {})
        observed = [row_identity(r, job['suite'], document, axes) for r in raw_rows]
        for row, identity in zip(raw_rows, observed):
            parameters = _effective_parameters(row, job['suite'], job)
            if parameters: identity['parameters'] = parameters
        encode = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
        wanted, actual = Counter(map(encode, job['expected_rows'])), Counter(map(encode, observed))
        inventory = {'expected_count': sum(wanted.values()), 'observed_count': sum(actual.values()),
                     'exact_match': wanted == actual, 'expected_rows': job['expected_rows'],
                     'expected_slots': job.get('expected_slots', []), 'expected_source': job.get('expected_source'),
                     'configuration': job.get('configuration', {}),
                     'observed_rows': observed, 'missing': list((wanted - actual).elements()),
                     'unexpected': list((actual - wanted).elements())}
        if wanted != actual:
            qualified['correctness']['status'] = 'failed'
            qualified['correctness']['errors'].append('Expanded row identity/multiplicity mismatch')
        for row in qualified.get('rows', []):
            if any(t['sample_count'] != job['expected_repeats'] for t in row.get('timing_records', [])):
                qualified['correctness']['status'] = 'failed'
                qualified['correctness']['errors'].append('Timing sample count differs from selected CLI repeats')
        for row in raw_rows:
            if row.get('status') == 'not-applicable': continue
            section = row.get('clean_public_fit') if job['suite'] == 'six-family-cv' else row.get('timing')
            warmups, missing = timing_warmups(section)
            if (any(type(value) is not int or value != job['expected_warmup'] for value in warmups)
                    or (job.get('require_warmup_metadata') and (not warmups or missing))):
                qualified['correctness']['status'] = 'failed'
                qualified['correctness']['errors'].append('Observed timing warmup differs from selected CLI warmup or is missing')
        arguments = document.get('benchmark', {}).get('arguments', {})
        for key, value in job.get('expected_arguments', {}).items():
            if type(arguments.get(key)) is not type(value) or arguments.get(key) != value:
                qualified['correctness']['status'] = 'failed'
                qualified['correctness']['errors'].append('Raw CLI configuration differs for ' + key)
        if '--device' in job['argv'] or '--cuda-device' in job['argv']:
            expected_device = 'cpu' if job['backend'] == 'numpy' else f"cuda:{job['configuration']['device_ordinal']}"
            for row in raw_rows:
                if row.get('status') == 'not-applicable': continue
                devices, timing_devices = [], []
                def visit(value, timing=False):
                    if isinstance(value, dict):
                        if (timing and job['suite'] == 'workflow-families' and row.get('case') == 'formula'
                                and value.get('scope') == 'host-formula-model-matrix-preparation-only'):
                            if value.get('synchronized_device') != 'cpu': devices.append('invalid host formula timing device')
                            return
                        for key, child in value.items():
                            if key in ('device', 'executed_device', 'observed_array_device', 'synchronized_device') and isinstance(child, str):
                                devices.append(child)
                                if timing and key == 'synchronized_device': timing_devices.append(child)
                            elif isinstance(child, (dict, list)): visit(child, timing)
                    elif isinstance(value, list):
                        for child in value: visit(child, timing)
                # Inspect result placement and synchronization, not host oracle/report
                # arrays or intentionally mixed host/device numerical trace stages.
                for key in ('device', 'executed_device'):
                    if isinstance(row.get(key), str): devices.append(row[key])
                visit(row.get('backend', {}))
                execution_device = row.get('execution', {}).get('observed_array_device')
                if isinstance(execution_device, str): devices.append(execution_device)
                visit(row.get('timing', {}), timing=True)
                if (not devices or any(device != expected_device for device in devices)
                        or (job['backend'] != 'numpy' and not timing_devices)):
                    qualified['correctness']['status'] = 'failed'
                    qualified['correctness']['errors'].append('Observed concrete device differs from selected ordinal or is absent')
        if axes and job['suite'] == 'six-family-cv' and 'grid_size' in axes:
            for row in raw_rows:
                grids = row.get('case', {}).get('grid', {}).get('parameters', {})
                grid_key = 'Cs' if row.get('model_id') == 'LogisticRegressionCV' else {'PenalizedGLM_CV': 'alpha_grid', 'CoxPHCV': 'penalties'}.get(row.get('model_id'), 'alphas')
                expected_grid = cv_grid(grid_key, axes['grid_size'])
                actual_grid = grids.get(grid_key, [])
                if (len(actual_grid) != len(expected_grid) or
                        any(type(value) not in (int, float) or not math.isclose(value, wanted, rel_tol=1e-12, abs_tol=0.) for value, wanted in zip(actual_grid, expected_grid))):
                    qualified['correctness']['status'] = 'failed'
                    qualified['correctness']['errors'].append('Actual CV candidate grid differs from selected configuration')
        qualified['expanded_inventory'] = inventory
        return qualified
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {'correctness': {'status': 'failed', 'errors': [f'Missing/malformed artifact: {type(exc).__name__}: {exc}']},
                'expanded_inventory': {'expected_count': job['expected_row_count'], 'exact_match': False},
                'performance': {'eligible': False, 'qualification': 'diagnostic_only',
                                'exclusive_gpu_ownership': 'UNVERIFIED_EXCLUSIVITY'}}


def source_files(root=ROOT):
    paths = [p for p in (root / 'statgpu').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    dependencies = [*RUNNERS.values(), 'run_qualification_campaign', 'benchmark_qualification',
                    'benchmark_cv_models', 'cv_external', 'cv_source', 'learning_execution_trace',
                    'run_full_benchmark_matrix', 'independent_benchmark_registry', 'run_independent_benchmarks']
    paths += [root / 'dev/benchmarks' / (name + '.py') for name in dependencies]
    paths += [root / name for name in ('dev/benchmarks/full_coverage_manifest.json',
              'dev/benchmarks/cv_source_schema.json', 'dev/benchmarks/independent_benchmark_catalog.json', 'pyproject.toml', '.claude/skills/benchmark/schema.md')]
    return sorted(set(p for p in paths if p.is_file()))


def fingerprint(root=ROOT):
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in source_files(root)}
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {'git_commit': git('rev-parse', 'HEAD'), 'working_tree': 'clean' if git('status', '--porcelain') == '' else 'dirty-or-unknown',
            'content_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), 'files_sha256': hashes}


def source_changed(before, after):
    return any(before.get(key) != after.get(key) for key in ('git_commit', 'content_sha256'))


def resource_guard(destination, threads, min_free_mib=512):
    if threads > cpu_limits()['thread_cap']:
        return 'CPU affinity/quota no longer permits selected thread profile'
    if shutil.disk_usage(destination).free < min_free_mib * 1024**2:
        return 'Output filesystem free-space reserve exhausted'
    for path in cgroup_paths('memory'):
        for limit_name, usage_name in (('memory.max', 'memory.current'),
                                       ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            try:
                maximum = (path / limit_name).read_text().strip()
                current = int((path / usage_name).read_text())
                if maximum != 'max' and int(maximum) - current < 256 * 1024**2:
                    return 'Cgroup memory headroom below 256 MiB reserve'
            except (OSError, ValueError):
                pass
    return None


def launch(command, log, timeout, env, guard):
    start = time.monotonic()
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == 'posix'))
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() - start >= timeout:
                    reason = 'timeout'
                    break
                violation = guard()
                if violation:
                    reason = 'resource_guard: ' + violation
                    break
                time.sleep(min(0.1, timeout / 5))
        except KeyboardInterrupt:
            reason = 'interrupted'
        finally:
            leader_exited = process.poll() is not None
            if os.name == 'posix':
                # start_new_session made this PID our own process-group ID.
                # Always clean that group, even after poll() reaped its leader:
                # descendants can otherwise keep inherited log/artifact handles.
                # Never target an NVML/Docker PID or any externally found group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    if leader_exited and reason is None:
                        reason = 'process_group_cleanup: descendants outlived exited leader'
                except ProcessLookupError:
                    pass  # The entire owned group has already disappeared.
            elif not leader_exited:
                process.kill()
            rc = process.wait()
    return {'exit_code': rc, 'reason': reason, 'elapsed_seconds': time.monotonic() - start}


def freeze_and_archive(destination):
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    hashes = {p.relative_to(destination).as_posix(): sha256_file(p) for p in files}
    write_json(destination / 'SHA256SUMS.json', hashes)
    for path in [*files, destination / 'SHA256SUMS.json']:
        path.chmod(0o444)
    archive = destination.with_name(destination.name + '.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as bundle:
            bundle.add(destination, arcname=destination.name, recursive=True)
    digest = sha256_file(archive)
    checksum = archive.with_name(archive.name + '.sha256')
    with checksum.open('x') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    archive.chmod(0o444)
    checksum.chmod(0o444)
    return archive


def validate_destination(path):
    destination = path.expanduser().resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError('Output must be outside the checkout')
    for candidate in (destination, destination.with_name(destination.name + '.tar.gz'), destination.with_name(destination.name + '.tar.gz.sha256')):
        if candidate.exists():
            raise ValueError('Refusing existing output/archive: ' + str(candidate))
    if not destination.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    return destination


def execute(plan, destination, timeout):
    budget = plan.get('execution_budget', {}).get('max_invocations', 21)
    if not 1 <= budget <= 64 or len(plan['jobs']) > budget:
        raise ValueError('Selected execution exceeds the bounded invocation budget; partition the plan before execution')
    if len({job['id'] for job in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('Duplicate execution job identity')
    destination.mkdir(exist_ok=False)
    report = {'schema_version': 'statgpu-qualification-campaign-v1', 'status': 'failed', 'results': [],
              'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'performance_eligible': False,
              'canonical_dashboard_ready': False, 'comparisons': [],
              'capability_exclusions': plan.get('capability_exclusions', []),
              'excluded_slot_count': plan.get('excluded_slot_count', 0),
              'coverage_scope': 'Selected supported slots only; excluded capability/resource configurations are not passes',
              'batch': plan.get('batch', {}), 'execution_budget': plan.get('execution_budget', {}),
              'gpu_process_ownership': 'unverified; unknown or namespace-hidden PIDs do not fail correctness',
              'gpu_peak_memory': 'not_measured', 'stop_reason': None}
    before = None
    try:
        write_json(destination / 'plan.json', plan)
        before = fingerprint()
        write_json(destination / 'source-before.json', before)
        if not before.get('git_commit'):
            raise ValueError('Exact source commit unavailable')
        if plan.get('source_locked') and source_changed(plan['source'], before):
            raise ValueError('Source changed since configuration plan generation')
        for relative in before['files_sha256']:
            target = destination / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream, (ROOT / relative).open('rb') as original:
                shutil.copyfileobj(original, stream, length=1024 * 1024)
            if sha256_file(target) != before['files_sha256'][relative]:
                raise ValueError('Source changed during snapshot copy')
        bootstrap = destination / 'child_bootstrap.py'
        with bootstrap.open('x') as stream:
            stream.write(CHILD_BOOTSTRAP)
        write_json(destination / 'runtime.json', {'python': sys.version, 'platform': platform.platform(),
            'cpu_limits': cpu_limits(), 'memory_policy': 'Child RSS is process high-water, GPU allocator peaks not measured',
            'thread_env_before_import': True})
        for job in plan['jobs']:
            now = fingerprint()
            if source_changed(before, now):
                report['stop_reason'] = 'source_changed'
                break
            violation = resource_guard(destination, job['cpu_threads'])
            if violation:
                report['stop_reason'] = 'resource_guard: ' + violation
                break
            directory = destination / job['id']
            directory.mkdir()
            output = directory / 'raw.json'
            argv = [*job['argv'], '--output', str(output)]
            command = [sys.executable, str(bootstrap), str(ROOT), job['module'], job['backend'],
                       str(job['cpu_threads']), str(directory / 'runtime'), *argv]
            environment = os.environ.copy()
            environment.update({key: str(job['cpu_threads']) for key in THREAD_ENV})
            environment.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONDONTWRITEBYTECODE='1')
            write_json(directory / 'invocation.json', {'command': command, 'thread_env': {k: environment[k] for k in THREAD_ENV}, 'expected_rows': job['expected_rows'], 'expected_slots': job.get('expected_slots', []), 'configuration': job.get('configuration', {}), 'source': now})
            write_json(directory / 'source-before.json', now)
            try:
                process = launch(command, directory / 'stdout-stderr.log', timeout, environment,
                                 lambda: resource_guard(destination, job['cpu_threads']))
            except Exception as exc:
                process = {'exit_code': None, 'reason': f'{type(exc).__name__}: {exc}'}
            write_json(directory / 'process.json', process)
            after = fingerprint()
            write_json(directory / 'source-after.json', after)
            qualification = assess(output, job, process['exit_code'], before['git_commit'])
            if process.get('reason'):
                qualification['correctness']['status'] = 'failed'
                qualification['correctness']['errors'].append(process['reason'])
            runtime_file = directory / 'runtime.after.json'
            try:
                runtime = read_json(runtime_file)
                verified = runtime.get('before', {}).get('cap_verified') and runtime.get('after', {}).get('cap_verified')
                if not verified:
                    raise ValueError('Actual thread pools not verified before and after workload')
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                qualification['correctness']['status'] = 'failed'
                qualification['correctness']['errors'].append(str(exc))
            if source_changed(before, after):
                qualification['correctness']['status'] = 'failed'
                qualification['correctness']['errors'].append('Source changed during invocation')
                report['stop_reason'] = 'source_changed'
            write_json(directory / 'qualification.json', qualification)
            status = qualification['correctness']['status']
            report['results'].append({'id': job['id'], 'status': status, 'exit_code': process['exit_code'],
                                      'qualification': str((directory / 'qualification.json').relative_to(destination))})
            print(job['id'] + ': ' + status, flush=True)
            if process.get('reason') == 'interrupted':
                report['stop_reason'] = 'interrupted'
            if process['exit_code'] == 86 or str(process.get('reason', '')).startswith('resource_guard:'):
                report['stop_reason'] = process.get('reason') or 'resource_guard: child thread pool violation'
            if report['stop_reason']:
                break
        if not report['stop_reason'] and len(report['results']) == len(plan['jobs']) and all(r['status'] == 'complete' for r in report['results']):
            report['status'] = 'complete'
    except BaseException as exc:
        report['stop_reason'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            after = fingerprint()
            write_json(destination / 'source-after.json', after)
            report['source_changed'] = before is not None and source_changed(before, after)
            if report['source_changed']:
                report['status'] = 'failed'
                report['stop_reason'] = report['stop_reason'] or 'source_changed'
        except Exception as exc:
            report['status'] = 'failed'
            report['final_source_error'] = str(exc)
        report['planned_invocations'] = len(plan['jobs'])
        report['attempted_invocations'] = len(report['results'])
        write_json(destination / 'summary.json', report)
        archive = freeze_and_archive(destination)
        print('Evidence archive: ' + str(archive), flush=True)
    return 0 if report['status'] == 'complete' else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--execute', action='store_true', help='Explicitly launch the plan; default only prints JSON')
    result.add_argument('--output-dir', type=Path, help='Fresh nonexisting directory outside checkout; parent must exist')
    result.add_argument('--suites', nargs='+', choices=list(RUNNERS))
    result.add_argument('--backends', nargs='+', choices=['numpy', 'cupy', 'torch'], default=['numpy', 'cupy', 'torch'])
    result.add_argument('--scale', choices=['representative', 'medium', 'large'], default='representative')
    result.add_argument('--profile', choices=['representative', 'expanded'], default='representative', help='Expanded explicitly plans three seeds and three bounded scale shapes with <=8-case jobs; never executes without --execute')
    result.add_argument('--config', action='append', default=[], metavar='SUITE:NAME=JSON', help='Named direct-runner configuration; repeat to enumerate finite solver/dtype/weight/sparse/covariance/CV profiles')
    result.add_argument('--seeds', nargs='+', type=int, help='Distinct independent seeds, at most 32; omitted preserves per-suite defaults')
    result.add_argument('--shapes', action='append', default=[], metavar='SUITE=NxP,NxP', help='Explicit shapes, at most eight per suite; classification uses Nx0')
    result.add_argument('--dtypes', nargs='+', choices=['float32', 'float64'], help='Explicit input precision; unsupported suite capabilities fail closed')
    result.add_argument('--weights', nargs='+', choices=['none', 'analytic'], help='Explicit weighting; only reviewed weighted fixture combinations accepted')
    result.add_argument('--max-invocations', type=int, default=21, help='Explicit execution count budget in 1..64; plan-only may enumerate more, execution must fit the selected batch')
    result.add_argument('--batch-index', type=int, default=0, help='Zero-based deterministic invocation partition')
    result.add_argument('--batch-count', type=int, default=1)
    result.add_argument('--resamples', type=int, help='Statistical resampling count, 2..1000')
    result.add_argument('--targets', type=int, help='Workflow target count, 1..16')
    result.add_argument('--alpha', type=float, help='Gaussian ridge alpha; OLS remains unpenalized')
    result.add_argument('--knockoff-p', type=int, default=20)
    result.add_argument('--selection-seeds', nargs='+', type=int, help='Knockoff-only independent trials, at most 32; not timing repeats')
    result.add_argument('--max-pairwise-elements', type=int, default=4000000, help='May tighten, never exceed the reviewed 4M campaign allocation bound')
    result.add_argument('--device', type=int, help='Concrete CUDA ordinal; six-family-cv supports only 0')
    result.add_argument('--cases', action='append', default=[], metavar='SUITE=CASE,CASE')
    result.add_argument('--cpu-threads', nargs='+', type=int, choices=[1, 2, 4, 8], default=[1])
    result.add_argument('--timeout', type=float, default=1800)
    result.add_argument('--warmup', type=int, choices=range(0, 11), default=None)
    result.add_argument('--repeats', type=int, choices=range(1, 11), default=None)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if not args.execute:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        if args.output_dir is None:
            raise ValueError('--execute requires --output-dir')
        destination = validate_destination(args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        argument_parser.error(str(exc))
    # Absolute script invocation must be just as usable as python -m.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return execute(plan, destination, args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
