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
# Larger learning invocations are a deliberately bounded menu, never all 44 cases.
SCALABLE_LEARNING = {'kde', 'kernel_regression_nw', 'pca', 'incremental_pca',
                     'truncated_svd', 'kmeans', 'minibatch_kmeans', 'nystroem',
                     'bspline_basis', 'natural_spline', 'cyclic_spline',
                     'bandwidth_scott', 'bandwidth_silverman', 'bandwidth_nrd',
                     'bandwidth_nrd0'}
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


def expected_rows(suite, backend, cases, n, p, seed):
    rows = []
    for case in cases:
        actual_n, actual_p = n, p
        if suite == 'model-families' and case.startswith('panel:'):
            periods = max(5, p + 2)
            actual_n = max(p + 3, n // periods) * periods
        if suite == 'learning-families':
            actual_p = 20 if case.startswith('knockoff') else 1 if case in ONE_FEATURE_LEARNING or case.startswith('bandwidth') else p
        row = {'case': case, 'backend': backend, 'n': actual_n, 'p': actual_p,
               'input_dtype': 'float64', 'seed': seed}
        if suite == 'classification-metrics':
            row.pop('p')
        if suite == 'statistical-families' and case in HOST_ONLY and backend != 'numpy':
            # Current producer omits problem dimensions for authorized host-only NA.
            row = {'case': case, 'backend': backend, 'not_applicable': True}
        if suite == 'gaussian-refresh':
            for dtype in ('float32', 'float64'):
                for weighted in (False, True):
                    rows.append(dict(row, input_dtype=dtype, weighted=weighted))
        elif suite == 'classification-metrics':
            for dtype in ('float32', 'float64'):
                for scenario in SCENARIOS:
                    rows.append(dict(row, input_dtype=dtype, scenario=scenario))
        else:
            rows.append(row)
    return rows


def build_plan(args):
    entries = registry()
    suites = args.suites or list(RUNNERS)
    selectors = parse_selectors(args.cases, entries)
    if len(suites) != len(set(suites)) or len(args.backends) != len(set(args.backends)):
        raise ValueError('Duplicate suites/backends are not allowed')
    if set(selectors) - set(suites):
        raise ValueError('Case selectors must belong to selected suites')
    if args.scale != 'representative' and not args.suites:
        raise ValueError('Medium/large scaling requires explicit --suites and selective --cases')
    limits = cpu_limits()
    if 1 not in args.cpu_threads or len(args.cpu_threads) != len(set(args.cpu_threads)):
        raise ValueError('--cpu-threads must include baseline 1 without duplicates')
    if max(args.cpu_threads) > 1 and 'numpy' not in args.backends:
        raise ValueError('Optional CPU thread profiles require NumPy in --backends')
    if max(args.cpu_threads) > limits['thread_cap']:
        raise ValueError('CPU thread profile exceeds actual affinity/cgroup quota')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800:
        raise ValueError('Per-invocation timeout must be > 0 and <= 1800 seconds')
    jobs = []
    for threads in sorted(args.cpu_threads):
        for suite in suites:
            cases = selectors.get(suite, entries[suite]['case_ids'])
            if suite == 'six-family-cv' and suite in selectors and cases != entries[suite]['case_ids']:
                raise ValueError('CV runner only supports its six-family registry, not case filtering')
            if args.scale != 'representative':
                if suite != 'six-family-cv' and (suite not in selectors or len(cases) > 8):
                    raise ValueError('Medium/large requires 1-8 exact cases per selected suite')
                if suite == 'learning-families' and not set(cases) <= SCALABLE_LEARNING:
                    raise ValueError('Selected learning case is outside the bounded scaling menu')
            scale_index = ('representative', 'medium', 'large').index(args.scale)
            n, p = {
                'gaussian-refresh': ((1024, 16), (8192, 64), (32768, 128)),
                'classification-metrics': ((1000, 0), (10000, 0), (100000, 0)),
                'model-families': ((500, 8), (1024, 8), (2048, 16)),
                'statistical-families': ((128, 4), (1024, 4), (2048, 8)),
                'learning-families': ((256, 4), (1024, 4), (2000, 8)),
                'workflow-families': ((256, 6), (1024, 6), (2048, 8)),
                'six-family-cv': ((240, 16), (1024, 16), (2048, 16)),
            }[suite][scale_index]
            shapes = [(n, p)]
            if args.scale == 'representative':
                shapes = {
                    'gaussian-refresh': [(1024, 16), (8192, 64), (32768, 128)],
                    'classification-metrics': [(1000, 0), (10000, 0), (100000, 0)],
                    'statistical-families': [(128, 4), (1024, 4)],
                    'workflow-families': [(256, 6), (1024, 6)],
                    'six-family-cv': [(240, 16), (1024, 16)],
                }.get(suite, shapes)
            default_warmup, default_repeats = (2, 5) if suite in ('gaussian-refresh', 'classification-metrics') else (1, 5) if suite == 'statistical-families' else (1, 3)
            warmup = default_warmup if args.warmup is None else args.warmup
            repeats = default_repeats if args.repeats is None else args.repeats
            seed = 20260807 if suite == 'six-family-cv' else 42 if suite == 'learning-families' else 20261020
            for backend in args.backends:
                if threads != 1 and backend != 'numpy':
                    continue
                argv = ['--backends', backend, '--require-backends', '--seed', str(seed),
                        '--warmup', str(warmup), '--repeats', str(repeats)]
                if suite == 'gaussian-refresh':
                    argv += ['--models', *cases, '--shapes', *[f'{nn}x{pp}' for nn, pp in shapes], '--dtypes', 'float32', 'float64', '--weights', 'none', 'analytic']
                elif suite == 'classification-metrics':
                    argv += ['--operations', *cases, '--sizes', *[str(nn) for nn, _ in shapes], '--dtypes', 'float32', 'float64', '--scenarios', *SCENARIOS]
                elif suite in ('model-families', 'learning-families'):
                    argv += ['--cases', *cases, '--n', str(n), '--p', str(p)]
                    if suite == 'learning-families':
                        argv += ['--knockoff-p', '20', '--max-pairwise-elements', '4000000']
                else:
                    argv += ['--sizes', *[str(nn) for nn, _ in shapes], '--features', str(p)]
                    if suite != 'six-family-cv':
                        argv += ['--cases', *cases]
                    if suite == 'statistical-families':
                        argv += ['--resamples', '100']
                    if suite == 'workflow-families':
                        argv += ['--targets', '4']
                expected = [row for nn, pp in shapes for row in expected_rows(suite, backend, cases, nn, pp, seed)]
                jobs.append({'id': f'{len(jobs):03d}-{suite}-{backend}-t{threads}',
                    'suite': suite, 'backend': backend, 'cpu_threads': threads,
                    'scale': args.scale, 'shapes': [{'n': nn, 'p': pp} for nn, pp in shapes], 'module': 'dev.benchmarks.' + RUNNERS[suite],
                    'argv': argv, 'cases': cases, 'expected_rows': expected,
                    'expected_row_count': len(expected), 'expected_repeats': repeats, 'expected_warmup': warmup,
                    'input_precision': sorted({r['input_dtype'] for r in expected if 'input_dtype' in r}),
                    'compute_precision': 'Not inferred from input/output dtype; review scoped numerical observations in raw artifact',
                    'resource_bounds': {'max_pairwise_elements': 4000000 if suite == 'learning-families' else None,
                                        'timeout_seconds': args.timeout}})
    return {'schema_version': 'statgpu-qualification-plan-v1', 'plan_only': not args.execute,
            'serial': True, 'cpu_limits': limits, 'jobs': jobs,
            'expected_total_rows': sum(j['expected_row_count'] for j in jobs),
            'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'canonical_dashboard_ready': False,
            'note': 'Representative preserves the registry/default fullscan shapes, dtypes and cases, not exhaustive APIs. Extra thread profiles are NumPy-only and remain separate.'}


def row_identity(row, suite, document):
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
    return identity


def assess(output, job, exit_code, expected_source):
    from dev.benchmarks.benchmark_qualification import evaluate_artifact
    try:
        document = read_json(output)
        qualified = evaluate_artifact(document, job['suite'], job['backend'], exit_code=exit_code,
                                      expected_source=expected_source, expected_cases=job['cases'])
        observed = [row_identity(r, job['suite'], document) for r in document.get('results', document.get('rows', []))]
        encode = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
        wanted, actual = Counter(map(encode, job['expected_rows'])), Counter(map(encode, observed))
        inventory = {'expected_count': sum(wanted.values()), 'observed_count': sum(actual.values()),
                     'exact_match': wanted == actual, 'expected_rows': job['expected_rows'],
                     'observed_rows': observed, 'missing': list((wanted - actual).elements()),
                     'unexpected': list((actual - wanted).elements())}
        if wanted != actual:
            qualified['correctness']['status'] = 'failed'
            qualified['correctness']['errors'].append('Expanded row identity/multiplicity mismatch')
        for row in qualified.get('rows', []):
            if any(t['sample_count'] != job['expected_repeats'] for t in row.get('timing_records', [])):
                qualified['correctness']['status'] = 'failed'
                qualified['correctness']['errors'].append('Timing sample count differs from selected CLI repeats')
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
                    'benchmark_cv_models', 'cv_external', 'cv_source', 'learning_execution_trace']
    paths += [root / 'dev/benchmarks' / (name + '.py') for name in dependencies]
    paths += [root / name for name in ('dev/benchmarks/full_coverage_manifest.json',
              'dev/benchmarks/cv_source_schema.json', 'pyproject.toml', '.claude/skills/benchmark/schema.md')]
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
    destination.mkdir(exist_ok=False)
    report = {'schema_version': 'statgpu-qualification-campaign-v1', 'status': 'failed', 'results': [],
              'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'performance_eligible': False,
              'canonical_dashboard_ready': False, 'comparisons': [],
              'gpu_process_ownership': 'unverified; unknown or namespace-hidden PIDs do not fail correctness',
              'gpu_peak_memory': 'not_measured', 'stop_reason': None}
    before = None
    try:
        write_json(destination / 'plan.json', plan)
        before = fingerprint()
        write_json(destination / 'source-before.json', before)
        if not before.get('git_commit'):
            raise ValueError('Exact source commit unavailable')
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
            write_json(directory / 'invocation.json', {'command': command, 'thread_env': {k: environment[k] for k in THREAD_ENV}, 'expected_rows': job['expected_rows']})
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
