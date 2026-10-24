#!/usr/bin/env python3
"""Plan or explicitly execute reviewed independent protocols, never arbitrary CLI.

Planning is standard-library-only. Runs are serial, exact-source, bounded and
strictly diagnostic. Historical inventory provenance is not execution evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from dev.benchmarks.independent_benchmark_registry import (
    ADAPTERS, ROOT, TIERS, build_plan, validate_artifact, verify_catalog_sources,
)
from dev.benchmarks.run_qualification_campaign import (
    THREAD_ENV, cpu_limits, freeze_and_archive, launch, read_json, resource_guard,
    sha256_file, validate_destination, write_json,
)

# Only this bootstrap imports numerical packages, in an explicitly requested
# child. The parent/plan path never imports any benchmark workload.
CHILD_BOOTSTRAP = r'''
import importlib, importlib.metadata, importlib.util, json, os, platform, resource, runpy, shutil, sys, traceback
from pathlib import Path
root, spec_path, runtime_prefix, output_path = sys.argv[1:5]
sys.path.insert(0, root)
spec = json.loads(Path(spec_path).read_text())
report = {'python': platform.python_version(), 'requested_backends': spec['backends'],
          'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
          'device_evidence_scope': 'availability and memory prerequisites only, NOT executed numerical-device proof',
          'gpu_peak_memory': 'not_measured', 'device_prerequisites': {}}
rc = 1
class PrerequisiteBlocked(RuntimeError): pass
class ResourceGuard(RuntimeError): pass
try:
    for package in spec.get('required_python', []):
        if importlib.util.find_spec(package) is None:
            raise PrerequisiteBlocked('Required Python dependency unavailable: ' + package)
    for executable in spec.get('required_executables', []):
        if shutil.which(executable) is None:
            raise PrerequisiteBlocked('Required executable unavailable: ' + executable)
    import numpy
    import scipy
    import threadpoolctl
    import statgpu
    report['statgpu_import_path'] = str(Path(statgpu.__file__).resolve())
    if not Path(statgpu.__file__).resolve().is_relative_to(Path(root).resolve() / 'statgpu'):
        raise PrerequisiteBlocked('statgpu imported outside the exact checkout')
    if importlib.util.find_spec('torch') is not None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if 'cupy' in spec['backends']:
        import cupy as cp
        cp.cuda.Device(0).use()
        free, total = cp.cuda.runtime.memGetInfo()
        if free < spec['budget']['minimum_gpu_free_bytes']:
            raise ResourceGuard('CuPy free device memory below reserve')
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'unknown')
        report['device_prerequisites']['cupy'] = {'device': 'cuda:0', 'name': name.decode() if isinstance(name, bytes) else str(name), 'free_bytes': int(free), 'total_bytes': int(total)}
    if 'torch' in spec['backends']:
        import torch
        if not torch.cuda.is_available():
            raise PrerequisiteBlocked('Physical Torch CUDA is required; CPU fallback forbidden')
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info(0)
        if free < spec['budget']['minimum_gpu_free_bytes']:
            raise ResourceGuard('Torch free device memory below reserve')
        report['device_prerequisites']['torch'] = {'device': 'cuda:0', 'name': torch.cuda.get_device_name(0), 'free_bytes': int(free), 'total_bytes': int(total)}
    def snapshot():
        pools = threadpoolctl.threadpool_info()
        if not pools or any(type(p.get('num_threads')) is not int or p['num_threads'] != 1 for p in pools):
            raise ResourceGuard('Numerical thread pools were not observed at the serial bound')
        torch_threads = None
        if 'torch' in sys.modules:
            torch_threads = {'intraop': sys.modules['torch'].get_num_threads(), 'interop': sys.modules['torch'].get_num_interop_threads()}
            if any(v != 1 for v in torch_threads.values()):
                raise ResourceGuard('Torch thread pools exceeded the serial bound')
        return {'pools': pools, 'torch_threads': torch_threads, 'cap_verified': True}
    report['before'] = snapshot()
    with open(runtime_prefix + '.before.json', 'x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    sys.argv = [spec['module'], *spec['argv']]
    if spec['contract'] == 'memory-cleanup':
        # Original runner has no JSON option. Capture its original measurement
        # dictionaries without changing its workload/timing/cleanup operations.
        module = importlib.import_module(spec['module'])
        original = module._fit_one
        rows, counters = [], {}
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            key = (result['model'], result['cleanup'])
            index = counters.get(key, 0)
            counters[key] = index + 1
            rows.append(dict(result, sample_index=index, phase='warmup' if index < spec['configuration']['warmup_runs'] else 'measurement'))
            return result
        module._fit_one = observed
        try:
            module.main()
        finally:
            module._fit_one = original
        with open(output_path, 'x') as stream:
            json.dump({'schema_version': 'statgpu-memory-pool-capture-v1', 'rows': rows,
                       'memory_scope': 'CuPy pool used/total snapshots; not peak VRAM or OOM capacity',
                       'correctness_status': 'unavailable', 'performance_eligible': False}, stream, indent=2, allow_nan=False)
        rc = 0
    else:
        try:
            runpy.run_module(spec['module'], run_name='__main__')
            rc = 0
        except SystemExit as exc:
            rc = int(exc.code or 0)
    report['after'] = snapshot()
except (ImportError, PrerequisiteBlocked) as exc:
    rc = 77
    report['blocked_reason'] = type(exc).__name__ + ': ' + str(exc)
    traceback.print_exc()
except BaseException as exc:
    rc = 86 if isinstance(exc, ResourceGuard) else 1
    report['error'] = type(exc).__name__ + ': ' + str(exc)
    traceback.print_exc()
finally:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report['process_peak_rss_bytes'] = int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    report['memory_scope'] = 'Child process high-water RSS only, not per-method memory or subprocess aggregate'
    report['packages'] = {}
    for name in ('numpy', 'scipy', 'torch', 'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'threadpoolctl'):
        try: report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['packages'][name] = None
    report['exit_code'] = rc
    with open(runtime_prefix + '.after.json', 'x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
sys.exit(rc)
'''


def source_fingerprint(root=ROOT):
    """Live source identity includes independent helpers and compiled sources."""
    root = Path(root)
    files = []
    for directory in ('statgpu', 'dev/benchmarks'):
        files.extend(p for p in (root / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix.lower() in ('.py', '.pyx', '.pxd', '.c', '.h', '.cpp', '.json', '.r', '.sh', '.so'))
    files.extend(root / p for p in ('pyproject.toml', '.claude/skills/benchmark/SKILL.md', '.claude/skills/benchmark/schema.md'))
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(files)) if p.is_file()}
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {'git_commit': git('rev-parse', 'HEAD'),
            'working_tree': 'clean' if git('status', '--porcelain') == '' else 'dirty-or-unknown',
            'files_sha256': hashes,
            'content_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def source_is_same(before, after):
    return all(before.get(k) == after.get(k) for k in ('git_commit', 'working_tree', 'content_sha256'))


def _host_memory_guard(minimum):
    try:
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(values['MemAvailable'].strip().split()[0]) * 1024
        if available < minimum:
            return 'Host MemAvailable is below registered reserve'
    except (OSError, ValueError, KeyError):
        return 'Host available-memory preflight is unavailable'
    return None


def assess_artifact(job, folder, process):
    """Qualify a child protocol, retaining raw bytes and rejecting partial output."""
    folder = Path(folder)
    result = {'lane_id': job['id'], 'config_id': job['config_id'], 'status': 'failed',
              'performance_eligible': False, 'canonical_dashboard_ready': False,
              'device_claim': 'unavailable', 'correctness_status': 'unavailable',
              'process': process, 'raw_sha256': None}
    raw = folder / 'raw.json'
    if raw.is_file():
        result['raw_sha256'] = sha256_file(raw)
    if process.get('reason') or process.get('exit_code') != 0:
        result['status'] = 'blocked' if process.get('exit_code') in (77, 86) else 'failed'
        result['reason'] = process.get('reason') or 'Child exit code ' + str(process.get('exit_code'))
        return result
    try:
        if raw.is_symlink() or not raw.is_file():
            raise ValueError('Missing regular raw artifact')
        if raw.stat().st_size > 64 * 1024**2:
            raise ValueError('Raw artifact exceeds 64 MiB protocol budget')
        runtime = read_json(folder / 'runtime.after.json')
        before = read_json(folder / 'runtime.before.json')
        if runtime.get('exit_code') != 0 or runtime.get('after', {}).get('cap_verified') is not True or before.get('before', {}).get('cap_verified') is not True:
            raise ValueError('Missing/failed actual thread-bound runtime evidence')
        payload = read_json(raw)
        result.update(validate_artifact(job['id'], payload, job['configuration']))
        result['runtime'] = runtime
        result['raw_validation_tier'] = payload.get('validation_tier', payload.get('validation', {}).get('tier'))
        result['status'] = 'complete'
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
    return result


def execute(plan, destination, timeout=1800, max_invocations=21):
    """Execute canonical selected protocols; caller must have execution authority.

    The plan is rederived and compared before output creation, so a deserialized
    plan cannot inject a module, command, output path, budget or environment.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError('timeout must be finite, positive and at most 1800 seconds')
    if type(max_invocations) is not int or not 1 <= max_invocations <= 64:
        raise ValueError('max_invocations must be 1..64; split larger executions into batches')
    if sum(j.get('adapter') is not None for j in plan.get('jobs', [])) > max_invocations:
        raise ValueError('Selected configurations exceed explicit invocation budget; split the plan')
    expected_sha = plan.get('expected_sha')
    if not isinstance(expected_sha, str) or re.fullmatch(r'[0-9a-f]{40}', expected_sha) is None:
        raise ValueError('Execution requires a full exact expected SHA')
    environment_id = plan.get('environment_id')
    if not isinstance(environment_id, str) or not environment_id.strip() or len(environment_id) > 256:
        raise ValueError('Execution requires an explicit verified environment ID')
    lanes, configurations = [], {}
    for job in plan.get('jobs', []):
        if job['id'] not in lanes:
            lanes.append(job['id'])
        configurations.setdefault(job['id'], []).append(job['configuration'])
    canonical = build_plan(lanes, expected_sha=expected_sha, configurations=configurations,
                           validation_tier=plan.get('validation_tier'), environment_id=environment_id)
    if canonical != plan:
        raise ValueError('Execution plan differs from reviewed registry; arbitrary commands/arguments forbidden')
    destination = validate_destination(Path(destination))
    before = source_fingerprint()
    if before['git_commit'] != expected_sha or before['working_tree'] != 'clean':
        raise ValueError('Execution requires a clean checkout at the exact expected SHA')
    if cpu_limits()['thread_cap'] < 1:
        raise ValueError('Serial CPU budget is unavailable')
    destination.mkdir(exist_ok=False)
    report = {'schema_version': 'statgpu-independent-execution-v1', 'status': 'failed',
              'validation_tier': plan['validation_tier'], 'environment_id': environment_id,
              'coverage_status': plan['coverage_status'], 'results': [],
              'performance_eligible': False, 'canonical_dashboard_ready': False,
              'exclusive_gpu_ownership': 'UNVERIFIED_EXCLUSIVITY',
              'gpu_peak_memory': 'not_measured', 'stop_reason': None}
    write_json(destination / 'plan.json', plan)
    write_json(destination / 'source-before.json', before)
    write_json(destination / 'audit-drift.json', verify_catalog_sources())
    try:
        for job in plan['jobs']:
            if job['adapter'] is None:
                report['results'].append({'lane_id': job['id'], 'config_id': job['config_id'],
                    'status': job['disposition'], 'reason': job['blocked_reason'], 'performance_eligible': False})
                continue
            current = source_fingerprint()
            if not source_is_same(before, current):
                report['stop_reason'] = 'Source changed before next lane'
                break
            folder = destination / (job['id'] + '-' + job['config_id'][:12])
            folder.mkdir(exist_ok=False)
            spec = dict(job['adapter'])
            spec['configuration'] = job['configuration']
            substitutions = {'output': str(folder / 'raw.json'), 'expected_sha': expected_sha,
                             'validation_tier': plan['validation_tier']}
            spec['argv'] = [value.format(**substitutions) for value in spec['arguments']]
            write_json(folder / 'protocol.json', spec)
            guard = lambda: resource_guard(destination, 1) or _host_memory_guard(spec['budget']['minimum_host_available_bytes'])
            preflight = guard()
            if preflight:
                report['stop_reason'] = 'Resource preflight: ' + preflight
                result = {'lane_id': job['id'], 'config_id': job['config_id'], 'status': 'blocked',
                          'reason': preflight, 'performance_eligible': False, 'canonical_dashboard_ready': False}
            else:
                env = os.environ.copy()
                env.update({key: '1' for key in THREAD_ENV})
                env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', CMAKE_BUILD_PARALLEL_LEVEL='1')
                env['PYTHONPATH'] = str(ROOT)
                with tempfile.TemporaryDirectory(prefix='independent-cache-', dir=destination) as cache:
                    env.update(TMPDIR=cache, XDG_CACHE_HOME=cache, TORCHINDUCTOR_CACHE_DIR=cache + '/inductor',
                               TRITON_CACHE_DIR=cache + '/triton', CUDA_CACHE_PATH=cache + '/cuda')
                    command = [sys.executable, '-B', '-c', CHILD_BOOTSTRAP, str(ROOT),
                               str(folder / 'protocol.json'), str(folder / 'runtime'), str(folder / 'raw.json')]
                    process = launch(command, folder / 'stdout.log', min(timeout, spec['budget']['timeout_seconds']), env, guard)
                result = assess_artifact(job, folder, process)
                if process.get('reason') == 'interrupted' or (process.get('reason') or '').startswith('resource_guard:') or process.get('exit_code') == 86:
                    report['stop_reason'] = process.get('reason') or 'Child resource guard tripped'
            after_job = source_fingerprint()
            write_json(folder / 'source-after.json', after_job)
            if not source_is_same(before, after_job):
                result.update(status='invalidated', reason='Source changed during lane', correctness_status='unavailable', performance_eligible=False)
                report['stop_reason'] = 'Source changed during execution'
            result['validation_tier'] = plan['validation_tier']
            result['environment_id'] = environment_id
            write_json(folder / 'qualified.json', result)
            if (folder / 'raw.json').is_file() and not (folder / 'raw.json').is_symlink():
                (folder / 'raw.json').chmod(0o444)
            report['results'].append(result)
            if report['stop_reason']:
                break
        after = source_fingerprint()
        write_json(destination / 'source-after.json', after)
        if not source_is_same(before, after):
            report['stop_reason'] = 'Source changed during execution'
        completed = len(report['results']) == len(plan['jobs']) and all(r['status'] == 'complete' for r in report['results'])
        report['status'] = 'complete' if completed and not report['stop_reason'] else 'partial'
    except BaseException as exc:
        report['stop_reason'] = type(exc).__name__ + ': ' + str(exc)
        if not (destination / 'source-after.json').exists():
            write_json(destination / 'source-after.json', source_fingerprint())
    finally:
        write_json(destination / 'summary.json', report)
        freeze_and_archive(destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', nargs='+')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--configurations', type=Path, help='JSON mapping selected lane IDs to bounded configuration lists')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--expected-sha')
    parser.add_argument('--environment-id')
    parser.add_argument('--validation-tier', choices=TIERS, default='local-full')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--max-invocations', type=int, default=21)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.lanes, expected_sha=args.expected_sha,
                          validation_tier=args.validation_tier, environment_id=args.environment_id,
                          configurations=read_json(args.configurations) if args.configurations else None)
        if not args.execute:
            if args.output_root is not None:
                parser.error('--output-root is for explicit execution; planning writes only stdout')
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.output_root is None:
            parser.error('--execute requires a fresh external --output-root')
        if args.lanes is None:
            parser.error('--execute requires explicit --lanes; default planning never launches every workload')
        report = execute(plan, args.output_root, args.timeout, args.max_invocations)
        print(json.dumps({'status': report['status'], 'output': str(args.output_root),
                          'performance_eligible': False}, sort_keys=True))
        return 0 if report['status'] == 'complete' else 2
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
