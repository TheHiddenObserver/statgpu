#!/usr/bin/env python3
"""Plan (default) or explicitly execute the ten observed thread-bound reruns.

This stdlib-only selector is deliberately not a general campaign interface. It
accepts one immutable historical campaign and one clean repaired source commit.
It never changes original evidence, invokes SSH, installs packages, runs an
independent/expanded profile, or claims performance certification. Planning
launches Git identity checks only, never numerical imports or workload children.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.metadata
import importlib.util
from pathlib import Path
import signal
import sys

import batch_driver as driver

BASE_SHA = '91df0d9ebc35a49bde917383ba0680806016f3ae'
REPAIRED_SHA = 'f90f5f8b42b62334e6c746ef3488f8a18a1f6217'
OLD_PLAN_SHA256 = 'b1074e3af4ed87df4c57a8427e1f825155b4faeb29b60d2aabc5506379be5705'
OLD_SEAL_SHA256 = '8c206754739bd1e6f631823b5c02805eea93c4a737591f517cd296e20cac237c'
POLICY = 'all-backends-optional-torch-v1'
EXPECTED_IDS = (
    '0001-gaussian-refresh-cupy-t1-s20261020-a315a9763b9343ef',
    '0006-model-families-numpy-t1-s20261020-bf1993112d0bc4f4',
    '0007-model-families-cupy-t1-s20261020-b18e6f9432a314f9',
    '0009-statistical-families-numpy-t1-s20261020-12332f01e48adafc',
    '0010-statistical-families-cupy-t1-s20261020-f887d7b6e4b54b4d',
    '0012-learning-families-numpy-t1-s42-8e34d5dca8287036',
    '0013-learning-families-cupy-t1-s42-e368d05d893148e3',
    '0015-workflow-families-numpy-t1-s20261020-21998ad046530cdf',
    '0018-six-family-cv-numpy-t1-s20260807-460f9917bfb98906',
    '0019-six-family-cv-cupy-t1-s20260807-b4df8750c5c83038',
)


def original_selection(evidence):
    evidence = Path(evidence).expanduser().absolute()
    if any(path.is_symlink() for path in (evidence, *evidence.parents)):
        raise ValueError('Historical evidence and its ancestors must not be symlinks')
    evidence = evidence.resolve()
    if driver.file_hash(evidence / 'SHA256SUMS.json') != OLD_SEAL_SHA256:
        raise ValueError('Historical native evidence seal differs from the reviewed immutable pin')
    driver.verify_seal(evidence, archive=True)
    if driver.file_hash(evidence / 'plan.json') != OLD_PLAN_SHA256:
        raise ValueError('Historical native plan differs from the reviewed immutable pin')
    original = driver.read_json(evidence / 'plan.json')
    before = driver.read_json(evidence / 'source-before.json')
    after = driver.read_json(evidence / 'source-after.json')
    if (original.get('profile') != 'representative' or len(original.get('jobs', [])) != 21
            or original.get('expected_total_rows') != 1692
            or original.get('source', {}).get('git_commit') != BASE_SHA
            or before != after or before.get('git_commit') != BASE_SHA
            or before.get('working_tree') != 'clean'):
        raise ValueError('Historical representative campaign/source identity does not match')
    selected, observations = [], []
    for job in original['jobs']:
        runtime = driver.read_json(evidence / job['id'] / 'runtime.after.json')
        values = runtime.get('after', {}).get('torch_threads')
        if values is not None and any(type(v) is not int or not 1 <= v <= job['cpu_threads']
                                      for v in values.values()):
            selected.append(copy.deepcopy(job))
            observations.append({'id': job['id'], 'requested_threads': job['cpu_threads'],
                                 'observed_after_torch_threads': values})
    if tuple(job['id'] for job in selected) != EXPECTED_IDS:
        raise ValueError('Observed violations do not equal the exact reviewed ten-job selection')
    if any(job['backend'] not in ('numpy', 'cupy') or job['cpu_threads'] != 1 for job in selected):
        raise ValueError('Rerun scope is restricted to the observed NumPy/CuPy t1 jobs')
    return original, selected, observations


def load_native(repo):
    # Only the exact checked stdlib orchestrator is imported in this process.
    repo = Path(repo).resolve()
    for name, imported in list(sys.modules.items()):
        if name == 'dev' or name.startswith('dev.') or name == 'statgpu' or name.startswith('statgpu.'):
            paths = ([getattr(imported, '__file__', None)] + list(getattr(imported, '__path__', [])))
            paths = [Path(value).resolve() for value in paths if value is not None]
            if not paths or any(not value.is_relative_to(repo) for value in paths):
                raise ValueError('Existing module import is outside the exact checkout: ' + name)
    # assess() imports dev.benchmarks.benchmark_qualification in this parent.
    # Do not rely on cwd, PYTHONPATH, or the interpreter's installation.
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    path = repo / 'dev/benchmarks/run_qualification_campaign.py'
    spec = importlib.util.spec_from_file_location('_exact_thread_rerun_campaign', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT != Path(repo).resolve() or POLICY not in module.CHILD_BOOTSTRAP:
        raise ValueError('Native runner is not the pinned all-backend thread-policy repair')
    validator = importlib.import_module('dev.benchmarks.benchmark_qualification')
    if Path(validator.__file__).resolve() != repo / 'dev/benchmarks/benchmark_qualification.py':
        raise ValueError('Native qualifier imported outside the exact checkout')
    return module


def tooling_identity():
    return {path.name: driver.file_hash(path) for path in (Path(__file__), Path(driver.__file__))}


def import_environment(repo):
    """Read metadata only; device availability is checked by the ten jobs."""
    if sys.version_info < (3, 9):
        raise ValueError('The reviewed selector requires Python 3.9 or newer')
    package = Path(repo) / 'statgpu/__init__.py'
    if package.is_symlink() or not package.is_file():
        raise ValueError('Pinned statgpu package entrypoint is unavailable')
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'statsmodels', 'threadpoolctl',
                 'torch', 'cupy', 'cupy-cuda11x', 'cupy-cuda12x'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable,
            'statgpu_import_candidate': str(package.resolve()),
            'statgpu_entrypoint_sha256': driver.file_hash(package),
            'child_import_root': str(Path(repo).resolve()), 'package_versions': versions,
            'numerical_imports_performed': False, 'gpu_probe_performed': False,
            'device_validation': 'Only within the explicitly requested ten native jobs; --require-backends retained'}


def build_plan(repo, evidence, timeout=1800):
    timeout = driver.checked_timeout(timeout)
    repo = Path(repo).expanduser().resolve()
    source = driver.source_identity(repo, REPAIRED_SHA)
    original, jobs, observations = original_selection(evidence)
    native = load_native(repo)
    native_source = native.fingerprint()
    if native_source['git_commit'] != REPAIRED_SHA or native_source['working_tree'] != 'clean':
        raise ValueError('Repaired native source fingerprint is not clean and pinned')
    compact_source = {key: native_source[key] for key in ('git_commit', 'content_sha256', 'working_tree')}
    for job in jobs:
        job['expected_source'] = copy.deepcopy(compact_source)
        job['resource_bounds']['timeout_seconds'] = timeout
        for slot in job['expected_slots']:
            slot['source'] = copy.deepcopy(compact_source)
    plan = copy.deepcopy(original)
    plan.update(jobs=jobs, source=compact_source, source_locked=True,
                plan_only=True, expected_total_rows=sum(job['expected_row_count'] for job in jobs),
                cpu_limits=native.cpu_limits(),
                batch={'index': 0, 'count': 1, 'all_invocations': 10,
                       'all_expected_rows': sum(job['expected_row_count'] for job in jobs)},
                execution_budget={'max_invocations': 10, 'selected_invocations': 10,
                                  'within_budget': True, 'worst_case_timeout_seconds': 10 * timeout},
                note='Exact ten historical thread-bound violations only. Original arguments and row/slot identities are preserved; source pins name the new native repair commit. No expanded/independent jobs or performance promotion.')
    plan['thread_bound_rerun'] = {
        'schema_version': 'statgpu-exact-ten-thread-rerun-v1', 'policy': POLICY,
        'original_commit': BASE_SHA, 'repaired_commit': REPAIRED_SHA,
        'original_plan_sha256': OLD_PLAN_SHA256, 'original_seal_sha256': OLD_SEAL_SHA256,
        'original_job_ids': list(EXPECTED_IDS), 'observed_violations': observations,
        'tooling_sha256': tooling_identity(),
        'native_bootstrap_sha256': driver.hashlib.sha256(native.CHILD_BOOTSTRAP.encode()).hexdigest(),
        'timeout_seconds_per_job': timeout,
        'source_identity': source,
        'import_environment': import_environment(repo),
        'operator_attestation': None,
        'performance_eligible': False, 'canonical_dashboard_ready': False,
        'historical_evidence_modified': False,
    }
    if driver.source_identity(repo, REPAIRED_SHA) != source:
        raise ValueError('Source changed while preparing the exact-ten plan')
    return plan, source, native


@contextmanager
def one_interrupt_then_finalize():
    """First termination triggers owned-child cleanup; repeats cannot cut sealing."""
    previous = {}
    numbers = [getattr(signal, name) for name in ('SIGINT', 'SIGTERM', 'SIGHUP')
               if hasattr(signal, name)]
    def interrupted(number, _frame):
        for item in numbers:
            signal.signal(item, signal.SIG_IGN)
        raise KeyboardInterrupt('Termination signal ' + str(number))
    try:
        for number in numbers:
            previous[number] = signal.signal(number, interrupted)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def output_path(output, repo, evidence):
    output = driver.external_new(output, repo)
    evidence = Path(evidence).resolve()
    if output.is_relative_to(evidence) or evidence.is_relative_to(output):
        raise ValueError('New output and immutable historical evidence must be disjoint')
    return output


def save_inputs(output, plan, source, evidence):
    driver.write_json(output / 'plan.json', plan)
    driver.write_json(output / 'source-identity.json', source)
    # Preserve the exact controlling script bytes as well as their hashes.
    tooling = output / 'tooling'
    tooling.mkdir()
    for path in (Path(__file__), Path(driver.__file__)):
        with (tooling / path.name).open('xb') as stream:
            stream.write(path.read_bytes())
    with (output / 'original-plan.json').open('xb') as stream:
        stream.write((Path(evidence) / 'plan.json').read_bytes())
    if tooling_identity() != plan['thread_bound_rerun']['tooling_sha256']:
        raise ValueError('External tooling changed while preparing output')


def run(repo, evidence, output, *, execute=False, timeout=1800):
    output = output_path(output, repo, evidence)
    plan, source, native = build_plan(repo, evidence, timeout)
    plan['plan_only'] = not execute
    output.mkdir(exist_ok=False)
    report = {'schema_version': 'statgpu-exact-ten-thread-rerun-attempt-v1',
              'status': 'failed', 'executed': execute, 'native_directory': 'native' if execute else None,
              'planned_invocations': 10, 'attempted_invocations': 0,
              'original_commit': BASE_SHA, 'repaired_commit': REPAIRED_SHA,
              'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'performance_eligible': False,
              'canonical_dashboard_ready': False, 'error': None}
    try:
        save_inputs(output, plan, source, evidence)
        if not execute:
            report['status'] = 'plan_only'
        else:
            # No custom launch/assess implementation: retain native raw output,
            # qualification, source snapshots, own-group cleanup and archives.
            with one_interrupt_then_finalize():
                code = native.execute(plan, output / 'native', timeout)
            driver.verify_seal(output / 'native', archive=True)
            summary = driver.read_json(output / 'native' / 'summary.json')
            report['attempted_invocations'] = summary.get('attempted_invocations', 0)
            report['native_summary_sha256'] = driver.file_hash(output / 'native' / 'summary.json')
            report['native_archive_sha256'] = driver.file_hash(output / 'native.tar.gz')
            if (code == 0 and summary.get('status') == 'complete'
                    and [result.get('id') for result in summary.get('results', [])] == list(EXPECTED_IDS)
                    and all(result.get('status') == 'complete' for result in summary['results'])):
                report['status'] = 'complete'
            else:
                report['error'] = summary.get('stop_reason') or 'Native qualification did not complete all ten jobs'
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        if isinstance(exc, KeyboardInterrupt):
            report['status'] = 'interrupted'
    finally:
        with driver.finish_without_signals():
            try:
                after = driver.source_identity(repo, REPAIRED_SHA)
                driver.write_json(output / 'source-identity-after.json', after)
                if after != source or tooling_identity() != plan['thread_bound_rerun']['tooling_sha256']:
                    raise ValueError('Source or external tooling changed during operation')
                # Any new evidence cannot silently replace or edit the old data.
                original_selection(evidence)
            except Exception as exc:
                report.update(status='failed', error='Final identity check: ' + str(exc))
            driver.write_json(output / 'summary.json', report)
            driver.seal_directory(output, archive=execute)
    print(str(output / 'summary.json'))
    return 0 if report['status'] in ('plan_only', 'complete') else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--repo', type=Path, required=True, help='Clean checkout at the exact repaired commit')
    result.add_argument('--evidence', type=Path, required=True, help='Original registered-representative-0000/native directory, with adjacent native archive')
    result.add_argument('--output-dir', type=Path, required=True, help='Unique external output path; must not exist')
    result.add_argument('--timeout', type=float, default=1800, help='Per-job seconds, greater than zero and at most 1800')
    result.add_argument('--execute', action='store_true', help='Explicitly authorize this invocation of exactly the selected ten jobs')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    sys.dont_write_bytecode = True
    try:
        return run(args.repo, args.evidence, args.output_dir, execute=args.execute, timeout=args.timeout)
    except (OSError, ValueError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
