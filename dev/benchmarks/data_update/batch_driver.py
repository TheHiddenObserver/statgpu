#!/usr/bin/env python3
"""Plan, explicitly run/resume bounded batches, and aggregate immutable evidence.

This external stdlib-only driver never installs packages, uses SSH, executes
arbitrary commands from a plan, or promotes diagnostic evidence to performance
certification. Native repository planners, executors, validators and finalizers
remain authoritative. No benchmark is run by `plan` or `aggregate`.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile

SCHEMA = 'statgpu-batch-plan-v1'
TERMINAL = {'complete', 'failed', 'blocked', 'interrupted'}
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
SAFE_ID = re.compile(r'[a-z0-9][a-z0-9-]*\Z')
OWNED_FLAGS = {'--execute', '--output-dir', '--max-invocations', '--batch-index', '--batch-count', '--timeout'}
SOURCE_SUFFIXES = {'.py', '.pyx', '.pxd', '.c', '.h', '.cpp', '.json', '.r', '.sh', '.so'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            bad(value)
        return parsed
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024**2:
        raise ValueError('Expected bounded regular JSON file: ' + str(path))
    result = json.loads(path.read_text(encoding='utf8'), object_pairs_hook=pairs,
                        parse_constant=bad, parse_float=number)
    if not isinstance(result, dict):
        raise ValueError('Expected an object JSON root: ' + str(path))
    return result


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def regular_tree(root):
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink is not allowed in immutable evidence: ' + str(path))
        if path.is_file():
            yield path


def seal_directory(root, *, archive=False):
    """Exclusive creation; no replacement of any original byte or old seal."""
    root = Path(root)
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in regular_tree(root)}
    write_json(root / 'SHA256SUMS.json', hashes)
    for path in regular_tree(root):
        path.chmod(0o444)
    if archive:
        target = root.with_name(root.name + '.tar.gz')
        with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as bundle:
            bundle.add(root, arcname=root.name)
        with target.with_name(target.name + '.sha256').open('x') as stream:
            stream.write(file_hash(target) + '  ' + target.name + '\n')
        target.chmod(0o444)
        target.with_name(target.name + '.sha256').chmod(0o444)


def verify_seal(root, *, archive=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Missing regular sealed directory: ' + str(root))
    hashes = read_json(root / 'SHA256SUMS.json')
    actual = {p.relative_to(root).as_posix(): file_hash(p) for p in regular_tree(root)
              if p != root / 'SHA256SUMS.json'}
    if hashes != actual:
        raise ValueError('Immutable evidence hash/file inventory mismatch: ' + str(root))
    if archive:
        target = root.with_name(root.name + '.tar.gz')
        checksum = target.with_name(target.name + '.sha256')
        if target.is_symlink() or checksum.is_symlink() or not target.is_file() or not checksum.is_file():
            raise ValueError('Missing completed archive: ' + str(root))
        if checksum.read_text().strip() != file_hash(target) + '  ' + target.name:
            raise ValueError('Archive digest mismatch: ' + str(root))


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.PIPE,
                                   timeout=15, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).decode().strip()


def source_identity(repo, expected_sha=None):
    """Pin full tracked tree plus live compiled/numerical source contents."""
    repo = Path(repo).resolve()
    sha = git(repo, 'rev-parse', 'HEAD')
    if expected_sha is not None and sha != expected_sha:
        raise ValueError('Checkout HEAD differs from the full expected SHA')
    if git(repo, 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('A clean execution checkout is required; use an external plan/result directory')
    index_flags = git(repo, 'ls-files', '-v', '-z').split('\0')
    if any(entry and not entry.startswith('H ') for entry in index_flags):
        raise ValueError('Hidden index flags (assume-unchanged/skip-worktree) forbid clean-source attestation')
    names = git(repo, 'ls-files', '-z').split('\0')
    paths = {repo / name for name in names if name}
    for directory in ('statgpu', 'dev/benchmarks'):
        paths.update(p for p in (repo / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix.lower() in SOURCE_SUFFIXES)
    committed = {}
    for record in git(repo, 'ls-tree', '-r', '-z', 'HEAD').split('\0'):
        if record:
            header, name = record.split('\t', 1)
            mode, kind, object_id = header.split()
            if kind != 'blob' or mode not in ('100644', '100755'):
                raise ValueError('Only regular committed source files are supported')
            committed[name] = object_id
    hashes = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Execution source must contain regular tracked/source files: ' + str(path))
        name = path.relative_to(repo).as_posix()
        content = path.read_bytes()
        if name in committed:
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if blob != committed[name]:
                raise ValueError('Tracked working bytes differ from pinned Git blob: ' + name)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return {'git_commit': sha, 'git_tree': git(repo, 'rev-parse', 'HEAD^{tree}'),
            'working_tree': 'clean', 'content_sha256': digest(hashes), 'files_sha256': hashes,
            'scope': 'all tracked files plus live numerical/benchmark source and compiled extensions'}


def load_protocols(repo):
    """Only pure-stdlib planning/orchestration modules, never workload modules."""
    repo = Path(repo).resolve()
    existing = sys.modules.get('dev.benchmarks.run_qualification_campaign')
    if existing and Path(existing.__file__).resolve().parents[2] != repo:
        raise ValueError('One source checkout per driver process; start a fresh invocation')
    sys.path.insert(0, str(repo))
    registered = importlib.import_module('dev.benchmarks.run_qualification_campaign')
    independent = importlib.import_module('dev.benchmarks.independent_benchmark_registry')
    executor = importlib.import_module('dev.benchmarks.run_independent_benchmarks')
    for module in (registered, independent, executor):
        if Path(module.__file__).resolve().parents[2] != repo:
            raise ValueError('Protocol imported outside the exact checkout')
    return registered, independent, executor


def external_new(path, repo):
    path = Path(path).expanduser().absolute()
    # Resolving an existing symlink would conceal a result-root alias.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Output paths and their ancestors must not be symlinks')
    path = path.resolve()
    if path.is_relative_to(Path(repo).resolve()):
        raise ValueError('Plans and results must be outside the source checkout')
    if not path.parent.is_dir():
        raise ValueError('Output parent must already exist')
    for candidate in (path, path.with_name(path.name + '.tar.gz'), path.with_name(path.name + '.tar.gz.sha256')):
        if candidate.exists():
            raise ValueError('Refusing existing output or archive: ' + str(candidate))
    return path


def checked_budget(value, label='batch size'):
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError(label + ' must be in 1..64')
    return value


def checked_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1800:
        raise ValueError('Timeout must be finite, positive and <=1800 seconds')
    return value


def registered_args(module, argv, budget, timeout, index=0, count=1):
    if not isinstance(argv, list) or not all(type(arg) is str for arg in argv):
        raise ValueError('Registered configuration requires a list of CLI strings')
    if any(arg.split('=')[0] in OWNED_FLAGS for arg in argv):
        raise ValueError('Execution/output/batch/timeout flags are owned by the driver')
    parser = module.parser()
    parser.allow_abbrev = False
    try:
        args = parser.parse_args([*argv, '--max-invocations', str(budget), '--timeout', str(timeout),
                                  '--batch-index', str(index), '--batch-count', str(count)])
    except SystemExit as exc:
        raise ValueError('Invalid registered configuration') from exc
    if args.execute or args.output_dir is not None:
        raise ValueError('Nested execution/output is forbidden')
    return args


def partition_registered(plan, budget):
    """Exactly mirror the repository's modulo batch partition, including slots."""
    checked_budget(budget)
    count = math.ceil(len(plan['jobs']) / budget)
    if not count:
        raise ValueError('Empty registered plan')
    for index in range(count):
        child = dict(plan)
        child['jobs'] = plan['jobs'][index::count]
        child['expected_total_rows'] = sum(j['expected_row_count'] for j in child['jobs'])
        child['batch'] = dict(plan['batch'], index=index, count=count)
        child['execution_budget'] = dict(plan['execution_budget'], selected_invocations=len(child['jobs']),
            within_budget=True, worst_case_timeout_seconds=sum(j['resource_bounds']['timeout_seconds'] for j in child['jobs']))
        yield index, count, child


def plan_inventory(registered, independent, options):
    """Return fixed canonical batch configs/native plans. No workloads launched."""
    result, profiles = [], {}
    for profile in ('representative', 'expanded', 'custom'):
        if profile not in options['registered_profiles']:
            continue
        argv = options['registered_profiles'][profile]
        native = registered.build_plan(registered_args(registered, argv, options['batch_size'], options['timeout']))
        profiles[profile] = {'invocations': len(native['jobs']), 'expected_slots': native['expected_total_rows'],
                             'excluded_slots': native['excluded_slot_count']}
        for index, count, child in partition_registered(native, options['batch_size']):
            config = {'kind': 'registered', 'profile': profile, 'argv': argv, 'batch_index': index,
                      'batch_count': count, 'budget': options['batch_size'], 'timeout': options['timeout']}
            result.append((f'registered-{profile}-{index:04d}', config, child))
    inventory = independent.build_plan(expected_sha=options['expected_sha'], environment_id=options['environment_id'],
                                       validation_tier=options['validation_tier'])
    for job in inventory['jobs']:
        config = {'kind': 'independent', 'lane': job['id'], 'expected_sha': options['expected_sha'],
                  'environment_id': options['environment_id'], 'validation_tier': options['validation_tier'],
                  'budget': options['batch_size'], 'timeout': options['timeout']}
        native = independent.build_plan([job['id']], expected_sha=options['expected_sha'],
                    environment_id=options['environment_id'], validation_tier=options['validation_tier'])
        result.append(('independent-' + job['id'], config, native))
    return result, {'registered_profiles': profiles, 'independent': inventory['counts']}


def validate_options(options):
    keys = {'expected_sha', 'environment_id', 'validation_tier', 'batch_size', 'timeout', 'registered_profiles'}
    if not isinstance(options, dict) or set(options) != keys:
        raise ValueError('Unexpected plan options')
    if not isinstance(options['expected_sha'], str) or not HEX40.fullmatch(options['expected_sha']):
        raise ValueError('A full 40-character lowercase expected SHA is required')
    if not isinstance(options['environment_id'], str) or not options['environment_id'].strip() or len(options['environment_id']) > 256:
        raise ValueError('A nonempty explicit environment ID is required')
    if options['validation_tier'] not in ('local-minimal', 'local-full', 'remote-full'):
        raise ValueError('Unknown validation tier')
    checked_budget(options['batch_size']); checked_timeout(options['timeout'])
    profiles = options['registered_profiles']
    if not isinstance(profiles, dict) or not profiles or set(profiles) - {'representative', 'expanded', 'custom'}:
        raise ValueError('Unknown or empty registered profile set')
    if 'custom' in profiles and len(profiles) != 1:
        raise ValueError('Custom configuration cannot be mixed with predefined profiles')


def batch_metadata(identifier, config, native, config_sha256, plan_sha256):
    jobs = native['jobs']
    admitted = config['kind'] == 'registered' or jobs[0]['adapter'] is not None
    return {'id': identifier, 'kind': config['kind'],
            'config_sha256': config_sha256, 'plan_sha256': plan_sha256,
            'invocations': len(jobs) if admitted else 0,
            'expected_slots': native.get('expected_total_rows', sum(len(j.get('expected_cells', [])) for j in jobs)),
            'admitted': admitted, 'blocked_reason': None if admitted else jobs[0]['blocked_reason'],
            'prerequisites': (['clean pinned checkout', 'fresh output', 'installed runner dependencies',
                'requested physical GPUs; no fallback', 'native CPU/cgroup/memory/disk guards'] if config['kind'] == 'registered'
                else jobs[0]['adapter']['prerequisites'] +
                    ['Python: ' + p for p in jobs[0]['adapter'].get('required_python', [])] +
                    ['Executable: ' + p for p in jobs[0]['adapter'].get('required_executables', [])]
                    if admitted else ['Not admitted: ' + jobs[0]['blocked_reason']])}


def create_plan(repo, output, options):
    validate_options(options)
    repo = Path(repo).resolve()
    source = source_identity(repo, options['expected_sha'])
    registered, independent, _ = load_protocols(repo)
    batches, inventory = plan_inventory(registered, independent, options)
    if source_identity(repo, options['expected_sha']) != source:
        raise ValueError('Source changed while planning')
    output = external_new(output, repo)
    output.mkdir()
    manifest = {'schema_version': SCHEMA, 'created_at': utc_now(), 'repo_hint': str(repo),
                'driver_sha256': file_hash(__file__), 'options': options, 'source': source,
                'inventory': inventory, 'batches': [], 'performance_eligible': False,
                'machine_verified_exclusivity': False, 'canonical_dashboard_ready': False,
                'coverage_scope': 'Finite registered configurations and all 41 audited independent lanes; not all public APIs'}
    (output / 'batches').mkdir()
    for identifier, config, native in batches:
        directory = output / 'batches' / identifier
        directory.mkdir()
        write_json(directory / 'config.json', config)
        write_json(directory / 'plan.json', native)
        manifest['batches'].append(batch_metadata(identifier, config, native,
            file_hash(directory / 'config.json'), file_hash(directory / 'plan.json')))
    write_json(output / 'plan.json', manifest)
    commands = [
        '# Explicit single-batch command templates; this file never auto-executes.',
        '# Set DRIVER, PLAN, REPO and RESULTS to your verified paths before copying ONE selected command.',
        '# No loop or execute-all command is provided. Each invocation remains separately selected.',
        '# Catalog admitted/ready is a code-review admission, NOT verified runtime prerequisite availability.',
        '# Timeout totals below are conservative ceilings, not predicted durations.',
    ]
    for batch in manifest['batches']:
        native = read_json(output / 'batches' / batch['id'] / 'plan.json')
        cost = batch['invocations'] * options['timeout']
        commands += ['', '# ' + batch['id'] + ': ' + str(batch['invocations']) +
                     ' admitted invocations; timeout ceiling ' + str(cost) + ' seconds.',
                     '# Prerequisites: ' + '; '.join(batch['prerequisites']),
                     'python3 "$DRIVER" execute --repo "$REPO" --plan-dir "$PLAN" --results-dir "$RESULTS" --batch-id ' +
                     batch['id'] + ' --max-invocations ' + str(max(1, batch['invocations']))]
    (output / 'commands.txt').write_text('\n'.join(commands) + '\n', encoding='utf8')
    seal_directory(output)
    return manifest


def load_plan(directory, repo=None, *, canonical_check=True):
    directory = Path(directory).resolve()
    verify_seal(directory)
    manifest = read_json(directory / 'plan.json')
    if manifest.get('schema_version') != SCHEMA or manifest.get('driver_sha256') != file_hash(__file__):
        raise ValueError('Plan schema/driver fingerprint mismatch; create a new plan')
    validate_options(manifest['options'])
    ids = [b['id'] for b in manifest['batches']]
    if len(ids) != len(set(ids)) or any(not SAFE_ID.fullmatch(i) for i in ids):
        raise ValueError('Unsafe or duplicate batch IDs')
    for batch in manifest['batches']:
        for name in ('config', 'plan'):
            if file_hash(directory / 'batches' / batch['id'] / (name + '.json')) != batch[name + '_sha256']:
                raise ValueError('Batch digest mismatch')
    repo = Path(repo or manifest['repo_hint']).resolve()
    current = source_identity(repo, manifest['options']['expected_sha'])
    if current != manifest['source']:
        raise ValueError('Pinned source/tree/content changed; never reuse the old plan')
    protocols = load_protocols(repo)
    if canonical_check:
        batches, inventory = plan_inventory(protocols[0], protocols[1], manifest['options'])
        if inventory != manifest['inventory'] or [i for i, _, _ in batches] != ids:
            raise ValueError('Plan inventory differs from the current pinned registry')
        indexed = {b['id']: b for b in manifest['batches']}
        for identifier, config, native in batches:
            batch = indexed[identifier]
            expected_batch = batch_metadata(identifier, config, native, batch['config_sha256'], batch['plan_sha256'])
            if batch != expected_batch:
                raise ValueError('Batch budgets/admission/slot metadata differ from canonical plan: ' + identifier)
            stored_config = read_json(directory / 'batches' / identifier / 'config.json')
            stored_native = read_json(directory / 'batches' / identifier / 'plan.json')
            # Planning CPU visibility is not a source or command identity. The
            # unchanged native executor rechecks actual limits before each child.
            if config['kind'] == 'registered':
                native = {k: v for k, v in native.items() if k != 'cpu_limits'}
                stored_native = {k: v for k, v in stored_native.items() if k != 'cpu_limits'}
            if config != stored_config or native != stored_native:
                raise ValueError('Plan differs from reviewed protocol: ' + identifier)
    return manifest, repo, protocols


def select_batches(manifest, identifiers, max_invocations):
    checked_budget(max_invocations, 'selected invocation budget')
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError('Explicit unique --batch-id values are required; no execute-all mode')
    known = {b['id']: b for b in manifest['batches']}
    if any(i not in known for i in identifiers):
        raise ValueError('Unknown batch ID')
    selected = [known[i] for i in identifiers]
    if sum(b['invocations'] for b in selected) > max_invocations:
        raise ValueError('Selected batches exceed the explicit bounded invocation budget')
    return selected


def completed_attempt(path, batch, plan_sha256):
    verify_seal(path, archive=True)
    result = read_json(Path(path) / 'summary.json')
    if (result.get('schema_version') != 'statgpu-batch-attempt-v1' or result.get('batch_id') != batch['id']
            or result.get('plan_sha256') != plan_sha256 or result.get('batch_plan_sha256') != batch['plan_sha256']
            or result.get('status') not in TERMINAL):
        raise ValueError('Unknown/partial or mismatched existing attempt; preserve it and use a fresh results root')
    return result


@contextmanager
def terminating_signals():
    previous = {}
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('Termination signal ' + str(signum))
    try:
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, interrupted)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def finish_without_signals():
    previous = {}
    try:
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            if hasattr(signal, name):
                number = getattr(signal, name)
                previous[number] = signal.signal(number, signal.SIG_IGN)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def operator_metadata(path):
    if path is None:
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or not {'operator', 'statement', 'recorded_at', 'scope'} <= set(value)
            or set(value) - {'operator', 'statement', 'recorded_at', 'scope', 'environment_id', 'started_at', 'ended_at', 'device_uuid'}):
        raise ValueError('Attestation requires only operator, statement, recorded_at and scope strings')
    if any(type(v) is not str or not v.strip() or len(v) > 4000 for v in value.values()):
        raise ValueError('Attestation fields must be nonempty bounded strings')
    if any('REPLACE_' in text.upper() for text in value.values()):
        raise ValueError('Replace every attestation template placeholder before running')
    stamps = {}
    now = datetime.now(timezone.utc)
    for name in ('recorded_at', 'started_at', 'ended_at'):
        if name in value:
            try:
                stamp = datetime.fromisoformat(value[name].replace('Z', '+00:00'))
            except ValueError as exc:
                raise ValueError('Attestation timestamps must be ISO8601: ' + name) from exc
            if stamp.tzinfo is None or stamp > now:
                raise ValueError('Attestation timestamps must be timezone-aware observations, not future intervals: ' + name)
            stamps[name] = stamp
    if stamps.get('started_at', stamps['recorded_at']) > stamps['recorded_at']:
        raise ValueError('Attestation observation start cannot follow recorded_at')
    if 'ended_at' in stamps and (stamps['ended_at'] > stamps['recorded_at']
            or ('started_at' in stamps and stamps['ended_at'] < stamps['started_at'])):
        raise ValueError('Attestation observed interval must end between its start and recorded_at; omit ended_at before a new run')
    return {'kind': 'operator_provided_unverified', 'machine_verified': False,
            'applies_to': 'this new batch attempt only; never historical raw evidence', 'report': value}


def run_batches(plan_dir, results_dir, identifiers, *, repo=None, resume=False, max_invocations=21,
                attestation_path=None):
    manifest, repo, (registered, independent, executor) = load_plan(plan_dir, repo)
    selected = select_batches(manifest, identifiers, max_invocations)
    plan_dir = Path(plan_dir).resolve()
    plan_hash = file_hash(plan_dir / 'plan.json')
    results_dir = Path(results_dir).expanduser().absolute()
    if any(p.is_symlink() for p in (results_dir, *results_dir.parents)) or results_dir.resolve().is_relative_to(repo):
        raise ValueError('Results must be outside source, with no symlink ancestors')
    if results_dir.resolve().is_relative_to(plan_dir) or plan_dir.is_relative_to(results_dir.resolve()):
        raise ValueError('Plan and result roots must be disjoint')
    results_dir.mkdir(exist_ok=True)
    attestation = operator_metadata(attestation_path)
    if (attestation is not None and 'environment_id' in attestation['report']
            and attestation['report']['environment_id'] != manifest['options']['environment_id']):
        raise ValueError('Operator report environment_id differs from the immutable plan')
    # Preflight the entire selection before starting any costly batch.
    pending, reports = [], []
    for batch in selected:
        target = results_dir / batch['id']
        if target.exists():
            if not resume:
                raise ValueError('Existing attempt: use resume to skip sealed attempts, never overwrite')
            reports.append(completed_attempt(target, batch, plan_hash))
        else:
            external_new(target, repo)
            pending.append(batch)
    for batch in pending:
        source = source_identity(repo, manifest['options']['expected_sha'])
        if source != manifest['source']:
            raise ValueError('Pinned source changed before next batch')
        target = results_dir / batch['id']
        target.mkdir()
        report = {'schema_version': 'statgpu-batch-attempt-v1', 'batch_id': batch['id'],
                  'plan_sha256': plan_hash, 'batch_plan_sha256': batch['plan_sha256'],
                  'source_before': source, 'source_after': None, 'started_at': utc_now(),
                  'environment_id': manifest['options']['environment_id'],
                  'validation_tier': manifest['options']['validation_tier'],
                  'status': 'failed', 'error': None, 'native_directory': 'native',
                  'native_summary_sha256': None, 'native_archive_sha256': None,
                  'performance_eligible': False, 'machine_verified_exclusivity': False,
                  'canonical_dashboard_ready': False}
        write_json(target / 'started.json', report)
        if attestation is not None:
            write_json(target / 'operator_attestation.json', attestation)
        native = read_json(plan_dir / 'batches' / batch['id'] / 'plan.json')
        config = read_json(plan_dir / 'batches' / batch['id'] / 'config.json')
        try:
            with terminating_signals():
                if not batch['admitted']:
                    report.update(status='blocked', error=batch['blocked_reason'])
                else:
                    # Only canonical native dispatch, with native resource guards,
                    # own-session children, qualification, snapshots and archives.
                    if config['kind'] == 'registered':
                        registered.execute(native, target / 'native', config['timeout'])
                    else:
                        executor.execute(native, target / 'native', config['timeout'], config['budget'])
                    child = read_json(target / 'native' / 'summary.json')
                    statuses = [r.get('status') for r in child.get('results', [])]
                    stop = str(child.get('stop_reason') or '')
                    if child.get('status') == 'complete' and len(statuses) == len(native['jobs']) and all(s == 'complete' for s in statuses):
                        report['status'] = 'complete'
                    elif 'interrupt' in stop.lower() or 'KeyboardInterrupt' in stop:
                        report['status'] = 'interrupted'
                    elif statuses and all(s == 'blocked' for s in statuses) or 'resource' in stop.lower():
                        report['status'] = 'blocked'
                    report['error'] = child.get('stop_reason')
        except KeyboardInterrupt as exc:
            report.update(status='interrupted', error=str(exc))
        except Exception as exc:
            report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        finally:
            # SIGKILL or host loss cannot finalize; resume detects those attempts
            # as unknown partial and refuses to rerun or replace them silently.
            with finish_without_signals():
                try:
                    report['source_after'] = source_identity(repo, manifest['options']['expected_sha'])
                    if report['source_after'] != source:
                        raise ValueError('Source changed during execution')
                except Exception as exc:
                    report.update(status='failed', error='Source verification: ' + str(exc))
                for field, path in [('native_summary_sha256', target / 'native' / 'summary.json'),
                                    ('native_archive_sha256', target / 'native.tar.gz')]:
                    if path.is_file() and not path.is_symlink():
                        report[field] = file_hash(path)
                report['finished_at'] = utc_now()
                write_json(target / 'summary.json', report)
                seal_directory(target, archive=True)
        reports.append(report)
        print(batch['id'] + ': ' + report['status'], flush=True)
        if report['status'] == 'interrupted' or (report['error'] and any(
                token in report['error'].lower() for token in ('source', 'resource'))):
            break
    return reports


def _artifact_paths(folder, qualification):
    raw = folder / 'raw.json'
    return {'raw_path': str(raw) if raw.is_file() else None,
            'raw_sha256': file_hash(raw) if raw.is_file() and not raw.is_symlink() else None,
            'qualification_path': str(qualification) if qualification.is_file() else None,
            'qualification_sha256': file_hash(qualification) if qualification.is_file() and not qualification.is_symlink() else None}


def registered_status(job, folder, native_summary, module):
    result = next((r for r in native_summary.get('results', []) if r.get('id') == job['id']), None)
    if not folder.exists():
        return 'not_run', 'No attempt directory', {}, None
    try:
        process = read_json(folder / 'process.json')
        qualified = read_json(folder / 'qualification.json')
        raw = folder / 'raw.json'
        if process.get('exit_code') in (77, 86) or str(process.get('reason') or '').startswith('resource_guard:'):
            return 'blocked', process.get('reason') or 'Prerequisite/resource guard', qualified, None
        reassessed = module.assess(raw, job, process.get('exit_code'), job['expected_source']['git_commit'])
        before = read_json(folder / 'runtime.before.json')
        after = read_json(folder / 'runtime.after.json')
        good = (result is not None and result.get('status') == 'complete'
                and qualified.get('correctness', {}).get('status') == 'complete'
                and reassessed.get('correctness', {}).get('status') == 'complete'
                and reassessed.get('expanded_inventory', {}).get('exact_match') is True
                and before.get('before', {}).get('cap_verified') is True
                and after.get('after', {}).get('cap_verified') is True
                and after.get('exit_code') == 0 and process.get('exit_code') == 0 and not process.get('reason'))
        return ('complete' if good else 'failed'), None if good else 'Native qualification/process/runtime is incomplete or failed', qualified, reassessed
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return 'failed', 'Missing/malformed native evidence: ' + str(exc), {}, None


def native_source_matches(path, expected, exact_native):
    try:
        observed = read_json(path)
        hashes = observed.get('files_sha256')
        if not isinstance(hashes, dict) or not hashes:
            return False
        return (observed == exact_native
                and observed.get('git_commit') == expected['git_commit']
                and observed.get('working_tree') == 'clean'
                and all(expected['files_sha256'].get(name) == value for name, value in hashes.items())
                and observed.get('content_sha256') == hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def aggregate(plan_dir, results_dir, output, *, repo=None):
    manifest, repo, (registered, independent, executor) = load_plan(plan_dir, repo)
    plan_dir, results_dir = Path(plan_dir).resolve(), Path(results_dir).resolve()
    plan_hash = file_hash(plan_dir / 'plan.json')
    output = external_new(output, repo)
    if output.is_relative_to(plan_dir) or output.is_relative_to(results_dir):
        raise ValueError('Aggregate must be separate from immutable plans and results')
    ledger = {'schema_version': 'statgpu-batch-coverage-ledger-v1', 'created_at': utc_now(),
              'plan_sha256': plan_hash, 'source': manifest['source'], 'inventory': manifest['inventory'],
              'batches': [], 'slots': [], 'performance_eligible': False, 'machine_verified_exclusivity': False,
              'canonical_dashboard_ready': False, 'coverage_scope': manifest['coverage_scope'],
              'notes': ['complete means native protocol/qualification completion, not universal correctness or performance eligibility',
                        'independent correctness_status may remain unavailable even for a complete protocol',
                        'missing slots and resource/capability exclusions remain separate; no old raw artifact is upgraded']}
    exact_native_sources = {'registered': registered.fingerprint(), 'independent': executor.source_fingerprint()}
    excluded_profiles = set()
    for batch in manifest['batches']:
        target = results_dir / batch['id']
        native = read_json(plan_dir / 'batches' / batch['id'] / 'plan.json')
        config = read_json(plan_dir / 'batches' / batch['id'] / 'config.json')
        batch_status = 'not_run' if batch['admitted'] else 'blocked'
        reason = batch['blocked_reason']
        attempt = None
        if target.exists():
            try:
                attempt = completed_attempt(target, batch, plan_hash)
                batch_status, reason = attempt['status'], attempt.get('error')
            except (OSError, ValueError, KeyError) as exc:
                batch_status, reason = 'unknown_partial', str(exc)
        ledger['batches'].append(dict(batch, status=batch_status, reason=reason, evidence_directory=str(target)))
        summary = {}
        if attempt and (target / 'native' / 'summary.json').is_file():
            try:
                summary = read_json(target / 'native' / 'summary.json')
                if file_hash(target / 'native' / 'summary.json') != attempt['native_summary_sha256']:
                    raise ValueError('Native summary identity mismatch')
            except (OSError, ValueError, KeyError) as exc:
                batch_status, reason = 'unknown_partial', str(exc)
                summary = {}
        native_finalized = not batch['admitted']
        finalizer_error = None
        if attempt is not None and batch['admitted']:
            try:
                verify_seal(target / 'native', archive=True)
                if file_hash(target / 'native.tar.gz') != attempt.get('native_archive_sha256'):
                    raise ValueError('Native archive identity mismatch')
                native_finalized = True
            except (OSError, ValueError, KeyError) as exc:
                finalizer_error = 'Native finalizer did not complete intact: ' + str(exc)
        ledger['batches'][-1].update(status=batch_status, reason=reason)
        source_valid = (attempt is not None and native_finalized and attempt.get('source_before') == manifest['source']
                        and attempt.get('source_after') == manifest['source']
                        and summary.get('source_changed') is not True
                        and (not batch['admitted'] or all(native_source_matches(target / 'native' / name, manifest['source'], exact_native_sources[config['kind']])
                             for name in ('source-before.json', 'source-after.json'))))
        if attempt is not None and not source_valid:
            batch_status, reason = 'failed', finalizer_error or 'Execution source provenance does not match the immutable plan'
            ledger['batches'][-1].update(status=batch_status, reason=reason)
        for job in native['jobs']:
            common = {'batch_id': batch['id'], 'lane': config['kind'], 'profile': config.get('profile'),
                      'job_id': job['id'], 'configuration_id': job.get('configuration_id', job.get('config_id')),
                      'source_commit': manifest['source']['git_commit'], 'source_tree': manifest['source']['git_tree'],
                      'source_content_sha256': manifest['source']['content_sha256'],
                      'status': batch_status, 'reason': reason, 'performance_eligible': False}
            if config['kind'] == 'registered':
                folder = target / 'native' / job['id']
                common.update(_artifact_paths(folder, folder / 'qualification.json'))
                qualified, reassessed = {}, None
                if source_valid and batch_status != 'unknown_partial':
                    state, why, qualified, reassessed = registered_status(job, folder, summary, registered)
                    if state == 'complete' and not all(native_source_matches(folder / name, manifest['source'], exact_native_sources['registered'])
                            for name in ('source-before.json', 'source-after.json')):
                        state, why = 'failed', 'Per-job source identity mismatch or missing'
                    common.update(status=state, reason=why)
                missing = Counter((reassessed or qualified).get('expanded_inventory', {}).get('missing', []))
                for index, expected in enumerate(job['expected_rows']):
                    slot = dict(common, slot_id=batch['id'] + '/' + job['expected_slots'][index]['id'], expected=expected,
                                expected_status='supported', correctness_status=common['status'])
                    key = json.dumps(expected, sort_keys=True, allow_nan=False)
                    slot['missing'] = (common['status'] in ('not_run', 'unknown_partial') or missing[key] > 0
                                       or common['raw_path'] is None or (common['status'] != 'complete'
                                       and not (reassessed or qualified).get('expanded_inventory', {}).get('observed_rows')))
                    if missing[key] > 0:
                        missing[key] -= 1
                    if expected.get('not_applicable') is True:
                        slot.update(status='excluded', expected_status='not_applicable', correctness_status='not_applicable',
                                    protocol_observation_status=common['status'],
                                    reason='Registered host-only case is not a supported GPU numerical slot')
                    ledger['slots'].append(slot)
            else:
                folder = target / 'native' / (job['id'] + '-' + job['config_id'][:12])
                common.update(_artifact_paths(folder, folder / 'qualified.json'))
                correctness = 'unavailable'
                if job['adapter'] is None:
                    common.update(status='excluded' if job['disposition'] == 'deduplicated' else 'blocked', reason=job['blocked_reason'])
                elif source_valid and batch_status != 'unknown_partial' and folder.exists():
                    try:
                        qualified = read_json(folder / 'qualified.json')
                        result = next((r for r in summary.get('results', []) if r.get('config_id') == job['config_id']), {})
                        if qualified.get('status') == 'blocked':
                            common.update(status='blocked', reason=qualified.get('reason'))
                        else:
                            rechecked = executor.assess_artifact(job, folder, qualified.get('process', {}))
                            valid = (result.get('status') == 'complete' and qualified.get('status') == 'complete'
                                     and rechecked.get('status') == 'complete' and qualified.get('raw_sha256') == common['raw_sha256']
                                     and native_source_matches(folder / 'source-after.json', manifest['source'], exact_native_sources['independent']))
                            common.update(status='complete' if valid else 'failed', reason=None if valid else rechecked.get('reason', 'Native qualification incomplete'))
                            correctness = rechecked.get('correctness_status', 'unavailable')
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        common.update(status='failed', reason=str(exc))
                elif source_valid and batch_status != 'unknown_partial':
                    common.update(status='not_run', reason='No independent attempt directory')
                for index, cell in enumerate(job['expected_cells'] or [None]):
                    ledger['slots'].append(dict(common, slot_id=batch['id'] + '/' + job['config_id'] + '/' + str(index),
                        expected={'cell': cell}, expected_status='supported' if job['adapter'] else 'blocked_or_deduplicated',
                        correctness_status=correctness,
                        missing=(job['adapter'] is not None and common['status'] != 'complete')))
        profile = config.get('profile')
        if config['kind'] == 'registered' and profile not in excluded_profiles:
            excluded_profiles.add(profile)
            for index, exclusion in enumerate(native.get('capability_exclusions', [])):
                for seed in exclusion['seeds']:
                    for backend in exclusion['backends']:
                        ledger['slots'].append({'slot_id': f'excluded/{profile}/{index}/{seed}/{backend}',
                            'batch_id': None, 'lane': 'registered', 'profile': profile, 'job_id': None,
                            'configuration_id': None, 'expected': dict(exclusion, seed=seed, backend=backend),
                            'expected_status': 'excluded', 'status': 'excluded', 'reason': exclusion['reason'],
                            'missing': False, 'correctness_status': 'not_applicable', 'performance_eligible': False,
                            'source_commit': manifest['source']['git_commit'], 'source_tree': manifest['source']['git_tree'],
                            'source_content_sha256': manifest['source']['content_sha256']})
    ledger['counts'] = {'batches': dict(Counter(b['status'] for b in ledger['batches'])),
                        'slots': dict(Counter(s['status'] for s in ledger['slots'])),
                        'missing_supported_slots': sum(s['missing'] for s in ledger['slots'] if s['expected_status'] == 'supported')}
    ledger['full_catalog_complete'] = False
    ledger['all_admitted_slots_complete'] = all(s['status'] == 'complete' for s in ledger['slots'] if s['expected_status'] == 'supported')
    output.mkdir()
    write_json(output / 'coverage_ledger.json', ledger)
    write_json(output / 'summary.json', {k: v for k, v in ledger.items() if k not in ('slots', 'batches')})
    seal_directory(output)
    return ledger


def parser():
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = root.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='No workloads; include every independent lane', allow_abbrev=False)
    plan.add_argument('--repo', required=True, type=Path)
    plan.add_argument('--expected-sha', required=True)
    plan.add_argument('--environment-id', required=True)
    plan.add_argument('--validation-tier', choices=['local-minimal', 'local-full', 'remote-full'], default='local-full')
    plan.add_argument('--output-dir', required=True, type=Path)
    plan.add_argument('--profile', choices=['representative', 'expanded', 'both'], default='both')
    plan.add_argument('--registered-config', type=Path, help='Optional {"argv": [...]} for explicit bounded scales/cases/thread profiles')
    plan.add_argument('--batch-size', type=int, default=21)
    plan.add_argument('--timeout', type=float, default=1800)
    plan.add_argument('--warmup', type=int, choices=range(0, 11))
    plan.add_argument('--repeats', type=int, choices=range(1, 11))
    plan.add_argument('--cpu-threads', nargs='+', type=int, choices=[1, 2, 4, 8], help='Must include 1 and fit observed native quota; separate CPU profiles, no extra parallel jobs')
    for name in ('execute', 'resume'):
        run = sub.add_parser(name, help='Explicit IDs only; never execute all', allow_abbrev=False)
        run.add_argument('--plan-dir', required=True, type=Path)
        run.add_argument('--results-dir', required=True, type=Path)
        run.add_argument('--repo', type=Path, help='Relocated exact clean checkout, not an alternate source')
        run.add_argument('--batch-id', action='append', required=True)
        run.add_argument('--max-invocations', type=int, default=21)
        run.add_argument('--operator-attestation', type=Path)
    agg = sub.add_parser('aggregate', help='No workloads; include not-run/blocked/missing slots', allow_abbrev=False)
    agg.add_argument('--plan-dir', required=True, type=Path)
    agg.add_argument('--results-dir', required=True, type=Path)
    agg.add_argument('--output-dir', required=True, type=Path)
    agg.add_argument('--repo', type=Path)
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == 'plan':
            if args.registered_config:
                if args.warmup is not None or args.repeats is not None or args.cpu_threads is not None:
                    raise ValueError('Put registered controls in the config or flags, not both')
                config = read_json(args.registered_config)
                if not isinstance(config, dict) or set(config) != {'argv'}:
                    raise ValueError('Registered config must contain only argv')
                profiles = {'custom': config['argv']}
            else:
                names = ['representative', 'expanded'] if args.profile == 'both' else [args.profile]
                extra = []
                for name in ('warmup', 'repeats'):
                    if getattr(args, name) is not None:
                        extra += ['--' + name, str(getattr(args, name))]
                if args.cpu_threads is not None:
                    extra += ['--cpu-threads', *map(str, args.cpu_threads)]
                profiles = {name: ([] if name == 'representative' else ['--profile', 'expanded']) + extra for name in names}
            value = create_plan(args.repo, args.output_dir, {'expected_sha': args.expected_sha,
                'environment_id': args.environment_id, 'validation_tier': args.validation_tier,
                'registered_profiles': profiles, 'batch_size': args.batch_size, 'timeout': args.timeout})
            print(json.dumps({'plan': str(args.output_dir), 'inventory': value['inventory'], 'batch_ids': [b['id'] for b in value['batches']]}, indent=2))
            return 0
        if args.command in ('execute', 'resume'):
            values = run_batches(args.plan_dir, args.results_dir, args.batch_id, repo=args.repo,
                resume=args.command == 'resume', max_invocations=args.max_invocations,
                attestation_path=args.operator_attestation)
            return 0 if values and all(r['status'] == 'complete' for r in values) else 2
        ledger = aggregate(args.plan_dir, args.results_dir, args.output_dir, repo=args.repo)
        print(json.dumps({'output': str(args.output_dir), 'counts': ledger['counts'], 'performance_eligible': False}, indent=2))
        return 0 if ledger['all_admitted_slots_complete'] else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
