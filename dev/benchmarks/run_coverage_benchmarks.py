#!/usr/bin/env python3
"""Auditable benchmark-suite dispatcher; never converts missing coverage into PASS.

Default is inventory-only. Use --execute to launch reviewed local runners.
No SSH, credential access, shell expansion, dependency installation or publication.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('full_coverage_manifest.json')
REVIEWED_RUNNERS = {
    'benchmark_release_refresh', 'benchmark_classification_metrics',
    'benchmark_workflow_coverage', 'benchmark_statistical_coverage',
    'benchmark_model_coverage', 'benchmark_learning_coverage', 'benchmark_cv_coverage',
}
FAIL = {'fail', 'failed', 'failure', 'error', 'incorrect', 'timeout'}
INCOMPLETE = {'partial', 'skipped', 'skip', 'unavailable', 'blocked', 'not_run', 'pending'}


def read_manifest(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    entries = value.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest requires nonempty entries list')
    ids = [e.get('id') for e in entries]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('entry IDs must be unique nonempty strings')
    return value


def command(entry, output, backend, smoke):
    if entry.get('safe_to_execute') is not True:
        raise ValueError('Entry has not passed local-execution review')
    template = entry.get('smoke_argv') if smoke else entry.get('argv')
    if not isinstance(template, list) or not all(isinstance(x, str) for x in template):
        raise ValueError('missing explicit invocation for selected profile')
    replacements = {'{python}': sys.executable, '{output}': str(output), '{backend}': backend}
    args = [replacements.get(x, x) for x in template]
    if len(args) < 3 or args[0] != sys.executable:
        raise ValueError('Only this Python executable may be launched')
    if args[1] == '-m':
        prefix = 'dev.benchmarks.'
        if not args[2].startswith(prefix) or args[2][len(prefix):] not in REVIEWED_RUNNERS:
            raise ValueError('Module not in reviewed runner allowlist')
    else:
        script = Path(args[1])
        resolved = (ROOT / script).resolve()
        if (script.is_absolute() or resolved.parent != ROOT / 'dev' / 'benchmarks'
                or resolved.suffix != '.py' or resolved.stem not in REVIEWED_RUNNERS):
            raise ValueError('Script not in reviewed runner allowlist')
    if '{output}' not in template or args.count('--output') + args.count('--out') != 1:
        raise ValueError('An explicit unique output file is required')
    if any('{' in x or '}' in x for x in args):
        raise ValueError('Unresolved template token')
    return args


def disposition(document):
    """Read all explicit status fields, not just a process's successful exit."""
    found = []
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'status' and isinstance(item, str):
                    found.append(item.lower())
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(document)
    if any(s in FAIL for s in found):
        return 'failed'
    if any(s in INCOMPLETE for s in found):
        return 'partial'
    if not any(s in {'pass', 'passed', 'success', 'complete'} for s in found):
        return 'failed'
    return 'passed'


def fingerprint(manifest):
    paths = [Path(manifest).resolve(), Path(__file__).resolve()]
    paths += sorted((ROOT/'statgpu').rglob('*.py'))
    paths += [Path(__file__).with_name(name+'.py') for name in sorted(REVIEWED_RUNNERS)]
    paths += [Path(__file__).with_name(name) for name in (
        'benchmark_cv_models.py', 'cv_external.py', 'cv_source.py',
        'cv_source_schema.json', 'learning_execution_trace.py',
    )]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        tracked_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
    except (OSError, subprocess.CalledProcessError):
        head, tracked_dirty = None, True
    return {'git_commit': head, 'tracked_worktree_dirty': tracked_dirty,
            'content_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'files_sha256': hashes,
            'scope': 'Python numerical sources, available reviewed runners, dispatcher and manifest; includes ignored files'}


def launch(args, output, timeout, log):
    if output.exists():
        raise ValueError('Refusing existing output: use a fresh output directory')
    start = time.monotonic()
    with log.open('wb') as stream:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=(os.name == 'posix'))
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
            return {'status': 'failed', 'reason': 'timeout', 'elapsed_seconds': time.monotonic()-start}
    row = {'exit_code': rc, 'elapsed_seconds': time.monotonic()-start}
    if not output.exists():
        return dict(row, status='failed', reason='Runner did not produce JSON')
    try:
        document = json.loads(output.read_text(encoding='utf8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        status = disposition(document)
    except (ValueError, OSError) as exc:
        return dict(row, status='failed', reason=f'Invalid result JSON: {exc}')
    if rc != 0:
        row['reported_artifact_status'] = status
        status = 'failed'
        row['reason'] = 'Runner exited nonzero; retained JSON cannot override process failure'
    # Missing dependencies with a successful process may be partial; crashes never are.
    return dict(row, status=status, artifact_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Use explicit tiny CPU commands; not performance evidence')
    parser.add_argument('--suites', nargs='+')
    parser.add_argument('--backend', choices=['numpy','cupy','torch'], default='numpy')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--allow-incomplete', action='store_true', help='Permit partial/skipped/blocked exit, never numerical failures')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    manifest = read_manifest(args.manifest)
    entries = manifest['entries']
    if args.suites:
        unknown = set(args.suites)-{e['id'] for e in entries}
        if unknown:
            parser.error('Unknown suites: '+', '.join(sorted(unknown)))
        entries = [e for e in entries if e['id'] in args.suites]
    if not args.execute:
        print(json.dumps({'inventory_only': True, 'entries': entries}, indent=2, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        parser.error('--execute requires --output-dir')
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error('Use a new or empty output directory; results are never overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    before = fingerprint(args.manifest)
    results = []
    for index, entry in enumerate(entries):
        output = destination/f'{index:03d}.json'
        log = destination/f'{index:03d}.log'
        row = {'id': entry['id'], 'family': entry.get('family'),
               'artifact': output.name, 'log': log.name,
               'declared_evidence_gaps': entry.get('evidence_gaps', [])}
        try:
            cmd = command(entry, output, args.backend, args.smoke)
        except ValueError as exc:
            row.update(status='blocked', reason=str(exc))
        else:
            row['command'] = cmd
            try:
                row.update(launch(cmd, output, args.timeout, log))
            except Exception as exc:
                row.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        results.append(row)
        print(entry['id']+': '+row['status'], flush=True)
    after = fingerprint(args.manifest)
    changed = before['git_commit'] != after['git_commit'] or before['content_sha256'] != after['content_sha256']
    failed = changed or any(r['status']=='failed' for r in results)
    incomplete = any(r['status'] in {'blocked','partial'} for r in results)
    status = 'failed' if failed else 'partial' if incomplete else 'complete'
    report = {'schema_version': 'statgpu-coverage-execution-v1', 'status': status,
              'profile': 'CPU-harness-smoke' if args.smoke else 'benchmark-candidate',
              'source': before, 'source_after': after, 'source_changed': changed,
              'results': results, 'manifest_capability_count': len(manifest.get('capabilities', [])),
              'capability_status_counts': dict(Counter(c.get('support_status', 'unknown') for c in manifest.get('capabilities', []))),
              'global_uncovered_axes': manifest.get('global_uncovered_axes', []),
              'selected_suite_ids': [e['id'] for e in entries],
              'canonical_dashboard_ready': False,
              'note': 'Complete means selected invocations succeeded, NOT all API combinations or physical backend coverage.'}
    (destination/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n',encoding='utf8')
    return 2 if failed or (incomplete and not args.allow_incomplete) else 0


if __name__ == '__main__':
    raise SystemExit(main())
