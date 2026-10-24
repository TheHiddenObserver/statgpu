"""Explicit, provenance-preserving coordinator for registered and independent plans.

Planning never imports numerical runners. This is not a claim that every public
API/configuration is covered, nor a means of granting performance eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from dev.benchmarks import run_qualification_campaign as registered

SCHEMA = 'statgpu-full-benchmark-config-v1'
TOP_KEYS = {'schema_version', 'registered', 'independent'}
INDEPENDENT_KEYS = {'lanes', 'validation_tier', 'environment_id', 'expected_sha', 'configurations', 'max_invocations'}


def load_configuration(path=None):
    config = ({'schema_version': SCHEMA, 'registered': {'argv': []},
               'independent': {'lanes': None}} if path is None else registered.read_json(path))
    if not isinstance(config, dict) or set(config) - TOP_KEYS or config.get('schema_version') != SCHEMA:
        raise ValueError('Invalid full benchmark configuration schema/keys')
    if not any(key in config for key in ('registered', 'independent')):
        raise ValueError('At least one explicit lane is required')
    if 'registered' in config:
        lane = config['registered']
        if not isinstance(lane, dict) or set(lane) != {'argv'}:
            raise ValueError('Registered lane accepts only an argv list')
        argv = lane['argv']
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError('Registered argv must be a string list')
        # Output/execution are owned by this coordinator, not nested input.
        if any(arg.split('=')[0] in {'--execute', '--output-dir'} for arg in argv):
            raise ValueError('Nested execution/output flags are forbidden')
    if 'independent' in config:
        lane = config['independent']
        if not isinstance(lane, dict) or set(lane) - INDEPENDENT_KEYS:
            raise ValueError('Unknown independent lane option')
        budget = lane.get('max_invocations', 21)
        if type(budget) is not int or not 1 <= budget <= 64:
            raise ValueError('Independent max_invocations must be 1..64')
        lanes = lane.get('lanes')
        if lanes is not None and (not isinstance(lanes, list) or not lanes or
                                  not all(isinstance(item, str) and item for item in lanes) or
                                  len(lanes) != len(set(lanes))):
            raise ValueError('Independent lanes must be unique nonempty names')
    return config


def build_plan(config):
    plans = {}
    if 'registered' in config:
        args = registered.parser().parse_args(config['registered']['argv'])
        plans['registered'] = registered.build_plan(args)
    if 'independent' in config:
        from dev.benchmarks import independent_benchmark_registry as independent
        settings = {key: value for key, value in config['independent'].items() if key != 'max_invocations'}
        plans['independent'] = independent.build_plan(**settings)
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'schema_version': 'statgpu-full-benchmark-plan-v1',
            'configuration': config, 'configuration_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
            'plan_only': True, 'lanes': plans, 'source': registered.fingerprint(),
            'coverage_scope': 'Explicit registered configurations and independently catalogued protocols, not all APIs',
            'performance_status': 'UNVERIFIED_EXCLUSIVITY', 'performance_eligible': False,
            'canonical_dashboard_ready': False, 'comparisons': []}


def execute(plan, destination, timeout=1800):
    """Dispatch approved child protocols; retain partial/blocked result identity."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError('Timeout must be finite, >0 and <=1800 seconds')
    independent_config = plan['configuration'].get('independent')
    if independent_config is not None and not independent_config.get('lanes'):
        raise ValueError('Execution requires explicit independent lanes; inventory-all is plan-only')
    destination = registered.validate_destination(Path(destination))
    destination.mkdir()
    report = {'schema_version': 'statgpu-full-benchmark-result-v1', 'status': 'failed',
              'lanes': {}, 'coverage_claim': 'selected protocols only; no full API/catalog certification',
              'full_catalog_complete': False, 'performance_status': 'UNVERIFIED_EXCLUSIVITY',
              'performance_eligible': False, 'canonical_dashboard_ready': False, 'comparisons': []}
    before = registered.fingerprint()
    try:
        registered.write_json(destination / 'plan.json', plan)
        registered.write_json(destination / 'source-before.json', before)
        if not isinstance(plan.get('source'), dict) or registered.source_changed(plan['source'], before):
            raise ValueError('Source changed since plan construction')
        if not before.get('git_commit') or before.get('working_tree') != 'clean':
            raise ValueError('Execution requires exact clean source')
        for name, child_plan in plan['lanes'].items():
            now = registered.fingerprint()
            if registered.source_changed(before, now):
                raise ValueError('Source changed between lanes')
            error = registered.resource_guard(destination, 1)
            if error:
                raise ValueError('Resource guard: ' + error)
            child_destination = destination / name
            if name == 'registered':
                bounds = [job.get('resource_bounds', {}).get('timeout_seconds', timeout)
                          for job in child_plan.get('jobs', [])]
                if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in bounds):
                    raise ValueError('Invalid planned child timeout')
                effective_timeout = min([timeout, *bounds])
                code = registered.execute(child_plan, child_destination, effective_timeout)
                child_report = registered.read_json(child_destination / 'summary.json')
            else:
                from dev.benchmarks import run_independent_benchmarks as independent
                effective_timeout = timeout
                child_report = independent.execute(child_plan, child_destination, timeout=timeout,
                                                   max_invocations=independent_config.get('max_invocations', 21))
                code = 0 if child_report.get('status') == 'complete' else 2
            report['lanes'][name] = {'exit_code': code, 'evidence_directory': name,
                                     'status': 'complete' if code == 0 else 'incomplete_or_failed',
                                     'coverage_status': child_report.get('coverage_status', 'selected_protocols_only'),
                                     'effective_timeout_seconds': effective_timeout}
            stop_reason = str(child_report.get('stop_reason') or '')
            if child_report.get('source_changed') or any(token in stop_reason.lower() for token in ('resource', 'source_changed', 'source changed', 'interrupt')):
                raise ValueError('Child safety stop: ' + stop_reason)
            # Numerical/coverage failures do not silently erase subsequent lanes.
        report['status'] = ('complete_for_selected_protocols' if
            len(report['lanes']) == len(plan['lanes']) and all(
                lane['exit_code'] == 0 for lane in report['lanes'].values()) else 'incomplete_or_failed')
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        after = registered.fingerprint()
        report['source_changed'] = registered.source_changed(before, after)
        if report['source_changed']:
            report['status'] = 'failed'
        registered.write_json(destination / 'source-after.json', after)
        registered.write_json(destination / 'summary.json', report)
        registered.freeze_and_archive(destination)
    return 0 if report['status'] == 'complete_for_selected_protocols' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Strict JSON configuration; default emits all-inventory plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 1800:
        parser.error('Timeout must be >0 and <=1800 seconds')
    try:
        plan = build_plan(load_configuration(args.config))
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.output_dir is None:
            parser.error('--execute requires --output-dir')
        return execute(plan, args.output_dir, args.timeout)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
