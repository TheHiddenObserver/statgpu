"""Synthetic orchestration/ledger tests. No numerical benchmark workloads."""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / 'benchmarks/data_update'
sys.path.insert(0, str(TOOLS))

import batch_driver as d

SHA = 'a' * 40
SOURCE = {'git_commit': SHA, 'git_tree': 'b' * 40, 'working_tree': 'clean',
          'content_sha256': d.hashlib.sha256(json.dumps({'synthetic.py': 'c' * 64}, sort_keys=True).encode()).hexdigest(), 'files_sha256': {'synthetic.py': 'c' * 64}}


def options():
    return {'expected_sha': SHA, 'environment_id': 'synthetic-test-environment',
            'validation_tier': 'local-full', 'batch_size': 21, 'timeout': 1,
            'registered_profiles': {'representative': []}}


def job(i=0):
    row = {'case': 'synthetic', 'backend': 'numpy', 'n': 16, 'p': 2, 'seed': i, 'input_dtype': 'float64'}
    return {'id': f'job-{i}', 'configuration_id': str(i), 'expected_rows': [row],
            'expected_row_count': 1, 'expected_slots': [{'id': f'job-{i}-r0', 'row': row}],
            'expected_source': {'git_commit': SHA}, 'resource_bounds': {'timeout_seconds': 1},
            'suite': 'synthetic', 'backend': 'numpy', 'cases': ['synthetic']}


def native_plan(count=1):
    return {'jobs': [job(i) for i in range(count)], 'expected_total_rows': count,
            'batch': {'index': 0, 'count': 1, 'all_invocations': count, 'all_expected_rows': count},
            'execution_budget': {'max_invocations': 21, 'selected_invocations': count, 'within_budget': count <= 21,
                                 'worst_case_timeout_seconds': count},
            'capability_exclusions': [], 'excluded_slot_count': 0, 'cpu_limits': {'thread_cap': 1}}


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.plans = self.root / 'plan'
        self.plans.mkdir()
        self.results = self.root / 'results'
        self.config = {'kind': 'registered', 'profile': 'representative', 'argv': [], 'batch_index': 0,
                       'batch_count': 1, 'budget': 21, 'timeout': 1}
        self.native = native_plan()
        self.identifier = 'registered-representative-0000'
        directory = self.plans / 'batches' / self.identifier
        directory.mkdir(parents=True)
        d.write_json(directory / 'config.json', self.config)
        d.write_json(directory / 'plan.json', self.native)
        self.batch = d.batch_metadata(self.identifier, self.config, self.native,
                                     d.file_hash(directory / 'config.json'), d.file_hash(directory / 'plan.json'))
        self.manifest = {'schema_version': d.SCHEMA, 'driver_sha256': d.file_hash(d.__file__),
                         'options': options(), 'source': SOURCE, 'batches': [self.batch],
                         'repo_hint': str(self.repo), 'inventory': {}, 'coverage_scope': 'synthetic'}
        d.write_json(self.plans / 'plan.json', self.manifest)
        d.seal_directory(self.plans)
        self.registered = types.SimpleNamespace(execute=self.fake_execute, assess=self.fake_assess, fingerprint=lambda: SOURCE)
        self.protocols = (self.registered, types.SimpleNamespace(), types.SimpleNamespace(source_fingerprint=lambda: SOURCE))

    def fake_assess(self, *_args):
        return {'correctness': {'status': 'complete'}, 'expanded_inventory': {'exact_match': True, 'missing': []}}

    def fake_execute(self, plan, destination, _timeout):
        destination.mkdir()
        d.write_json(destination / 'plan.json', plan)
        d.write_json(destination / 'source-before.json', SOURCE)
        d.write_json(destination / 'source-after.json', getattr(self, 'native_after', SOURCE))
        rows = []
        for item in plan['jobs']:
            folder = destination / item['id']
            folder.mkdir()
            d.write_json(folder / 'raw.json', {'synthetic': True})
            d.write_json(folder / 'source-before.json', SOURCE)
            d.write_json(folder / 'source-after.json', SOURCE)
            d.write_json(folder / 'process.json', {'exit_code': 0, 'reason': None})
            d.write_json(folder / 'qualification.json', self.fake_assess())
            d.write_json(folder / 'runtime.before.json', {'before': {'cap_verified': True}})
            d.write_json(folder / 'runtime.after.json', {'after': {'cap_verified': True}, 'exit_code': 0})
            rows.append({'id': item['id'], 'status': 'complete'})
        d.write_json(destination / 'summary.json', {'status': 'complete', 'results': rows, 'stop_reason': None})
        d.seal_directory(destination, archive=True)
        return 0

    @contextmanager
    def patched(self):
        with mock.patch.object(d, 'load_plan', return_value=(self.manifest, self.repo, self.protocols)), \
                mock.patch.object(d, 'source_identity', return_value=SOURCE):
            yield

    def test_requires_explicit_selection(self):
        for ids in ([], [self.identifier, self.identifier], ['all'], ['../escape']):
            with self.assertRaises(ValueError):
                d.select_batches(self.manifest, ids, 21)

    def test_budget_is_total_selected_invocations(self):
        second = dict(self.batch, id='registered-expanded-0000', invocations=21)
        self.manifest['batches'].append(second)
        with self.assertRaises(ValueError):
            d.select_batches(self.manifest, [self.identifier, second['id']], 21)
        self.assertEqual(len(d.select_batches(self.manifest, [self.identifier, second['id']], 22)), 2)

    def test_partition_exact_once_and_bounded(self):
        full = native_plan(990)
        partitions = list(d.partition_registered(full, 21))
        self.assertEqual(len(partitions), 48)
        ids = [j['id'] for _, _, p in partitions for j in p['jobs']]
        self.assertEqual(len(ids), 990)
        self.assertEqual(set(ids), {j['id'] for j in full['jobs']})
        self.assertTrue(all(len(p['jobs']) <= 21 for _, _, p in partitions))
        self.assertEqual(sum(p['expected_total_rows'] for _, _, p in partitions), 990)

    def test_execute_archives_then_resume_skips(self):
        with self.patched(), mock.patch.object(self.registered, 'execute', wraps=self.fake_execute) as execute:
            report = d.run_batches(self.plans, self.results, [self.identifier])
            self.assertEqual(report[0]['status'], 'complete')
            raw = self.results / self.identifier / 'native' / 'job-0' / 'raw.json'
            before = raw.read_bytes()
            resumed = d.run_batches(self.plans, self.results, [self.identifier], resume=True)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(resumed, report)
            self.assertEqual(raw.read_bytes(), before)
            d.verify_seal(self.results / self.identifier, archive=True)

    def test_execute_refuses_existing_even_complete(self):
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            with self.assertRaises(ValueError):
                d.run_batches(self.plans, self.results, [self.identifier])

    def test_resume_refuses_unknown_partial(self):
        target = self.results / self.identifier
        target.mkdir(parents=True)
        (target / 'raw.json').write_text('{}')
        with self.patched(), self.assertRaises((ValueError, OSError)):
            d.run_batches(self.plans, self.results, [self.identifier], resume=True)
        self.assertEqual((target / 'raw.json').read_text(), '{}')

    def test_resume_refuses_tampered_result(self):
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            path = self.results / self.identifier / 'native' / 'job-0' / 'raw.json'
            path.chmod(0o644)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                d.run_batches(self.plans, self.results, [self.identifier], resume=True)

    def test_resume_never_retries_known_failure(self):
        def failure(*_args):
            raise RuntimeError('synthetic runner failure')
        self.registered.execute = failure
        with self.patched():
            report = d.run_batches(self.plans, self.results, [self.identifier])
            self.assertEqual(report[0]['status'], 'failed')
            self.registered.execute = mock.Mock(side_effect=AssertionError('must not retry'))
            d.run_batches(self.plans, self.results, [self.identifier], resume=True)
            self.registered.execute.assert_not_called()

    def test_keyboard_interrupt_is_archived(self):
        self.registered.execute = mock.Mock(side_effect=KeyboardInterrupt('synthetic interruption'))
        with self.patched():
            report = d.run_batches(self.plans, self.results, [self.identifier])
            self.assertEqual(report[0]['status'], 'interrupted')
            d.verify_seal(self.results / self.identifier, archive=True)

    def test_source_change_fails_before_launch(self):
        with self.patched(), mock.patch.object(d, 'source_identity', return_value=dict(SOURCE, git_tree='d' * 40)), \
                self.assertRaises(ValueError):
            d.run_batches(self.plans, self.results, [self.identifier])
        self.assertFalse((self.results / self.identifier).exists())

    def test_all_not_run_slots_are_present(self):
        with self.patched():
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'not_run': 1})
        self.assertEqual(ledger['counts']['missing_supported_slots'], 1)
        self.assertFalse(ledger['performance_eligible'])
        self.assertFalse(ledger['all_admitted_slots_complete'])

    def test_complete_is_revalidated_not_exit_code_only(self):
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            self.registered.assess = lambda *_: {'correctness': {'status': 'failed'}, 'expanded_inventory': {'exact_match': False, 'missing': []}}
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'failed': 1})
        self.assertFalse(ledger['all_admitted_slots_complete'])

    def test_success_stays_diagnostic(self):
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'complete': 1})
        self.assertFalse(ledger['performance_eligible'])
        self.assertFalse(ledger['full_catalog_complete'])
        self.assertTrue(ledger['slots'][0]['raw_sha256'])

    def test_native_source_invalidation_dominates_success(self):
        self.native_after = dict(SOURCE, git_commit='d' * 40)
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'failed': 1})
        self.assertFalse(ledger['all_admitted_slots_complete'])
        self.assertIn('source provenance', ledger['slots'][0]['reason'])

    def test_missing_native_archive_cannot_complete_slots(self):
        def incomplete_finalizer(plan, destination, timeout):
            self.fake_execute(plan, destination, timeout)
            destination.with_name(destination.name + '.tar.gz').unlink()
            raise OSError('Synthetic native finalizer failure')
        self.registered.execute = incomplete_finalizer
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier])
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'failed': 1})
        self.assertFalse(ledger['all_admitted_slots_complete'])
        self.assertIn('Native finalizer', ledger['slots'][0]['reason'])
        self.assertTrue(Path(ledger['slots'][0]['raw_path']).is_file())

    def test_host_only_na_never_counts_as_gpu_support(self):
        self.native['jobs'][0]['expected_rows'][0] = {'case': 'host-only', 'backend': 'cupy', 'not_applicable': True}
        path = self.plans / 'batches' / self.identifier / 'plan.json'
        path.chmod(0o644)
        path.write_text(json.dumps(self.native))
        with self.patched():
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        row = ledger['slots'][0]
        self.assertEqual(row['status'], 'excluded')
        self.assertEqual(row['expected_status'], 'not_applicable')
        self.assertEqual(row['protocol_observation_status'], 'not_run')
        self.assertEqual(ledger['counts']['missing_supported_slots'], 0)

    def test_unknown_partial_ledger_cannot_pass(self):
        (self.results / self.identifier).mkdir(parents=True)
        with self.patched():
            ledger = d.aggregate(self.plans, self.results, self.root / 'aggregate')
        self.assertEqual(ledger['counts']['slots'], {'unknown_partial': 1})

    def test_plan_hash_corruption_rejected_before_import(self):
        path = self.plans / 'batches' / self.identifier / 'plan.json'
        path.chmod(0o644)
        path.write_text('{}')
        with mock.patch.object(d, 'load_protocols') as load, self.assertRaises(ValueError):
            d.load_plan(self.plans, self.repo)
        load.assert_not_called()

    def test_resealed_forged_budget_is_rejected(self):
        manifest_path = self.plans / 'plan.json'
        manifest_path.chmod(0o644)
        forged = copy.deepcopy(self.manifest)
        forged['batches'][0]['invocations'] = 0
        manifest_path.write_text(json.dumps(forged))
        (self.plans / 'SHA256SUMS.json').unlink()
        d.seal_directory(self.plans)
        with mock.patch.object(d, 'source_identity', return_value=SOURCE), \
                mock.patch.object(d, 'load_protocols', return_value=self.protocols), \
                mock.patch.object(d, 'plan_inventory', return_value=([(self.identifier, self.config, self.native)], {})), \
                self.assertRaisesRegex(ValueError, 'budgets/admission'):
            d.load_plan(self.plans, self.repo)

    def test_attestation_is_new_metadata_only(self):
        attestation = self.root / 'attestation.json'
        d.write_json(attestation, {'operator': 'test operator', 'statement': 'I observed the device.',
                                  'recorded_at': '2026-01-01T00:00:00Z', 'scope': 'new batch', 'device_uuid': 'GPU-synthetic'})
        with self.patched():
            d.run_batches(self.plans, self.results, [self.identifier], attestation_path=attestation)
        meta = d.read_json(self.results / self.identifier / 'operator_attestation.json')
        self.assertFalse(meta['machine_verified'])
        raw = d.read_json(self.results / self.identifier / 'native' / 'job-0' / 'raw.json')
        self.assertEqual(raw, {'synthetic': True})

    def test_attestation_environment_mismatch_prevents_execution(self):
        path = self.root / 'attestation.json'
        d.write_json(path, {'operator': 'synthetic operator', 'statement': 'new run',
                           'recorded_at': '2026-01-01T00:00:00Z', 'scope': 'new batch', 'environment_id': 'other-environment'})
        with self.patched(), self.assertRaisesRegex(ValueError, 'environment_id differs'):
            d.run_batches(self.plans, self.results, [self.identifier], attestation_path=path)
        self.assertFalse((self.results / self.identifier).exists())

    def test_attestation_placeholders_and_invalid_times_rejected(self):
        for field, value in [('operator', 'REPLACE_OPERATOR'), ('recorded_at', 'not-a-date'),
                             ('recorded_at', '2026-01-01T00:00:00'), ('ended_at', '9999-01-01T00:00:00Z')]:
            path = self.root / ('attestation-' + field + '-' + str(len(value)) + '.json')
            doc = {'operator': 'test', 'statement': 'new run', 'recorded_at': '2026-01-01T00:00:00Z', 'scope': 'selected batch'}
            doc[field] = value
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                d.operator_metadata(path)

    def test_seal_rejects_symlink(self):
        folder = self.root / 'symlink-evidence'
        folder.mkdir()
        (folder / 'external').symlink_to(self.plans / 'plan.json')
        with self.assertRaises(ValueError):
            d.seal_directory(folder)

    def test_no_output_inside_checkout(self):
        with self.assertRaises(ValueError):
            d.external_new(self.repo / 'artifact', self.repo)


class ValidationTests(unittest.TestCase):
    def test_finite_timeout_and_hard_maximum(self):
        for value in (0, -1, 1801, float('nan'), float('inf'), True, '10'):
            with self.assertRaises(ValueError):
                d.checked_timeout(value)
        self.assertEqual(d.checked_timeout(1800), 1800)

    def test_budget_not_bool_and_max64(self):
        for value in (True, 0, 65, 1.0):
            with self.assertRaises(ValueError):
                d.checked_budget(value)

    def test_strict_json_duplicates_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test.json'
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '[]', 'null', '42'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    d.read_json(path)

    def test_truncated_native_source_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.json'
            expected = copy.deepcopy(SOURCE)
            expected['files_sha256']['measured.py'] = 'd' * 64
            expected['content_sha256'] = d.hashlib.sha256(json.dumps(expected['files_sha256'], sort_keys=True).encode()).hexdigest()
            d.write_json(path, SOURCE)
            self.assertFalse(d.native_source_matches(path, expected, expected))
            path.write_text('[]')
            self.assertFalse(d.native_source_matches(path, expected, expected))

    def test_argument_injection_and_abbreviations_forbidden(self):
        def parser():
            p = argparse.ArgumentParser()
            p.add_argument('--execute', action='store_true')
            p.add_argument('--output-dir')
            p.add_argument('--max-invocations')
            p.add_argument('--timeout')
            p.add_argument('--batch-index')
            p.add_argument('--batch-count')
            return p
        module = types.SimpleNamespace(parser=parser)
        for argv in (['--execute'], ['--exec'], ['--output-dir=/tmp/x'], ['--unknown'], ['; touch /tmp/never'], ['--batch-count', '0']):
            with self.assertRaises(ValueError):
                d.registered_args(module, argv, 21, 1)

    @unittest.skipUnless(os.name == 'posix', 'POSIX signal contract')
    def test_sigterm_maps_to_cleanup_path(self):
        with self.assertRaises(KeyboardInterrupt), d.terminating_signals():
            os.kill(os.getpid(), signal.SIGTERM)

    def test_options_require_full_pin_and_explicit_environment(self):
        for key, value in [('expected_sha', 'abc123'), ('environment_id', ''), ('batch_size', 65), ('timeout', 1801)]:
            data = options(); data[key] = value
            with self.assertRaises(ValueError):
                d.validate_options(data)

    def test_git_source_hidden_modification_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
            git('init', '-q')
            (root / 'source.py').write_text('VALUE = 1\n')
            git('add', 'source.py')
            git('-c', 'user.email=synthetic@example.invalid', '-c', 'user.name=Synthetic Test', 'commit', '-qm', 'fixture')
            before = d.source_identity(root, git('rev-parse', 'HEAD'))
            self.assertEqual(before['working_tree'], 'clean')
            git('update-index', '--assume-unchanged', 'source.py')
            (root / 'source.py').write_text('VALUE = 2\n')
            self.assertEqual(git('status', '--porcelain'), '')
            with self.assertRaisesRegex(ValueError, 'Hidden index flags'):
                d.source_identity(root, git('rev-parse', 'HEAD'))


if __name__ == '__main__':
    unittest.main()
