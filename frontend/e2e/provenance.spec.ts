import { expect, test } from '@playwright/test';
import type { Run } from '../src/schema';
import { cpuThreadClass, cpuThreadLabel, sourceProvenance } from '../src/provenance';
import { createDefaultState, setSelectedModel, setSelectedMetricScope, toggleScaleKey, resetDownstreamFilters } from '../src/state';

function run(requested?: unknown, observed?: unknown): Run {
  return {
    run_id: 'run-1', env_id: 'test', category_ids: ['diagnostic_evidence'], model_id: 'model',
    framework: 'sklearn', backend: null, comparison_id: 'comparison', case_id: 'case', method_config_id: 'method',
    scale: { scale_key: 'n10_p2', n_samples: 10, n_features: 2, label: '10×2' },
    source: { source_id: 'source', file: 'preview.json', original_path: '/original/preview.json', sha256: 'a'.repeat(64), date: '2026-10-25', parser: 'qualification_preview', parser_version: '1.0' },
    parameters: { raw_path: '/raw/data.json', raw_sha256: 'b'.repeat(64), external_adapter: 'sklearn-scoped-cv-v2' },
    metrics: {},
    evidence: { qualification: 'diagnostic_only', correctness_status: 'complete', validation_status: 'validated_diagnostic', performance_eligible: false, canonical_dashboard_ready: false, machine_verified_exclusivity: false, raw_pointer: '/rows/0/timing', source_commit: 'c'.repeat(40), cpu_threads_requested: requested, cpu_threads_observed: observed },
  };
}

test('CPU limit classification requires observations and never upgrades one-thread data', () => {
  for (const backend of ['numpy', 'cupy', 'torch', null] as const) {
    const value = run(1, [1, 1, 1, 1]); value.backend = backend;
    expect(cpuThreadClass(value)).toBe('single');
    expect(cpuThreadLabel(value)).toContain('1 thread (observed limit)');
  }
  expect(cpuThreadClass(run(4, [4, 4]))).toBe('multi');
  expect(cpuThreadClass(run(undefined, 2))).toBe('multi');
  expect(cpuThreadClass(run(1, [1, 2]))).toBe('conflicting');
  expect(cpuThreadClass(run(4, [1, 1]))).toBe('conflicting');
  expect(cpuThreadClass(run(4, [2, 4]))).toBe('conflicting');
  for (const observed of [undefined, null, [], [1, null], [1, '1'], true, '1', 0, -1, 1.5, NaN, Infinity, { threads: 4 }]) {
    expect(cpuThreadClass(run(4, observed))).toBe('unknown');
  }
  expect(cpuThreadClass(run('4', [4]))).toBe('unknown');
  const legacy = run(); delete legacy.evidence;
  expect(cpuThreadClass(legacy)).toBe('unknown');
});

test('upstream filter changes clear thread selections, including repeated selections', () => {
  const value = run(1, [1]);
  const state = createDefaultState([{ env_id: 'test', label: 'test', gpu: 'none', cpu: 'test' }], [value]);
  expect(state.selectedCpuThreads).toBe('all');
  for (const reset of [
    () => setSelectedModel(state, 'model'),
    () => setSelectedMetricScope(state, 'all'),
    () => toggleScaleKey(state, 'n10_p2'),
    () => resetDownstreamFilters(state, { clearMetricScope: true }),
    () => resetDownstreamFilters(state, { clearCpuThreads: true }),
  ]) {
    state.selectedCpuThreads = 'multi'; reset(); expect(state.selectedCpuThreads).toBe('all');
  }
});

test('source provenance preserves raw producer identity, pointers, hashes and adapter separately', () => {
  const details = Object.fromEntries(sourceProvenance(run(1, [1])));
  expect(details['Raw producer commit (not renderer)']).toBe('c'.repeat(40));
  expect(details['Source SHA256']).toBe('a'.repeat(64));
  expect(details['Raw SHA256']).toBe('b'.repeat(64));
  expect(details['Raw path']).toBe('/raw/data.json');
  expect(details['Raw pointer']).toBe('/rows/0/timing');
  expect(details['Run ID']).toBe('run-1');
  expect(details['Source ID']).toBe('source');
  expect(details['External adapter']).toBe('sklearn-scoped-cv-v2');
  const missing = run(); delete missing.evidence; delete missing.parameters;
  expect(Object.fromEntries(sourceProvenance(missing))['Raw producer commit (not renderer)']).toBe('Not recorded');
});
