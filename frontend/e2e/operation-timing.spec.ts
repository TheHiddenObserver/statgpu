import { test, expect } from '@playwright/test';
import type { Run } from '../src/schema';
import { canDisplaySpeedup, isPerformanceQualified, timingValue, timingScopeLabel } from '../src/timing';
import { selectTimingRuns, formatGroupLabel, formatTimingTooltip } from '../src/charts/TimingChart';
import { selectSpeedupRuns } from '../src/charts/SpeedupChart';
import { getRunMetricScopes, getPrimaryMetricScope } from '../src/metricScopes';
import { createDefaultState } from '../src/state';

function run(id = 'gpu'): Run {
  return {
    run_id: id, env_id: 'test', benchmark_session_id: 'session', category_ids: ['diagnostic_evidence'],
    model_id: 'ClassificationMetrics', comparison_id: 'comparison', case_id: 'case', method_config_id: 'metric',
    framework: 'statgpu', backend: id === 'cpu' ? 'numpy' : 'cupy',
    scale: { scale_key: 'n100_p0', n_samples: 100, n_features: 0, feature_dimension_kind: 'not_applicable', label: '100 samples · p not applicable' },
    source: { source_id: 'source', file: 'raw.json', date: '2026-10-25', parser: 'qualification_preview', parser_version: '1.0' },
    metrics: {
      operation_timing: { duration_ms: id === 'cpu' ? 20 : 10, scope: 'classification_metric', statistic: 'median_of_recorded_samples', unit: 'ms', quality: 'measured', source_file: 'raw.json' },
      speedup: { value: 2, reference_run_id: 'cpu', reference_backend: 'numpy', reference_framework: 'statgpu', reported_semantics: 'computed', quality: 'computed', source_file: 'raw.json' },
    },
    evidence: { qualification: 'diagnostic_only', correctness_status: 'pass', validation_status: 'validated_diagnostic', performance_eligible: false, canonical_dashboard_ready: false, machine_verified_exclusivity: false, raw_performance_eligible: true, raw_pointer: '/rows/0/seconds', source_commit: 'a'.repeat(40), source_fingerprint: 'b'.repeat(64) },
  };
}
function qualified(id: string): Run {
  const value = run(id);
  Object.assign(value.evidence!, { qualification: 'machine_verified', validation_status: 'pass', performance_eligible: true, canonical_dashboard_ready: true, machine_verified_exclusivity: true });
  return value;
}
const envs = [{ env_id: 'test', label: 'Test', gpu: 'GPU', cpu: 'CPU' }];

test('operation duration and labels preserve scope without a fit metric or p1', () => {
  const value = run();
  expect(timingValue(value)).toBe(10);
  expect(timingScopeLabel(value)).toBe('classification metric');
  expect(formatGroupLabel(value, true)).toContain('classification metric');
  expect(value.metrics.timing).toBeUndefined();
  expect([...getRunMetricScopes(value)]).toEqual(['operation']);
  expect(getPrimaryMetricScope(value)).toBe('operation');
  expect(value.scale.n_features).toBe(0);
  expect(selectTimingRuns([value], createDefaultState(envs, [value])).runs).toHaveLength(1);
});

test('diagnostic and operator-attested rows never gain speedups from a raw eligibility flag', () => {
  const value = run();
  expect(isPerformanceQualified(value)).toBe(false);
  expect(canDisplaySpeedup(value, [value, run('cpu')])).toBe(false);
  value.evidence!.qualification = 'operator_attested';
  value.evidence!.operator_attestation = { statement: 'exclusive' };
  expect(selectSpeedupRuns([value], createDefaultState(envs, [value])).runs).toEqual([]);
});

test('new ratios require positive comparable qualified references, explicit provenance, and matching ratios', () => {
  const gpu = qualified('gpu'); const cpu = qualified('cpu');
  expect(canDisplaySpeedup(gpu, [gpu, cpu])).toBe(true);
  expect(canDisplaySpeedup(gpu, [gpu])).toBe(false);
  for (const change of [
    (r: Run) => { r.metrics.operation_timing!.scope = 'inference'; },
    (r: Run) => { r.metrics.operation_timing!.statistic = 'minimum'; },
    (r: Run) => { r.evidence!.raw_performance_eligible = false; },
    (r: Run) => { r.evidence!.source_commit = null; },
    (r: Run) => { r.benchmark_session_id = 'different'; },
    (r: Run) => { r.metrics.operation_timing!.duration_ms = 21; },
  ]) {
    const ref = qualified('cpu'); change(ref);
    expect(canDisplaySpeedup(gpu, [gpu, ref])).toBe(false);
  }
});

test('missing, zero, nonfinite or negative timings are never charted', () => {
  const value = run(); value.metrics = {};
  expect(timingValue(value)).toBeNull();
  expect(selectTimingRuns([value], createDefaultState(envs, [value])).runs).toEqual([]);
  for (const duration of [0, -1, NaN, Infinity]) {
    const invalid = run(); invalid.metrics.operation_timing!.duration_ms = duration;
    expect(timingValue(invalid)).toBeNull();
  }
});

test('legacy v1.1 fit timing and reported speedup remain supported', () => {
  const value = run(); delete value.evidence; delete value.metrics.operation_timing;
  value.metrics.timing = { fit_time_ms: 30, quality: 'measured', source_file: 'legacy.json' };
  value.metrics.speedup!.reported_semantics = 'reported_by_runner';
  expect(timingValue(value)).toBe(30);
  expect(timingScopeLabel(value)).toBe('Fit');
  expect(canDisplaySpeedup(value, [value])).toBe(true);
});

test('operation records cannot gain legacy speedups by dropping evidence', () => {
  const value = run(); delete value.evidence;
  expect(canDisplaySpeedup(value, [value])).toBe(false);
});


// Inert punctuation only: these fixtures never construct executable content.
for (const field of ['model', 'variant', 'scope', 'statistic', 'solver', 'scale'] as const) {
  test(`tooltip treats ${field} punctuation as text`, () => {
    const value = run();
    const punctuation = `A&B<C>D"E'F`;
    if (field === 'model') value.model_id = punctuation;
    if (field === 'variant') value.variant = punctuation;
    if (field === 'scope') value.metrics.operation_timing!.scope = punctuation;
    if (field === 'statistic') value.metrics.operation_timing!.statistic = punctuation;
    if (field === 'solver') value.solver_display = punctuation;
    if (field === 'scale') value.scale.label = punctuation;
    const html = formatTimingTooltip([{
      axisValueLabel: formatGroupLabel(value, false),
      seriesName: 'CuPy', color: '#12Ab34', value: 10,
    }]);
    expect(html).toContain('A&amp;B&lt;C&gt;D&quot;E&#39;F');
    expect(html).not.toContain(punctuation);
    expect(html).toContain('10.00 ms');
  });
}

test('tooltip escapes series text and permits only palette hex marker colors', () => {
  const html = formatTimingTooltip([{
    seriesName: `A&B<C>D"E'F`, color: 'not a palette color', value: 2,
  }]);
  expect(html).toBe('<span style="color:#fff">●</span> A&amp;B&lt;C&gt;D&quot;E&#39;F: <b>2.00 ms</b>');
  expect(formatTimingTooltip([{ seriesName: 'CPU', color: '#abc', value: 1 }])).toContain('color:#abc');
});

test('tooltip retains empty and missing-value behavior without nonfinite numbers', () => {
  expect(formatTimingTooltip([])).toBe('No data');
  for (const value of [null, NaN, Infinity]) {
    expect(formatTimingTooltip([{ seriesName: 'CPU', color: '#abc', value }])).toBe('No data');
  }
});
