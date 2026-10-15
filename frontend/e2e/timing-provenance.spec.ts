import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import type { BenchmarkData } from '../src/schema';
import { loadChartRenderer, TestElement, withTestDocument } from './helpers/chart-renderer';

// This is the same strict-generated canonical bundle served by the dashboard.
const data: BenchmarkData = JSON.parse(readFileSync(
  new URL('../public/data/benchmark_data.json', import.meta.url), 'utf8',
));
const sessions = [
  ['remote-p100-cv-20260807', 'cv_benchmark_20260807.json'],
  ['remote-p100-pr116-20260807', 'cv_benchmark_pr116_p100.json'],
];
let renderer: Awaited<ReturnType<typeof loadChartRenderer>>;
test.beforeAll(async () => { renderer = await loadChartRenderer(); });

for (const view of ['focused', 'full'] as const) {
  test(`real timing renderer retains canonical CV session provenance in ${view} view`, () => {
    const originalData = JSON.stringify(data);
    const environments = renderer.groupPhysicalEnvironments(data.environments);
    const state = renderer.createDefaultState(environments, data.runs);
    state.selectedCategoryIds = new Set(['linear_models']);
    state.selectedMetricScope = 'cross_validation';
    state.selectedModelId = 'RidgeCV';
    state.chartViewMode = view;
    const runs = renderer.filterRuns(data.runs, state);

    expect(environments).toHaveLength(1);
    expect(state.selectedEnvId).toBe('remote-p100');
    expect(state.selectedEnvIds.size).toBe(8);
    expect(runs).toHaveLength(6);
    expect(runs.map(run => [run.run_id, run.metrics.timing!.fit_time_ms])).toEqual([
      ['8c9d73de17cd5fbc', 9.062319993972778],
      ['7329f20ec3127d26', 17.41701364517212],
      ['a5a6aed51f5d17af', 12.491881847381592],
      ['dda93f32f6315ff8', 9.013503789901733],
      ['5d24f0906f96641a', 17.37356185913086],
      ['749946278ccf048d', 12.376457452774048],
    ]);

    renderer.capturedOptions.length = 0;
    const element = new TestElement('div');
    renderer.renderTimingChart(element as unknown as HTMLElement, runs, state, []);
    const option = renderer.capturedOptions[0];
    expect(element.dataset.timingGroups).toBe('6');
    expect(element.dataset.timingDisplayed).toBe('6');
    expect(option.xAxis.data).toHaveLength(6);
    expect(option.series).toHaveLength(3);

    for (const run of runs) {
      const series = option.series.find(item => item.name === `${run.backend}/${run.implementation}`)!;
      const index = series.data.indexOf(run.metrics.timing!.fit_time_ms);
      expect(index).toBeGreaterThanOrEqual(0);
      const label = option.xAxis.data[index];
      expect(label).toContain('Ridge CV (mean_squared_error)');
      expect(label).toContain(`Session: ${run.env_id}`);
      expect(label).toContain(`Source: ${run.source.file}`);
      if (view === 'full') expect(label).toContain('240×16');
      const tooltip = option.tooltip.formatter(option.series.map(item => ({
        seriesName: item.name, value: item.data[index], color: '#123456',
        dataIndex: index, axisValueLabel: 'Truncated chart label…',
      })));
      expect(tooltip).toContain(`Session: ${run.env_id}`);
      expect(tooltip).toContain(`Source: ${run.source.file}`);
      expect(tooltip).toContain(`${run.metrics.timing!.fit_time_ms.toFixed(2)} ms`);
      expect(tooltip).not.toContain(sessions.find(([id]) => id !== run.env_id)![1]);
    }

    expect(JSON.stringify(data)).toBe(originalData);
  });

  test(`accessible timing table retains canonical CV session provenance in ${view} view`, () => {
    const originalData = JSON.stringify(data);
    const environments = renderer.groupPhysicalEnvironments(data.environments);
    const state = renderer.createDefaultState(environments, data.runs);
    state.selectedCategoryIds = new Set(['linear_models']);
    state.selectedMetricScope = 'cross_validation';
    state.selectedModelId = 'RidgeCV';
    state.chartViewMode = view;
    const runs = renderer.filterRuns(data.runs, state);
    expect(runs).toHaveLength(6);
    withTestDocument(() => {
      const fallback = renderer.renderChartDataFallback(runs, runs, state) as unknown as TestElement;
      const details = fallback.findAll(el => el.attributes.id === 'timing-chart-data')[0];
      expect(details.children[0].textContent).toBe('Fit Time chart data (6 rows)');
      expect(details.findAll(el => el.tag === 'table')).toHaveLength(0);
      details.open = true;
      details.events.get('toggle')!();
      details.events.get('toggle')!();
      expect(details.findAll(el => el.tag === 'table')).toHaveLength(1);
      const headers = details.findAll(el => el.tag === 'th');
      expect(headers.map(el => el.textContent)).toEqual([
        'Model', 'Variant', 'Penalty', 'Solver', 'Backend / reference', 'Scale',
        'Benchmark session', 'Source', 'Time', 'Reference',
      ]);
      expect(headers.every(el => el.attributes.scope === 'col')).toBe(true);
      const rows = details.findAll(el => el.tag === 'tbody')[0].children;
      expect(rows).toHaveLength(6);
      for (const run of runs) {
        const row = rows.find(el => el.attributes['data-run-id'] === run.run_id)!;
        expect(row.children.map(el => el.textContent)).toEqual([
          'Ridge CV', 'mean_squared_error', 'l2', 'Cross-validation',
          `${run.backend}/${run.implementation}`, '240×16', run.env_id, run.source.file,
          `${run.metrics.timing!.fit_time_ms.toFixed(3)} ms`, 'measured',
        ]);
      }
    });
    expect(JSON.stringify(data)).toBe(originalData);
  });
}

test('timing labels prefer explicit session identity and escape provenance in tooltips', () => {
  const original = data.runs.find(run => run.model_id === 'RidgeCV')!;
  const run = {
    ...original,
    benchmark_session_id: 'session <b>2026</b> & "test"',
    source: { ...original.source, file: '<img src=x onerror=alert(1)>.json' },
  };
  const state = renderer.createDefaultState(data.environments, [run]);
  renderer.capturedOptions.length = 0;
  renderer.renderTimingChart(new TestElement('div') as unknown as HTMLElement, [run], state, []);
  const option = renderer.capturedOptions[0];
  expect(option.xAxis.data[0]).toContain(`Session: ${run.benchmark_session_id}`);
  const tooltip = option.tooltip.formatter([{
    dataIndex: 0, seriesName: '<b>numpy</b>', value: 1, color: '#123456',
  }]);
  expect(tooltip).toContain('session &lt;b&gt;2026&lt;/b&gt; &amp; &quot;test&quot;');
  expect(tooltip).toContain('&lt;img src=x onerror=alert(1)&gt;.json');
  expect(tooltip).toContain('&lt;b&gt;numpy&lt;/b&gt;');
  expect(tooltip).not.toContain('<img');
  withTestDocument(() => {
    const fallback = renderer.renderChartDataFallback([run], [], state) as unknown as TestElement;
    const details = fallback.findAll(el => el.attributes.id === 'timing-chart-data')[0];
    details.open = true;
    details.events.get('toggle')!();
    expect(details.textContent).toContain(run.benchmark_session_id);
    expect(details.textContent).toContain(run.source.file);
    expect(details.findAll(el => el.tag === 'img' || el.tag === 'b')).toHaveLength(0);
  });
});
