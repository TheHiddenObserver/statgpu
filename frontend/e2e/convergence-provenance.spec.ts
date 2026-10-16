import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import type { BenchmarkData, Run } from '../src/schema';
import type { AppState } from '../src/state';
import { loadChartRenderer, TestElement, withTestDocument } from './helpers/chart-renderer';

// Exercise the actual panel against the same strict-generated bundle as the app.
const data: BenchmarkData = JSON.parse(readFileSync(
  new URL('../public/data/benchmark_data.json', import.meta.url), 'utf8',
));
const convergenceRuns = data.runs.filter(run => {
  const metric = run.metrics.convergence;
  return metric && (metric.n_iter_mean != null || metric.converged_rate != null);
});
const headers = [
  'Model', 'Scale', 'Backend', 'Benchmark session / 基准会话', 'Source / 来源',
  'N Iter (mean±std)', 'Converged Rate',
];
let renderer: Awaited<ReturnType<typeof loadChartRenderer>>;
test.beforeAll(async () => { renderer = await loadChartRenderer(); });

function expectedCells(run: Run): string[] {
  const metric = run.metrics.convergence!;
  return [
    run.model_id, run.scale.label, run.backend ?? run.framework,
    run.benchmark_session_id ?? run.env_id, run.source.file,
    metric.n_iter_mean != null
      ? `${metric.n_iter_mean.toFixed(1)}${metric.n_iter_std != null ? ` ± ${metric.n_iter_std.toFixed(1)}` : ''}`
      : '-',
    metric.converged_rate != null ? `${(metric.converged_rate * 100).toFixed(0)}%` : '-',
  ];
}

function renderPanel(runs: Run[], state: AppState, onUpdate = () => {}): TestElement | null {
  return renderer.renderConvergencePanel(runs, state, onUpdate) as unknown as TestElement | null;
}

function renderedCells(panel: TestElement): string[][] {
  return panel.findAll(el => el.tag === 'tr')
    .filter(row => row.children.some(cell => cell.tag === 'td'))
    .map(row => row.children.map(cell => cell.textContent));
}

test('all 21 canonical cross-session convergence pairs have distinct visible provenance without changing values or order', () => {
  const originalData = JSON.stringify(data);
  const groups = new Map<string, Run[]>();
  for (const run of convergenceRuns) {
    const identity = JSON.stringify([run.model_id, run.scale.label, run.backend ?? run.framework]);
    groups.set(identity, [...(groups.get(identity) ?? []), run]);
  }
  const collisions = [...groups.values()].filter(runs =>
    new Set(runs.map(run => run.benchmark_session_id ?? run.env_id)).size > 1,
  );
  expect(collisions).toHaveLength(21);
  expect(collisions.every(runs => runs.length === 2)).toBe(true);

  const state = renderer.createDefaultState(renderer.groupPhysicalEnvironments(data.environments), data.runs);
  state.expandedPanels.add('convergence');
  state.panelLimits.convergence = Infinity;
  withTestDocument(() => {
    const panel = renderPanel(data.runs, state)!;
    const cells = renderedCells(panel);
    expect(cells).toHaveLength(55);
    expect(cells).toEqual(convergenceRuns.map(expectedCells));
    const columnHeaders = panel.findAll(el => el.tag === 'th');
    expect(columnHeaders.map(el => el.textContent)).toEqual(headers);
    expect(columnHeaders.every(el => el.attributes.scope === 'col')).toBe(true);
    for (const pair of collisions) {
      expect(new Set(pair.map(run => JSON.stringify(expectedCells(run).slice(0, 5)))).size).toBe(2);
      for (const run of pair) {
        expect(run.metrics.convergence!.source_file).toBe(run.source.file);
        expect(cells.filter(row => JSON.stringify(row) === JSON.stringify(expectedCells(run)))).toHaveLength(1);
      }
    }
  });
  expect(JSON.stringify(data)).toBe(originalData);
});

test('convergence provenance follows hardware, model, backend, external, and session filters without adding sorting', () => {
  const originalData = JSON.stringify(data);
  const state = renderer.createDefaultState(renderer.groupPhysicalEnvironments(data.environments), data.runs);
  state.selectedCategoryIds = new Set(['linear_models']);
  state.selectedMetricScope = 'cross_validation';
  state.selectedModelId = 'RidgeCV';
  state.expandedPanels.add('convergence');
  expect(state.selectedEnvId).toBe('remote-p100');
  expect(state.selectedEnvIds.size).toBe(8);
  const expectedIds = [
    '8c9d73de17cd5fbc', '7329f20ec3127d26', 'a5a6aed51f5d17af',
    'dda93f32f6315ff8', '5d24f0906f96641a', '749946278ccf048d',
  ];
  withTestDocument(() => {
    const runs = renderer.filterRuns(data.runs, state);
    expect(runs.map(run => run.run_id)).toEqual(expectedIds);
    expect(renderedCells(renderPanel(runs, state)!)).toEqual(runs.map(expectedCells));
    // Overview sorting has never sorted metric panels. Preserve input order.
    state.sortColumn = 'time_ms';
    state.sortDir = 'desc';
    expect(renderedCells(renderPanel(runs, state)!)).toEqual(runs.map(expectedCells));
    state.showExternal.add('sklearn');
    state.selectedBackends.add('numpy');
    const withExternal = renderer.filterRuns(data.runs, state);
    expect(withExternal.map(run => run.run_id)).toEqual([
      '8c9d73de17cd5fbc', 'ccc57cda0d82c9c4', 'dda93f32f6315ff8', '1811661bc8edaa90',
    ]);
    expect(renderedCells(renderPanel(withExternal, state)!)).toEqual(withExternal.map(expectedCells));
    state.selectedEnvIds = new Set(['remote-p100-pr116-20260807']);
    const oneSession = renderer.filterRuns(data.runs, state);
    expect(oneSession.map(run => run.run_id)).toEqual(['dda93f32f6315ff8', '1811661bc8edaa90']);
    expect(renderedCells(renderPanel(oneSession, state)!)).toEqual(oneSession.map(expectedCells));
  });
  expect(JSON.stringify(data)).toBe(originalData);
});

test('convergence provenance survives pagination and repeated collapse/expand without dropping rows', () => {
  const originalData = JSON.stringify(data);
  const state = renderer.createDefaultState(data.environments, data.runs);
  let updates = 0;
  withTestDocument(() => {
    const render = () => renderPanel(data.runs, state, () => { updates += 1; })!;
    let panel = render();
    expect(renderedCells(panel)).toHaveLength(0);
    const toggle = panel.findAll(el => el.tag === 'button')[0];
    expect(toggle.attributes['aria-expanded']).toBe('false');
    expect(toggle.attributes['aria-controls']).toBe('convergence-panel-body');
    toggle.events.get('click')!();
    panel = render();
    expect(renderedCells(panel)).toEqual(convergenceRuns.slice(0, 30).map(expectedCells));
    panel.findAll(el => el.tag === 'button' && el.textContent === 'Show all 55')[0].events.get('click')!();
    panel = render();
    expect(renderedCells(panel)).toEqual(convergenceRuns.map(expectedCells));
    panel.findAll(el => el.tag === 'button')[0].events.get('click')!();
    panel = render();
    expect(renderedCells(panel)).toHaveLength(0);
    panel.findAll(el => el.tag === 'button')[0].events.get('click')!();
    panel = render();
    expect(renderedCells(panel)).toEqual(convergenceRuns.map(expectedCells));
    panel.findAll(el => el.tag === 'button' && el.textContent === 'Show first 30')[0].events.get('click')!();
    expect(renderedCells(render())).toEqual(convergenceRuns.slice(0, 30).map(expectedCells));
    expect(updates).toBe(5);
  });
  expect(JSON.stringify(data)).toBe(originalData);
});

test('convergence keeps missing and zero metric behavior, prefers explicit sessions, and renders provenance as text', () => {
  const original = convergenceRuns[0];
  const base = {
    ...original,
    benchmark_session_id: 'session <b>2026</b> & "test"',
    source: { ...original.source, file: '<img src=x onerror=alert(1)>.json' },
  };
  const runs: Run[] = [
    { ...base, metrics: {} },
    { ...base, metrics: { convergence: { quality: 'reported', n_iter_std: 2 } } },
    { ...base, metrics: { convergence: { quality: 'reported', n_iter_mean: 0, n_iter_std: 0, converged_rate: 0 } } },
    { ...base, metrics: { convergence: { quality: 'reported', n_iter_mean: 12.34, n_iter_std: 0.56 } } },
    { ...base, benchmark_session_id: undefined, metrics: { convergence: { quality: 'reported', converged_rate: 0.875 } } },
  ];
  const before = JSON.stringify(runs);
  const state = renderer.createDefaultState(data.environments, runs);
  state.expandedPanels.add('convergence');
  withTestDocument(() => {
    expect(renderPanel(runs.slice(0, 2), state)).toBeNull();
    const panel = renderPanel(runs, state)!;
    const cells = renderedCells(panel);
    expect(cells).toHaveLength(3);
    expect(cells.map(row => row.slice(5))).toEqual([
      ['0.0 ± 0.0', '0%'], ['12.3 ± 0.6', '-'], ['-', '88%'],
    ]);
    expect(cells.map(row => row[3])).toEqual([base.benchmark_session_id, base.benchmark_session_id, base.env_id]);
    expect(cells.every(row => row[4] === base.source.file)).toBe(true);
    expect(panel.findAll(el => el.tag === 'img' || el.tag === 'b')).toHaveLength(0);
  });
  expect(JSON.stringify(runs)).toBe(before);
});
