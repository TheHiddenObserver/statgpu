import './style.css';
import './metric-scope.css';

import type { ECharts } from './echarts';
import type { BenchmarkData, ParseReport, Run, SourceInventory } from './schema';
import { filterRuns, loadBenchmarkBundle } from './data';
import { createDefaultState } from './state';
import type { AppState } from './state';
import { h, clear } from './utils/dom';
import { enhanceDashboardAccessibility } from './accessibility';
import { renderHeader, renderLoadingHeader } from './components/Header';
import { renderSidebar } from './components/Sidebar';
import { renderFilterBar } from './components/FilterBar';
import { renderOverviewTable } from './components/OverviewTable';
import { renderSummaryCards } from './components/SummaryCards';
import { emptyStateMessage } from './components/EmptyState';

type ChartRuntime = typeof import('./charts/runtime');

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

let data: BenchmarkData | null = null;
let parseReport: ParseReport | null = null;
let sourceInventory: SourceInventory | null = null;
let state: AppState | null = null;
let chartRuntime: ChartRuntime | null = null;

/** Track ECharts instances for cleanup before re-render */
const chartInstances: ECharts[] = [];

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function renderApp(): HTMLElement {
  const app = h('div', { id: 'app-root' });
  app.appendChild(renderHeader(data!, parseReport, sourceInventory, state!, update));
  app.appendChild(renderBody());
  return app;
}

function renderBody(): HTMLElement {
  const body = h('div', { class: 'body' });
  body.appendChild(renderSidebar(data!, state!, update));

  const right = h('div', { class: 'content-column' });
  // Summary cards + footer persist across filter updates; only main is re-rendered
  right.appendChild(renderSummaryCards(data!, parseReport, data!.runs));
  const main = renderMain();
  right.appendChild(main);
  right.appendChild(renderFooter());

  body.appendChild(right);
  return body;
}

function renderMain(): HTMLElement {
  const main = h('main', { class: 'main', id: 'dashboard-main', tabindex: '-1' });
  const filtered = getFilteredRuns();
  main.appendChild(renderFilterBar(data!.runs, data!, state!, update));
  main.appendChild(renderChartArea(filtered));
  main.appendChild(
    chartRuntime!.renderChartDataFallback(
      filtered,
      focusedSpeedupRuns(filtered),
      state!,
    ),
  );
  main.appendChild(renderOverviewTable(filtered, state!, update));
  enhanceDashboardAccessibility(main);
  return main;
}

let renderEpoch = 0;

function usesDefaultSurvivalImplementation(): boolean {
  return Boolean(
    state &&
    state.chartViewMode === 'focused' &&
    state.selectedCategoryIds.size === 1 &&
    state.selectedCategoryIds.has('survival'),
  );
}

function focusedSpeedupRuns(filtered: Run[]): Run[] {
  if (!usesDefaultSurvivalImplementation()) return filtered;
  return filtered.filter(
    (run) => !(
      run.framework === 'statgpu' &&
      run.backend === 'numpy' &&
      run.implementation === 'numba'
    ),
  );
}

function renderChartArea(filtered: Run[]): HTMLElement {
  const area = h('div', { class: 'chart-area', 'aria-label': 'Benchmark charts' });
  const timingDiv = h('div', {
    id: 'timing-chart',
    class: 'chart-container',
    role: 'img',
    'aria-label': 'Fit Time chart',
    'aria-describedby': 'timing-chart-data',
  });
  const speedupDiv = h('div', {
    id: 'speedup-chart',
    class: 'chart-container',
    role: 'img',
    'aria-label': 'Speedup vs Reference chart',
    'aria-describedby': 'speedup-chart-data',
  });
  area.appendChild(timingDiv);
  area.appendChild(speedupDiv);

  const speedupRuns = focusedSpeedupRuns(filtered);
  const defaultSurvivalOnly = usesDefaultSurvivalImplementation();
  speedupDiv.dataset.implementationScope = defaultSurvivalOnly ? 'default-only' : 'all';

  const epoch = ++renderEpoch;
  requestAnimationFrame(() => {
    if (epoch !== renderEpoch || !timingDiv.isConnected || !speedupDiv.isConnected) return;
    chartRuntime!.renderTimingChart(timingDiv, filtered, state!, chartInstances);
    chartRuntime!.renderSpeedupChart(speedupDiv, speedupRuns, state!, chartInstances);
    if (defaultSurvivalOnly) {
      const aria = speedupDiv.getAttribute('aria-label') ?? 'Speedup vs Reference chart';
      speedupDiv.setAttribute('aria-label', `${aria}; default NumPy implementation only`);
    }
  });
  return area;
}

function renderFooter(): HTMLElement {
  const footer = h('div', { class: 'dashboard-footer' });
  const guideUrl = new URL('../en/guides/benchmarks', window.location.href).toString();

  const links: [string, string][] = [
    ['Benchmark guide', guideUrl],
    ['Raw data (JSON)', 'data/benchmark_data.json'],
    ['Parse report (JSON)', 'data/parse_report.json'],
    ['Source inventory (JSON)', 'data/source_inventory.json'],
    [
      'Catalog policy',
      'https://github.com/TheHiddenObserver/statgpu/blob/master/dev/benchmarks/benchmark_source_catalog.json',
    ],
    [
      'Coverage matrix',
      'https://github.com/TheHiddenObserver/statgpu/blob/master/dev/benchmarks/benchmark_coverage_matrix.json',
    ],
    [
      'GitHub source',
      'https://github.com/TheHiddenObserver/statgpu/tree/master/dev/benchmarks',
    ],
  ];
  for (const [label, href] of links) {
    const a = h('a', { href, target: '_blank', rel: 'noopener' }, label);
    footer.appendChild(a);
  }

  const inventorySuffix = sourceInventory
    ? ` · Inventory ${sourceInventory.inventory_version}`
    : '';
  const meta = h(
    'span',
    {},
    `Schema ${data!.schema_version}${inventorySuffix} · ${data!.meta.git_sha}`,
  );
  footer.appendChild(meta);

  return footer;
}

// ---------------------------------------------------------------------------
// State & update loop
// ---------------------------------------------------------------------------

function getFilteredRuns(): Run[] {
  if (!data || !state) return [];
  return filterRuns(data.runs, state);
}

function disposeCharts(): void {
  if (!chartRuntime) return;
  for (const id of ['timing-chart', 'speedup-chart']) {
    const el = document.getElementById(id);
    if (!el) continue;
    const chart = chartRuntime.echarts.getInstanceByDom(el);
    if (chart && !chart.isDisposed()) chart.dispose();
  }
  chartInstances.length = 0;
}

function resizeCharts(): void {
  for (const chart of chartInstances) {
    if (!chart.isDisposed()) chart.resize();
  }
}

function update(): void {
  const main = document.querySelector('.main') as HTMLElement | null;
  if (!main) return;

  disposeCharts();

  // Compute filtered runs once per update, pass to all renderers
  const allRuns = data!.runs;
  const filtered = filterRuns(allRuns, state!);

  clear(main);
  main.appendChild(renderFilterBar(allRuns, data!, state!, update));
  main.appendChild(renderChartArea(filtered));
  main.appendChild(
    chartRuntime!.renderChartDataFallback(
      filtered,
      focusedSpeedupRuns(filtered),
      state!,
    ),
  );
  main.appendChild(renderOverviewTable(filtered, state!, update));
  enhanceDashboardAccessibility(main);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function renderLoadingBody(): HTMLElement {
  const shell = h('div', { class: 'dashboard-loading-shell' });
  const sidebar = h('div', {
    class: 'loading-placeholder loading-sidebar',
    'aria-hidden': 'true',
  });
  const content = h('div', { class: 'loading-content' });
  content.appendChild(
    h(
      'div',
      { class: 'loading-copy', role: 'status' },
      'Loading benchmark evidence… / 正在加载基准数据…',
    ),
  );
  content.appendChild(
    h('div', {
      class: 'loading-placeholder loading-summary',
      'aria-hidden': 'true',
    }),
  );
  content.appendChild(
    h('div', {
      class: 'loading-placeholder loading-chart',
      'aria-hidden': 'true',
    }),
  );
  shell.appendChild(sidebar);
  shell.appendChild(content);
  return shell;
}

async function init(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) return;

  clear(root);
  root.appendChild(renderLoadingHeader());
  root.appendChild(renderLoadingBody());

  try {
    const [bundle, runtime] = await Promise.all([
      loadBenchmarkBundle(),
      import('./charts/runtime'),
    ]);
    chartRuntime = runtime;
    data = bundle.data;
    parseReport = bundle.parseReport;
    sourceInventory = bundle.sourceInventory;
    state = createDefaultState(data.environments, data.runs);
    const appEl = renderApp();
    clear(root);
    (root as HTMLElement).appendChild(appEl);
    // renderApp() already renders with default state — no extra update() needed
    window.addEventListener('resize', resizeCharts);
  } catch (err) {
    clear(root);
    const msg = emptyStateMessage(
      `Failed to load benchmark data: ${err instanceof Error ? err.message : String(err)}`,
    );
    msg.style.color = '#c96f73';
    const hint = h(
      'small',
      {},
      'Make sure to run: python dev/benchmarks/generate_benchmark_data.py',
    );
    msg.appendChild(h('br'));
    msg.appendChild(hint);
    root.appendChild(msg);
  }
}

init();
