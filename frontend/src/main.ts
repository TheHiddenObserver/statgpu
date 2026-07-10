import './style.css';

import * as echarts from 'echarts';
import type { BenchmarkData, ParseReport, Run } from './schema';
import { fetchBenchmarkData, fetchParseReport, filterRuns } from './data';
import { createDefaultState } from './state';
import type { AppState } from './state';
import { h, clear } from './utils/dom';
import { renderHeader } from './components/Header';
import { renderSidebar } from './components/Sidebar';
import { renderFilterBar } from './components/FilterBar';
import { renderOverviewTable } from './components/OverviewTable';
import { renderSummaryCards } from './components/SummaryCards';
import { renderTimingChart } from './charts/TimingChart';
import { renderSpeedupChart } from './charts/SpeedupChart';
import { emptyStateMessage } from './components/EmptyState';

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

let data: BenchmarkData | null = null;
let parseReport: ParseReport | null = null;
let state: AppState = createDefaultState();

/** Track ECharts instances for cleanup before re-render */
const chartInstances: echarts.ECharts[] = [];

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

function renderApp(): HTMLElement {
  const app = h('div', { id: 'app-root' });
  app.appendChild(renderHeader(data!, parseReport, state, update));
  app.appendChild(renderBody());
  return app;
}

function renderBody(): HTMLElement {
  const body = h('div', { class: 'body' });
  body.appendChild(renderSidebar(data!, state, update));

  const right = h('div', { style: 'flex:1; display:flex; flex-direction:column; overflow:hidden;' });
  // Summary cards + footer persist across filter updates; only main is re-rendered
  right.appendChild(renderSummaryCards(data!, parseReport, data!.runs));
  const main = renderMain();
  right.appendChild(main);
  right.appendChild(renderFooter());

  body.appendChild(right);
  return body;
}

function renderMain(): HTMLElement {
  const main = h('div', { class: 'main' });
  const filtered = getFilteredRuns();
  main.appendChild(renderFilterBar(data!.runs, data!, state, update));
  main.appendChild(renderChartArea(filtered));
  main.appendChild(renderOverviewTable(filtered, state, update));
  return main;
}

function renderChartArea(filtered: Run[]): HTMLElement {
  const area = h('div', { class: 'chart-area' });
  const timingDiv = h('div', { id: 'timing-chart', class: 'chart-container' });
  const speedupDiv = h('div', { id: 'speedup-chart', class: 'chart-container' });
  area.appendChild(timingDiv);
  area.appendChild(speedupDiv);

  setTimeout(() => {
    renderTimingChart(timingDiv, filtered, state, chartInstances);
    renderSpeedupChart(speedupDiv, filtered, state, chartInstances);
  }, 0);
  return area;
}

function renderFooter(): HTMLElement {
  const footer = h('div', { class: 'dashboard-footer' });

  const links: [string, string][] = [
    ['Benchmark guide', '../../en/guides/benchmarks.html'],
    ['Raw data (JSON)', 'data/benchmark_data.json'],
    ['Parse report (JSON)', 'data/parse_report.json'],
    [
      'GitHub source',
      'https://github.com/TheHiddenObserver/statgpu/tree/master/dev/benchmarks',
    ],
  ];
  for (const [label, href] of links) {
    const a = h('a', { href, target: '_blank', rel: 'noopener' }, label);
    footer.appendChild(a);
  }

  const meta = h('span', {}, `Schema ${data!.schema_version} · ${data!.meta.git_sha}`);
  footer.appendChild(meta);

  return footer;
}

// ---------------------------------------------------------------------------
// State & update loop
// ---------------------------------------------------------------------------

function getFilteredRuns(): Run[] {
  if (!data) return [];
  return filterRuns(data.runs, state);
}

function disposeCharts(): void {
  for (const id of ['timing-chart', 'speedup-chart']) {
    const el = document.getElementById(id);
    if (!el) continue;
    const chart = echarts.getInstanceByDom(el);
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
  const filtered = filterRuns(allRuns, state);

  clear(main);
  main.appendChild(renderFilterBar(allRuns, data!, state, update));
  main.appendChild(renderChartArea(filtered));
  main.appendChild(renderOverviewTable(filtered, state, update));
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) return;

  clear(root);
  root.appendChild(emptyStateMessage('Loading benchmark data...'));

  try {
    data = await fetchBenchmarkData();
    // parseReport is non-critical; fetch separately so failure doesn't block dashboard
    parseReport = await fetchParseReport().catch(() => null);
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
    msg.style.color = '#ff4d4f';
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
