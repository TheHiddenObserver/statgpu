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
  main.appendChild(renderFilterBar(data!.runs, data!, state, update));
  main.appendChild(renderChartArea());
  main.appendChild(renderOverviewTable(getFilteredRuns(), state, update));
  return main;
}

function renderChartArea(): HTMLElement {
  const area = h('div', { class: 'chart-area' });
  const timingDiv = h('div', { id: 'timing-chart', class: 'chart-container' });
  const speedupDiv = h('div', { id: 'speedup-chart', class: 'chart-container' });
  area.appendChild(timingDiv);
  area.appendChild(speedupDiv);

  setTimeout(() => {
    const filtered = getFilteredRuns();
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

function update(): void {
  const main = document.querySelector('.main') as HTMLElement | null;
  if (!main) return;

  for (const chart of chartInstances) {
    if (!chart.isDisposed()) chart.dispose();
  }
  chartInstances.length = 0;

  clear(main);
  main.appendChild(renderFilterBar(data!.runs, data!, state, update));
  main.appendChild(renderChartArea());
  main.appendChild(renderOverviewTable(getFilteredRuns(), state, update));
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) return;

  root.innerHTML =
    '<div class="empty-state">Loading benchmark data...</div>';

  try {
    [data, parseReport] = await Promise.all([
      fetchBenchmarkData(),
      fetchParseReport(),
    ]);
    const appEl = renderApp();
    clear(root);
    (root as HTMLElement).appendChild(appEl);
    update();
  } catch (err) {
    root.innerHTML = `<div class="empty-state" style="color:#ff4d4f;">
      Failed to load benchmark data: ${err instanceof Error ? err.message : String(err)}<br/>
      <small>Make sure to run: python dev/benchmarks/generate_benchmark_data.py</small>
    </div>`;
  }
}

init();
