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
// Components
// ---------------------------------------------------------------------------

function renderApp(): HTMLElement {
  const app = h('div', { id: 'app-root' });
  app.style.cssText = 'display:flex; flex-direction:column; height:100vh;';
  app.appendChild(renderHeader(data!, parseReport, state, update));
  app.appendChild(renderBody());
  return app;
}

function renderBody(): HTMLElement {
  const body = h('div', { class: 'body' });
  body.style.cssText = 'display:flex; flex:1; overflow:hidden;';

  body.appendChild(renderSidebar(data!, state, update));
  body.appendChild(renderMain());
  return body;
}

function renderMain(): HTMLElement {
  const main = h('div', { class: 'main' });
  main.style.cssText = 'flex:1; display:flex; flex-direction:column; overflow-y:auto; padding:12px;';

  const allRuns = data!.runs;
  main.appendChild(renderFilterBar(allRuns, data!, state, update));
  main.appendChild(renderChartArea());
  main.appendChild(renderOverviewTable(getFilteredRuns(), state, update));

  return main;
}

function renderChartArea(): HTMLElement {
  const area = h('div', { class: 'chart-area' });
  area.style.cssText = 'display:flex; gap:8px; margin-bottom:8px;';

  const timingDiv = h('div', { id: 'timing-chart', style: 'flex:1; height:350px; border:1px solid #eee; border-radius:4px;' });
  const speedupDiv = h('div', { id: 'speedup-chart', style: 'flex:1; height:350px; border:1px solid #eee; border-radius:4px;' });

  area.appendChild(timingDiv);
  area.appendChild(speedupDiv);

  // Render charts after DOM update
  setTimeout(() => {
    const filtered = getFilteredRuns();
    renderTimingChart(timingDiv, filtered, state, chartInstances);
    renderSpeedupChart(speedupDiv, filtered, state, chartInstances);
  }, 0);
  return area;
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

  // Dispose old ECharts instances before clearing DOM
  for (const chart of chartInstances) {
    if (!chart.isDisposed()) chart.dispose();
  }
  chartInstances.length = 0;

  clear(main);
  const allRuns = data!.runs;
  main.appendChild(renderFilterBar(allRuns, data!, state, update));
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
    '<div style="padding:40px; text-align:center; color:#999;">Loading benchmark data...</div>';

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
    root.innerHTML = `<div style="padding:40px; text-align:center; color:#ff4d4f;">
      Failed to load benchmark data: ${err instanceof Error ? err.message : String(err)}<br/>
      <small>Make sure to run: python dev/benchmarks/generate_benchmark_data.py</small>
    </div>`;
  }
}

init();
