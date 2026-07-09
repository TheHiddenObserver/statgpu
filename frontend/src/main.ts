import * as echarts from 'echarts';
import type { BenchmarkData, Run } from './schema';
import { fetchBenchmarkData, fetchParseReport, filterRuns, createDefaultState, getUniqueValues, getUniqueScaleKeys } from './data';
import type { AppState } from './data';

/** Color palette */
const COLORS: Record<string, string> = {
  numpy: '#5470c6',
  cupy: '#91cc75',
  torch: '#fac858',
  sklearn: '#ee6666',
  statsmodels: '#73c0de',
  glmnet: '#3ba272',
  r: '#3ba272',
};

/** Quality badge colors */
const QUALITY_COLORS: Record<string, string> = {
  measured: '#52c41a',
  reported: '#faad14',
  computed: '#1890ff',
  partial: '#ff4d4f',
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

let data: BenchmarkData | null = null;
let state: AppState = createDefaultState();

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function h(tag: string, attrs: Record<string, string> = {}, ...children: (string | Node)[]): HTMLElement {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) el.append(typeof c === 'string' ? document.createTextNode(c) : c);
  return el;
}

function clear(el: HTMLElement) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function renderApp(): HTMLElement {
  const app = h('div', { id: 'app-root' });
  app.style.cssText = 'display:flex; flex-direction:column; height:100vh;';

  app.appendChild(renderHeader());
  app.appendChild(renderBody());

  return app;
}

function renderHeader(): HTMLElement {
  const header = h('div', { class: 'header' });
  header.style.cssText = `
    display:flex; align-items:center; justify-content:space-between;
    padding:8px 16px; background:#1a1a2e; color:#eee; font-size:14px;
    border-bottom:2px solid #16213e;
  `;

  const left = h('div');
  left.innerHTML = `<strong style="font-size:16px;">statgpu</strong> <span style="color:#888;">Benchmark Dashboard</span>`;

  const right = h('div', { style: 'display:flex; align-items:center; gap:12px;' });

  // Hardware selector
  const hwLabel = h('span', {}, 'Environment: ');
  const hwSelect = h('select', { id: 'env-select', style: 'padding:4px 8px; border-radius:4px;' });
  for (const env of data?.environments ?? []) {
    const opt = h('option', { value: env.env_id }, env.label);
    if (env.env_id === state.selectedEnvId) opt.setAttribute('selected', '');
    hwSelect.appendChild(opt);
  }
  hwSelect.addEventListener('change', () => {
    state.selectedEnvId = (hwSelect as HTMLSelectElement).value;
    update();
  });
  right.appendChild(hwLabel);
  right.appendChild(hwSelect);

  // Parse report
  fetchParseReport().then(report => {
    const info = h('span', { style: 'color:#666; font-size:12px;' },
      `${report.runs_generated} runs from ${report.files_parsed}/${report.files_seen} files`
    );
    right.appendChild(info);
  });

  header.appendChild(left);
  header.appendChild(right);
  return header;
}

function renderBody(): HTMLElement {
  const body = h('div', { class: 'body' });
  body.style.cssText = 'display:flex; flex:1; overflow:hidden;';

  body.appendChild(renderSidebar());
  body.appendChild(renderMain());
  return body;
}

function renderSidebar(): HTMLElement {
  const sidebar = h('div', { class: 'sidebar' });
  sidebar.style.cssText = `
    width:200px; min-width:200px; background:#f5f5f5; border-right:1px solid #ddd;
    padding:8px; overflow-y:auto; font-size:13px;
  `;

  const title = h('div', { style: 'font-weight:bold; margin-bottom:6px;' }, 'Categories');
  sidebar.appendChild(title);

  // Search
  const search = h('input', {
    type: 'text', placeholder: 'Search...',
    style: 'width:100%; padding:4px; margin-bottom:6px; border:1px solid #ccc; border-radius:4px;',
  });
  sidebar.appendChild(search);

  // Category checkboxes
  const catContainer = h('div', { id: 'category-list' });
  for (const cat of data?.categories ?? []) {
    const row = h('div', { style: 'display:flex; align-items:center; gap:4px; padding:2px 0; cursor:pointer;' });
    const cb = h('input', {
      type: 'checkbox', id: `cat-${cat.category_id}`, value: cat.category_id,
    }) as HTMLInputElement;
    if (state.selectedCategoryIds.has(cat.category_id)) cb.checked = true;
    cb.addEventListener('change', () => {
      if (cb.checked) state.selectedCategoryIds.add(cat.category_id);
      else state.selectedCategoryIds.delete(cat.category_id);
      update();
    });
    const label = h('label', { for: `cat-${cat.category_id}` }, ` ${cat.name_zh}`);
    row.appendChild(cb);
    row.appendChild(label);
    catContainer.appendChild(row);
  }
  sidebar.appendChild(catContainer);

  // Select all / none
  const btnRow = h('div', { style: 'display:flex; gap:4px; margin-top:6px;' });
  const allBtn = h('button', { style: 'font-size:11px; padding:2px 6px;' }, 'All');
  allBtn.addEventListener('click', () => {
    for (const cat of data?.categories ?? []) state.selectedCategoryIds.add(cat.category_id);
    update();
  });
  const noneBtn = h('button', { style: 'font-size:11px; padding:2px 6px;' }, 'None');
  noneBtn.addEventListener('click', () => {
    state.selectedCategoryIds.clear();
    update();
  });
  btnRow.appendChild(allBtn);
  btnRow.appendChild(noneBtn);
  sidebar.appendChild(btnRow);

  return sidebar;
}

function renderMain(): HTMLElement {
  const main = h('div', { class: 'main' });
  main.style.cssText = 'flex:1; display:flex; flex-direction:column; overflow-y:auto; padding:12px;';

  main.appendChild(renderFilterBar());
  main.appendChild(renderChartArea());
  main.appendChild(renderOverviewTable());

  return main;
}

function renderFilterBar(): HTMLElement {
  const bar = h('div', { class: 'filter-bar' });
  bar.style.cssText = 'display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #eee; margin-bottom:8px; font-size:13px;';

  const filtered = getFilteredRuns();

  // Model selector
  const modelIds = getUniqueValues(filtered, 'model_id');
  if (modelIds.length > 0) {
    bar.appendChild(h('span', {}, 'Model:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const m of modelIds) {
      const opt = h('option', { value: m }, m);
      if (m === state.selectedModelId) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedModelId = (sel as HTMLSelectElement).value || null;
      state.selectedPenalty = null;
      state.selectedSolver = null;
      update();
    });
    bar.appendChild(sel);
  }

  // Penalty selector (appears after model selected)
  if (state.selectedModelId) {
    const penalties = getUniqueValues(filtered.filter(r => r.model_id === state.selectedModelId), 'penalty');
    bar.appendChild(h('span', {}, 'Penalty:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const p of penalties) {
      const opt = h('option', { value: p }, p || 'none');
      if (p === state.selectedPenalty) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedPenalty = (sel as HTMLSelectElement).value || null;
      state.selectedSolver = null;
      update();
    });
    bar.appendChild(sel);
  }

  // Solver selector (appears after penalty selected)
  if (state.selectedPenalty) {
    const solvers = getUniqueValues(
      filtered.filter(r => r.model_id === state.selectedModelId && r.penalty === state.selectedPenalty),
      'solver'
    );
    bar.appendChild(h('span', {}, 'Solver:'));
    const sel = h('select', { style: 'padding:2px 6px;' });
    sel.appendChild(h('option', { value: '' }, 'All'));
    for (const s of solvers) {
      const opt = h('option', { value: s }, s);
      if (s === state.selectedSolver) opt.setAttribute('selected', '');
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      state.selectedSolver = (sel as HTMLSelectElement).value || null;
      update();
    });
    bar.appendChild(sel);
  }

  // Scale chips
  const scaleKeys = getUniqueScaleKeys(filtered);
  if (scaleKeys.length > 0) {
    bar.appendChild(h('span', {}, 'Scale:'));
    for (const sk of scaleKeys.slice(0, 15)) {
      const chip = h('span', {
        style: `display:inline-block; padding:2px 6px; margin:1px; border-radius:4px; cursor:pointer;
          font-size:11px; border:1px solid #ccc;
          ${state.selectedScaleKeys.has(sk) ? 'background:#1890ff; color:#fff; border-color:#1890ff;' : ''}`,
      }, data?.runs.find(r => r.scale.scale_key === sk)?.scale.label ?? sk);
      chip.addEventListener('click', () => {
        if (state.selectedScaleKeys.has(sk)) state.selectedScaleKeys.delete(sk);
        else state.selectedScaleKeys.add(sk);
        update();
      });
      bar.appendChild(chip);
    }
  }

  // Backend radio
  bar.appendChild(h('span', {}, '| Backend:'));
  for (const bk of ['all', 'numpy', 'cupy', 'torch']) {
    const label = bk === 'all' ? 'All' : bk;
    const radio = h('label', { style: 'margin:0 4px; cursor:pointer; font-size:12px;' });
    const inp = h('input', { type: 'radio', name: 'backend', value: bk }) as HTMLInputElement;
    if (bk === 'all' && state.selectedBackends.size === 0) inp.checked = true;
    if (bk !== 'all' && state.selectedBackends.has(bk)) inp.checked = true;
    inp.addEventListener('change', () => {
      if (bk === 'all') state.selectedBackends.clear();
      else {
        state.selectedBackends.clear();
        state.selectedBackends.add(bk);
      }
      update();
    });
    radio.appendChild(inp);
    radio.appendChild(document.createTextNode(label));
    bar.appendChild(radio);
  }

  // External frameworks
  bar.appendChild(h('span', {}, '| Ext:'));
  for (const ext of ['sklearn', 'glmnet', 'statsmodels']) {
    const lbl = h('label', { style: 'margin:0 4px; cursor:pointer; font-size:12px;' });
    const cb = h('input', { type: 'checkbox', value: ext }) as HTMLInputElement;
    if (state.showExternal.has(ext)) cb.checked = true;
    cb.addEventListener('change', () => {
      if (cb.checked) state.showExternal.add(ext);
      else state.showExternal.delete(ext);
      update();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(ext));
    bar.appendChild(lbl);
  }

  return bar;
}

function renderChartArea(): HTMLElement {
  const area = h('div', { class: 'chart-area' });
  area.style.cssText = 'display:flex; gap:8px; margin-bottom:8px;';

  const timingDiv = h('div', { id: 'timing-chart', style: 'flex:1; height:350px; border:1px solid #eee; border-radius:4px;' });
  const speedupDiv = h('div', { id: 'speedup-chart', style: 'flex:1; height:350px; border:1px solid #eee; border-radius:4px;' });

  area.appendChild(timingDiv);
  area.appendChild(speedupDiv);

  // Render charts after DOM update
  setTimeout(() => renderCharts(), 0);
  return area;
}

function renderCharts() {
  const filtered = getFilteredRuns();
  renderTimingChart(filtered);
  renderSpeedupChart(filtered);
}

function renderTimingChart(runs: Run[]) {
  const el = document.getElementById('timing-chart') as HTMLElement | null;
  if (!el) return;
  let chart = echarts.getInstanceByDom(el);
  if (!chart) chart = echarts.init(el);

  const timingRuns = runs.filter(r => r.metrics.timing);
  if (timingRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: { text: 'No timing data', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    });
    return;
  }

  // Sort: model + penalty + scale, then numpy/cupy/torch
  timingRuns.sort((a, b) => {
    const ak = `${a.model_id}-${a.penalty}-${a.scale.scale_key}-${a.backend}`;
    const bk = `${b.model_id}-${b.penalty}-${b.scale.scale_key}-${b.backend}`;
    return ak.localeCompare(bk);
  });

  const categories = timingRuns.map(r =>
    `${r.model_id.replace('Penalized', '').replace('Regression', '')}+${r.penalty ?? 'none'} ${r.scale.label}`
  );
  const backendOrder = ['numpy', 'cupy', 'torch'];
  const series = backendOrder.map(bk => ({
    name: bk,
    type: 'bar' as const,
    data: timingRuns.filter(r => r.backend === bk).map(r => r.metrics.timing!.fit_time_ms),
    itemStyle: { color: COLORS[bk] || '#999' },
  }));

  // Add external frameworks
  for (const ext of state.showExternal) {
    const extRuns = timingRuns.filter(r => r.framework === ext);
    if (extRuns.length > 0) {
      series.push({
        name: ext,
        type: 'bar' as const,
        data: extRuns.map(r => r.metrics.timing!.fit_time_ms),
        itemStyle: { color: COLORS[ext] || '#999' },
      });
    }
  }

  chart.setOption({
    title: { text: 'Fit Time (ms)', left: 'center', textStyle: { fontSize: 13 } },
    tooltip: {
      trigger: 'axis',
      formatter: (params: { seriesName: string; value: number; color: string }[]) => {
        return params.map(p =>
          `<span style="color:${p.color}">●</span> ${p.seriesName}: <b>${p.value.toFixed(2)}ms</b>`
        ).join('<br/>');
      },
    },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 10, right: 10, top: 40, bottom: 30 },
    xAxis: { type: 'category', data: categories, axisLabel: { fontSize: 10, rotate: 45 } },
    yAxis: { type: 'log', name: 'ms', axisLabel: { fontSize: 10 } },
    series,
  }, true);
}

function renderSpeedupChart(runs: Run[]) {
  const el = document.getElementById('speedup-chart') as HTMLElement | null;
  if (!el) return;
  let chart = echarts.getInstanceByDom(el);
  if (!chart) chart = echarts.init(el);

  const speedupRuns = runs.filter(r => r.metrics.speedup && r.backend !== 'numpy');
  if (speedupRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: { text: 'No speedup data', left: 'center', top: 'center', textStyle: { color: '#999', fontSize: 14 } },
    });
    return;
  }

  speedupRuns.sort((a, b) => (b.metrics.speedup?.value ?? 0) - (a.metrics.speedup?.value ?? 0));
  const topN = speedupRuns.slice(0, 30);
  const labels = topN.map(r =>
    `${r.model_id.replace('Penalized', '').replace('Regression', '')}+${r.penalty} [${r.solver_display ?? r.solver}] ${r.scale.label}`
  );

  chart.setOption({
    title: { text: 'GPU Speedup vs CPU', left: 'center', textStyle: { fontSize: 13 } },
    tooltip: {
      trigger: 'axis',
      formatter: (params: { value: number; color: string }[]) => {
        const v = params[0].value;
        const label = v > 1 ? 'faster' : v < 1 ? 'slower' : 'same';
        return `<b>${v.toFixed(2)}x</b> (${label})`;
      },
    },
    grid: { left: 10, right: 10, top: 40, bottom: 30 },
    xAxis: { type: 'value', name: 'speedup', axisLabel: { fontSize: 10 } },
    yAxis: { type: 'category', data: labels.reverse(), axisLabel: { fontSize: 10 } },
    series: [{
      type: 'bar',
      data: topN.reverse().map(r => ({
        value: r.metrics.speedup!.value,
        itemStyle: { color: r.metrics.speedup!.value >= 1 ? '#52c41a' : '#ff4d4f' },
      })),
      markLine: {
        silent: true,
        data: [{ xAxis: 1, lineStyle: { color: '#999', type: 'dashed' } }],
        label: { formatter: '1.0x' },
      },
    }],
  }, true);
}

function renderOverviewTable(): HTMLElement {
  const container = h('div', { class: 'table-container' });
  container.style.cssText = 'font-size:12px;';

  const filtered = getFilteredRuns();
  const title = h('div', { style: 'font-weight:bold; margin-bottom:4px;' },
    `Showing ${Math.min(filtered.length, 200)} of ${filtered.length} runs`
  );
  container.appendChild(title);

  const table = h('table', { style: 'width:100%; border-collapse:collapse;' });
  const thead = h('thead');
  const headerRow = h('tr');
  const cols = ['Model', 'Penalty', 'Solver', 'Backend', 'Scale', 'Time (ms)', 'Speedup', 'Quality', 'Source'];
  for (const col of cols) {
    const th = h('th', { style: 'padding:4px 6px; border-bottom:2px solid #ddd; text-align:left; position:sticky; top:0; background:#fff;' }, col);
    th.addEventListener('click', () => {
      // Simple sort toggle
      (th as HTMLElement).dataset.sortDir = (th as HTMLElement).dataset.sortDir === 'asc' ? 'desc' : 'asc';
      update();
    });
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = h('tbody');
  const displayRuns = filtered.slice(0, 200);
  for (const r of displayRuns) {
    const row = h('tr', { style: 'border-bottom:1px solid #eee;' });
    const t = r.metrics.timing;
    const s = r.metrics.speedup;
    const cells = [
      r.model_id.replace('Penalized', '').replace('Regression', ''),
      r.penalty ?? '-',
      r.solver_display ?? r.solver ?? '-',
      r.backend ?? r.framework,
      r.scale.label,
      t ? `${t.fit_time_ms.toFixed(2)}±${(t.std_ms ?? 0).toFixed(1)}` : '-',
      s ? `${s.value.toFixed(1)}x` : '-',
      t?.quality ?? s?.quality ?? '-',
      r.source.file,
    ];
    for (const c of cells) {
      const td = h('td', { style: 'padding:3px 6px;' }, String(c));
      if (c && typeof c === 'string' && c.includes('x') && s && s.value >= 2) {
        td.style.cssText += '; font-weight:bold; color:#52c41a;';
      }
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  container.appendChild(table);

  // Show more button
  if (filtered.length > 200) {
    const btn = h('button', { style: 'margin-top:6px; padding:4px 12px;' }, `Show all ${filtered.length} runs`);
    btn.addEventListener('click', () => {
      // Re-render with all rows (simple approach: rebuild with larger slice)
      const tableContainer = document.querySelector('.table-container') as HTMLElement | null;
      if (tableContainer) {
        clear(tableContainer);
        tableContainer.appendChild(renderOverviewTableAll(filtered));
      }
    });
    container.appendChild(btn);
  }

  // Accuracy panel
  const accRuns = filtered.filter(r => r.metrics.accuracy);
  if (accRuns.length > 0) {
    const accToggle = h('div', {
      style: 'margin-top:4px; color:#1890ff; cursor:pointer; font-size:12px;',
    }, '▶ Accuracy (l2_diff vs reference)');
    accToggle.addEventListener('click', () => {
      const accPanel = document.getElementById('accuracy-panel');
      if (accPanel) {
        accPanel.style.display = accPanel.style.display === 'none' ? 'block' : 'none';
        accToggle.textContent = accPanel.style.display === 'none'
          ? '▶ Accuracy (l2_diff vs reference)'
          : '▼ Accuracy (l2_diff vs reference)';
      }
    });
    container.appendChild(accToggle);

    const accPanel = h('div', { id: 'accuracy-panel', style: 'display:none; margin-top:4px;' });
    const accTable = h('table', { style: 'width:100%; border-collapse:collapse; font-size:11px;' });
    const accHeader = h('tr');
    for (const hdr of ['Model', 'Reference', 'L2 diff', 'Max abs diff', 'Status']) {
      accHeader.appendChild(h('th', { style: 'padding:2px 6px; border-bottom:1px solid #ddd; text-align:left;' }, hdr));
    }
    accTable.appendChild(accHeader);
    for (const r of accRuns.slice(0, 30)) {
      const a = r.metrics.accuracy!;
      const accRow = h('tr');
      const l2 = a.coef_l2_diff ?? 0;
      const status = l2 < 1e-5 ? 'PASS' : l2 < 1e-3 ? 'WARN' : 'FAIL';
      const statusColor = l2 < 1e-5 ? '#52c41a' : l2 < 1e-3 ? '#faad14' : '#ff4d4f';
      for (const c of [r.model_id, a.reference ?? 'sklearn', l2.toExponential(2), (a.coef_max_abs_diff ?? 0).toExponential(2)]) {
        accRow.appendChild(h('td', { style: 'padding:2px 6px;' }, String(c)));
      }
      const statusTd = h('td', { style: `padding:2px 6px; color:${statusColor}; font-weight:bold;` }, status);
      accRow.appendChild(statusTd);
      accTable.appendChild(accRow);
    }
    accPanel.appendChild(accTable);
    container.appendChild(accPanel);
  }

  return container;
}

function renderOverviewTableAll(runs: Run[]): HTMLElement {
  // Full table render (used when user clicks "Show all")
  return renderOverviewTable(); // Simplified: re-renders the same component with all data
}

// ---------------------------------------------------------------------------
// State & update loop
// ---------------------------------------------------------------------------

function getFilteredRuns(): Run[] {
  if (!data) return [];
  return filterRuns(data.runs, state);
}

function update() {
  const main = document.querySelector('.main') as HTMLElement | null;
  if (!main) return;
  clear(main);
  main.appendChild(renderFilterBar());
  main.appendChild(renderChartArea());
  main.appendChild(renderOverviewTable());
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  const root = document.getElementById('app');
  if (!root) return;

  root.innerHTML = '<div style="padding:40px; text-align:center; color:#999;">Loading benchmark data...</div>';

  try {
    data = await fetchBenchmarkData();
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
