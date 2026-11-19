import { cpuThreadLabel, sourceProvenance } from '../provenance';
import { timingValue, timingQuality, evidenceLabel, canDisplaySpeedup } from '../timing';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { h } from '../utils/dom';
import { setSortColumn, setTableLimit } from '../state';
import { emptyFilterMessage } from './EmptyState';
import {
  getMetricScopeLabel,
  getPrimaryMetricScope,
} from '../data';
import {
  formatModelName,
  formatTime,
  formatSpeedup,
  formatQuality,
} from '../utils/format';
import { renderValidationPanel } from './panels/ValidationPanel';
import { renderAccuracyPanel } from './panels/AccuracyPanel';
import { renderInferencePanel } from './panels/InferencePanel';
import { renderCrossValidationPanel } from './panels/CrossValidationPanel';
import { renderPredictionPanel } from './panels/PredictionPanel';
import { renderConvergencePanel } from './panels/ConvergencePanel';
import { renderSelectionPanel } from './panels/SelectionPanel';

// ---------------------------------------------------------------------------
// Keyed column descriptor
// ---------------------------------------------------------------------------

interface Column {
  key: string;
  label: string;
  sortValue: (r: Run) => string | number | null;
  render: (r: Run) => string;
  style?: (r: Run) => string | null;
}

function formatScope(run: Run): string {
  const scope = getPrimaryMetricScope(run);
  const label = getMetricScopeLabel(scope);
  const timingScope = run.metrics.operation_timing?.scope ?? run.parameters?.timing_scope;
  if (timingScope == null) return label;
  const detail = String(timingScope).replace(/_/g, ' ');
  return detail.toLowerCase() === label.toLowerCase() ? label : `${label} · ${detail}`;
}

function buildColumns(runs: Run[]): Column[] {
  return [
    { key: 'model_id', label: 'Model',
      sortValue: r => r.model_id,
      render: r => formatModelName(r.model_id) },
    { key: 'metric_scope', label: 'Scope',
      sortValue: r => getPrimaryMetricScope(r),
      render: r => formatScope(r) },
    { key: 'penalty', label: 'Penalty',
      sortValue: r => r.penalty ?? '',
      render: r => r.penalty ?? '-' },
    { key: 'solver', label: 'Solver',
      sortValue: r => r.solver ?? '',
      render: r => r.solver_display ?? r.solver ?? '-' },
    { key: 'backend_framework', label: 'Backend',
      sortValue: r => r.backend ?? r.framework,
      render: r => r.implementation
        ? `${r.backend ?? r.framework} (${r.implementation})`
        : (r.backend ?? r.framework) },
    { key: 'scale', label: 'Scale',
      sortValue: r => r.scale.label,
      render: r => r.scale.label },
    { key: 'time_ms', label: 'Time (ms)',
      sortValue: r => timingValue(r),
      render: r => {
        const value = timingValue(r);
        return value == null ? '-' : formatTime(value, r.metrics.operation_timing ? undefined : r.metrics.timing?.std_ms);
      }},
    { key: 'speedup', label: 'Speedup',
      sortValue: r => canDisplaySpeedup(r, runs) ? r.metrics.speedup!.value : null,
      render: r => {
        const s = r.metrics.speedup;
        return s && canDisplaySpeedup(r, runs) ? formatSpeedup(s.value) : '-';
      },
      style: r => canDisplaySpeedup(r, runs) && (r.metrics.speedup?.value ?? 0) >= 2
        ? 'font-weight:bold; color:#52c41a;' : null },
    { key: 'quality', label: 'Quality',
      sortValue: r => timingQuality(r) ?? r.metrics.speedup?.quality ?? '',
      render: r => formatQuality(
        timingQuality(r), canDisplaySpeedup(r, runs) ? r.metrics.speedup?.quality : undefined) },
    { key: 'evidence', label: 'Evidence',
      sortValue: r => r.evidence?.qualification ?? '',
      render: r => evidenceLabel(r) },
    { key: 'timing_statistic', label: 'Timing statistic',
      sortValue: r => r.metrics.operation_timing?.statistic ?? '',
      render: r => r.metrics.operation_timing?.statistic ?? '-' },
    { key: 'cpu_threads', label: 'Host CPU limits',
      sortValue: r => cpuThreadLabel(r),
      render: r => cpuThreadLabel(r) },
    { key: 'source', label: 'Source / provenance',
      sortValue: r => r.source.file,
      render: r => r.source.file },
  ];
}

// ---------------------------------------------------------------------------
// Table renderer
// ---------------------------------------------------------------------------

export function renderOverviewTable(
  runs: Run[],
  state: AppState,
  onUpdate: () => void,
): HTMLElement {
  const container = h('div', { class: 'table-container' });
  const filtered = runs;

  if (filtered.length === 0) {
    container.appendChild(emptyFilterMessage());
    return container;
  }

  // Metric panels belong to the filtered model rows, not below hundreds of
  // overview records. Keeping their headers above the table makes existing
  // inference/CV-adjacent metrics discoverable without changing the data model.
  const panelStack = h('div', { class: 'metric-panel-stack' });
  const panels = [
    renderValidationPanel(filtered, state, onUpdate),
    renderAccuracyPanel(filtered, state, onUpdate),
    renderInferencePanel(filtered, state, onUpdate),
    renderCrossValidationPanel(filtered, state, onUpdate),
    renderPredictionPanel(filtered, state, onUpdate),
    renderConvergencePanel(filtered, state, onUpdate),
    renderSelectionPanel(filtered, state, onUpdate),
  ];
  for (const panel of panels) {
    if (panel) panelStack.appendChild(panel);
  }
  if (panelStack.childElementCount > 0) container.appendChild(panelStack);

  const displayCount =
    state.tableLimit === Infinity
      ? filtered.length
      : Math.min(filtered.length, state.tableLimit);
  const title = h('div', { class: 'overview-table-title' },
    `Showing ${displayCount} of ${filtered.length} runs`);
  container.appendChild(title);

  const columns = buildColumns(runs);
  const table = h('table');
  const thead = h('thead');
  const headerRow = h('tr');

  for (const col of columns) {
    const arrow = state.sortColumn === col.key
      ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
    const th = h('th', {}, col.label + arrow);
    th.addEventListener('click', () => {
      setSortColumn(state, col.key);
      onUpdate();
    });
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Sort: null-last, run_id tie-break
  const sorted = [...filtered];
  if (state.sortColumn) {
    const col = columns.find(c => c.key === state.sortColumn);
    if (col) {
      sorted.sort((a, b) => {
        const va = col.sortValue(a);
        const vb = col.sortValue(b);
        // null always last
        if (va == null && vb == null) return a.run_id.localeCompare(b.run_id);
        if (va == null) return 1;
        if (vb == null) return -1;
        if (va < vb) return state.sortDir === 'asc' ? -1 : 1;
        if (va > vb) return state.sortDir === 'asc' ? 1 : -1;
        return a.run_id.localeCompare(b.run_id);
      });
    }
  }

  const tbody = h('tbody');
  const displayRuns = sorted.slice(0, state.tableLimit);
  for (const r of displayRuns) {
    const row = h('tr', { style: 'border-bottom:1px solid #eee;' });
    for (const col of columns) {
      const val = col.render(r);
      const td = h('td', {}, col.key === 'source' ? '' : val);
      if (col.key === 'source') {
        td.classList.add('source-provenance');
        td.appendChild(h('div', {}, `${r.source.file} · ${r.source.source_id}`));
        td.appendChild(h('div', {}, `Run ${r.run_id} · raw commit ${r.evidence?.source_commit?.slice(0, 12) ?? 'not recorded'}`));
        if (r.parameters?.external_adapter) td.appendChild(h('div', {}, `Adapter: ${r.parameters.external_adapter}`));
        const details = h('details');
        details.appendChild(h('summary', {}, 'Full source / raw provenance'));
        for (const [label, value] of sourceProvenance(r)) details.appendChild(h('div', {}, `${label}: ${value}`));
        td.appendChild(details);
      }
      const extraStyle = col.style?.(r);
      if (extraStyle) {
        td.style.cssText += `; ${extraStyle}`;
      }
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  container.appendChild(table);

  // Show more / Show first 200
  if (state.tableLimit === 200 && filtered.length > 200) {
    const btn = h('button', { style: 'margin-top:6px; padding:4px 12px;' },
      `Show all ${filtered.length} runs`);
    btn.addEventListener('click', () => {
      setTableLimit(state, Infinity);
      onUpdate();
    });
    container.appendChild(btn);
  } else if (state.tableLimit === Infinity && filtered.length > 200) {
    const btn = h('button', { style: 'margin-top:6px; padding:4px 12px;' },
      'Show first 200');
    btn.addEventListener('click', () => {
      setTableLimit(state, 200);
      onUpdate();
    });
    container.appendChild(btn);
  }

  return container;
}
