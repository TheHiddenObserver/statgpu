import type { Run } from '../schema';
import type { AppState } from '../state';
import { h } from '../utils/dom';
import { setSortColumn } from '../state';
import { emptyFilterMessage } from './EmptyState';
import {
  formatModelName,
  formatTime,
  formatSpeedup,
  formatQuality,
} from '../utils/format';

// ---------------------------------------------------------------------------
// Sort helper
// ---------------------------------------------------------------------------

function getSortValue(r: Run, key: string): string | number {
  switch (key) {
    case 'model_id':
      return r.model_id;
    case 'penalty':
      return r.penalty ?? '';
    case 'solver':
      return r.solver ?? '';
    case 'backend_framework':
      return r.backend ?? r.framework;
    case 'scale':
      return r.scale.label;
    case 'time_ms':
      return r.metrics.timing?.fit_time_ms ?? 0;
    case 'speedup':
      return r.metrics.speedup?.value ?? 0;
    case 'quality':
      return (
        r.metrics.timing?.quality ?? r.metrics.speedup?.quality ?? ''
      );
    case 'source':
      return r.source.file;
    default:
      return '';
  }
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

  // Empty state
  if (filtered.length === 0) {
    container.appendChild(emptyFilterMessage());
    return container;
  }

  const displayCount =
    state.tableLimit === Infinity
      ? filtered.length
      : Math.min(filtered.length, state.tableLimit);
  const title = h(
    'div',
    { style: 'font-weight:bold; margin-bottom:4px;' },
    `Showing ${displayCount} of ${filtered.length} runs`,
  );
  container.appendChild(title);

  const table = h('table');
  const thead = h('thead');
  const headerRow = h('tr');
  const cols = [
    'Model',
    'Penalty',
    'Solver',
    'Backend',
    'Scale',
    'Time (ms)',
    'Speedup',
    'Quality',
    'Source',
  ];
  const colKeyMap: Record<string, string> = {
    Model: 'model_id',
    Penalty: 'penalty',
    Solver: 'solver',
    Backend: 'backend_framework',
    Scale: 'scale',
    'Time (ms)': 'time_ms',
    Speedup: 'speedup',
    Quality: 'quality',
    Source: 'source',
  };
  for (const col of cols) {
    const ck = colKeyMap[col];
    const arrow =
      state.sortColumn === ck
        ? state.sortDir === 'asc'
          ? ' ▲'
          : ' ▼'
        : '';
    const th = h('th', {}, col + arrow);
    th.addEventListener('click', () => {
      setSortColumn(state, ck);
      onUpdate();
    });
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Sort filtered runs
  const sorted = [...filtered];
  if (state.sortColumn) {
    const key = state.sortColumn;
    sorted.sort((a, b) => {
      const va = getSortValue(a, key);
      const vb = getSortValue(b, key);
      if (va < vb) return state.sortDir === 'asc' ? -1 : 1;
      if (va > vb) return state.sortDir === 'asc' ? 1 : -1;
      return 0;
    });
  }

  const tbody = h('tbody');
  const displayRuns = sorted.slice(0, state.tableLimit);
  for (const r of displayRuns) {
    const row = h('tr', {
      style: 'border-bottom:1px solid #eee;',
    });
    const t = r.metrics.timing;
    const s = r.metrics.speedup;
    const cells = [
      formatModelName(r.model_id),
      r.penalty ?? '-',
      r.solver_display ?? r.solver ?? '-',
      r.backend ?? r.framework,
      r.scale.label,
      t ? formatTime(t.fit_time_ms, t.std_ms) : '-',
      s ? formatSpeedup(s.value) : '-',
      formatQuality(t?.quality, s?.quality),
      r.source.file,
    ];
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const td = h('td', {}, String(c));
      // Highlight the Speedup column (index 6) when speedup >= 2x
      if (i === 6 && s && s.value >= 2) {
        td.style.cssText +=
          '; font-weight:bold; color:#52c41a;';
      }
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  container.appendChild(table);

  // Show more button
  if (state.tableLimit === 200 && filtered.length > 200) {
    const btn = h(
      'button',
      { style: 'margin-top:6px; padding:4px 12px;' },
      `Show all ${filtered.length} runs`,
    );
    btn.addEventListener('click', () => {
      state.tableLimit = Infinity;
      onUpdate();
    });
    container.appendChild(btn);
  } else if (
    state.tableLimit === Infinity &&
    filtered.length > 200
  ) {
    const btn = h(
      'button',
      { style: 'margin-top:6px; padding:4px 12px;' },
      'Show first 200',
    );
    btn.addEventListener('click', () => {
      state.tableLimit = 200;
      onUpdate();
    });
    container.appendChild(btn);
  }

  // Accuracy panel
  const accRuns = filtered.filter((r) => r.metrics.accuracy);
  if (accRuns.length > 0) {
    const accToggle = h(
      'div',
      {
        style:
          'margin-top:4px; color:#1890ff; cursor:pointer; font-size:12px;',
      },
      '▶ Accuracy (l2_diff vs reference)',
    );
    accToggle.addEventListener('click', () => {
      const accPanel = document.getElementById('accuracy-panel');
      if (accPanel) {
        accPanel.style.display =
          accPanel.style.display === 'none' ? 'block' : 'none';
        accToggle.textContent =
          accPanel.style.display === 'none'
            ? '▶ Accuracy (l2_diff vs reference)'
            : '▼ Accuracy (l2_diff vs reference)';
      }
    });
    container.appendChild(accToggle);

    const accPanel = h('div', {
      id: 'accuracy-panel',
      style: 'display:none; margin-top:4px;',
    });
    const accTable = h('table', {
      style:
        'width:100%; border-collapse:collapse; font-size:11px;',
    });
    const accHeader = h('tr');
    for (const hdr of [
      'Model',
      'Reference',
      'L2 diff',
      'Max abs diff',
      'Status',
    ]) {
      accHeader.appendChild(
        h(
          'th',
          {
            style:
              'padding:2px 6px; border-bottom:1px solid #ddd; text-align:left;',
          },
          hdr,
        ),
      );
    }
    accTable.appendChild(accHeader);
    for (const r of accRuns.slice(0, 30)) {
      const a = r.metrics.accuracy!;
      const accRow = h('tr');
      const l2 = a.coef_l2_diff ?? 0;
      const status =
        l2 < 1e-5 ? 'PASS' : l2 < 1e-3 ? 'WARN' : 'FAIL';
      const statusColor =
        l2 < 1e-5 ? '#52c41a' : l2 < 1e-3 ? '#faad14' : '#ff4d4f';
      for (const c of [
        r.model_id,
        a.reference ?? 'sklearn',
        l2.toExponential(2),
        (a.coef_max_abs_diff ?? 0).toExponential(2),
      ]) {
        accRow.appendChild(
          h('td', { style: 'padding:2px 6px;' }, String(c)),
        );
      }
      const statusTd = h(
        'td',
        {
          style: `padding:2px 6px; color:${statusColor}; font-weight:bold;`,
        },
        status,
      );
      accRow.appendChild(statusTd);
      accTable.appendChild(accRow);
    }
    accPanel.appendChild(accTable);
    container.appendChild(accPanel);
  }

  return container;
}
