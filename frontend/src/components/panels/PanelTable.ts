/** Shared collapsible panel with pagination. */

import { h } from '../../utils/dom';
import type { AppState } from '../../state';

export interface PanelColumn {
  key: string;
  label: string;
  render: (row: Record<string, unknown>) => string;
  style?: (row: Record<string, unknown>) => string;
}

export interface PanelTableOptions {
  panelId: string;
  title: string;
  columns: PanelColumn[];
  rows: Record<string, unknown>[];
  state: AppState;
  defaultLimit?: number;
}

export function renderPanelTable(opts: PanelTableOptions): HTMLElement {
  const { panelId, title, columns, rows, state, defaultLimit = 30 } = opts;
  const limit = state.panelLimits[panelId] ?? defaultLimit;
  const expanded = state.expandedPanels.has(panelId);
  const displayRows = limit === Infinity ? rows : rows.slice(0, limit);

  const container = h('div', { style: 'margin-top:6px; font-size:12px;' });

  // Toggle header
  const toggle = h('div', {
    style: 'color:#1890ff; cursor:pointer; font-weight:bold; margin-bottom:4px;',
  }, `${expanded ? '▼' : '▶'} ${title} (${rows.length})`);
  toggle.addEventListener('click', () => {
    if (expanded) {
      state.expandedPanels.delete(panelId);
    } else {
      state.expandedPanels.add(panelId);
    }
    // Trigger re-render via callback
    (toggle as any)._onToggle?.();
  });
  container.appendChild(toggle);

  if (!expanded) return container;

  // Table
  const table = h('table', {
    style: 'width:100%; border-collapse:collapse; font-size:11px;',
  });

  const thead = h('tr');
  for (const col of columns) {
    thead.appendChild(h('th', {
      style: 'padding:2px 6px; border-bottom:1px solid #ddd; text-align:left;',
    }, col.label));
  }
  table.appendChild(thead);

  for (const row of displayRows) {
    const tr = h('tr');
    for (const col of columns) {
      const cellStyle = col.style ? col.style(row) : 'padding:2px 6px;';
      tr.appendChild(h('td', { style: cellStyle }, col.render(row)));
    }
    table.appendChild(tr);
  }
  container.appendChild(table);

  // "Showing N of M" + "Show all" toggle
  if (rows.length > defaultLimit) {
    const showing = limit === Infinity ? rows.length : limit;
    const footer = h('div', { style: 'margin-top:4px; font-size:11px; color:#666;' });
    footer.appendChild(h('span', {}, `Showing ${showing} of ${rows.length}`));

    if (limit === Infinity) {
      const btn = h('button', { style: 'margin-left:8px; padding:1px 6px; font-size:11px;' }, 'Show first 30');
      btn.addEventListener('click', () => {
        state.panelLimits[panelId] = 30;
        (btn as any)._onToggle?.();
      });
      footer.appendChild(btn);
    } else {
      const btn = h('button', { style: 'margin-left:8px; padding:1px 6px; font-size:11px;' }, `Show all ${rows.length}`);
      btn.addEventListener('click', () => {
        state.panelLimits[panelId] = Infinity;
        (btn as any)._onToggle?.();
      });
      footer.appendChild(btn);
    }
    container.appendChild(footer);
  }

  return container;
}
