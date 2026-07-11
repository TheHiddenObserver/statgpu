/** Accuracy metrics panel — coef/bse diffs vs reference */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

export function renderAccuracyPanel(runs: Run[], state: AppState, onUpdate: () => void): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const r of runs) {
    const a = r.metrics.accuracy;
    if (!a) continue;
    const l2 = a.coef_l2_diff;
    const status = l2 == null ? 'N/A' : l2 < 1e-5 ? 'PASS' : l2 < 1e-3 ? 'WARN' : 'FAIL';
    const color = l2 == null ? '#888' : l2 < 1e-5 ? '#52c41a' : l2 < 1e-3 ? '#faad14' : '#ff4d4f';
    rows.push({
      model: r.model_id, reference: a.reference ?? '-',
      coef_l2: l2 != null ? l2.toExponential(2) : '-',
      coef_max_abs: a.coef_max_abs_diff != null ? a.coef_max_abs_diff.toExponential(2) : '-',
      bse_max_abs: a.bse_max_abs_diff != null ? a.bse_max_abs_diff.toExponential(2) : '-',
      coef_l2_rel: a.coef_l2_rel_error != null ? a.coef_l2_rel_error.toExponential(2) : '-',
      status, _color: color,
    });
  }
  if (rows.length === 0) return null;

  const panel = renderPanelTable({
    panelId: 'accuracy', title: 'Accuracy Metrics',
    columns: [
      { key: 'model', label: 'Model', render: r => String(r.model) },
      { key: 'reference', label: 'Reference', render: r => String(r.reference) },
      { key: 'coef_l2', label: 'L2 Diff', render: r => String(r.coef_l2) },
      { key: 'coef_max_abs', label: 'Max Abs Diff', render: r => String(r.coef_max_abs) },
      { key: 'bse_max_abs', label: 'BSE Max Abs Diff', render: r => String(r.bse_max_abs) },
      { key: 'coef_l2_rel', label: 'L2 Rel Error', render: r => String(r.coef_l2_rel) },
      { key: 'status', label: 'Status', render: r => String(r.status),
        style: r => `padding:2px 6px; color:${r._color}; font-weight:bold;` },
    ],
    rows, state, onToggle: onUpdate,
  });
  return panel;
}
