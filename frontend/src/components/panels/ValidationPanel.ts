/** Validation metrics panel — shows per-metric checks from validation.checks[] */

import type { Run, ValidationCheck } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

const STATUS_COLORS: Record<string, string> = {
  pass: '#52c41a', warn: '#faad14', fail: '#ff4d4f',
};

export function renderValidationPanel(
  runs: Run[],
  state: AppState,
  onUpdate: () => void,
): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const r of runs) {
    const v = r.metrics.validation;
    if (!v?.checks) continue;
    for (const ch of v.checks) {
      rows.push({
        model: r.model_id,
        variant: r.variant ?? '-',
        scale: r.scale.label,
        backend: r.backend ?? r.framework,
        reference: ch.reference ?? '-',
        metric: ch.metric,
        value: ch.value != null ? ch.value.toExponential(2) : '-',
        tolerance: ch.tolerance != null ? ch.tolerance.toExponential(2) : '-',
        status: ch.status,
        _color: STATUS_COLORS[ch.status] ?? '#888',
      });
    }
  }
  if (rows.length === 0) return null;

  const panel = renderPanelTable({
    panelId: 'validation',
    title: 'Validation Checks',
    columns: [
      { key: 'model', label: 'Model', render: r => String(r.model) },
      { key: 'variant', label: 'Variant', render: r => String(r.variant) },
      { key: 'scale', label: 'Scale', render: r => String(r.scale) },
      { key: 'backend', label: 'Backend', render: r => String(r.backend) },
      { key: 'reference', label: 'Reference', render: r => String(r.reference) },
      { key: 'metric', label: 'Metric', render: r => String(r.metric) },
      { key: 'value', label: 'Value', render: r => String(r.value) },
      { key: 'tolerance', label: 'Tolerance', render: r => String(r.tolerance) },
      {
        key: 'status', label: 'Status',
        render: r => String(r.status).toUpperCase(),
        style: r => `padding:2px 6px; color:${r._color}; font-weight:bold;`,
      },
    ],
    rows,
    state,
    onToggle: onUpdate,
  });

  return panel;
}
