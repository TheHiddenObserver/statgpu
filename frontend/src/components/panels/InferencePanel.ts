/** Inference metrics panel — standard errors, Wald statistics, and p-values. */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

function formatNumber(value: number | undefined): string {
  if (value == null) return '-';
  if (value === 0) return '0';
  const abs = Math.abs(value);
  return abs >= 1e4 || abs < 1e-3 ? value.toExponential(3) : value.toFixed(5);
}

export function renderInferencePanel(
  runs: Run[],
  state: AppState,
  onUpdate: () => void,
): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const run of runs) {
    const inference = run.metrics.inference;
    if (!inference) continue;
    const ok = inference.ok;
    rows.push({
      model: run.model_id,
      backend: run.implementation
        ? `${run.backend ?? run.framework} (${run.implementation})`
        : (run.backend ?? run.framework),
      scale: run.scale.label,
      bse: formatNumber(inference.bse),
      wald: formatNumber(inference.wald_stat),
      pValue: formatNumber(inference.p_value),
      status: ok == null ? 'N/A' : ok ? 'PASS' : 'FAIL',
      source: inference.source_file,
      _color: ok == null ? '#888' : ok ? '#52c41a' : '#ff4d4f',
    });
  }
  if (rows.length === 0) return null;

  return renderPanelTable({
    panelId: 'inference',
    title: 'Inference Metrics',
    columns: [
      { key: 'model', label: 'Model', render: row => String(row.model) },
      { key: 'backend', label: 'Backend', render: row => String(row.backend) },
      { key: 'scale', label: 'Scale', render: row => String(row.scale) },
      { key: 'bse', label: 'BSE', render: row => String(row.bse) },
      { key: 'wald', label: 'Wald Statistic', render: row => String(row.wald) },
      { key: 'pValue', label: 'p-value', render: row => String(row.pValue) },
      {
        key: 'status',
        label: 'Status',
        render: row => String(row.status),
        style: row => `padding:2px 6px; color:${row._color}; font-weight:bold;`,
      },
      { key: 'source', label: 'Source', render: row => String(row.source) },
    ],
    rows,
    state,
    onToggle: onUpdate,
  });
}
