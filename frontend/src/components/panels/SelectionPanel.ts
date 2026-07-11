/** Selection metrics panel — precision, recall, FDP, F1, n_selected */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

export function renderSelectionPanel(runs: Run[], state: AppState, onUpdate: () => void): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const r of runs) {
    const s = r.metrics.selection;
    if (!s) continue;
    const fdp = s.fdp;
    const target = s.target_fdr;
    const fdpColor = target != null
      ? (fdp != null && fdp <= target ? '#52c41a' : '#ff4d4f')
      : '#888';
    rows.push({
      model: r.model_id, variant: r.variant ?? '-', scale: r.scale.label,
      backend: r.backend ?? r.framework,
      precision: s.precision != null ? s.precision.toFixed(4) : '-',
      recall: s.recall != null ? s.recall.toFixed(4) : '-',
      f1: s.f1 != null ? s.f1.toFixed(4) : '-',
      fdp: fdp != null ? fdp.toFixed(4) : '-',
      fdr: s.estimated_fdr != null ? s.estimated_fdr.toFixed(4) : '-',
      n_selected: s.n_selected_mean != null ? String(Math.round(s.n_selected_mean)) : '-',
      target_fdr: target != null ? target.toFixed(2) : '-',
      _fdpColor: fdpColor,
    });
  }
  if (rows.length === 0) return null;
  const panel = renderPanelTable({
    panelId: 'selection', title: 'Selection Metrics',
    columns: [
      { key: 'model', label: 'Model', render: r => String(r.model) },
      { key: 'variant', label: 'Variant', render: r => String(r.variant) },
      { key: 'scale', label: 'Scale', render: r => String(r.scale) },
      { key: 'backend', label: 'Backend', render: r => String(r.backend) },
      { key: 'precision', label: 'Precision', render: r => String(r.precision) },
      { key: 'recall', label: 'Recall', render: r => String(r.recall) },
      { key: 'f1', label: 'F1', render: r => String(r.f1) },
      { key: 'fdp', label: 'FDP', render: r => String(r.fdp),
        style: r => `padding:2px 6px; color:${r._fdpColor}; font-weight:bold;` },
      { key: 'fdr', label: 'FDR', render: r => String(r.fdr) },
      { key: 'n_selected', label: 'N Selected', render: r => String(r.n_selected) },
      { key: 'target_fdr', label: 'Target FDR', render: r => String(r.target_fdr) },
    ],
    rows, state,
  });
  const toggleBtn = panel.querySelector('div') as HTMLElement | null;
  if (toggleBtn) (toggleBtn as any)._onToggle = onUpdate;
  const buttons = panel.querySelectorAll('button');
  for (const btn of buttons) (btn as any)._onToggle = onUpdate;
  return panel;
}
