/** Prediction metrics panel — C-index, MSE, alpha */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

export function renderPredictionPanel(runs: Run[], state: AppState, onUpdate: () => void): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const r of runs) {
    const p = r.metrics.prediction;
    if (!p) continue;
    rows.push({
      model: r.model_id, framework: r.framework,
      scale: r.scale.label,
      c_index: p.c_index != null ? p.c_index.toFixed(4) : '-',
      train_mse: p.train_mse != null ? p.train_mse.toFixed(4) : '-',
      test_mse: p.test_mse != null ? p.test_mse.toFixed(4) : '-',
      alpha: p.alpha_mean != null ? p.alpha_mean.toFixed(4) : '-',
    });
  }
  if (rows.length === 0) return null;
  const panel = renderPanelTable({
    panelId: 'prediction', title: 'Prediction Metrics',
    columns: [
      { key: 'model', label: 'Model', render: r => String(r.model) },
      { key: 'framework', label: 'Framework', render: r => String(r.framework) },
      { key: 'scale', label: 'Scale', render: r => String(r.scale) },
      { key: 'c_index', label: 'C-index', render: r => String(r.c_index) },
      { key: 'train_mse', label: 'Train MSE', render: r => String(r.train_mse) },
      { key: 'test_mse', label: 'Test MSE', render: r => String(r.test_mse) },
      { key: 'alpha', label: 'Alpha', render: r => String(r.alpha) },
    ],
    rows, state, onToggle: onUpdate,
  });
  return panel;
}
