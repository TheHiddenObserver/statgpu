/** Convergence stats panel — n_iter mean±std, converged rate */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

export function renderConvergencePanel(runs: Run[], state: AppState, onUpdate: () => void): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const r of runs) {
    const c = r.metrics.convergence;
    if (!c || (c.n_iter_mean == null && c.converged_rate == null)) continue;
    rows.push({
      model: r.model_id, scale: r.scale.label, backend: r.backend ?? r.framework,
      n_iter: c.n_iter_mean != null
        ? `${c.n_iter_mean.toFixed(1)}${c.n_iter_std != null ? ' ± ' + c.n_iter_std.toFixed(1) : ''}` : '-',
      converged: c.converged_rate != null ? `${(c.converged_rate * 100).toFixed(0)}%` : '-',
    });
  }
  if (rows.length === 0) return null;
  const panel = renderPanelTable({
    panelId: 'convergence', title: 'Convergence Stats',
    columns: [
      { key: 'model', label: 'Model', render: r => String(r.model) },
      { key: 'scale', label: 'Scale', render: r => String(r.scale) },
      { key: 'backend', label: 'Backend', render: r => String(r.backend) },
      { key: 'n_iter', label: 'N Iter (mean±std)', render: r => String(r.n_iter) },
      { key: 'converged', label: 'Converged Rate', render: r => String(r.converged) },
    ],
    rows, state,
  });
  const toggleBtn = panel.querySelector('div') as HTMLElement | null;
  if (toggleBtn) (toggleBtn as any)._onToggle = onUpdate;
  const buttons = panel.querySelectorAll('button');
  for (const btn of buttons) (btn as any)._onToggle = onUpdate;
  return panel;
}
