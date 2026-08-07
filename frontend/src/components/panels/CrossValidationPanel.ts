/** Cross-validation timing, selection, score, and failure diagnostics. */

import type { Run } from '../../schema';
import type { AppState } from '../../state';
import { renderPanelTable } from './PanelTable';

function selectedLabel(value: Record<string, unknown>): string {
  return Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, selected]) => `${key}=${String(selected)}`)
    .join(', ');
}

export function renderCrossValidationPanel(
  runs: Run[],
  state: AppState,
  onUpdate: () => void,
): HTMLElement | null {
  const rows: Record<string, unknown>[] = [];
  for (const run of runs) {
    const cv = run.metrics.cross_validation;
    if (!cv) continue;
    rows.push({
      model: run.model_id,
      framework: run.framework,
      backend: run.backend ?? run.framework,
      scale: run.scale.label,
      cv_time: cv.cv_evaluation_ms.toFixed(3),
      refit_time: cv.final_refit_ms.toFixed(3),
      total_time: cv.total_fit_ms.toFixed(3),
      selected: selectedLabel(cv.selected_parameters),
      validation_score: cv.validation_score.toFixed(6),
      final_score: cv.final_score.toFixed(6),
      scoring: `${cv.scoring_name} (${cv.scoring_direction})`,
      failures: `${cv.failed_candidates} candidates / ${cv.failed_folds} folds`,
      refit: cv.final_refit_converged ? 'yes' : 'no',
    });
  }
  if (rows.length === 0) return null;
  return renderPanelTable({
    panelId: 'cross-validation',
    title: 'Cross-validation Metrics',
    columns: [
      { key: 'model', label: 'Model', render: row => String(row.model) },
      { key: 'framework', label: 'Framework', render: row => String(row.framework) },
      { key: 'backend', label: 'Backend', render: row => String(row.backend) },
      { key: 'scale', label: 'Scale', render: row => String(row.scale) },
      { key: 'cv_time', label: 'CV evaluation (ms)', render: row => String(row.cv_time) },
      { key: 'refit_time', label: 'Final refit (ms)', render: row => String(row.refit_time) },
      { key: 'total_time', label: 'Total fit (ms)', render: row => String(row.total_time) },
      { key: 'selected', label: 'Selected parameters', render: row => String(row.selected) },
      { key: 'validation_score', label: 'Validation score', render: row => String(row.validation_score) },
      { key: 'final_score', label: 'Final score', render: row => String(row.final_score) },
      { key: 'scoring', label: 'Scoring', render: row => String(row.scoring) },
      { key: 'failures', label: 'Failures', render: row => String(row.failures) },
      { key: 'refit', label: 'Refit converged', render: row => String(row.refit) },
    ],
    rows,
    state,
    onToggle: onUpdate,
  });
}
