/** Accessible tabular alternative for ECharts content. */

import type { Run } from '../schema';
import type { AppState } from '../state';
import { selectTimingRuns } from '../charts/TimingChart';
import { selectSpeedupRuns } from '../charts/SpeedupChart';
import { h } from '../utils/dom';
import { formatModelName } from '../utils/format';

function identityCells(run: Run): string[] {
  return [
    formatModelName(run.model_id),
    run.variant ?? '—',
    run.penalty ?? '—',
    run.solver_display ?? run.solver ?? '—',
    run.framework === 'statgpu'
      ? [run.backend, run.implementation].filter(Boolean).join('/') || 'statgpu'
      : run.framework,
    run.scale.label,
  ];
}

interface TableValue {
  value: string;
  reference: string;
}

function renderTable(
  id: string,
  caption: string,
  valueLabel: string,
  runs: Run[],
  valueForRun: (run: Run) => TableValue,
): HTMLElement {
  const details = h('details', { class: 'chart-data-details', id }) as HTMLDetailsElement;
  details.appendChild(h('summary', {}, `${caption} (${runs.length} rows)`));

  let materialized = false;
  const materialize = (): void => {
    if (materialized) return;
    materialized = true;

    const wrapper = h('div', { class: 'chart-data-table-wrap' });
    const table = h('table', { class: 'chart-data-table' });
    table.appendChild(
      h(
        'caption',
        {},
        `${caption}. Full labels are shown here when chart labels are truncated.`,
      ),
    );

    const header = h('tr');
    for (const label of [
      'Model',
      'Variant',
      'Penalty',
      'Solver',
      'Backend / reference',
      'Scale',
      valueLabel,
      'Reference',
    ]) {
      header.appendChild(h('th', { scope: 'col' }, label));
    }
    const thead = h('thead');
    thead.appendChild(header);
    table.appendChild(thead);

    const tbody = h('tbody');
    for (const run of runs) {
      const { value, reference } = valueForRun(run);
      const tr = h('tr');
      for (const cell of [...identityCells(run), value, reference]) {
        tr.appendChild(h('td', {}, cell));
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrapper.appendChild(table);
    details.appendChild(wrapper);
  };

  details.addEventListener('toggle', () => {
    if (details.open) materialize();
  });
  return details;
}

export function renderChartDataFallback(
  timingSourceRuns: Run[],
  speedupSourceRuns: Run[],
  state: AppState,
): HTMLElement {
  const section = h('section', {
    class: 'chart-data-fallback',
    'aria-label': 'Accessible chart data',
  });
  section.appendChild(h('h2', { class: 'chart-data-heading' }, 'Chart data tables'));
  section.appendChild(
    h(
      'p',
      { class: 'chart-data-help' },
      'Use these tables for exact values and full labels; they follow the current filters and chart view.',
    ),
  );

  const timingRuns = selectTimingRuns(timingSourceRuns, state).runs;
  const speedupRuns = selectSpeedupRuns(speedupSourceRuns, state).runs;

  section.appendChild(
    renderTable(
      'timing-chart-data',
      'Fit Time chart data',
      'Time',
      timingRuns,
      run => ({
        value: `${run.metrics.timing!.fit_time_ms.toFixed(3)} ms`,
        reference: run.metrics.timing!.quality,
      }),
    ),
  );
  section.appendChild(
    renderTable(
      'speedup-chart-data',
      'Speedup chart data',
      'Speedup',
      speedupRuns,
      run => ({
        value: `${run.metrics.speedup!.value.toFixed(3)}×`,
        reference: `${run.metrics.speedup!.reference_framework} (${run.metrics.speedup!.reported_semantics})`,
      }),
    ),
  );
  return section;
}
