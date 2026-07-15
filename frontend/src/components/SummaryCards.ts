import type { BenchmarkData, ParseReport, Run } from '../schema';
import { h } from '../utils/dom';

/** Global dataset-level statistics — do NOT change with filters. */
export function renderSummaryCards(
  data: BenchmarkData,
  parseReport: ParseReport | null,
  runs: Run[],
): HTMLElement {
  const row = h('div', { class: 'summary-cards' });

  // Total runs
  row.appendChild(summaryCard(String(data.runs.length), 'Total runs'));

  // Parsed files
  const parsed = parseReport
    ? `${parseReport.files_parsed}/${parseReport.files_seen}`
    : '-';
  row.appendChild(summaryCard(parsed, 'Parsed files'));

  // Model categories
  row.appendChild(
    summaryCard(String(data.categories.length), 'Model categories'),
  );

  // Keep computed and runner-reported maxima separate. They can use different
  // reference frameworks and are therefore not interchangeable statistics.
  let fastestComputedVal = -Infinity;
  let fastestReportedVal = -Infinity;
  for (const r of runs) {
    if (
      r.framework !== 'statgpu' ||
      r.backend === 'numpy' ||
      !r.metrics.speedup
    ) {
      continue;
    }

    const value = r.metrics.speedup.value ?? 0;
    if (r.metrics.speedup.reported_semantics === 'computed') {
      if (value > fastestComputedVal) fastestComputedVal = value;
    } else if (value > fastestReportedVal) {
      fastestReportedVal = value;
    }
  }

  const hasComputed = fastestComputedVal > -Infinity;
  const hasReported = fastestReportedVal > -Infinity;
  let gpuValue = '-';
  let gpuLabel = 'Fastest GPU speedup';
  let gpuTitle = 'No GPU speedup data is available.';

  if (hasComputed && hasReported) {
    gpuValue = `${fastestComputedVal.toFixed(1)}× / ${fastestReportedVal.toFixed(1)}× Ⓡ`;
    gpuLabel = 'Fastest GPU speedup · computed / reported';
    gpuTitle =
      'Computed speedups are recalculated from matched timing runs. Ⓡ values are copied from benchmark-runner reports and may use a different reference.';
  } else if (hasComputed) {
    gpuValue = `${fastestComputedVal.toFixed(1)}×`;
    gpuLabel = 'Fastest computed GPU speedup';
    gpuTitle = 'Recalculated from matched reference and GPU timing runs.';
  } else if (hasReported) {
    gpuValue = `${fastestReportedVal.toFixed(1)}× Ⓡ`;
    gpuLabel = 'Fastest reported GPU speedup';
    gpuTitle = 'Copied from benchmark-runner output; not recomputed by the dashboard.';
  }

  row.appendChild(summaryCard(gpuValue, gpuLabel, gpuTitle));

  // External frameworks available
  const extFrameworks = new Set<string>();
  for (const r of runs) {
    if (r.framework !== 'statgpu') extFrameworks.add(r.framework);
  }
  row.appendChild(
    summaryCard(
      extFrameworks.size > 0 ? [...extFrameworks].join(', ') : 'None',
      'External frameworks',
    ),
  );

  // Latest generated timestamp
  const ts = data.generated;
  const tsDisplay =
    ts && !ts.startsWith('1970')
      ? new Date(ts).toLocaleDateString()
      : 'Deterministic build';
  row.appendChild(summaryCard(tsDisplay, 'Generated'));

  return row;
}

function summaryCard(value: string, label: string, title?: string): HTMLElement {
  const attrs: Record<string, string> = { class: 'summary-card' };
  if (title) attrs.title = title;
  const card = h('div', attrs);
  const v = h('div', { class: 'card-value' }, value);
  const l = h('div', { class: 'card-label' }, label);
  card.appendChild(v);
  card.appendChild(l);
  return card;
}
