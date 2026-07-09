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
  row.appendChild(
    summaryCard(String(data.runs.length), 'Total runs'),
  );

  // Parsed files
  const parsed = parseReport
    ? `${parseReport.files_parsed}/${parseReport.files_seen}`
    : '-';
  row.appendChild(summaryCard(parsed, 'Parsed files'));

  // Model categories
  row.appendChild(
    summaryCard(String(data.categories.length), 'Model categories'),
  );

  // Fastest GPU speedup (prefer computed over reported)
  const speedupRuns = runs
    .filter(
      (r) =>
        r.framework === 'statgpu' &&
        r.backend !== 'numpy' &&
        r.metrics.speedup,
    )
    .sort(
      (a, b) =>
        (b.metrics.speedup!.value ?? 0) -
        (a.metrics.speedup!.value ?? 0),
    );

  const fastestComputed = speedupRuns.find(
    (r) => r.metrics.speedup!.reported_semantics === 'computed',
  );
  const fastestAny = speedupRuns[0];

  const gpuLabel = fastestComputed
    ? 'Fastest computed GPU speedup'
    : 'Fastest GPU speedup';
  const gpuValue = (fastestComputed ?? fastestAny)?.metrics.speedup?.value;
  row.appendChild(
    summaryCard(
      gpuValue != null ? `${gpuValue.toFixed(1)}×` : '-',
      gpuLabel,
    ),
  );

  // External frameworks available
  const extFrameworks = new Set<string>();
  for (const r of runs) {
    if (r.framework !== 'statgpu') extFrameworks.add(r.framework);
  }
  row.appendChild(
    summaryCard(
      extFrameworks.size > 0
        ? [...extFrameworks].join(', ')
        : 'None',
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

function summaryCard(value: string, label: string): HTMLElement {
  const card = h('div', { class: 'summary-card' });
  const v = h('div', { class: 'card-value' }, value);
  const l = h('div', { class: 'card-label' }, label);
  card.appendChild(v);
  card.appendChild(l);
  return card;
}
