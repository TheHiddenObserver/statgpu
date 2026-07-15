import type { BenchmarkData, ParseReport, Run } from '../schema';
import { h } from '../utils/dom';

/** Global dataset-level statistics — these do not change with filters. */
export function renderSummaryCards(
  data: BenchmarkData,
  parseReport: ParseReport | null,
  runs: Run[],
): HTMLElement {
  const row = h('div', { class: 'summary-cards' });

  row.appendChild(
    summaryCard(
      String(data.runs.length),
      'Benchmark runs',
      'Number of normalized run records in the canonical dashboard bundle.',
    ),
  );

  const parsed = parseReport
    ? `${parseReport.files_parsed}/${parseReport.files_seen}`
    : '-';
  row.appendChild(
    summaryCard(
      parsed,
      'Sources parsed',
      'Canonical source files parsed successfully out of all registered source files.',
    ),
  );

  row.appendChild(
    summaryCard(
      String(data.categories.length),
      'Benchmark categories',
      'Number of statistical benchmark categories defined by Schema v1.1.',
    ),
  );

  // The headline follows the benchmark runner's published speedup. Computed
  // speedups remain available in charts and run-level records for auditing.
  let fastestReported: Run | null = null;
  for (const run of runs) {
    const speedup = run.metrics.speedup;
    if (
      run.framework !== 'statgpu' ||
      run.backend === 'numpy' ||
      !speedup ||
      speedup.reported_semantics !== 'reported_by_runner'
    ) {
      continue;
    }
    if (!fastestReported || speedup.value > fastestReported.metrics.speedup!.value) {
      fastestReported = run;
    }
  }

  const reportedSpeedup = fastestReported?.metrics.speedup;
  const gpuValue = reportedSpeedup ? `${reportedSpeedup.value.toFixed(1)}× Ⓡ` : '-';
  const gpuTitle = reportedSpeedup
    ? `Largest runner-reported GPU speedup. Reference: ${[
        reportedSpeedup.reference_framework,
        reportedSpeedup.reference_backend,
      ]
        .filter(Boolean)
        .join('/') || 'benchmark runner reference'}. Computed timing ratios remain available in the charts and raw data.`
    : 'No runner-reported GPU speedup is available.';
  row.appendChild(
    summaryCard(gpuValue, 'Fastest reported GPU speedup', gpuTitle),
  );

  const extFrameworks = [...new Set(
    runs.filter((run) => run.framework !== 'statgpu').map((run) => run.framework),
  )].sort();
  row.appendChild(
    summaryCard(
      extFrameworks.length > 0 ? `${extFrameworks.length} frameworks` : 'None',
      'External references',
      extFrameworks.length > 0
        ? `Available external reference frameworks: ${extFrameworks.join(', ')}.`
        : 'No external reference framework is present in the canonical bundle.',
    ),
  );

  const deterministic = !data.generated || data.generated.startsWith('1970');
  row.appendChild(
    summaryCard(
      deterministic ? 'Deterministic' : new Date(data.generated).toLocaleDateString(),
      deterministic ? 'Build mode' : 'Generated',
      deterministic
        ? 'The committed bundle uses stable ordering and deterministic metadata so CI can detect stale generated assets.'
        : 'Date on which the current benchmark bundle was generated.',
    ),
  );

  return row;
}

function summaryCard(value: string, label: string, title: string): HTMLElement {
  const card = h('div', {
    class: 'summary-card',
    title,
    'aria-label': `${label}: ${value}. ${title}`,
  });
  const v = h('div', { class: 'card-value' }, value);
  const l = h('div', { class: 'card-label' }, label);
  card.appendChild(v);
  card.appendChild(l);
  return card;
}
