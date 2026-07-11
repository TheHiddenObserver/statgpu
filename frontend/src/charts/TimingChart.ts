import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { COLORS } from '../utils/theme';
import { formatModelName } from '../utils/format';
import { emptyChartMessage } from '../components/EmptyState';
import { chartGroupIdentity, chartSeriesIdentity } from '../identity';

export function renderTimingChart(
  el: HTMLElement,
  runs: Run[],
  state: AppState,
  chartInstances: echarts.ECharts[],
): void {
  let chart = echarts.getInstanceByDom(el);
  if (!chart) {
    chart = echarts.init(el);
    chartInstances.push(chart);
  }

  const timingRuns = runs.filter((r) => r.metrics.timing);
  if (timingRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: {
        text: 'No timing data',
        left: 'center',
        top: 'center',
        textStyle: { color: '#999', fontSize: 14 },
      },
    });
    return;
  }

  // Build timing series metadata: canonical key for dedup, display label for UI
  interface TimingSeries {
    key: string;
    label: string;
    backend: string | null;
    framework: string;
    impl: string | null;
  }
  function makeSeries(r: Run): TimingSeries {
    const impl = r.implementation ?? null;
    const identity = [r.framework, r.backend ?? null, impl];
    return {
      key: JSON.stringify(identity),
      label: r.framework === 'statgpu'
        ? [r.backend, impl].filter(Boolean).join('/') || (r.backend ?? 'ext')
        : r.framework,
      backend: r.backend,
      framework: r.framework,
      impl,
    };
  }

  type GroupKey = string;
  const groups = new Map<GroupKey, { label: string; bySeries: Map<string, number> }>();
  const seriesMeta = new Map<string, TimingSeries>();

  for (const r of timingRuns) {
    const gk = JSON.stringify(chartGroupIdentity(r, false));
    if (!groups.has(gk)) {
      const variantSuffix = r.variant ? ` (${r.variant})` : '';
      groups.set(gk, {
        label: `${formatModelName(r.model_id)}${variantSuffix}+${r.penalty ?? 'none'} ${r.scale.label}`,
        bySeries: new Map(),
      });
    }
    const s = makeSeries(r);
    seriesMeta.set(s.key, s);
    groups.get(gk)!.bySeries.set(s.key, r.metrics.timing!.fit_time_ms);
  }

  const sortedGroups = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  const categories = sortedGroups.map(([, g]) => g.label);

  // Preferred series order uses display labels for matching
  const allKeys = new Set<string>();
  for (const [, g] of sortedGroups) {
    for (const sk of g.bySeries.keys()) allKeys.add(sk);
  }
  const preferredLabels = ['numpy', 'numpy/numba', 'cupy', 'torch', ...state.showExternal];
  const seriesOrder: TimingSeries[] = [];
  for (const lbl of preferredLabels) {
    for (const [key, s] of seriesMeta) {
      if (s.label === lbl && allKeys.has(key)) {
        seriesOrder.push(s);
      }
    }
  }
  // Append any remaining series not in preferred order
  for (const [key, s] of seriesMeta) {
    if (allKeys.has(key) && !seriesOrder.some(x => x.key === key)) {
      seriesOrder.push(s);
    }
  }

  const series = seriesOrder.map((s) => ({
    name: s.label,
    type: 'bar' as const,
    data: sortedGroups.map(([, g]) => g.bySeries.get(s.key) ?? null),
    itemStyle: { color: COLORS[s.label] || COLORS[s.backend || ''] || '#999' },
  }));

  chart.setOption(
    {
      title: {
        text: 'Fit Time (ms)',
        left: 'center',
        textStyle: { fontSize: 13 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (
          params: { seriesName: string; value: number | null; color: string }[],
        ) => {
          return (
            params
              .filter((p) => p.value != null)
              .map(
                (p) =>
                  `<span style="color:${p.color}">●</span> ${p.seriesName}: <b>${p.value!.toFixed(2)}ms</b>`,
              )
              .join('<br/>') || 'No data'
          );
        },
      },
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      grid: { left: 10, right: 10, top: 40, bottom: 30, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { fontSize: 10, rotate: 45 },
      },
      yAxis: {
        type: 'log',
        name: 'ms',
        axisLabel: { fontSize: 10 },
      },
      series,
    },
    true,
  );
}
