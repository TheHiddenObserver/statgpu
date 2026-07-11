import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { COLORS } from '../utils/theme';
import { formatModelName } from '../utils/format';
import { emptyChartMessage } from '../components/EmptyState';

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

  // Group by comparison key: model + penalty + solver + scale
  type GroupKey = string;
  const groups = new Map<
    GroupKey,
    { label: string; byBackend: Map<string, number> }
  >();
  for (const r of timingRuns) {
    const gk = `${r.model_id}|${r.variant ?? ''}|${r.method_config_id}|${r.penalty ?? 'none'}|${r.solver ?? 'auto'}|${r.scale.scale_key}`;
    if (!groups.has(gk)) {
      const variantSuffix = r.variant ? ` (${r.variant})` : '';
      groups.set(gk, {
        label: `${formatModelName(r.model_id)}${variantSuffix}+${r.penalty ?? 'none'} ${r.scale.label}`,
        byBackend: new Map(),
      });
    }
    const implSuffix = r.implementation ? `/${r.implementation}` : '';
    const be =
      r.framework === 'statgpu' ? `${r.backend ?? 'ext'}${implSuffix}` : r.framework;
    groups.get(gk)!.byBackend.set(be, r.metrics.timing!.fit_time_ms);
  }

  const sortedGroups = [...groups.entries()].sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const categories = sortedGroups.map(([, g]) => g.label);

  const allBackends = new Set<string>();
  for (const [, g] of sortedGroups) {
    for (const bk of g.byBackend.keys()) allBackends.add(bk);
  }
  const backendOrder = ['numpy', 'cupy', 'torch', ...state.showExternal].filter(
    (b) => allBackends.has(b),
  );

  const series = backendOrder.map((bk) => ({
    name: bk,
    type: 'bar' as const,
    data: sortedGroups.map(([, g]) => g.byBackend.get(bk) ?? null),
    itemStyle: { color: COLORS[bk] || '#999' },
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
