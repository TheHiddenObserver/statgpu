import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { formatModelName } from '../utils/format';

function formatSeries(run: Run): string {
  if (run.framework !== 'statgpu') return run.framework;
  return [run.backend, run.implementation].filter(Boolean).join('/') || 'statgpu';
}

function formatRunLabel(run: Run): string {
  const variant = run.variant ? ` (${run.variant})` : '';
  const penalty = run.penalty ? ` + ${run.penalty}` : '';
  const solver = run.solver_display ?? run.solver ?? 'unknown';
  const reported = run.metrics.speedup?.reported_semantics === 'computed' ? '' : ' Ⓡ';
  return `${formatModelName(run.model_id)}${variant}${penalty} [${solver}] · ${formatSeries(run)} · ${run.scale.label}${reported}`;
}

export function renderSpeedupChart(
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

  const speedupRuns = runs.filter(
    (r) => r.metrics.speedup,
  );
  if (speedupRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: {
        text: 'No speedup data',
        left: 'center',
        top: 'center',
        textStyle: { color: '#999', fontSize: 14 },
      },
    });
    return;
  }

  speedupRuns.sort(
    (a, b) =>
      (b.metrics.speedup?.value ?? 0) - (a.metrics.speedup?.value ?? 0),
  );
  const limit = state.speedupChartLimit > 0 ? state.speedupChartLimit : 30;
  const topN = speedupRuns.slice(0, limit);
  const reportedCount = topN.filter(
    (r) => r.metrics.speedup?.reported_semantics === 'reported_by_runner',
  ).length;

  const labels = topN.map(formatRunLabel);

  chart.setOption(
    {
      title: {
        text: 'Speedup vs Reference',
        subtext:
          reportedCount > 0
            ? 'Ⓡ = reported speedup'
            : '',
        left: 'center',
        textStyle: { fontSize: 13 },
        subtextStyle: { fontSize: 10, color: '#999' },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (
          params: { value: number; color: string; name: string; data: { refLabel?: string; semantics?: string } }[],
        ) => {
          const p = params[0];
          if (!p || p.value == null) return 'No data';
          const label = p.value > 1 ? 'faster' : p.value < 1 ? 'slower' : 'same';
          const reference = p.data?.refLabel ?? 'reference';
          const semantics = p.data?.semantics === 'reported_by_runner' ? 'reported' : 'computed';
          return `<b>${p.value.toFixed(2)}×</b> ${semantics} vs ${reference} (${label})`;
        },
      },
      grid: { left: 10, right: 10, top: reportedCount > 0 ? 50 : 40, bottom: 30, containLabel: true },
      xAxis: {
        type: 'value',
        name: 'speedup',
        axisLabel: { fontSize: 10 },
      },
      yAxis: {
        type: 'category',
        data: labels.reverse(),
        axisLabel: { fontSize: 10 },
      },
      series: [
        {
          type: 'bar',
          data: topN.reverse().map((r) => {
            const sp = r.metrics.speedup!;
            const isReported = sp.reported_semantics === 'reported_by_runner';
            const val = sp.value;
            const refLabel = [sp.reference_framework, sp.reference_backend].filter(Boolean).join('/');
            return {
              value: val,
              refLabel,
              semantics: sp.reported_semantics,
              itemStyle: {
                color: val >= 1 ? '#52c41a' : '#ff4d4f',
                ...(isReported
                  ? {
                      decal: {
                        symbol: 'triangle' as const,
                        symbolSize: 0.6,
                        color: 'rgba(0,0,0,0.12)',
                        dashArrayX: [6, 0],
                      },
                    }
                  : {}),
              },
            };
          }),
          markLine: {
            silent: true,
            data: [
              {
                xAxis: 1,
                lineStyle: { color: '#999', type: 'dashed' },
              },
            ],
            label: { formatter: '1.0x' },
          },
        },
      ],
    },
    true,
  );
}
