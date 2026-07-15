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

  const speedupRuns = runs.filter((r) => r.metrics.speedup);
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

  const subtitleParts = ['solid gray line = 1× parity'];
  if (reportedCount > 0) subtitleParts.unshift('Ⓡ = reported by benchmark runner');
  const labels = topN.map(formatRunLabel);

  chart.setOption(
    {
      title: {
        text: 'Speedup vs Reference',
        subtext: subtitleParts.join(' · '),
        left: 'center',
        textStyle: { fontSize: 13 },
        subtextStyle: { fontSize: 10, color: '#8c8c8c' },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (
          params: {
            value: number;
            color: string;
            name: string;
            data: { refLabel?: string; semantics?: string };
          }[],
        ) => {
          const p = params[0];
          if (!p || p.value == null) return 'No data';
          const label = p.value > 1 ? 'faster' : p.value < 1 ? 'slower' : 'same';
          const reference = p.data?.refLabel ?? 'reference';
          const semantics =
            p.data?.semantics === 'reported_by_runner' ? 'reported' : 'computed';
          return `<b>${p.value.toFixed(2)}×</b> ${semantics} vs ${reference} (${label})`;
        },
      },
      grid: {
        left: 10,
        right: 18,
        top: 58,
        bottom: 30,
        containLabel: true,
      },
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
          barMaxWidth: 28,
          data: topN.reverse().map((r) => {
            const sp = r.metrics.speedup!;
            const isReported = sp.reported_semantics === 'reported_by_runner';
            const val = sp.value;
            const isFaster = val >= 1;
            const refLabel = [sp.reference_framework, sp.reference_backend]
              .filter(Boolean)
              .join('/');
            return {
              value: val,
              refLabel,
              semantics: sp.reported_semantics,
              itemStyle: {
                color: isFaster
                  ? isReported
                    ? '#73d13d'
                    : '#52c41a'
                  : isReported
                    ? '#ff7875'
                    : '#ff4d4f',
                borderColor: isReported
                  ? isFaster
                    ? '#389e0d'
                    : '#cf1322'
                  : 'transparent',
                borderWidth: isReported ? 1 : 0,
                opacity: isReported ? 0.9 : 1,
              },
            };
          }),
          markLine: {
            silent: true,
            symbol: ['none', 'none'],
            data: [
              {
                xAxis: 1,
                lineStyle: {
                  color: 'rgba(89, 89, 89, 0.72)',
                  type: 'solid',
                  width: 1.5,
                },
              },
            ],
            label: { show: false },
          },
        },
      ],
    },
    true,
  );
}
