import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { formatModelName } from '../utils/format';

export function renderSpeedupChart(
  el: HTMLElement,
  runs: Run[],
  _state: AppState,
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
  const topN = speedupRuns.slice(0, 30);
  const reportedCount = topN.filter(
    (r) => r.metrics.speedup?.reported_semantics === 'reported_by_runner',
  ).length;

  const labels = topN.map((r) => {
    const isComputed =
      r.metrics.speedup?.reported_semantics === 'computed';
    const suffix = isComputed ? '' : ' Ⓡ';
    return `${formatModelName(r.model_id)}+${r.penalty} [${r.solver_display ?? r.solver}] ${r.scale.label}${suffix}`;
  });

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
          params: { value: number; color: string; name: string }[],
        ) => {
          const p = params[0];
          if (!p || p.value == null) return 'No data';
          const label =
            p.value > 1 ? 'faster' : p.value < 1 ? 'slower' : 'same';
          const isReported = p.name.includes('Ⓡ');
          const kind = isReported
            ? 'reported by solver benchmark'
            : 'computed vs NumPy';
          return `<b>${p.value.toFixed(2)}×</b> ${kind} (${label})`;
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
            const isReported =
              r.metrics.speedup?.reported_semantics ===
              'reported_by_runner';
            const val = r.metrics.speedup!.value;
            return {
              value: val,
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
