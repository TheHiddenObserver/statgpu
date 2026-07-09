import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';

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
    (r) => r.metrics.speedup && r.backend !== 'numpy',
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
    (a, b) => (b.metrics.speedup?.value ?? 0) - (a.metrics.speedup?.value ?? 0),
  );
  const topN = speedupRuns.slice(0, 30);
  const labels = topN.map(
    (r) =>
      `${r.model_id.replace('Penalized', '').replace('Regression', '')}+${r.penalty} [${r.solver_display ?? r.solver}] ${r.scale.label}`,
  );

  chart.setOption(
    {
      title: {
        text: 'GPU Speedup vs CPU',
        left: 'center',
        textStyle: { fontSize: 13 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (
          params: { value: number; color: string }[],
        ) => {
          const v = params[0]?.value;
          if (v == null) return 'No data';
          const label = v > 1 ? 'faster' : v < 1 ? 'slower' : 'same';
          return `<b>${v.toFixed(2)}x</b> (${label})`;
        },
      },
      grid: { left: 10, right: 10, top: 40, bottom: 30 },
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
          data: topN.reverse().map((r) => ({
            value: r.metrics.speedup!.value,
            itemStyle: {
              color:
                r.metrics.speedup!.value >= 1 ? '#52c41a' : '#ff4d4f',
            },
          })),
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
