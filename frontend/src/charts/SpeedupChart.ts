import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { formatModelName } from '../utils/format';
import { CHART_STYLE } from '../utils/theme';

function formatSeries(run: Run): string {
  if (run.framework !== 'statgpu') return run.framework;
  return [run.backend, run.implementation].filter(Boolean).join('/') || 'statgpu';
}

function formatRunLabel(run: Run): string {
  const variant = run.variant ? ` (${run.variant})` : '';
  const penalty = run.penalty && run.penalty !== 'none' ? ` + ${run.penalty}` : '';
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

  el.dataset.parityStyle = 'dashed';
  el.dataset.parityLabelPlacement = 'axis-bottom';
  el.setAttribute(
    'aria-label',
    'Speedup vs Reference chart — dashed 1× parity line labeled near the horizontal axis; values to the right are faster',
  );

  const speedupRuns = runs.filter((run) => run.metrics.speedup);
  if (speedupRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: {
        text: 'No speedup data',
        left: 'center',
        top: 'center',
        textStyle: { color: CHART_STYLE.muted, fontSize: 14 },
      },
    });
    return;
  }

  speedupRuns.sort(
    (a, b) =>
      (b.metrics.speedup?.value ?? 0) - (a.metrics.speedup?.value ?? 0),
  );
  const limit = state.chartViewMode === 'focused' ? 18 : state.speedupChartLimit;
  const topN = speedupRuns.slice(0, limit);
  const reportedCount = topN.filter(
    (run) => run.metrics.speedup?.reported_semantics === 'reported_by_runner',
  ).length;

  const subtitleParts = ['dashed line = 1× parity'];
  if (reportedCount > 0) subtitleParts.unshift('Ⓡ = runner-reported');
  if (speedupRuns.length > topN.length) {
    subtitleParts.push(`showing top ${topN.length}/${speedupRuns.length}`);
  }
  const labels = topN.map(formatRunLabel);

  chart.setOption(
    {
      title: {
        text: 'Speedup vs Reference',
        subtext: subtitleParts.join(' · '),
        left: 'center',
        textStyle: { fontSize: 13, color: CHART_STYLE.text },
        subtextStyle: { fontSize: 10, color: CHART_STYLE.muted },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: CHART_STYLE.tooltipBackground,
        borderWidth: 0,
        textStyle: { color: '#fff' },
        formatter: (
          params: {
            value: number;
            color: string;
            name: string;
            data: { refLabel?: string; semantics?: string };
          }[],
        ) => {
          const param = params[0];
          if (!param || param.value == null) return 'No data';
          const label = param.value > 1 ? 'faster' : param.value < 1 ? 'slower' : 'same';
          const reference = param.data?.refLabel ?? 'reference';
          const semantics =
            param.data?.semantics === 'reported_by_runner' ? 'runner-reported' : 'computed';
          return `<b>${param.value.toFixed(2)}×</b> ${semantics} vs ${reference} (${label})`;
        },
      },
      grid: {
        left: 12,
        right: 20,
        top: 66,
        bottom: 46,
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        min: 0,
        axisLine: { lineStyle: { color: CHART_STYLE.axis } },
        axisTick: { lineStyle: { color: CHART_STYLE.axis } },
        axisLabel: {
          fontSize: 10,
          color: CHART_STYLE.text,
          formatter: (value: number) => (value === 0 ? '' : `${value}×`),
        },
        splitLine: { lineStyle: { color: CHART_STYLE.grid } },
      },
      yAxis: {
        type: 'category',
        data: labels.reverse(),
        axisLine: { lineStyle: { color: CHART_STYLE.axis } },
        axisTick: { lineStyle: { color: CHART_STYLE.axis } },
        axisLabel: {
          fontSize: 10,
          color: CHART_STYLE.text,
          width: 285,
          overflow: 'truncate',
        },
      },
      series: [
        {
          type: 'bar',
          barMaxWidth: 24,
          data: topN.reverse().map((run) => {
            const speedup = run.metrics.speedup!;
            const isReported = speedup.reported_semantics === 'reported_by_runner';
            const value = speedup.value;
            const isFaster = value >= 1;
            const refLabel = [speedup.reference_framework, speedup.reference_backend]
              .filter(Boolean)
              .join('/');
            return {
              value,
              refLabel,
              semantics: speedup.reported_semantics,
              itemStyle: {
                color: isFaster
                  ? isReported
                    ? CHART_STYLE.speedupReported
                    : CHART_STYLE.speedupComputed
                  : isReported
                    ? CHART_STYLE.slowdownReported
                    : CHART_STYLE.slowdownComputed,
                borderColor: isReported
                  ? isFaster
                    ? '#4f8763'
                    : '#ae5f64'
                  : 'transparent',
                borderWidth: isReported ? 1 : 0,
                borderRadius: [0, 4, 4, 0],
                opacity: isReported ? 0.92 : 1,
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
                  color: CHART_STYLE.parity,
                  type: 'dashed',
                  width: 1.4,
                  opacity: 0.9,
                },
                label: {
                  show: true,
                  formatter: '1×',
                  position: 'insideStartTop',
                  distance: 9,
                  offset: [10, 0],
                  color: CHART_STYLE.parity,
                  fontSize: 10,
                  fontWeight: 600,
                  backgroundColor: 'rgba(255, 255, 255, 0.97)',
                  borderColor: 'rgba(122, 132, 151, 0.35)',
                  borderWidth: 1,
                  padding: [2, 5],
                  borderRadius: 4,
                },
              },
            ],
          },
        },
      ],
    },
    true,
  );
}
