import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { formatModelName } from '../utils/format';
import { CHART_STYLE } from '../utils/theme';

interface SpeedupSelection {
  runs: Run[];
  notes: string[];
}

interface SpeedupTooltipParam {
  value: number;
  color: string;
  name: string;
  data: { refLabel?: string; semantics?: string };
}

interface TooltipSize {
  contentSize: number[];
  viewSize: number[];
}

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

function selectSpeedupRuns(runs: Run[], state: AppState): SpeedupSelection {
  const speedupRuns = runs.filter((run) => run.metrics.speedup);
  if (state.chartViewMode === 'full' || speedupRuns.length === 0) {
    return { runs: speedupRuns, notes: ['Full matrix'] };
  }

  let focused = speedupRuns;
  const notes = ['Focused'];

  // Match the timing-chart semantics: when scale is not explicitly selected,
  // choose one representative workload instead of mixing every scale.
  if (state.selectedScaleKeys.size === 0) {
    const scales = new Map<string, Run['scale']>();
    for (const run of focused) scales.set(run.scale.scale_key, run.scale);
    const representative = [...scales.values()].sort((a, b) => {
      const workloadDiff = b.n_samples * b.n_features - a.n_samples * a.n_features;
      if (workloadDiff !== 0) return workloadDiff;
      if (b.n_samples !== a.n_samples) return b.n_samples - a.n_samples;
      return b.n_features - a.n_features;
    })[0];
    if (representative) {
      focused = focused.filter(
        (run) => run.scale.scale_key === representative.scale_key,
      );
      notes.push(representative.label);
    }
  } else {
    notes.push('selected scale filter');
  }

  // Prefer dispatch/Auto(best) rows where the current domain provides them.
  // Domains without dispatch rows retain all methods at the representative scale.
  const dispatchRows = focused.filter(
    (run) => run.solver_kind === 'dispatch' || run.solver === 'auto',
  );
  if (dispatchRows.length > 0) {
    focused = dispatchRows;
    notes.push('Auto/best solver rows');
  }

  return { runs: focused, notes };
}

function placeTooltip(
  point: number[],
  _params: unknown,
  _dom: HTMLElement,
  _rect: unknown,
  size: TooltipSize,
): [number, number] {
  const margin = 14;
  const titleBand = 74;
  const [contentWidth, contentHeight] = size.contentSize;
  const [viewWidth, viewHeight] = size.viewSize;

  // Keep the tooltip out of the long y-axis label area. It is docked to the
  // right edge and placed in the vertical half opposite the hovered bar.
  const x = Math.max(margin, viewWidth - contentWidth - margin);
  const topY = titleBand;
  const bottomY = Math.max(titleBand, viewHeight - contentHeight - margin);
  const y = point[1] < viewHeight / 2 ? bottomY : topY;
  return [x, y];
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

  const selection = selectSpeedupRuns(runs, state);
  const selectedRuns = [...selection.runs].sort(
    (a, b) =>
      (b.metrics.speedup?.value ?? 0) - (a.metrics.speedup?.value ?? 0),
  );
  const isFocused = state.chartViewMode === 'focused';
  const limit = isFocused ? 18 : state.speedupChartLimit;
  const chartRuns = selectedRuns.slice(0, limit);
  const displayRuns = [...chartRuns].reverse();
  const hasScroll = !isFocused && displayRuns.length > 18;

  el.dataset.parityStyle = 'dashed';
  el.dataset.parityLabelPlacement = 'axis-bottom';
  el.dataset.tooltipPlacement = 'opposite-corner';
  el.dataset.chartView = state.chartViewMode;
  el.dataset.speedupRows = String(selectedRuns.length);
  el.dataset.speedupDisplayed = String(chartRuns.length);
  el.setAttribute(
    'aria-label',
    `Speedup vs Reference chart — ${isFocused ? 'focused representative view' : 'full matrix view'}; tooltip is confined to the chart and docked away from labels; dashed 1× parity line labeled near the horizontal axis; values to the right are faster`,
  );

  if (selectedRuns.length === 0) {
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

  const reportedCount = chartRuns.filter(
    (run) => run.metrics.speedup?.reported_semantics === 'reported_by_runner',
  ).length;
  const subtitleParts = [...selection.notes];
  if (reportedCount > 0) subtitleParts.push('Ⓡ = runner-reported');
  subtitleParts.push('dashed line = 1× parity');
  if (selectedRuns.length > chartRuns.length) {
    subtitleParts.push(`showing top ${chartRuns.length}/${selectedRuns.length}`);
  } else if (hasScroll) {
    subtitleParts.push('scroll to browse');
  }

  const visibleWindowStart = Math.max(0, displayRuns.length - 18);

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
        trigger: 'item',
        confine: true,
        enterable: false,
        transitionDuration: 0,
        position: placeTooltip,
        backgroundColor: CHART_STYLE.tooltipBackground,
        borderWidth: 0,
        padding: [8, 10],
        textStyle: { color: '#fff', fontSize: 12, lineHeight: 18 },
        extraCssText:
          'max-width: 360px; white-space: normal; overflow-wrap: anywhere; pointer-events: none; box-shadow: 0 6px 18px rgba(22, 27, 45, 0.22);',
        formatter: (params: SpeedupTooltipParam | SpeedupTooltipParam[]) => {
          const param = Array.isArray(params) ? params[0] : params;
          if (!param || param.value == null) return 'No data';
          const label = param.value > 1 ? 'faster' : param.value < 1 ? 'slower' : 'same';
          const reference = param.data?.refLabel ?? 'reference';
          const semantics =
            param.data?.semantics === 'reported_by_runner' ? 'runner-reported' : 'computed';
          return `<b>${param.value.toFixed(2)}×</b><br>${semantics} vs ${reference} (${label})`;
        },
      },
      grid: {
        left: 12,
        right: hasScroll ? 38 : 20,
        top: 66,
        bottom: 46,
        containLabel: true,
      },
      dataZoom: hasScroll
        ? [
            {
              type: 'inside',
              yAxisIndex: 0,
              startValue: visibleWindowStart,
              endValue: displayRuns.length - 1,
            },
            {
              type: 'slider',
              yAxisIndex: 0,
              right: 4,
              width: 12,
              startValue: visibleWindowStart,
              endValue: displayRuns.length - 1,
              showDetail: false,
              brushSelect: false,
            },
          ]
        : [],
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
        data: displayRuns.map(formatRunLabel),
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
          data: displayRuns.map((run) => {
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
