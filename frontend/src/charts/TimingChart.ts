import * as echarts from 'echarts';
import type { Run } from '../schema';
import type { AppState } from '../state';
import { CHART_STYLE, COLORS } from '../utils/theme';
import { formatModelName } from '../utils/format';
import { chartGroupIdentity } from '../identity';

interface TimingSelection {
  runs: Run[];
  notes: string[];
  penaltyScope: 'penalized-only' | 'all';
}

function groupKey(run: Run): string {
  return JSON.stringify(chartGroupIdentity(run, false));
}

function shouldFocusPenalizedRows(state: AppState): boolean {
  return (
    state.selectedCategoryIds.size === 1 &&
    state.selectedCategoryIds.has('penalized_glm') &&
    state.selectedPenalty === null
  );
}

function selectTimingRuns(runs: Run[], state: AppState): TimingSelection {
  const timingRuns = runs.filter((run) => run.metrics.timing);
  if (state.chartViewMode === 'full' || timingRuns.length === 0) {
    return { runs: timingRuns, notes: ['Full matrix'], penaltyScope: 'all' };
  }

  let focused = timingRuns;
  const notes = ['Focused'];
  let penaltyScope: TimingSelection['penaltyScope'] = 'all';

  // Penalized GLM has a dedicated category, while unpenalized GLM is available
  // in the separate GLM category. Keep the default focused view on true
  // penalty comparisons without changing the table or filter state. An explicit
  // penalty selection always takes precedence, and Full matrix retains all rows.
  if (shouldFocusPenalizedRows(state)) {
    const penalizedRows = focused.filter(
      (run) => Boolean(run.penalty) && run.penalty !== 'none',
    );
    if (penalizedRows.length > 0) {
      focused = penalizedRows;
      penaltyScope = 'penalized-only';
      notes.push('penalized rows only');
    }
  }

  // When the user has not explicitly selected scales, use the largest workload
  // represented in the current filter context. This keeps the default chart
  // legible without changing the table or the actual filter state.
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

  // Prefer dispatch/Auto(best) groups where they exist. Match by canonical
  // chart-group identity so any external reference rows in the same group stay
  // visible. Domains without dispatch rows retain their complete focused view.
  const dispatchGroupKeys = new Set(
    focused
      .filter(
        (run) =>
          run.framework === 'statgpu' &&
          (run.solver_kind === 'dispatch' || run.solver === 'auto'),
      )
      .map(groupKey),
  );
  if (dispatchGroupKeys.size > 0) {
    focused = focused.filter((run) => dispatchGroupKeys.has(groupKey(run)));
    notes.push('Auto/best solver groups');
  }

  return { runs: focused, notes, penaltyScope };
}

function formatGroupLabel(run: Run, focused: boolean): string {
  const variant = run.variant ? ` (${run.variant})` : '';
  const model = `${formatModelName(run.model_id)}${variant}`;
  const penalty = run.penalty && run.penalty !== 'none' ? run.penalty : null;
  const solver = run.solver_display ?? run.solver ?? 'unknown';

  if (focused) {
    const solverPart = run.solver === 'auto' || run.solver_kind === 'dispatch'
      ? null
      : solver;
    return [model, penalty, solverPart].filter(Boolean).join(' · ');
  }

  const firstLine = [model, penalty].filter(Boolean).join(' · ');
  return `${firstLine}\n${solver} · ${run.scale.label}`;
}

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

  const selection = selectTimingRuns(runs, state);
  const timingRuns = selection.runs;
  el.dataset.chartView = state.chartViewMode;
  el.dataset.penaltyScope = selection.penaltyScope;
  el.setAttribute(
    'aria-label',
    `Fit Time chart — ${state.chartViewMode === 'focused' ? 'focused representative view' : 'full matrix view'}${selection.penaltyScope === 'penalized-only' ? '; penalized rows only' : ''}`,
  );

  if (timingRuns.length === 0) {
    chart.clear();
    chart.setOption({
      title: {
        text: 'No timing data',
        left: 'center',
        top: 'center',
        textStyle: { color: CHART_STYLE.muted, fontSize: 14 },
      },
    });
    return;
  }

  interface TimingSeries {
    key: string;
    label: string;
    backend: string | null;
    framework: string;
    impl: string | null;
  }
  function makeSeries(run: Run): TimingSeries {
    const impl = run.implementation ?? null;
    const identity = [run.framework, run.backend ?? null, impl];
    return {
      key: JSON.stringify(identity),
      label:
        run.framework === 'statgpu'
          ? [run.backend, impl].filter(Boolean).join('/') || (run.backend ?? 'statgpu')
          : run.framework,
      backend: run.backend,
      framework: run.framework,
      impl,
    };
  }

  type GroupKey = string;
  const groups = new Map<GroupKey, { label: string; bySeries: Map<string, number> }>();
  const seriesMeta = new Map<string, TimingSeries>();
  const isFocused = state.chartViewMode === 'focused';

  for (const run of timingRuns) {
    const key = groupKey(run);
    if (!groups.has(key)) {
      groups.set(key, {
        label: formatGroupLabel(run, isFocused),
        bySeries: new Map(),
      });
    }
    const series = makeSeries(run);
    seriesMeta.set(series.key, series);
    groups.get(key)!.bySeries.set(series.key, run.metrics.timing!.fit_time_ms);
  }

  const limit = isFocused ? 14 : state.timingChartGroupLimit;
  const allGroups = [...groups.entries()].sort(([, a], [, b]) =>
    a.label.localeCompare(b.label),
  );
  const sortedGroups = allGroups.slice(0, limit);
  const categories = sortedGroups.map(([, group]) => group.label);
  const subtitleParts = [...selection.notes];
  if (allGroups.length > sortedGroups.length) {
    subtitleParts.push(`showing ${sortedGroups.length}/${allGroups.length} groups`);
  }

  const allKeys = new Set<string>();
  for (const [, group] of sortedGroups) {
    for (const seriesKey of group.bySeries.keys()) allKeys.add(seriesKey);
  }
  const preferredLabels = ['numpy', 'numpy/numba', 'cupy', 'torch', ...state.showExternal];
  const seriesOrder: TimingSeries[] = [];
  for (const label of preferredLabels) {
    for (const [key, series] of seriesMeta) {
      if (series.label === label && allKeys.has(key)) seriesOrder.push(series);
    }
  }
  for (const [key, series] of seriesMeta) {
    if (allKeys.has(key) && !seriesOrder.some((item) => item.key === key)) {
      seriesOrder.push(series);
    }
  }

  const series = seriesOrder.map((item) => ({
    name: item.label,
    type: 'bar' as const,
    barMaxWidth: 22,
    data: sortedGroups.map(([, group]) => group.bySeries.get(item.key) ?? null),
    itemStyle: {
      color: COLORS[item.label] || COLORS[item.backend || ''] || '#8a93a3',
      borderRadius: [3, 3, 0, 0],
    },
  }));

  chart.setOption(
    {
      title: {
        text: 'Fit Time (ms)',
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
            seriesName: string;
            value: number | null;
            color: string;
            axisValueLabel?: string;
          }[],
        ) => {
          const heading = params[0]?.axisValueLabel
            ? `<b>${params[0].axisValueLabel.replace('\n', ' · ')}</b><br/>`
            : '';
          const values = params
            .filter((param) => param.value != null)
            .map(
              (param) =>
                `<span style="color:${param.color}">●</span> ${param.seriesName}: <b>${param.value!.toFixed(2)} ms</b>`,
            )
            .join('<br/>');
          return `${heading}${values || 'No data'}`;
        },
      },
      legend: {
        bottom: 2,
        textStyle: { fontSize: 11, color: CHART_STYLE.text },
        itemWidth: 18,
        itemHeight: 9,
      },
      grid: { left: 12, right: 12, top: 64, bottom: 62, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLine: { lineStyle: { color: CHART_STYLE.axis } },
        axisTick: { alignWithLabel: true, lineStyle: { color: CHART_STYLE.axis } },
        axisLabel: {
          fontSize: 10,
          color: CHART_STYLE.text,
          rotate: isFocused ? 30 : 38,
          width: 118,
          overflow: 'truncate',
          hideOverlap: true,
        },
      },
      yAxis: {
        type: 'log',
        name: 'ms',
        nameTextStyle: { color: CHART_STYLE.muted },
        axisLine: { show: false },
        axisLabel: { fontSize: 10, color: CHART_STYLE.text },
        splitLine: { lineStyle: { color: CHART_STYLE.grid } },
      },
      series,
    },
    true,
  );
}
