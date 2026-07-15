/** UI state model and mutation helpers */

import type { Environment, Run } from './schema';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ChartViewMode = 'focused' | 'full';

export interface AppState {
  selectedCategoryIds: Set<string>;
  selectedEnvId: string | null;
  selectedModelId: string | null;
  selectedVariant: string | null;
  selectedPenalty: string | null;
  selectedSolver: string | null;
  selectedScaleKeys: Set<string>;
  selectedBackends: Set<string>;
  showExternal: Set<string>;
  chartViewMode: ChartViewMode;
  tableLimit: number;
  sortColumn: string | null;
  sortDir: 'asc' | 'desc';
  expandedPanels: Set<string>;
  panelLimits: Record<string, number>;
  timingChartGroupLimit: number;
  speedupChartLimit: number;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createDefaultState(envs: Environment[], runs: Run[] = []): AppState {
  if (envs.length === 0) throw new Error('environments must have at least 1 entry');

  const runEnvIds = new Set(runs.map(run => run.env_id));
  const preferredEnvId = envs.find(
    env => env.env_id === 'remote-p100' && (runs.length === 0 || runEnvIds.has(env.env_id)),
  )?.env_id;
  const firstEnvWithRuns = envs.find(env => runEnvIds.has(env.env_id))?.env_id;
  const defaultEnvId = preferredEnvId ?? firstEnvWithRuns ?? envs[0].env_id;

  const availableCategories = new Set(
    runs
      .filter(run => run.env_id === defaultEnvId)
      .flatMap(run => run.category_ids),
  );
  const firstAvailableCategory = availableCategories.values().next().value as string | undefined;
  const defaultCategory = availableCategories.has('penalized_glm')
    ? 'penalized_glm'
    : firstAvailableCategory ?? (runs.length === 0 ? 'penalized_glm' : null);

  return {
    selectedCategoryIds: defaultCategory ? new Set([defaultCategory]) : new Set(),
    selectedEnvId: defaultEnvId,
    selectedModelId: null,
    selectedVariant: null,
    selectedPenalty: null,
    selectedSolver: null,
    selectedScaleKeys: new Set(),
    selectedBackends: new Set(),
    showExternal: new Set(),
    chartViewMode: 'focused',
    tableLimit: 200,
    sortColumn: null,
    sortDir: 'asc',
    expandedPanels: new Set(),
    panelLimits: {},
    timingChartGroupLimit: 30,
    speedupChartLimit: 24,
  };
}

// ---------------------------------------------------------------------------
// Cascade reset
// ---------------------------------------------------------------------------

export function resetDownstreamFilters(state: AppState, opts: {
  clearModel?: boolean;
  clearVariant?: boolean;
  clearPenalty?: boolean;
  clearSolver?: boolean;
  clearScale?: boolean;
  clearBackend?: boolean;
  clearExternal?: boolean;
}): void {
  if (opts.clearModel)    state.selectedModelId = null;
  if (opts.clearVariant)  state.selectedVariant = null;
  if (opts.clearPenalty)  state.selectedPenalty = null;
  if (opts.clearSolver)   state.selectedSolver = null;
  if (opts.clearScale)    state.selectedScaleKeys.clear();
  if (opts.clearBackend)  state.selectedBackends.clear();
  if (opts.clearExternal) state.showExternal.clear();
}

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------

export function setSelectedModel(state: AppState, modelId: string | null): void {
  state.selectedModelId = modelId;
  resetDownstreamFilters(state, { clearVariant: true, clearPenalty: true, clearSolver: true, clearScale: true });
}

export function setSelectedVariant(state: AppState, variant: string | null): void {
  state.selectedVariant = variant;
  resetDownstreamFilters(state, { clearPenalty: true, clearSolver: true, clearScale: true });
}

export function setSelectedPenalty(state: AppState, penalty: string | null): void {
  state.selectedPenalty = penalty;
  resetDownstreamFilters(state, { clearSolver: true, clearScale: true });
}

export function setSelectedSolver(state: AppState, solver: string | null): void {
  state.selectedSolver = solver;
  resetDownstreamFilters(state, { clearScale: true });
}

export function toggleScaleKey(state: AppState, key: string): void {
  if (state.selectedScaleKeys.has(key)) {
    state.selectedScaleKeys.delete(key);
  } else {
    state.selectedScaleKeys.add(key);
  }
}

export function setBackend(state: AppState, backend: 'numpy' | 'cupy' | 'torch' | null): void {
  state.selectedBackends.clear();
  if (backend) state.selectedBackends.add(backend);
}

export function toggleExternal(state: AppState, framework: string): void {
  if (state.showExternal.has(framework)) {
    state.showExternal.delete(framework);
  } else {
    state.showExternal.add(framework);
  }
}

export function setChartViewMode(state: AppState, mode: ChartViewMode): void {
  state.chartViewMode = mode;
}

export function setTableLimit(state: AppState, limit: number): void {
  state.tableLimit = limit;
}

export function setSortColumn(state: AppState, column: string | null): void {
  if (state.sortColumn === column) {
    state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    state.sortColumn = column;
    state.sortDir = 'asc';
  }
}
