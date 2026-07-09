/** UI state model and mutation helpers */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AppState {
  selectedCategoryIds: Set<string>;
  selectedEnvId: string | null;
  selectedModelId: string | null;
  selectedPenalty: string | null;
  selectedSolver: string | null;
  selectedScaleKeys: Set<string>;
  selectedBackends: Set<string>;
  showExternal: Set<string>;
  showInference: boolean;
  tableLimit: number;
  sortColumn: string | null;
  sortDir: 'asc' | 'desc';
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createDefaultState(): AppState {
  return {
    selectedCategoryIds: new Set(['penalized_glm']),
    selectedEnvId: 'remote-p100',
    selectedModelId: null,
    selectedPenalty: null,
    selectedSolver: null,
    selectedScaleKeys: new Set(),
    selectedBackends: new Set(),
    showExternal: new Set(),
    showInference: false,
    tableLimit: 200,
    sortColumn: null,
    sortDir: 'asc',
  };
}

// ---------------------------------------------------------------------------
// Mutation helpers — mutate state in place, caller invokes onUpdate()
// ---------------------------------------------------------------------------

export function setSelectedModel(
  state: AppState,
  modelId: string | null,
): void {
  state.selectedModelId = modelId;
  state.selectedPenalty = null;
  state.selectedSolver = null;
}

export function setSelectedPenalty(
  state: AppState,
  penalty: string | null,
): void {
  state.selectedPenalty = penalty;
  state.selectedSolver = null;
}

export function setSelectedSolver(
  state: AppState,
  solver: string | null,
): void {
  state.selectedSolver = solver;
}

export function toggleScaleKey(state: AppState, key: string): void {
  if (state.selectedScaleKeys.has(key)) {
    state.selectedScaleKeys.delete(key);
  } else {
    state.selectedScaleKeys.add(key);
  }
}

export function setBackend(
  state: AppState,
  backend: 'numpy' | 'cupy' | 'torch' | null,
): void {
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

export function setTableLimit(state: AppState, limit: number): void {
  state.tableLimit = limit;
}

export function setSortColumn(
  state: AppState,
  column: string | null,
): void {
  if (state.sortColumn === column) {
    // toggle direction on same column
    state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    state.sortColumn = column;
    state.sortDir = 'asc';
  }
}
