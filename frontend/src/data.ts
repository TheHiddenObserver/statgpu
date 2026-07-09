import type { BenchmarkData, ParseReport, Run } from './schema';

const DATA_URL = `${import.meta.env.BASE_URL}data/benchmark_data.json`;
const REPORT_URL = `${import.meta.env.BASE_URL}data/parse_report.json`;

let cachedData: BenchmarkData | null = null;
let cachedReport: ParseReport | null = null;

export async function fetchBenchmarkData(): Promise<BenchmarkData> {
  if (cachedData) return cachedData;
  const resp = await fetch(DATA_URL);
  if (!resp.ok) throw new Error(`Failed to load benchmark data: ${resp.status}`);
  cachedData = await resp.json();
  return cachedData!;
}

export async function fetchParseReport(): Promise<ParseReport> {
  if (cachedReport) return cachedReport;
  const resp = await fetch(REPORT_URL);
  if (!resp.ok) return { files_seen: 0, files_parsed: 0, files_skipped: 0, runs_generated: 0, warnings: [] };
  cachedReport = await resp.json();
  return cachedReport!;
}

export function getUniqueValues(runs: Run[], field: string): string[] {
  const values = new Set<string>();
  for (const r of runs) {
    const val = (r as unknown as Record<string, unknown>)[field];
    if (val !== null && val !== undefined) values.add(String(val));
  }
  return [...values].sort();
}

export function getUniqueScaleKeys(runs: Run[]): string[] {
  const keys = new Set<string>();
  for (const r of runs) keys.add(r.scale.scale_key);
  return [...keys].sort();
}

export function filterRuns(runs: Run[], state: AppState): Run[] {
  return runs.filter(r => {
    // Category filter
    if (state.selectedCategoryIds.size > 0) {
      const hasCat = r.category_ids.some(cid => state.selectedCategoryIds.has(cid));
      if (!hasCat) return false;
    }

    // Environment filter
    if (state.selectedEnvId && r.env_id !== state.selectedEnvId) return false;

    // Model filter
    if (state.selectedModelId && r.model_id !== state.selectedModelId) return false;

    // Penalty filter
    if (state.selectedPenalty && r.penalty !== state.selectedPenalty) return false;

    // Solver filter
    if (state.selectedSolver && r.solver !== state.selectedSolver) return false;

    // Scale filter
    if (state.selectedScaleKeys.size > 0 && !state.selectedScaleKeys.has(r.scale.scale_key)) return false;

    // Backend filter (statgpu only)
    if (state.selectedBackends.size > 0 && r.framework === 'statgpu' && r.backend) {
      if (!state.selectedBackends.has(r.backend)) return false;
    }

    // External framework filter
    if (r.framework !== 'statgpu' && state.showExternal.size > 0) {
      if (!state.showExternal.has(r.framework)) return false;
    } else if (r.framework !== 'statgpu' && state.showExternal.size === 0) {
      return false; // External hidden by default
    }

    return true;
  });
}

/** Simple reactive state for Phase 2 */
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
}

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
  };
}
