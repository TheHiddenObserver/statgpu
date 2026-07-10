import type { BenchmarkData, ParseReport, Run } from './schema';
import type { AppState } from './state';

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

/** Precompute scale_key → label map for O(1) chip label lookup */
let scaleLabelMap: Map<string, string> | null = null;

export function getScaleLabelMap(runs: Run[]): Map<string, string> {
  if (scaleLabelMap) return scaleLabelMap;
  scaleLabelMap = new Map();
  for (const r of runs) {
    if (!scaleLabelMap.has(r.scale.scale_key)) {
      scaleLabelMap.set(r.scale.scale_key, r.scale.label);
    }
  }
  return scaleLabelMap;
}

/** Reset cached scale label map — call when switching data sources */
export function resetScaleLabelMap(): void {
  scaleLabelMap = null;
}

export interface FilterOptions {
  ignoreScale?: boolean;
}

export function filterRuns(
  runs: Run[],
  state: AppState,
  opts?: FilterOptions,
): Run[] {
  return runs.filter(r => {
    // Category filter — empty set means no categories selected = empty results
    if (state.selectedCategoryIds.size === 0) return false;
    const hasCat = r.category_ids.some(cid => state.selectedCategoryIds.has(cid));
    if (!hasCat) return false;

    // Environment filter
    if (state.selectedEnvId && r.env_id !== state.selectedEnvId) return false;

    // Model filter
    if (state.selectedModelId && r.model_id !== state.selectedModelId) return false;

    // Penalty filter
    if (state.selectedPenalty && r.penalty !== state.selectedPenalty) return false;

    // Solver filter
    if (state.selectedSolver && r.solver !== state.selectedSolver) return false;

    // Scale filter (skipped when deriving scale options)
    if (
      !opts?.ignoreScale &&
      state.selectedScaleKeys.size > 0 &&
      !state.selectedScaleKeys.has(r.scale.scale_key)
    )
      return false;

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

