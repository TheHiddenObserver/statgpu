import type { BenchmarkData, ParseReport, Run, FilterOptions } from './schema';
import type { AppState } from './state';

const DATA_URL = `${import.meta.env.BASE_URL}data/benchmark_data.json`;
const REPORT_URL = `${import.meta.env.BASE_URL}data/parse_report.json`;
const INVENTORY_URL = `${import.meta.env.BASE_URL}data/source_inventory.json`;

let cachedData: BenchmarkData | null = null;
let cachedReport: ParseReport | null = null;

export async function fetchBenchmarkData(): Promise<BenchmarkData> {
  if (cachedData) return cachedData;
  const resp = await fetch(DATA_URL);
  if (!resp.ok) throw new Error(`Failed to load benchmark data: ${resp.status}`);
  cachedData = await resp.json();
  // Schema version check
  const SUPPORTED = '1.1.0';
  if (cachedData!.schema_version !== SUPPORTED) {
    throw new Error(`Unsupported schema ${cachedData!.schema_version}; expected ${SUPPORTED}`);
  }
  return cachedData!;
}

export async function fetchParseReport(): Promise<ParseReport | null> {
  if (cachedReport) return cachedReport;
  try {
    const resp = await fetch(REPORT_URL);
    if (!resp.ok) return null;
    const raw = await resp.json();
    if (raw.report_version !== '2.0') return null;
    cachedReport = raw;
    return cachedReport!;
  } catch {
    return null;
  }
}

let cachedInventory: import('./schema').SourceInventory | null = null;

export async function fetchSourceInventory(): Promise<import('./schema').SourceInventory | null> {
  if (cachedInventory) return cachedInventory;
  try {
    const resp = await fetch(INVENTORY_URL);
    if (!resp.ok) return null;
    const raw = await resp.json();
    if (raw.inventory_version !== '1.0') return null;
    cachedInventory = raw;
    return cachedInventory!;
  } catch {
    return null;
  }
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

export function resetScaleLabelMap(): void {
  scaleLabelMap = null;
}

export function filterRuns(
  runs: Run[],
  state: AppState,
  opts?: FilterOptions,
): Run[] {
  return runs.filter(r => {
    // Category filter
    if (state.selectedCategoryIds.size === 0) return false;
    const hasCat = r.category_ids.some(cid => state.selectedCategoryIds.has(cid));
    if (!hasCat) return false;

    // Environment filter
    if (state.selectedEnvId && r.env_id !== state.selectedEnvId) return false;

    // Model filter
    if (state.selectedModelId && r.model_id !== state.selectedModelId) return false;

    // Variant filter (NEW)
    if (state.selectedVariant && r.variant !== state.selectedVariant) return false;

    // Penalty filter
    if (state.selectedPenalty && r.penalty !== state.selectedPenalty) return false;

    // Solver filter
    if (state.selectedSolver && r.solver !== state.selectedSolver) return false;

    // Scale filter
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

    // External framework filter (unchanged semantics: empty = hide all)
    if (!opts?.ignoreExternal) {
      if (r.framework !== 'statgpu' && state.showExternal.size > 0) {
        if (!state.showExternal.has(r.framework)) return false;
      } else if (r.framework !== 'statgpu' && state.showExternal.size === 0) {
        return false;
      }
    }

    return true;
  });
}
