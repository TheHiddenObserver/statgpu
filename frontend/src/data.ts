import type {
  FilterOptions,
  Run,
} from './schema';
import type { AppState } from './state';
import {
  StaticJsonBenchmarkProvider,
  type BenchmarkBundle,
  type BenchmarkDataProvider,
  type BenchmarkLoadOptions,
} from './providers/benchmark';
import { runHasMetricScope } from './metric-scope';

export { getUniqueScaleKeys } from './scales';
export {
  getMetricScopeLabel,
  getPrimaryMetricScope,
  getRunMetricScopes,
  isCrossValidationRun,
  isInferenceRun,
  runHasMetricScope,
} from './metric-scope';

let activeProvider: BenchmarkDataProvider = new StaticJsonBenchmarkProvider({
  baseUrl: import.meta.env.BASE_URL,
});

export function setBenchmarkDataProvider(provider: BenchmarkDataProvider): void {
  activeProvider = provider;
}

export function loadBenchmarkBundle(
  options?: BenchmarkLoadOptions,
): Promise<BenchmarkBundle> {
  return activeProvider.loadBundle(options);
}

export function getUniqueValues(runs: Run[], field: string): string[] {
  const values = new Set<string>();
  for (const r of runs) {
    const val = (r as unknown as Record<string, unknown>)[field];
    if (val !== null && val !== undefined) values.add(String(val));
  }
  return [...values].sort();
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
    if (state.selectedEnvId && !state.selectedEnvIds.has(r.env_id)) return false;

    // Metric-scope filter. Inference and CV remain attached to their model
    // categories rather than being treated as separate statistical families.
    if (
      !opts?.ignoreMetricScope &&
      state.selectedMetricScope !== 'all' &&
      !runHasMetricScope(r, state.selectedMetricScope)
    )
      return false;

    // Model filter
    if (state.selectedModelId && r.model_id !== state.selectedModelId) return false;

    // Variant filter
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

    // External framework filter (empty = hide all)
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
