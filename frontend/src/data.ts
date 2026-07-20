import type {
  BenchmarkData,
  FilterOptions,
  MetricScope,
  ParseReport,
  Run,
} from './schema';
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
  const scales = new Map<string, Run['scale']>();
  for (const run of runs) {
    if (!scales.has(run.scale.scale_key)) {
      scales.set(run.scale.scale_key, run.scale);
    }
  }
  return [...scales.values()]
    .sort((a, b) =>
      a.n_samples - b.n_samples ||
      a.n_features - b.n_features ||
      a.label.localeCompare(b.label) ||
      a.scale_key.localeCompare(b.scale_key),
    )
    .map(scale => scale.scale_key);
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

function parameterText(run: Run, key: string): string {
  const value = run.parameters?.[key];
  return value == null ? '' : String(value).toLowerCase();
}

export function isInferenceRun(run: Run): boolean {
  const timingScope = parameterText(run, 'timing_scope');
  return Boolean(
    run.metrics.inference ||
    run.parameters?.compute_inference === true ||
    run.parameters?.inference_method != null ||
    timingScope.includes('inference')
  );
}

export function isCrossValidationRun(run: Run): boolean {
  const explicitScopes = [
    parameterText(run, 'metric_scope'),
    parameterText(run, 'benchmark_scope'),
    parameterText(run, 'task_scope'),
    parameterText(run, 'timing_scope'),
  ];
  return Boolean(
    /(?:CV|CrossValidation)$/i.test(run.model_id) ||
    explicitScopes.some(value => value === 'cv' || value === 'cross_validation') ||
    run.parameters?.cv != null ||
    run.parameters?.cv_folds != null ||
    run.parameters?.fold_count != null ||
    run.parameters?.n_folds != null
  );
}

export function getRunMetricScopes(run: Run): Set<MetricScope> {
  const scopes = new Set<MetricScope>();
  const inference = isInferenceRun(run);
  const crossValidation = isCrossValidationRun(run);

  if (run.metrics.timing && !inference && !crossValidation) scopes.add('fit');
  if (crossValidation) scopes.add('cross_validation');
  if (inference) scopes.add('inference');
  if (run.metrics.prediction) scopes.add('prediction');
  if (run.metrics.selection) scopes.add('selection');

  return scopes;
}

export function runHasMetricScope(run: Run, scope: MetricScope): boolean {
  return scope === 'all' || getRunMetricScopes(run).has(scope);
}

export function getPrimaryMetricScope(run: Run): MetricScope {
  const scopes = getRunMetricScopes(run);
  for (const scope of [
    'inference',
    'cross_validation',
    'selection',
    'prediction',
    'fit',
  ] as MetricScope[]) {
    if (scopes.has(scope)) return scope;
  }
  return 'all';
}

export function getMetricScopeLabel(scope: MetricScope): string {
  const labels: Record<MetricScope, string> = {
    all: 'All',
    fit: 'Fit',
    cross_validation: 'Cross-validation',
    inference: 'Inference',
    prediction: 'Prediction',
    selection: 'Selection',
  };
  return labels[scope];
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
