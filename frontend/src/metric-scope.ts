import type { MetricScope, Run } from './schema';

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
    run.metrics.cross_validation ||
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
