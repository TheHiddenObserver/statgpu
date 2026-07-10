/** TypeScript types matching benchmark_frontend_schema.json v1.1.0 */

export type MetricQuality = 'measured' | 'reported' | 'computed' | 'partial';

export interface BenchmarkData {
  schema_version: string;
  generated: string;
  meta: Meta;
  environments: Environment[];
  categories: Category[];
  models: Model[];
  frameworks: Framework[];
  comparisons: Comparison[];
  runs: Run[];
}

export interface Meta {
  generator: string;
  git_sha: string;
  generation_id: string;
}

export interface Environment {
  env_id: string;
  label: string;
  gpu: string;
  cpu: string;
  host?: string;
}

export interface Category {
  category_id: string;
  name_zh: string;
  name_en: string;
}

export interface Model {
  model_id: string;
  primary_category_id: string;
  category_ids: string[];
  supports_penalty?: boolean;
  supports_inference?: boolean;
}

export interface Framework {
  framework_id: string;
  display_name: string;
  external: boolean;
  backend_policy: 'required' | 'forbidden' | 'optional';
  color?: string;
}

export interface Comparison {
  comparison_id: string;
  label: string;
  env_id: string;
}

export interface Run {
  run_id: string;
  benchmark_session_id?: string;
  env_id: string;
  category_ids: string[];
  model_id: string;
  loss?: string;
  penalty?: string | null;
  solver?: string;
  solver_display?: string;
  solver_kind?: 'dispatch' | 'manual' | 'internal' | null;
  comparison_id: string;
  case_id: string;
  method_config_id: string;
  variant?: string;
  implementation?: string;
  parameters?: Record<string, unknown>;
  replicate?: { n_runs: number; seed_count?: number; n_failed?: number };
  framework: string;
  backend: 'numpy' | 'cupy' | 'torch' | null;
  scale: Scale;
  source: Source;
  metrics: Metrics;
  quality?: Quality;
}

export interface Scale {
  scale_key: string;
  n_samples: number;
  n_features: number;
  label: string;
}

export interface Source {
  source_id: string;
  file: string;
  original_path?: string;
  sha256?: string;
  date: string;
  parser: string;
  parser_version: string;
}

export interface Metrics {
  timing?: TimingMetric;
  speedup?: SpeedupMetric;
  accuracy?: AccuracyMetric;
  inference?: InferenceMetric;
  convergence?: ConvergenceMetric;
  selection?: SelectionMetric;
  prediction?: PredictionMetric;
  validation?: ValidationMetric;
}

export interface TimingMetric {
  fit_time_ms: number;
  std_ms?: number;
  min_ms?: number;
  max_ms?: number;
  sample_count?: number;
  std_ddof?: 0 | 1;
  std_scope?: 'raw_measurements' | 'replicates';
  quality: MetricQuality;
  source_file: string;
}

export interface SpeedupMetric {
  value: number;
  reference_run_id?: string;
  reference_backend: 'numpy' | 'cupy' | 'torch' | null;
  reference_framework: string;
  reported_semantics: 'computed' | 'reported_by_runner';
  quality: MetricQuality;
  source_file: string;
}

export interface AccuracyMetric {
  coef_l2_diff?: number;
  coef_l2_diff_std?: number;
  coef_max_abs_diff?: number;
  coef_max_abs_diff_std?: number;
  coef_l2_rel_error?: number;
  coef_l2_rel_error_std?: number;
  bse_max_abs_diff?: number;
  bse_max_abs_diff_std?: number;
  reference?: string;
  quality: MetricQuality;
  source_file: string;
}

export interface InferenceMetric {
  bse?: number;
  wald_stat?: number;
  p_value?: number;
  ok?: boolean;
  quality: MetricQuality;
  source_file: string;
}

export interface ConvergenceMetric {
  n_iter_mean?: number;
  n_iter_std?: number;
  converged_rate?: number;
  quality?: MetricQuality;
  source_file?: string;
}

export interface SelectionMetric {
  precision?: number;
  precision_std?: number;
  recall?: number;
  recall_std?: number;
  fdp?: number;
  fdp_std?: number;
  f1?: number;
  f1_std?: number;
  jaccard_truth?: number;
  jaccard_truth_std?: number;
  estimated_fdr?: number;
  estimated_fdr_std?: number;
  target_fdr?: number;
  n_selected_mean?: number;
  n_selected_std?: number;
  quality: MetricQuality;
  source_file: string;
}

export interface PredictionMetric {
  train_mse?: number;
  train_mse_std?: number;
  test_mse?: number;
  test_mse_std?: number;
  test_mse_noiseless?: number;
  test_mse_noiseless_std?: number;
  c_index?: number;
  c_index_std?: number;
  alpha_mean?: number;
  alpha_std?: number;
  quality: MetricQuality;
  source_file: string;
}

export interface ValidationCheck {
  metric: string;
  operator?: 'le' | 'lt' | 'ge' | 'gt' | 'abs_le';
  status: 'pass' | 'warn' | 'fail';
  value?: number;
  tolerance?: number;
  reference?: string;
}

export interface ValidationMetric {
  status: 'pass' | 'warn' | 'fail';
  checks?: ValidationCheck[];
  quality: MetricQuality;
  source_file: string;
}

export interface Quality {
  status?: 'ok' | 'warning' | 'error';
  warnings?: string[];
}

/** Parse report v2 */
export interface ParseIssue {
  source_id?: string;
  file?: string;
  parser?: string;
  code: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
}

export interface ParseReport {
  report_version: '2.0';
  generation_id: string;
  files_seen: number;
  files_parsed: number;
  files_skipped: number;
  runs_generated: number;
  issues: ParseIssue[];
}

/** Source inventory */
export interface SourceInventory {
  inventory_version: '1.0';
  catalog_version: string;
  generation_id: string;
  catalog_total: number;
  eligible_total: number;
  registered_sources: number;
  available_sources: number;
  parsed_sources: number;
}

/** Filter context and options */
export interface FilterContext {
  externalFrameworkIds: ReadonlySet<string>;
  frameworksById: ReadonlyMap<string, Framework>;
  comparisonsById: ReadonlyMap<string, Comparison>;
}

export interface FilterOptions {
  ignoreScale?: boolean;
  ignoreExternal?: boolean;
}
