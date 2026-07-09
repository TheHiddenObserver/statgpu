/** TypeScript types matching benchmark_frontend_schema.json v1.0.0 */

export interface BenchmarkData {
  schema_version: string;
  generated: string;
  meta: Meta;
  environments: Environment[];
  categories: Category[];
  models: Model[];
  runs: Run[];
}

export interface Meta {
  generator: string;
  git_sha: string;
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
  loss?: string;
  supports_penalty?: boolean;
  supports_inference?: boolean;
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
  framework: 'statgpu' | 'sklearn' | 'statsmodels' | 'glmnet' | 'scipy' | 'r';
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
  file: string;
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
}

export interface TimingMetric {
  fit_time_ms: number;
  std_ms?: number;
  min_ms?: number;
  max_ms?: number;
  quality: 'measured' | 'reported' | 'computed' | 'partial';
  source_file: string;
}

export interface SpeedupMetric {
  value: number;
  reference_run_id?: string;
  reference_backend: 'numpy' | 'cupy' | 'torch' | null;
  reference_framework: 'statgpu' | 'sklearn' | 'statsmodels' | 'glmnet' | 'scipy' | 'r';
  reported_semantics: 'computed' | 'reported_by_runner';
  quality: 'measured' | 'reported' | 'computed' | 'partial';
  source_file: string;
}

export interface AccuracyMetric {
  coef_l2_diff?: number;
  coef_max_abs_diff?: number;
  bse_max_abs_diff?: number;
  reference?: string;
  quality: 'measured' | 'reported' | 'computed' | 'partial';
  source_file: string;
}

export interface InferenceMetric {
  bse?: number;
  wald_stat?: number;
  p_value?: number;
  ok?: boolean;
  quality: 'measured' | 'reported' | 'computed' | 'partial';
  source_file: string;
}

export interface ConvergenceMetric {
  n_iter?: number;
  converged?: boolean;
  quality?: 'measured' | 'reported' | 'computed' | 'partial';
  source_file?: string;
}

export interface Quality {
  status?: 'ok' | 'warning' | 'error';
  warnings?: string[];
}

/** Parse report */
export interface ParseReport {
  files_seen: number;
  files_parsed: number;
  files_skipped: number;
  runs_generated: number;
  warnings: { file: string; reason: string }[];
}
