/** Local diagnostic files are displayed, never promoted or uploaded. */
export const PREVIEW_VERSION = 'statgpu-benchmark-update-preview-v1';
export type PreviewObject = Record<string, unknown>;
export interface PreviewJob extends PreviewObject {
  job_id: string; suite_or_lane: string; validation_status: string; correctness_status: string;
  evidence_errors: string[]; performance_eligible: false; canonical_dashboard_ready: false;
}
export interface PreviewRecord extends PreviewJob {
  record_id: string; campaign_id: string; raw_path: string; raw_pointer: string; timing_key: string;
  timing_scope: string | null; unit: 'seconds' | 'milliseconds' | null; raw_value: unknown;
  context: PreviewObject; summary_seconds?: number; statistic?: string; sample_count?: number;
}
export interface DiagnosticPreview extends PreviewObject {
  schema_version: typeof PREVIEW_VERSION; campaigns: PreviewObject[]; jobs: PreviewJob[]; records: PreviewRecord[];
  performance_eligible: false; canonical_dashboard_ready: false; comparisons: [];
  exclusivity: PreviewObject & { machine_verified: false; machine_status: string; operator_attestation: unknown };
}
export function isObject(value: unknown): value is PreviewObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function requiredText(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`Missing or invalid ${label}.`);
}
function diagnosticFlags(value: PreviewObject): void {
  if (value.performance_eligible !== false || value.canonical_dashboard_ready !== false) {
    throw new Error('Preview must remain diagnostic only; performance promotion is not supported.');
  }
}
function validateNumbers(value: unknown): void {
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Nonfinite JSON numbers are not supported.');
  if (Array.isArray(value)) value.forEach(validateNumbers);
  else if (isObject(value)) Object.values(value).forEach(validateNumbers);
}
export function parseDiagnosticPreview(text: string): DiagnosticPreview {
  const value: unknown = JSON.parse(text);
  if (!isObject(value) || value.schema_version !== PREVIEW_VERSION) throw new Error(`Expected ${PREVIEW_VERSION}.`);
  validateNumbers(value);
  diagnosticFlags(value);
  if (!Array.isArray(value.campaigns) || value.campaigns.some(item => !isObject(item)) ||
      !Array.isArray(value.jobs) || !Array.isArray(value.records)) throw new Error('Expected campaigns, jobs, and records arrays.');
  if (!Array.isArray(value.comparisons) || value.comparisons.length !== 0) throw new Error('Diagnostic previews cannot contain performance comparisons.');
  if (!isObject(value.exclusivity) || value.exclusivity.machine_verified !== false) throw new Error('This preview format does not establish machine-verified exclusivity.');
  requiredText(value.exclusivity.machine_status, 'exclusivity status');
  const jobIds = new Set<string>();
  for (const item of value.jobs) {
    if (!isObject(item)) throw new Error('Invalid job record.');
    for (const key of ['job_id', 'suite_or_lane', 'validation_status', 'correctness_status']) requiredText(item[key], `job ${key}`);
    if (jobIds.has(item.job_id as string)) throw new Error('Duplicate job identity.');
    jobIds.add(item.job_id as string);
    diagnosticFlags(item);
    if (!Array.isArray(item.evidence_errors) || item.evidence_errors.some(error => typeof error !== 'string')) throw new Error('Invalid job evidence errors.');
  }
  const recordIds = new Set<string>();
  for (const item of value.records) {
    if (!isObject(item)) throw new Error('Invalid timing record.');
    for (const key of ['record_id', 'job_id', 'campaign_id', 'suite_or_lane', 'raw_path', 'raw_pointer', 'timing_key', 'correctness_status', 'validation_status']) requiredText(item[key], `record ${key}`);
    if (recordIds.has(item.record_id as string)) throw new Error('Duplicate timing record identity.');
    recordIds.add(item.record_id as string);
    if (!jobIds.has(item.job_id as string)) throw new Error('Timing record refers to an unknown job.');
    if (item.timing_scope !== null && typeof item.timing_scope !== 'string') throw new Error('Invalid timing scope.');
    if (![null, 'seconds', 'milliseconds'].includes(item.unit as string | null)) throw new Error('Invalid timing unit.');
    if (!isObject(item.context) || !('raw_value' in item)) throw new Error('Missing timing context or raw value.');
    if ('summary_seconds' in item && (typeof item.summary_seconds !== 'number' || item.summary_seconds <= 0 || !Number.isFinite(item.summary_seconds * 1000) || item.unit === null)) throw new Error('Invalid positive timing summary or unknown unit.');
    if ('summary_seconds' in item) requiredText(item.statistic, 'summary statistic');
    if ('sample_count' in item) {
      if (!Number.isInteger(item.sample_count) || (item.sample_count as number) < 0) throw new Error('Invalid sample count.');
      if (item.sample_count === 0 && (!Array.isArray(item.raw_value) || item.raw_value.length !== 0 || 'summary_seconds' in item)) {
        throw new Error('Zero samples require an empty raw array without a timing summary.');
      }
    }
    if (!Array.isArray(item.evidence_errors) || item.evidence_errors.some(error => typeof error !== 'string')) throw new Error('Invalid timing evidence errors.');
    diagnosticFlags(item);
  }
  return value as DiagnosticPreview;
}
export function previewScale(context: PreviewObject): string {
  const problem = isObject(context.problem) ? context.problem : {};
  const shape = context.shape ?? problem.shape;
  const shapeN = Array.isArray(shape) ? shape[0] : isObject(shape) ? shape.n_samples ?? shape.n : undefined;
  const shapeP = Array.isArray(shape) ? shape[1] : isObject(shape) ? shape.n_features ?? shape.p : undefined;
  const n = context.n_samples ?? context.n ?? problem.n_samples ?? problem.n ?? shapeN;
  const p = context.n_features ?? context.p ?? problem.n_features ?? problem.p ?? shapeP;
  return `${n == null ? 'n not recorded' : `n=${String(n)}`} · ${p === 0 ? 'p=0 (not applicable)' : p == null ? 'p not recorded' : `p=${String(p)}`}`;
}
export function previewDuration(record: PreviewRecord): string {
  const ms = record.summary_seconds == null ? NaN : record.summary_seconds * 1000;
  return !Number.isFinite(ms) || ms <= 0 ? 'No positive timing summary' : `${ms.toPrecision(6)} ms`;
}
export function previewOperation(context: PreviewObject): string {
  const benchmark = isObject(context.benchmark) ? context.benchmark : {};
  const identity = context.operation ?? context.model_id ?? context.model ?? context.case_id ?? context.case ?? context.public_api ?? benchmark.public_api;
  return [identity, context.backend ?? context.requested_backend]
    .filter(value => value != null).map(previewValue).join(' · ') || 'Not recorded';
}
export function previewValue(value: unknown): string {
  if (value === undefined || value === null) return 'Not recorded';
  return typeof value === 'string' ? value : JSON.stringify(value);
}
