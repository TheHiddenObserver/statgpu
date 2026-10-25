import type { Run } from './schema';
export type CpuThreadFilter = 'all' | 'single' | 'multi' | 'conflicting' | 'unknown';
export const CPU_THREAD_LABELS: Record<CpuThreadFilter, string> = {
  all: 'All CPU limits', single: '1 thread (observed limit)', multi: '>1 threads (observed limit)',
  conflicting: 'Mixed / conflicting limits', unknown: 'Not recorded / invalid',
};
/** Reported limits, not utilization or proof of parallel execution. */
export function cpuThreadClass(run: Run): Exclude<CpuThreadFilter, 'all'> {
  const observed = run.evidence?.cpu_threads_observed;
  const values: unknown[] = Array.isArray(observed) ? observed : [observed];
  const positiveInteger = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
  if (values.length === 0 || !values.every(positiveInteger)) return 'unknown';
  const limits = values as number[];
  if (new Set(limits).size !== 1) return 'conflicting';
  const requested = run.evidence?.cpu_threads_requested;
  if (requested !== undefined && requested !== null) {
    if (!positiveInteger(requested)) return 'unknown';
    if (requested !== limits[0]) return 'conflicting';
  }
  return limits[0] === 1 ? 'single' : 'multi';
}
export function provenanceValue(value: unknown): string {
  return value == null ? 'Not recorded' : typeof value === 'string' ? value : JSON.stringify(value);
}
export function cpuThreadLabel(run: Run): string {
  return `${CPU_THREAD_LABELS[cpuThreadClass(run)]}; requested ${provenanceValue(run.evidence?.cpu_threads_requested)}, observed ${provenanceValue(run.evidence?.cpu_threads_observed)}`;
}
/** Text only: untrusted source paths must not become executable links. */
export function sourceProvenance(run: Run): Array<[string, string]> {
  return [
    ['Source ID', run.source.source_id], ['Run ID', run.run_id],
    ['Session ID', provenanceValue(run.benchmark_session_id)],
    ['Raw producer commit (not renderer)', provenanceValue(run.evidence?.source_commit)],
    ['Source file', run.source.file], ['Source path', provenanceValue(run.source.original_path)],
    ['Source SHA256', provenanceValue(run.source.sha256)],
    ['Raw path', provenanceValue(run.parameters?.raw_path)],
    ['Raw pointer', provenanceValue(run.evidence?.raw_pointer)],
    ['Raw SHA256', provenanceValue(run.parameters?.raw_sha256)],
    ['Source fingerprint', provenanceValue(run.evidence?.source_fingerprint)],
    ['Parser', `${run.source.parser} ${run.source.parser_version}`],
    ['External adapter', provenanceValue(run.parameters?.external_adapter)],
  ];
}
