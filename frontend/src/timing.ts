/** Shared display and qualification rules. No timing or ratio is invented here. */
import type { Run } from './schema';
import { chartGroupIdentity } from './identity';

export function timingValue(run: Run): number | null {
  const value = run.metrics.operation_timing
    ? run.metrics.operation_timing.duration_ms
    : run.metrics.timing?.fit_time_ms;
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

export function timingScopeLabel(run: Run): string {
  const scope = run.metrics.operation_timing?.scope ?? run.parameters?.timing_scope;
  return typeof scope === 'string' && scope.length > 0 ? scope.replace(/_/g, ' ') : 'Fit';
}

export function timingQuality(run: Run): string | undefined {
  return run.metrics.operation_timing?.quality ?? run.metrics.timing?.quality;
}

export function evidenceLabel(run: Run): string {
  if (!run.evidence) return 'Legacy source';
  const evidence = run.evidence;
  const attested = evidence.operator_attestation != null && evidence.operator_attestation !== false;
  const exclusivity = evidence.machine_verified_exclusivity ? 'machine-verified exclusivity' : 'exclusivity not machine-verified';
  const qualification = evidence.qualification === 'machine_verified' ? 'Machine-verified evidence'
    : evidence.qualification === 'operator_attested' ? 'Operator-attested evidence' : 'Diagnostic only';
  return `${qualification}${attested ? ' · operator attestation recorded' : ''} · ${exclusivity}`;
}

export function isPerformanceQualified(run: Run): boolean {
  // Preserve existing v1.1 semantics. New operation records cannot omit evidence
  // to gain legacy eligibility; the normalized schema additionally requires it.
  if (!run.evidence) return !run.metrics.operation_timing;
  const e = run.evidence;
  return /^[0-9a-f]{40}$/.test(e.source_commit ?? '') &&
    /^[0-9a-f]{64}$/.test(e.source_fingerprint ?? '') && e.qualification === 'machine_verified' && e.performance_eligible === true &&
    e.raw_performance_eligible === true && e.canonical_dashboard_ready === true &&
    e.machine_verified_exclusivity === true && e.correctness_status === 'pass' &&
    e.validation_status === 'pass' && (e.evidence_errors?.length ?? 0) === 0;
}

export function canDisplaySpeedup(run: Run, availableRuns: Run[]): boolean {
  const speedup = run.metrics.speedup;
  if (!speedup || !Number.isFinite(speedup.value) || speedup.value <= 0 || !isPerformanceQualified(run)) return false;
  if (!run.evidence && !run.metrics.operation_timing) return true;
  // New evidence must identify a real qualified comparator, even for a ratio
  // reported by the runner. A raw eligibility boolean alone is not sufficient.
  const reference = availableRuns.find(item => item.run_id === speedup.reference_run_id);
  if (!run.evidence || !run.benchmark_session_id ||
      !/^[0-9a-f]{40}$/.test(run.evidence.source_commit ?? '') ||
      !/^[0-9a-f]{64}$/.test(run.evidence.source_fingerprint ?? '')) return false;
  if (!reference || !reference.evidence || !isPerformanceQualified(reference) ||
      timingValue(run) == null || timingValue(reference) == null) return false;
  if (JSON.stringify(chartGroupIdentity(run, true)) !== JSON.stringify(chartGroupIdentity(reference, true))) return false;
  if (reference.framework !== speedup.reference_framework || reference.backend !== speedup.reference_backend) return false;
  const operation = run.metrics.operation_timing;
  const referenceOperation = reference.metrics.operation_timing;
  const ratio = timingValue(reference)! / timingValue(run)!;
  if (Math.abs(speedup.value - ratio) > Math.max(1e-6, Math.abs(ratio) * 1e-6)) return false;
  return operation?.scope === referenceOperation?.scope && operation?.statistic === referenceOperation?.statistic &&
    run.evidence?.source_commit === reference.evidence.source_commit &&
    run.evidence?.source_fingerprint === reference.evidence.source_fingerprint;
}
