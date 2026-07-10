/** Chart and run identity helpers. Must match Python identity.py semantics. */

import type { Run } from './schema';

export function chartGroupIdentity(run: Run, includeSession: boolean): readonly unknown[] {
  const common: unknown[] = [
    run.comparison_id,
    run.env_id,
    run.model_id,
    run.case_id,
    run.method_config_id,
    run.variant ?? null,
    run.implementation ?? null,
    run.loss ?? null,
    run.penalty ?? null,
    run.solver ?? null,
    run.scale.scale_key,
  ];
  return includeSession
    ? [run.comparison_id, run.benchmark_session_id ?? null, ...common.slice(1)]
    : common;
}

export function chartSeriesIdentity(run: Run): readonly unknown[] {
  return [run.framework, run.backend];
}
