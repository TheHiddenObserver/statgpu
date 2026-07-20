import type { Run } from './schema';

/**
 * Return unique scale keys in numeric workload order.
 *
 * This helper is intentionally independent of browser/Vite globals so it can
 * be reused by the frontend and by the standalone E2E TypeScript contract.
 */
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
