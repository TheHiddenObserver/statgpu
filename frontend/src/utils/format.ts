/** Formatting utilities for display */

export function formatTime(ms: number, stdMs?: number): string {
  const std = stdMs != null ? `±${stdMs.toFixed(1)}` : '';
  return `${ms.toFixed(2)}${std}`;
}

export function formatSpeedup(value: number): string {
  return `${value.toFixed(1)}x`;
}

export function formatModelName(name: string): string {
  return name.replace('Penalized', '').replace('Regression', '');
}

export function formatQuality(
  timingQuality?: string,
  speedupQuality?: string,
): string {
  if (timingQuality && speedupQuality && timingQuality !== speedupQuality) {
    return `time:${timingQuality} · speedup:${speedupQuality}`;
  }
  return timingQuality || speedupQuality || '-';
}
