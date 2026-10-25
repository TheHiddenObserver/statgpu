import { test, expect } from '@playwright/test';
import { parseDiagnosticPreview, previewScale, previewDuration, previewOperation, PREVIEW_VERSION } from '../src/preview';

function fixture() {
  const base = { job_id: 'job-a', campaign_id: 'campaign', suite_or_lane: 'classification-metrics', validation_status: 'validated_diagnostic', correctness_status: 'complete', evidence_errors: [], performance_eligible: false, canonical_dashboard_ready: false };
  return { schema_version: PREVIEW_VERSION, campaigns: [{ campaign_id: 'campaign' }],
    jobs: [base, { ...base, job_id: 'job-blocked', suite_or_lane: 'blocked-lane', validation_status: 'blocked', correctness_status: 'unavailable' }, { ...base, job_id: 'job-not-run', suite_or_lane: 'not-run-lane', validation_status: 'not_run', correctness_status: 'unavailable' }],
    records: [{ ...base, record_id: 'record-a', raw_path: '/raw.json', raw_pointer: '/rows/0/seconds', timing_key: 'seconds', timing_scope: 'classification_metric', unit: 'seconds', raw_value: [0.01, 0.02], sample_count: 2, summary_seconds: 0.015, statistic: 'median_of_recorded_samples', context: { operation: 'accuracy', backend: 'cupy', n: 100, p: 0 }, raw_performance_eligible: true },
      { ...base, record_id: 'record-unknown', raw_path: '/raw.json', raw_pointer: '/rows/1/samples', timing_key: 'samples', timing_scope: null, unit: null, raw_value: [], summary_seconds: undefined as number | undefined, context: {} }],
    performance_eligible: false, canonical_dashboard_ready: false, comparisons: [],
    exclusivity: { machine_verified: false, machine_status: 'UNVERIFIED_EXCLUSIVITY', operator_attestation: { statement: 'Operator statement only' } },
  };
}

test('preview preserves raw status, absent timings, scope, and p0', () => {
  const result = parseDiagnosticPreview(JSON.stringify(fixture()));
  expect(previewScale(result.records[0].context)).toBe('n=100 · p=0 (not applicable)');
  expect(previewDuration(result.records[0])).toBe('15.0000 ms');
  expect(previewDuration(result.records[1])).toBe('No positive timing summary');
  expect(result.records[0].raw_performance_eligible).toBe(true);
  expect(result.records[0].performance_eligible).toBe(false);
  expect(previewScale({ problem: { shape: [120, 6] } })).toBe('n=120 · p=6');
  expect(previewScale({ shape: [99, 0] })).toBe('n=99 · p=0 (not applicable)');
  expect(previewScale({ problem: { shape: { n: 120, p: 6 } } })).toBe('n=120 · p=6');
  expect(previewScale({ shape: { n_samples: 120, n_features: 6 } })).toBe('n=120 · p=6');
});

test('preview rejects promotion, unknown jobs, duplicate records and malformed numeric summaries', () => {
  for (const change of [
    (v: ReturnType<typeof fixture>) => { v.performance_eligible = true; },
    (v: ReturnType<typeof fixture>) => { v.exclusivity.machine_verified = true; },
    (v: ReturnType<typeof fixture>) => { v.records[0].job_id = 'missing'; },
    (v: ReturnType<typeof fixture>) => { v.records.push(v.records[0]); },
    (v: ReturnType<typeof fixture>) => { v.records[0].summary_seconds = 0; },
    (v: ReturnType<typeof fixture>) => { v.records[0].summary_seconds = -1; },
    (v: ReturnType<typeof fixture>) => { v.records[0].summary_seconds = 1e308; },
  ]) { const value = fixture(); change(value); expect(() => parseDiagnosticPreview(JSON.stringify(value))).toThrow(); }
  expect(() => parseDiagnosticPreview(JSON.stringify(fixture()).replace('0.015', '1e400'))).toThrow(/Nonfinite/);
});

test('empty exported samples remain inspectable without a fabricated timing', () => {
  const value = fixture();
  Object.assign(value.records[1], { sample_count: 0 });
  const result = parseDiagnosticPreview(JSON.stringify(value));
  expect(result.records[1].raw_value).toEqual([]);
  expect(result.records[1].sample_count).toBe(0);
  expect(previewDuration(result.records[1])).toBe('No positive timing summary');
  for (const bad of [{ raw_value: [0.1] }, { raw_value: null }, { summary_seconds: 0.1, statistic: 'median', unit: 'seconds' }]) {
    const malformed = fixture();
    Object.assign(malformed.records[1], { sample_count: 0 }, bad);
    expect(() => parseDiagnosticPreview(JSON.stringify(malformed))).toThrow();
  }
});

test('recorded case IDs and public API names remain visible and overflow cannot render Infinity', () => {
  expect(previewOperation({ case_id: 'paired-t-test', backend: 'cupy' })).toBe('paired-t-test · cupy');
  expect(previewOperation({ benchmark: { public_api: 'KernelDensity.score_samples' }, backend: 'numpy' })).toBe('KernelDensity.score_samples · numpy');
  const value = parseDiagnosticPreview(JSON.stringify(fixture())).records[0];
  value.summary_seconds = 1e308;
  expect(previewDuration(value)).toBe('No positive timing summary');
});

test('local preview loads alongside canonical data, filters, clears, rejects a replacement, and recovers', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.header')).toBeVisible();
  const canonicalCount = await page.locator('.overview-table-title').textContent();
  await page.getByText('Local benchmark update preview', { exact: true }).click();
  const input = page.locator('#preview-file');
  const payload = { name: 'preview_data.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(fixture())) };
  const uploads: string[] = [];
  page.on('request', request => { if (request.method() !== 'GET') uploads.push(request.url()); });
  await input.setInputFiles(payload);
  await expect(page.locator('#preview-output')).toContainText('3 planned jobs · 2 recorded timing fields');
  await expect(page.locator('#preview-output')).toContainText('p=0 (not applicable)');
  await expect(page.locator('#preview-output')).toContainText('Unspecified scope (not fit)');
  await expect(page.locator('#preview-output')).toContainText('true; effective: false');
  await expect(page.locator('#preview-output')).toContainText('recorded, not machine verification');
  await page.locator('#preview-job-filter').selectOption('job-blocked');
  await expect(page.locator('#preview-records')).toContainText('No recorded timing fields');
  await expect(page.locator('#preview-output')).toContainText('blocked-lane');
  await page.locator('#preview-job-filter').selectOption('');
  await page.locator('#preview-search').fill('classification_metric');
  await expect(page.locator('#preview-records h3')).toContainText('(1 of 1)');
  await page.getByRole('button', { name: 'Clear local preview' }).click();
  await expect(page.locator('#preview-output')).toBeEmpty();
  await input.setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"schema_version":"invalid"}') });
  await expect(page.locator('#diagnostic-preview [role=status]')).toContainText('Preview not loaded');
  await input.setInputFiles(payload);
  await expect(page.locator('#preview-output')).toContainText('2 recorded timing fields');
  expect(await page.locator('.overview-table-title').textContent()).toBe(canonicalCount);
  expect(uploads).toEqual([]);
});

test('preview renders untrusted artifact text as text, not HTML', async ({ page }) => {
  const value = fixture(); value.records[0].context.operation = '<img src=x onerror="alert(1)">';
  await page.goto('/');
  await page.getByText('Local benchmark update preview', { exact: true }).click();
  await page.locator('#preview-file').setInputFiles({ name: 'safe.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(value)) });
  await expect(page.locator('#preview-output')).toContainText('<img src=x');
  await expect(page.locator('#preview-output img')).toHaveCount(0);
});

test('preview remains reachable if canonical bundle cannot load', async ({ page }) => {
  await page.route('**/data/benchmark_data.json', route => route.fulfill({ status: 503, body: 'unavailable' }));
  await page.goto('/');
  await page.getByText('Local benchmark update preview', { exact: true }).click();
  await page.locator('#preview-file').setInputFiles({ name: 'preview_data.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(fixture())) });
  await expect(page.locator('#preview-output')).toContainText('2 recorded timing fields');
});
