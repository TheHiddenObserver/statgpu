import { expect, test } from '@playwright/test';
import type { BenchmarkData, Run } from '../src/schema';

// Browser interaction coverage; pure helpers are covered separately. This test
// must run in a browser-capable environment, not be represented as pure-test QA.
const run: Run = {
  run_id: 'display-run', env_id: 'test', category_ids: ['diagnostic_evidence'], model_id: 'model',
  framework: 'statgpu', backend: 'numpy', comparison_id: 'comparison', case_id: 'case', method_config_id: 'method',
  scale: { scale_key: 'n10_p2', n_samples: 10, n_features: 2, label: '10×2' },
  source: { source_id: 'display-source', file: 'preview.json', original_path: '/original/preview.json', date: '2026-10-25', parser: 'qualification_preview', parser_version: '1.0' },
  parameters: { raw_path: '<img src=x onerror=alert(1)>', raw_sha256: 'b'.repeat(64) },
  metrics: { operation_timing: { duration_ms: 3, scope: 'fit', statistic: 'median', unit: 'ms', quality: 'measured', source_file: 'preview.json' } },
  evidence: { qualification: 'diagnostic_only', correctness_status: 'complete', validation_status: 'validated_diagnostic', performance_eligible: false, canonical_dashboard_ready: false, machine_verified_exclusivity: false, raw_pointer: '/rows/0/timing', source_commit: 'c'.repeat(40), cpu_threads_requested: 1, cpu_threads_observed: [1, 1] },
};
const data: BenchmarkData = {
  schema_version: '1.2.0', generated: '2026-10-25', meta: { generator: 'test', git_sha: 'renderer-test', generation_id: 'test' },
  environments: [{ env_id: 'test', label: 'diagnostic', cpu: 'recorded', gpu: 'recorded' }],
  categories: [{ category_id: 'diagnostic_evidence', name_en: 'Diagnostic', name_zh: '诊断' }],
  models: [{ model_id: 'model', primary_category_id: 'diagnostic_evidence', category_ids: ['diagnostic_evidence'] }],
  frameworks: [
    { framework_id: 'statgpu', display_name: 'statgpu', external: false, backend_policy: 'required' },
    { framework_id: 'sklearn', display_name: 'sklearn', external: true, backend_policy: 'forbidden' },
  ], comparisons: [], runs: [run, { ...run, run_id: 'external-run', framework: 'sklearn', backend: null, parameters: { ...run.parameters, external_adapter: 'sklearn-scoped-cv-v2' } }],
};

test('thread selector and expandable raw provenance survive repeated filters without unsafe links', async ({ page }) => {
  await page.route('**/data/benchmark_data.json', route => route.fulfill({ json: data }));
  await page.route('**/data/parse_report.json', route => route.fulfill({ status: 404 }));
  await page.route('**/data/source_inventory.json', route => route.fulfill({ status: 404 }));
  await page.goto('/');
  const threads = page.getByRole('combobox', { name: 'Host CPU thread limits' });
  await expect(threads).toHaveValue('all');
  await expect(threads.locator('option[value="multi"]')).toBeDisabled();
  await threads.selectOption('single');
  await page.getByRole('checkbox', { name: 'sklearn', exact: true }).check();
  await expect(page.locator('.overview-table-title')).toHaveText('Showing 2 of 2 runs');
  await page.getByRole('radio', { name: 'cupy', exact: true }).check();
  await expect(page.locator('.overview-table-title')).toHaveText('Showing 1 of 1 runs');
  await page.getByRole('radio', { name: 'All', exact: true }).check();
  await threads.selectOption('all');
  await threads.selectOption('single');
  const source = page.locator('.source-provenance').first();
  await source.getByText('Full source / raw provenance').click();
  await expect(source).toContainText('/rows/0/timing');
  await expect(source).toContainText('<img src=x onerror=alert(1)>');
  await expect(source.locator('img, a')).toHaveCount(0);
  await source.getByText('Full source / raw provenance').click();
  await expect(source.locator('details')).not.toHaveAttribute('open', '');
});
