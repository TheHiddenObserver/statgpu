import { test, expect } from '@playwright/test';

test.describe('Audited source inventory', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('uses literal inventory-v2 labels', async ({ page }) => {
    const inventory = page.locator('.inventory-meta');
    await expect(inventory).toBeVisible();
    await expect(inventory).toContainText(/\d+ registered/);
    await expect(inventory).toContainText(/\d+ eligible/);
    await expect(inventory).toContainText(/\d+ non-ready/);
    await expect(inventory).toContainText(/\d+ historical\/excluded/);
    await expect(inventory).toHaveAttribute('title', /discovered JSON artifacts/);
    await expect(inventory).toHaveAttribute('title', /unclassified/);
  });

  test('links the inventory, catalog policy, and coverage matrix', async ({ page }) => {
    await expect(page.getByRole('link', { name: 'Source inventory (JSON)' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Catalog policy' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Coverage matrix' })).toBeVisible();
    await expect(page.locator('.dashboard-footer')).toContainText('Inventory 2.0');
  });
});


test.describe('Canonical cross-validation evidence', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('preserves the pre-fix Torch failure and exposes the PR116 success', async ({ page }) => {
    const response = await page.request.get('/data/benchmark_data.json');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    const cvRuns = data.runs.filter((run: { metrics?: { cross_validation?: unknown } }) => run.metrics?.cross_validation);
    expect(cvRuns).toHaveLength(44);

    const historicalRuns = cvRuns.filter(
      (run: { comparison_id: string }) => run.comparison_id === 'cv-benchmark-20260807',
    );
    const repairedRuns = cvRuns.filter(
      (run: { comparison_id: string }) => run.comparison_id === 'cv-benchmark-pr116-20260807',
    );
    expect(historicalRuns).toHaveLength(22);
    expect(repairedRuns).toHaveLength(22);

    const failed = cvRuns.filter(
      (run: { metrics: { cross_validation: { status: string } } }) =>
        run.metrics.cross_validation.status === 'failed',
    );
    expect(failed).toHaveLength(1);
    expect(failed[0].comparison_id).toBe('cv-benchmark-20260807');
    expect(failed[0].model_id).toBe('LogisticRegressionCV');
    expect(failed[0].backend).toBe('torch');
    expect(failed[0].metrics.cross_validation.reason).toContain('CPU fallback is disabled');
    expect(failed[0].metrics.timing).toBeUndefined();

    const repairedTorch = repairedRuns.filter(
      (run: { model_id: string; backend: string | null }) =>
        run.model_id === 'LogisticRegressionCV' && run.backend === 'torch',
    );
    expect(repairedTorch).toHaveLength(1);
    expect(repairedTorch[0].metrics.cross_validation.status).toBe('success');
    expect(repairedTorch[0].metrics.cross_validation.selected_parameters).toEqual({ C: 0.1 });
    expect(repairedTorch[0].metrics.timing.fit_time_ms).toBeGreaterThan(0);

    // Session-level environments are intentionally grouped into one hardware
    // selector. The CV panel must therefore retain session/source identity so
    // historical and repaired evidence remain distinguishable.
    await expect(page.locator('#env-select')).toHaveValue('remote-p100');
    await expect(page.locator('#env-select option')).toHaveCount(1);
    await expect(page.locator('#env-select option')).toContainText(
      '8 benchmark sessions',
    );

    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-linear_models').check();
    const cvScope = page.locator('[data-metric-scope="cross_validation"]');
    await expect(cvScope).toBeEnabled();
    await expect(cvScope).toContainText(/CV \([1-9]\d*\)/);
    await cvScope.click();
    await page.getByLabel('Model', { exact: true }).selectOption('LogisticRegressionCV');

    const toggle = page.getByRole('button', { name: /Cross-validation Metrics/ });
    await expect(toggle).toBeVisible();
    await toggle.click();
    const panelBodyId = await toggle.getAttribute('aria-controls');
    expect(panelBodyId).toBeTruthy();
    const panel = page.locator(`#${panelBodyId}`);

    const torchRows = panel
      .locator('tr')
      .filter({ hasText: 'LogisticRegressionCV' })
      .filter({ hasText: 'torch' });
    await expect(torchRows).toHaveCount(2);

    const failedRow = torchRows.filter({ hasText: 'failed' });
    await expect(failedRow).toHaveCount(1);
    await expect(failedRow).toContainText('CPU fallback is disabled');
    await expect(failedRow).toContainText('remote-p100-cv-20260807');
    await expect(failedRow).toContainText('cv_benchmark_20260807.json');

    const repairedRow = torchRows.filter({ hasText: 'success' });
    await expect(repairedRow).toHaveCount(1);
    await expect(repairedRow).not.toContainText('CPU fallback is disabled');
    await expect(repairedRow).toContainText('remote-p100-pr116-20260807');
    await expect(repairedRow).toContainText('cv_benchmark_pr116_p100.json');
  });
});


test.describe('Grouped physical validation evidence', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('keeps session and source provenance in the Validation panel', async ({ page }) => {
    await expect(page.locator('#env-select')).toHaveValue('remote-p100');
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-panel').check();

    const toggle = page.getByRole('button', { name: /Validation Checks/ });
    await expect(toggle).toBeVisible();
    await toggle.click();
    const panelBodyId = await toggle.getAttribute('aria-controls');
    expect(panelBodyId).toBeTruthy();
    const panel = page.locator(`#${panelBodyId}`);

    await expect(panel.getByRole('columnheader', { name: 'Benchmark session' })).toBeVisible();
    await expect(panel.getByRole('columnheader', { name: 'Source' })).toBeVisible();
    // The panel shows the first 30 rows by default. Assert provenance on the
    // canonical PR122 evidence that is intentionally visible without changing
    // pagination state; later PR126 rows are available through "Show all".
    await expect(panel).toContainText('panel_stage_b_pr122_p100_20260809_2701aa9f.json');
    await expect(panel).toContainText('remote-p100-pr122-20260809-panel-stage-b-pr122');
  });
});
