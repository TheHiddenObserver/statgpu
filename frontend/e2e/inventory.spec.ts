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

  test('renders measured rows and preserves the explicit Torch failure', async ({ page }) => {
    const response = await page.request.get('/data/benchmark_data.json');
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    const cvRuns = data.runs.filter((run: { metrics?: { cross_validation?: unknown } }) => run.metrics?.cross_validation);
    expect(cvRuns).toHaveLength(22);
    const failed = cvRuns.filter((run: { metrics: { cross_validation: { status: string } } }) => run.metrics.cross_validation.status === 'failed');
    expect(failed).toHaveLength(1);
    expect(failed[0].model_id).toBe('LogisticRegressionCV');
    expect(failed[0].backend).toBe('torch');
    expect(failed[0].metrics.cross_validation.reason).toContain('CPU fallback is disabled');
    expect(failed[0].metrics.timing).toBeUndefined();

    await page.locator('#env-select').selectOption('remote-p100-cv-20260807');
    const cvScope = page.locator('[data-metric-scope="cross_validation"]');
    await expect(cvScope).toBeEnabled();
    await expect(cvScope).toContainText(/CV \([1-9]\d*\)/);
    await cvScope.click();

    const toggle = page.getByText(/Cross-validation Metrics \(\d+\)/);
    await expect(toggle).toBeVisible();
    await toggle.click();
    const panel = toggle.locator('..');
    const failedRow = panel.locator('tr').filter({ hasText: 'LogisticRegressionCV' }).filter({ hasText: 'torch' });
    await expect(failedRow).toHaveCount(1);
    await expect(failedRow).toContainText('failed');
    await expect(failedRow).toContainText('CPU fallback is disabled');
  });
});
