import { test, expect } from '@playwright/test';

test.describe('Benchmark data provider contract', () => {
  test('fails closed for an unsupported benchmark schema', async ({ page }) => {
    await page.route('**/data/benchmark_data.json', async route => {
      const response = await route.fetch();
      const body = await response.json();
      body.schema_version = '99.0.0';
      await route.fulfill({ response, json: body });
    });

    await page.goto('/');
    await expect(page.getByText(/Unsupported schema 99\.0\.0; expected 1\.1\.0/)).toBeVisible();
    await expect(page.locator('.header')).toHaveCount(0);
  });

  test('keeps valid data when optional metadata has another generation', async ({ page }) => {
    await page.route('**/data/parse_report.json', async route => {
      const response = await route.fetch();
      const body = await response.json();
      body.generation_id = '0'.repeat(64);
      await route.fulfill({ response, json: body });
    });

    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('.header-meta')).toHaveCount(1);
    await expect(page.locator('#timing-chart')).toBeVisible();
  });
});
