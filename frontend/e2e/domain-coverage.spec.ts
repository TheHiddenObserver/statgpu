import { test, expect } from '@playwright/test';

const categories = [
  'robust_quantile',
  'unsupervised',
  'ordered',
  'nonparametric',
  'panel',
  'covariance',
];

test.describe('Benchmark domain coverage', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('all published benchmark domains render rows', async ({ page }) => {
    for (const category of categories) {
      await page.getByRole('button', { name: 'None' }).click();
      await page.locator(`#cat-${category}`).check();
      await expect(page.locator('.table-container tbody tr').first()).toBeVisible({
        timeout: 5000,
      });
    }
  });

  test('ordered benchmarks expose inference metrics', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-ordered').check();
    await expect(page.getByText(/Inference Metrics \(\d+\)/)).toBeVisible({
      timeout: 5000,
    });
  });

  test('linear models include the June 2026 GLM benchmark sources', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-linear_models').check();

    const modelSelect = page.locator('.filter-bar select').first();
    await modelSelect.selectOption('PenalizedLinearRegression');
    await expect(page.locator('.table-container')).toContainText(
      /penalized_glm_perf_20260622\.json|glm_solver_20260623\.json/,
      { timeout: 5000 },
    );
  });
});
