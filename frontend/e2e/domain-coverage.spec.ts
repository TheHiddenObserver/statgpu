import { test, expect } from '@playwright/test';

const categories = [
  'robust_quantile',
  'survival',
  'unsupervised',
  'ordered',
  'nonparametric',
  'panel',
  'covariance',
  'anova',
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

  test('survival benchmarks expose the CoxPH Breslow variant', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-survival').check();

    const selects = page.locator('.filter-bar select');
    await selects.first().selectOption('CoxPH');
    await expect(page.getByText('Variant:', { exact: true })).toBeVisible();

    const variantSelect = page.locator('.filter-bar select').nth(1);
    await expect(variantSelect.locator('option[value="breslow"]')).toHaveCount(1);
    await variantSelect.selectOption('breslow');

    await expect(page.locator('.table-container')).toContainText(
      'loss_functions_20260623.json',
      { timeout: 5000 },
    );
    await expect(page.locator('input[value="statsmodels"]')).toBeVisible();
  });

  test('ordered benchmarks expose inference metrics', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-ordered').check();
    await expect(page.getByText(/Inference Metrics \(\d+\)/)).toBeVisible({
      timeout: 5000,
    });
  });

  test('ANOVA includes all functions and SciPy reference rows', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-anova').check();

    const table = page.locator('.table-container');
    await expect(table).toContainText('One Way ANOVA', { timeout: 5000 });
    await expect(table).toContainText('Two Way ANOVA', { timeout: 5000 });
    await expect(table).toContainText('Welch ANOVA', { timeout: 5000 });
    await expect(page.locator('input[value="scipy"]')).toBeVisible();
  });

  test('speedup summary uses the runner-reported headline', async ({ page }) => {
    const card = page.locator('.summary-card').filter({
      hasText: 'Fastest reported GPU speedup',
    });
    await expect(card).toBeVisible();
    await expect(card).toContainText('Ⓡ');
    await expect(card).not.toContainText('/');
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
