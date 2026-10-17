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

  test('unpenalized models hide an empty penalty selector and expose solver', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-robust_quantile').check();
    await page.locator('.filter-bar select').first().selectOption('QuantileRegression');

    await expect(page.getByText('Penalty:', { exact: true })).toHaveCount(0);
    const solverLabel = page.getByText('Solver:', { exact: true });
    await expect(solverLabel).toBeVisible();
    const solverSelect = solverLabel.locator('xpath=following-sibling::select[1]');
    await expect(solverSelect.locator('option[value="irls"]')).toHaveCount(1);
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

  test('metric scope exposes current inference and CV evidence', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-penalized_glm').check();

    const inference = page.locator('[data-metric-scope="inference"]');
    const cv = page.locator('[data-metric-scope="cross_validation"]');
    await expect(inference).toBeEnabled();
    await expect(cv).toBeEnabled();
    await expect(cv).toContainText(/CV \([1-9]\d*\)/);

    await cv.click();
    await expect(cv).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText(/Cross-validation Metrics \(\d+\)/)).toBeVisible();
    const cvScopeCells = page.locator('.table-container tbody tr td:nth-child(2)');
    await expect(cvScopeCells.first()).toContainText('Cross-validation');
    const cvScopes = await cvScopeCells.allTextContents();
    expect(cvScopes.length).toBeGreaterThan(0);
    expect(
      cvScopes.every(value => value.includes('Cross-validation')),
    ).toBeTruthy();

    await inference.click();
    await expect(inference).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText(/Inference Metrics \(\d+\)/)).toBeVisible();

    const scopeCells = page.locator('.table-container tbody tr td:nth-child(2)');
    await expect(scopeCells.first()).toContainText('Inference');
    const allScopes = await scopeCells.allTextContents();
    expect(allScopes.length).toBeGreaterThan(0);
    expect(allScopes.every(value => value.includes('Inference'))).toBeTruthy();

    const panelTop = await page.getByText(/Inference Metrics \(\d+\)/).boundingBox();
    const tableTitleTop = await page.locator('.overview-table-title').boundingBox();
    expect(panelTop).not.toBeNull();
    expect(tableTitleTop).not.toBeNull();
    expect(panelTop!.y).toBeLessThan(tableTitleTop!.y);
  });

  test('PR74 exposes sandwich, oracle, and bootstrap inference configurations', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-penalized_glm').check();

    const modelSelect = page.locator('.filter-bar select').first();
    await modelSelect.selectOption('PenalizedLogisticRegression');
    const logisticVariant = page.locator('.filter-bar select').nth(1);
    await expect(
      logisticVariant.locator('option[value="hc0-sandwich"]'),
    ).toHaveCount(1);
    await expect(
      logisticVariant.locator('option[value="oracle-inference"]'),
    ).toHaveCount(1);

    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-linear_models').check();
    await page
      .locator('.filter-bar select')
      .first()
      .selectOption('PenalizedLinearRegression');
    const linearVariant = page.locator('.filter-bar select').nth(1);
    await expect(
      linearVariant.locator('option[value="bootstrap-inference"]'),
    ).toHaveCount(1);
    await linearVariant.selectOption('bootstrap-inference');
    await expect(page.locator('.table-container')).toContainText(
      'ordered_inference_pr74.json',
      { timeout: 5000 },
    );
  });
});
