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

  test('metric scope exposes current inference and reserves the CV frontend contract', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-penalized_glm').check();

    const inference = page.locator('[data-metric-scope="inference"]');
    const cv = page.locator('[data-metric-scope="cross_validation"]');
    await expect(inference).toBeEnabled();
    await expect(cv).toBeDisabled();
    await expect(cv).toContainText('CV (0)');

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

  test('nonparametric GAM exposes both comparison variants and all scales', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-nonparametric').check();

    const modelSelect = page.locator('.filter-bar select').first();
    await modelSelect.selectOption('GAM');

    const variantSelect = page.locator('.filter-bar select').nth(1);
    await expect(
      variantSelect.locator('option[value="pygam-comparison"]'),
    ).toHaveCount(1);
    await expect(
      variantSelect.locator('option[value="aligned-pygam"]'),
    ).toHaveCount(1);

    const chips = page.locator('.scale-chip');
    await expect(chips.filter({ hasText: '1K×3' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '10K×5' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '100K×10' })).toHaveCount(1);
    await expect(page.locator('input[value="pygam"]')).toBeVisible();
  });

  test('unsupervised PCA exposes the complete and correctly labelled scale matrix', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-unsupervised').check();

    await page.locator('.filter-bar select').first().selectOption('PCA');
    const chips = page.locator('.scale-chip');
    await expect(chips.filter({ hasText: '1K×20' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '10K×50' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '100K×50' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '100K×100' })).toHaveCount(0);
    await expect(page.locator('input[value="sklearn"]')).toBeVisible();
  });

  test('unsupervised DBSCAN exposes both dimensional variants', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-unsupervised').check();

    await page.locator('.filter-bar select').first().selectOption('DBSCAN');
    const variantSelect = page.locator('.filter-bar select').nth(1);
    await expect(variantSelect.locator('option[value="10d"]')).toHaveCount(1);
    await expect(variantSelect.locator('option[value="50d"]')).toHaveCount(1);
    await variantSelect.selectOption('10d');

    const chips = page.locator('.scale-chip');
    await expect(chips.filter({ hasText: '1K×10' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '10K×10' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '100K×10' })).toHaveCount(1);
  });

  test('panel models expose both aligned source scales', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-panel').check();

    const modelSelect = page.locator('.filter-bar select').first();
    await modelSelect.selectOption('PanelOLS');

    const chips = page.locator('.scale-chip');
    await expect(chips.filter({ hasText: '10K×10' })).toHaveCount(1);
    await expect(chips.filter({ hasText: '100K×20' })).toHaveCount(1);
    await expect(page.locator('input[value="linearmodels"]')).toBeVisible();
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
