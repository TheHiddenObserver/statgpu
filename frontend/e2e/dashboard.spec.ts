import { test, expect } from '@playwright/test';

test.describe('Benchmark Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  test('dashboard loads with header, sidebar, charts, and table', async ({ page }) => {
    await expect(page.locator('.header')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
    await expect(page.locator('#timing-chart')).toBeVisible();
    await expect(page.locator('#speedup-chart')).toBeVisible();
    await expect(page.locator('.table-container')).toBeVisible();
    await expect(page.locator('.summary-cards')).toBeVisible();
  });

  test('charts default to focused mode and can expose the full matrix', async ({ page }) => {
    const focused = page.getByRole('button', { name: 'Focused' });
    const full = page.getByRole('button', { name: 'Full matrix' });
    await expect(focused).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#timing-chart')).toHaveAttribute('data-chart-view', 'focused');

    await full.click();
    await expect(full).toHaveAttribute('aria-pressed', 'true');
    await expect(focused).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('#timing-chart')).toHaveAttribute('data-chart-view', 'full');
  });

  test('speedup chart exposes a dashed 1x parity reference', async ({ page }) => {
    const chart = page.locator('#speedup-chart');
    await expect(chart).toHaveAttribute('data-parity-style', 'dashed');
    await expect(chart).toHaveAttribute('aria-label', /dashed 1× parity line/);
  });

  test('category filter — clear all shows empty state, re-select restores data', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await expect(page.getByText(/No runs match/i)).toBeVisible({ timeout: 5000 });

    await page.locator('#cat-penalized_glm').check();
    await expect(page.locator('.table-container tbody tr').first()).toBeVisible({ timeout: 5000 });
  });

  test('progressive filter — model selection reveals penalty dropdown', async ({ page }) => {
    await page.locator('.filter-bar select').first().selectOption({ index: 1 });
    await expect(page.getByText('Penalty:')).toBeVisible();
    const penaltySelect = page.locator('.filter-bar select').nth(1);
    if ((await penaltySelect.count()) > 0) {
      await penaltySelect.selectOption({ index: 1 });
    }
  });

  test('scale chips are multi-selectable with .active class', async ({ page }) => {
    const chips = page.locator('.scale-chip');
    const count = await chips.count();
    if (count >= 2) {
      await chips.first().click();
      await expect(chips.first()).toHaveClass(/active/);
      await chips.nth(1).click();
      await expect(chips.first()).toHaveClass(/active/);
      await expect(chips.nth(1)).toHaveClass(/active/);
    }
  });

  test('table pagination — Show all and Show first 200', async ({ page }) => {
    const showAllBtn = page.getByText('Show all', { exact: false });
    if (await showAllBtn.isVisible()) {
      await showAllBtn.click();
      await expect(page.getByText('Show first 200', { exact: false })).toBeVisible({ timeout: 5000 });
      await page.getByText('Show first 200', { exact: false }).click();
      await expect(page.getByText('Show all', { exact: false })).toBeVisible({ timeout: 5000 });
    }
  });

  test('current external framework checkboxes toggle table visibility', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-robust_quantile').check();

    const sklearnCheckbox = page.locator('input[value="sklearn"]');
    await expect(sklearnCheckbox).toBeVisible();
    await expect(sklearnCheckbox).not.toBeChecked();

    await sklearnCheckbox.check();
    await expect(sklearnCheckbox).toBeChecked();
    await expect(page.locator('.table-container')).toContainText('sklearn', { timeout: 5000 });

    await sklearnCheckbox.uncheck();
    await expect(sklearnCheckbox).not.toBeChecked();
    await expect(page.locator('.table-container')).not.toContainText('sklearn', { timeout: 5000 });
  });

  test('removed April frameworks are not offered', async ({ page }) => {
    for (const framework of ['glmnet', 'lifelines', 'scikit_survival', 'knockpy']) {
      await expect(page.locator(`input[value="${framework}"]`)).toHaveCount(0);
    }
  });

  test('table header click toggles sort direction', async ({ page }) => {
    const modelHeader = page.getByRole('columnheader', { name: /Model/ });
    await modelHeader.click();
    await expect(modelHeader).toContainText('▲');
    await modelHeader.click();
    await expect(modelHeader).toContainText('▼');
  });

  test('backend radio — select numpy only', async ({ page }) => {
    const numpyRadio = page.locator('input[value="numpy"]');
    await numpyRadio.check();
    await expect(numpyRadio).toBeChecked();
    const allRadio = page.locator('input[value="all"]');
    await allRadio.check();
    await expect(allRadio).toBeChecked();
  });

  test('environment selector is present with options', async ({ page }) => {
    const envSelect = page.locator('#env-select');
    await expect(envSelect).toBeVisible();
    const options = await envSelect.locator('option').count();
    expect(options).toBeGreaterThan(0);
  });

  test('summary cards show global statistics', async ({ page }) => {
    const cards = page.locator('.summary-card');
    await expect(cards).toHaveCount(6);
    await expect(cards.first().locator('.card-value')).not.toBeEmpty();
  });

  test('changing model clears incompatible scale filters', async ({ page }) => {
    const modelSelect = page.locator('.filter-bar select').first();

    await modelSelect.selectOption('PenalizedLinearRegression');
    const chip = page.locator('.scale-chip').first();
    if (await chip.count() > 0) {
      await chip.click();
      await expect(chip).toHaveClass(/active/);
    }

    await modelSelect.selectOption('PenalizedLogisticRegression');
    await expect(page.locator('.scale-chip.active')).toHaveCount(0);
    await expect(page.locator('.table-container tbody tr').first()).toBeVisible({ timeout: 5000 });
  });
});
