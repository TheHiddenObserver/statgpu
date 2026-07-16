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

  test('sidebar category labels use the English dashboard locale', async ({ page }) => {
    await expect(page.locator('label[for="cat-survival"]')).toHaveText('Survival Analysis');
    await expect(page.locator('label[for="cat-robust_quantile"]')).toHaveText('Robust/Quantile');
    await expect(page.locator('label[for="cat-linear_models"]')).toHaveText('Linear Models');
    await expect(page.locator('#category-list')).not.toContainText('生存分析');
  });

  test('focused charts use a representative subset and full matrix expands it', async ({ page }) => {
    const focused = page.getByRole('button', { name: 'Focused' });
    const full = page.getByRole('button', { name: 'Full matrix' });
    const timingChart = page.locator('#timing-chart');
    const speedupChart = page.locator('#speedup-chart');

    await expect(focused).toHaveAttribute('aria-pressed', 'true');
    await expect(timingChart).toHaveAttribute('data-chart-view', 'focused');
    await expect(speedupChart).toHaveAttribute('data-chart-view', 'focused');

    const focusedRows = Number(await speedupChart.getAttribute('data-speedup-rows'));
    expect(focusedRows).toBeGreaterThan(0);

    await full.click();
    await expect(full).toHaveAttribute('aria-pressed', 'true');
    await expect(focused).toHaveAttribute('aria-pressed', 'false');
    await expect(timingChart).toHaveAttribute('data-chart-view', 'full');
    await expect(speedupChart).toHaveAttribute('data-chart-view', 'full');

    const fullRows = Number(await speedupChart.getAttribute('data-speedup-rows'));
    expect(fullRows).toBeGreaterThanOrEqual(focusedRows);
  });

  test('speedup chart exposes a dashed parity line with its label near the axis', async ({ page }) => {
    const chart = page.locator('#speedup-chart');
    await expect(chart).toHaveAttribute('data-parity-style', 'dashed');
    await expect(chart).toHaveAttribute('data-parity-label-placement', 'axis-bottom');
    await expect(chart).toHaveAttribute('aria-label', /labeled near the horizontal axis/);
  });

  test('speedup tooltip is confined and docked away from method labels', async ({ page }) => {
    const chart = page.locator('#speedup-chart');
    await expect(chart).toHaveAttribute('data-tooltip-placement', 'opposite-corner');
    await expect(chart).toHaveAttribute('aria-label', /tooltip is confined to the chart and docked away from labels/);
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

  test('summary cards use compact, self-explanatory global statistics', async ({ page }) => {
    const cards = page.locator('.summary-card');
    await expect(cards).toHaveCount(6);
    await expect(page.getByText('Benchmark runs', { exact: true })).toBeVisible();
    await expect(page.getByText('Sources parsed', { exact: true })).toBeVisible();
    await expect(page.getByText('Benchmark categories', { exact: true })).toBeVisible();
    await expect(page.getByText('Fastest reported GPU speedup', { exact: true })).toBeVisible();
    await expect(page.getByText('External references', { exact: true })).toBeVisible();
    await expect(page.getByText('Build mode', { exact: true })).toBeVisible();
    await expect(page.getByText('Fastest GPU speedup · computed / reported')).toHaveCount(0);
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
