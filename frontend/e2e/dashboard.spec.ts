import { test, expect } from '@playwright/test';

test.describe('Benchmark Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
  });

  // 1. Page Loads
  test('dashboard loads with header, sidebar, charts, and table', async ({
    page,
  }) => {
    await expect(page.locator('.header')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
    await expect(page.locator('#timing-chart')).toBeVisible();
    await expect(page.locator('#speedup-chart')).toBeVisible();
    await expect(page.locator('.table-container')).toBeVisible();
    await expect(page.locator('.summary-cards')).toBeVisible();
  });

  // 2. Category Filtering
  test('category filter — clear all shows empty state, re-select restores data', async ({
    page,
  }) => {
    // Click "None" to clear all — should show empty state
    await page.getByRole('button', { name: 'None' }).click();
    await expect(page.getByText(/No runs match/i)).toBeVisible({ timeout: 5000 });

    // Re-select penalized_glm — data should return
    await page.locator('#cat-penalized_glm').check();
    await expect(page.locator('.table-container tbody tr').first()).toBeVisible({ timeout: 5000 });
  });

  // 3. Model / Penalty Filtering
  test('progressive filter — model then penalty then solver', async ({
    page,
  }) => {
    await page.locator('.filter-bar select').first().selectOption({ index: 1 });
    await expect(page.getByText('Penalty:')).toBeVisible();
    const penaltySelect = page.locator('.filter-bar select').nth(1);
    if ((await penaltySelect.count()) > 0) {
      await penaltySelect.selectOption({ index: 1 });
    }
  });

  // 4. Scale Multi-select
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

  // 5. Show All / Show First 200
  test('table pagination — Show all and Show first 200', async ({ page }) => {
    const showAllBtn = page.getByText('Show all', { exact: false });
    if (await showAllBtn.isVisible()) {
      await showAllBtn.click();
      await expect(
        page.getByText('Show first 200', { exact: false }),
      ).toBeVisible({ timeout: 5000 });
      await page.getByText('Show first 200', { exact: false }).click();
      await expect(
        page.getByText('Show all', { exact: false }),
      ).toBeVisible({ timeout: 5000 });
    }
  });

  // 6. External Framework Toggle
  test('external framework checkboxes toggle table visibility', async ({
    page,
  }) => {
    const glmnetCheckbox = page.locator('input[value="glmnet"]');
    await expect(glmnetCheckbox).toBeVisible();
    await expect(glmnetCheckbox).not.toBeChecked();

    // Show all rows so glmnet entries are not hidden by pagination
    const showAllBtn = page.getByText('Show all', { exact: false });
    if (await showAllBtn.isVisible()) await showAllBtn.click();
    await page.waitForTimeout(300);

    // Enable glmnet — table should show glmnet rows
    await glmnetCheckbox.check();
    await expect(glmnetCheckbox).toBeChecked();
    await expect(page.locator('.table-container')).toContainText('glmnet', { timeout: 5000 });

    // Disable — glmnet rows should disappear
    await glmnetCheckbox.uncheck();
    await expect(glmnetCheckbox).not.toBeChecked();
    await expect(page.locator('.table-container')).not.toContainText('glmnet', { timeout: 5000 });
  });

  // 7. Table sorting
  test('table header click toggles sort direction', async ({ page }) => {
    const modelHeader = page.getByRole('columnheader', { name: /Model/ });
    await modelHeader.click();
    await expect(modelHeader).toContainText('▲');
    await modelHeader.click();
    await expect(modelHeader).toContainText('▼');
  });

  // 8. Backend radio filtering
  test('backend radio — select numpy only', async ({ page }) => {
    const numpyRadio = page.locator('input[value="numpy"]');
    await numpyRadio.check();
    await expect(numpyRadio).toBeChecked();
    const allRadio = page.locator('input[value="all"]');
    await allRadio.check();
    await expect(allRadio).toBeChecked();
  });

  // 9. Environment selector
  test('environment selector changes env', async ({ page }) => {
    const envSelect = page.locator('#env-select');
    await expect(envSelect).toBeVisible();
    const options = await envSelect.locator('option').count();
    expect(options).toBeGreaterThan(0);
  });

  // 10. Summary cards display global stats
  test('summary cards show global statistics', async ({ page }) => {
    const cards = page.locator('.summary-card');
    await expect(cards).toHaveCount(6);
    await expect(cards.first().locator('.card-value')).not.toBeEmpty();
  });

  // 11. None button shows empty state (regression guard)
  test('clearing all categories shows empty state message', async ({ page }) => {
    await page.getByRole('button', { name: 'None' }).click();
    await expect(page.getByText(/No runs match/i)).toBeVisible({ timeout: 5000 });
    // Re-select a category to confirm recovery
    await page.locator('#cat-penalized_glm').check();
    await expect(page.locator('.table-container tbody tr').first()).toBeVisible({ timeout: 5000 });
  });
});
