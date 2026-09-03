import { test, expect, type Page } from '@playwright/test';

async function openProduction(page: Page): Promise<void> {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('./');
  await expect(page.locator('.header')).toBeVisible({ timeout: 60_000 });
  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
}

test.describe('Deployed benchmark dashboard', () => {
  test('loads from the project Pages path with relative assets and metadata', async ({ page }) => {
    const failedResponses: string[] = [];
    page.on('response', response => {
      if (response.status() >= 400 && !response.url().endsWith('/favicon.ico')) {
        failedResponses.push(`${response.status()} ${response.url()}`);
      }
    });
    await openProduction(page);

    expect(new URL(page.url()).pathname).toBe('/statgpu/dashboard/');
    await expect(page.locator('html')).toHaveAttribute('lang', 'en');
    await expect(page.locator('#timing-chart')).toHaveAttribute('role', 'img');
    await expect(page.locator('#speedup-chart')).toHaveAttribute('role', 'img');
    await expect(page.getByRole('link', { name: 'Benchmark guide' })).toHaveAttribute(
      'href',
      'http://127.0.0.1:4173/statgpu/en/guides/benchmarks',
    );
    await expect(page.getByRole('link', { name: 'Raw data (JSON)' })).toHaveAttribute(
      'href',
      'data/benchmark_data.json',
    );

    const raw = await page.request.get(
      new URL('data/benchmark_data.json', page.url()).toString(),
    );
    expect(raw.ok()).toBeTruthy();
    expect(failedResponses).toEqual([]);
  });

  test('supports the CV filter cascade and deterministic upstream reset', async ({ page }) => {
    await openProduction(page);
    await page.locator('#env-select').selectOption('remote-p100-cv-20260807');
    await page.getByRole('button', { name: 'None' }).click();
    await page.locator('#cat-linear_models').check();
    await page.locator('[data-metric-scope="cross_validation"]').click();

    const model = page.getByLabel('Model', { exact: true });
    await model.selectOption('RidgeCV');
    const variant = page.getByLabel('Variant', { exact: true });
    await expect(variant).toBeVisible();
    await variant.selectOption({ index: 1 });
    await page.getByLabel('Penalty', { exact: true }).selectOption('l2');
    await page.getByLabel('Solver', { exact: true }).selectOption('cv');

    const scale = page.locator('.scale-chip').first();
    await expect(scale).toHaveAttribute('role', 'button');
    await scale.focus();
    await page.keyboard.press('Space');
    await expect(scale).toHaveAttribute('aria-pressed', 'true');

    const cupy = page.locator('input[name="backend"][value="cupy"]');
    await cupy.check();
    await expect(cupy).toBeChecked();
    const sklearn = page.locator('input[value="sklearn"]');
    if (await sklearn.count()) {
      await sklearn.check();
      await expect(sklearn).toBeChecked();
    }
    await expect(page.locator('.table-container tbody tr').first()).toBeVisible();

    await page.locator('#env-select').selectOption({ index: 0 });
    await expect(page.getByLabel('Model', { exact: true })).toHaveValue('');
    await expect(page.locator('.scale-chip[aria-pressed="true"]')).toHaveCount(0);
    await expect(page.locator('input[name="backend"][value="all"]')).toBeChecked();
  });

  test('exposes keyboard sorting, disclosure controls, skip link, and lazy chart tables', async ({ page }) => {
    await openProduction(page);
    const skip = page.getByRole('link', { name: 'Skip to benchmark results' });
    await skip.focus();
    await expect(skip).toBeFocused();
    await skip.press('Enter');
    await expect(page).toHaveURL(/#dashboard-main$/);

    const modelHeader = page.getByRole('columnheader', { name: /Sort by Model/ });
    await modelHeader.focus();
    await modelHeader.press('Enter');
    await expect(modelHeader).toHaveAttribute('aria-sort', 'ascending');
    await modelHeader.press('Space');
    await expect(modelHeader).toHaveAttribute('aria-sort', 'descending');

    await expect(page.locator('#timing-chart-data')).toBeVisible();
    await expect(page.locator('#speedup-chart-data')).toBeVisible();
    await expect(page.locator('#timing-chart-data table')).toHaveCount(0);
    await expect(page.locator('#speedup-chart-data table')).toHaveCount(0);

    await page.locator('#timing-chart-data summary').click();
    await expect(page.locator('#timing-chart-data table')).toBeVisible();
    await expect(page.locator('#timing-chart-data tbody tr').first()).toBeVisible();
    await expect(page.locator('#timing-chart-data caption')).toContainText('Full labels');

    await page.locator('#speedup-chart-data summary').click();
    await expect(page.locator('#speedup-chart-data table')).toBeVisible();
    await expect(page.locator('#speedup-chart-data tbody tr').first()).toBeVisible();
    await expect(page.locator('#speedup-chart-data caption')).toContainText('Full labels');

    const selects = page.locator('select');
    for (let i = 0; i < await selects.count(); i += 1) {
      const select = selects.nth(i);
      const id = await select.getAttribute('id');
      const aria = await select.getAttribute('aria-label');
      const labelled = id ? await page.locator(`label[for="${id}"]`).count() : 0;
      expect(Boolean(aria || labelled)).toBeTruthy();
    }
  });

  test('keeps explicit empty states and CV failure disclosure after refresh', async ({ page }) => {
    await openProduction(page);
    await page.getByRole('button', { name: 'None' }).click();
    await expect(page.getByText(/No runs match the current filters/i)).toBeVisible();

    await page.locator('#env-select').selectOption('remote-p100-cv-20260807');
    await page.locator('#cat-linear_models').check();
    await page.locator('[data-metric-scope="cross_validation"]').click();
    await page.getByLabel('Model', { exact: true }).selectOption('LogisticRegressionCV');
    const panelToggle = page.getByRole('button', { name: /Cross-validation Metrics/ });
    const panelBodyId = await panelToggle.getAttribute('aria-controls');
    expect(panelBodyId).toBeTruthy();
    await panelToggle.click();
    await expect(panelToggle).toHaveAttribute('aria-expanded', 'true');
    const failed = page
      .locator(`#${panelBodyId} tr`)
      .filter({ hasText: 'LogisticRegressionCV' })
      .filter({ hasText: 'torch' });
    await expect(failed).toHaveCount(1);
    await expect(failed).toContainText('failed');
    await expect(failed).toContainText('CPU fallback is disabled');

    await page.reload();
    await expect(page.locator('.header')).toBeVisible({ timeout: 15000 });
    await expect(page.locator('#timing-chart')).toBeVisible();
  });

  test('meets the dashboard text contrast contract', async ({ page }) => {
    await openProduction(page);
    const ratios = await page.evaluate(() => {
      function rgb(value: string): number[] {
        const match = value.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (!match) throw new Error(`Unsupported color: ${value}`);
        return match.slice(1, 4).map(Number);
      }
      function luminance(value: string): number {
        const [r, g, b] = rgb(value)
          .map(channel => channel / 255)
          .map(channel =>
            channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
          );
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }
      function contrast(foreground: string, background: string): number {
        const a = luminance(foreground);
        const b = luminance(background);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      }

      const body = getComputedStyle(document.body);
      const header = getComputedStyle(document.querySelector('.header')!);
      const muted = getComputedStyle(document.querySelector('.card-label')!);
      const card = getComputedStyle(document.querySelector('.summary-card')!);
      return {
        body: contrast(body.color, body.backgroundColor),
        header: contrast(header.color, header.backgroundColor),
        muted: contrast(muted.color, card.backgroundColor),
      };
    });
    expect(ratios.body).toBeGreaterThanOrEqual(4.5);
    expect(ratios.header).toBeGreaterThanOrEqual(4.5);
    expect(ratios.muted).toBeGreaterThanOrEqual(4.5);
  });
});
