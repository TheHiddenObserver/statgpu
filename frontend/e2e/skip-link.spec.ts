import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

test('dashboard HTML has a focusable skip destination before scripts run', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  expect(html.match(/id="dashboard-main"/g)).toHaveLength(1);
  expect(html).toMatch(/<main\b[^>]*id="dashboard-main"[^>]*tabindex="-1"/);
  expect(html).toContain('href="#dashboard-main"');
});

test('skip destination is the same focusable main through loading and filter changes', async ({ page }) => {
  let release: () => void = () => {};
  const waitForData = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/data/benchmark_data.json', async route => {
    await waitForData;
    await route.continue();
  });
  await page.goto('/');
  const main = page.locator('#dashboard-main');
  await expect(main).toHaveCount(1);
  await expect(main).toContainText('Loading benchmark evidence');
  const initialMain = await main.elementHandle();
  const skip = page.getByRole('link', { name: 'Skip to benchmark results' });
  await skip.focus();
  await skip.press('Enter');
  await expect(main).toBeFocused();
  release();
  await expect(page.locator('.header:not(.header-loading)')).toBeVisible({ timeout: 60_000 });
  await expect(main).toHaveCount(1);
  expect(await main.evaluate((element, initial) => element === initial, initialMain)).toBe(true);
  await page.getByRole('button', { name: 'None', exact: true }).click();
  await expect(main).toHaveCount(1);
  expect(await main.evaluate((element, initial) => element === initial, initialMain)).toBe(true);
  await skip.focus();
  await skip.press('Enter');
  await expect(main).toBeFocused();
});

test('failed loading retains the static skip destination', async ({ page }) => {
  await page.route('**/data/benchmark_data.json', route => route.fulfill({ status: 503, body: 'unavailable' }));
  await page.goto('/');
  const main = page.locator('#dashboard-main');
  await expect(main).toHaveCount(1);
  await expect(main).toContainText('Failed to load benchmark data');
  const skip = page.getByRole('link', { name: 'Skip to benchmark results' });
  await skip.focus();
  await skip.press('Enter');
  await expect(main).toBeFocused();
});
