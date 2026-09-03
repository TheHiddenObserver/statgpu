import { expect, test } from '@playwright/test';

test('renders inline and display LaTeX on documentation pages', async ({ page }) => {
  await page.goto('/statgpu/en/panel/random-effects');

  const equations = page.locator('mjx-container');
  await expect(equations.first()).toBeVisible({ timeout: 60_000 });
  expect(await equations.count()).toBeGreaterThan(5);
  expect(await page.locator('mjx-container[display="true"]').count()).toBeGreaterThan(0);
  await expect(page.locator('main')).not.toContainText('$$');
});
