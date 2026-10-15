import { expect, test } from '@playwright/test';
import { siteRoot } from './site';

for (const language of ['en', 'cn']) {
  test(`renders inline and display LaTeX on current ${language} model pages`, async ({ page }) => {
    for (const route of ['panel/random-effects', 'models/regression/linear-regression', 'models/regression/generalized-linear-model']) {
      await page.goto(`${siteRoot}${language}/${route}`);
      const main = page.getByRole('main');
      await expect(main.locator('mjx-container').first()).toBeVisible({ timeout: 60_000 });
      expect(await main.locator('mjx-container').count()).toBeGreaterThan(5);
      expect(await main.locator('mjx-container[display="true"]').count()).toBeGreaterThan(0);
      expect(await main.locator('mjx-container:not([display="true"])').count()).toBeGreaterThan(0);
      await expect(main).not.toContainText('$$');
    }
  });

  test(`renders the current ${language} solver algorithms and preserved anchors`, async ({ page }) => {
    await page.goto(`${siteRoot}${language}/guides/solver-algorithms`);
    const main = page.getByRole('main');
    await expect(main.getByRole('heading', { name: '1. Proximal IRLS-CD' })).toBeVisible();
    await expect(main.locator('h2').filter({ hasText: /^10\. exact/ })).toBeVisible();
    await expect(main.locator('#quantile-low-level-solvers')).toHaveCount(1);
    expect(await main.locator('mjx-container[display="true"]').count()).toBeGreaterThanOrEqual(20);
    await expect(main).not.toContainText('$$');
  });
}
